#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Coupled Lattice Boltzmann and Turing/Gray-Scott morphogenesis interactive GUI tool.

pub mod presets;
pub mod render;
pub mod state;
mod ui;

pub use presets::VortexMorphogenesisTheory;
pub use state::{
    VortexBrushMode, VortexDisplayMode, VortexKineticsKind, VortexPresetTag, VortexToolState,
};

use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui::{self, Color32, ColorImage};
use math_explorer::biology::morphogenesis::{GrayScottKinetics, SchnakenbergKinetics};
use math_explorer::vortex_morphogenesis::{
    CoupledVortexMorphogenesis, MorphogenKinetics, VELOCITY_SAFETY_CAP, VortexRegimePreset,
};
use scientific_metadata::theory::TheoryDescribable;
use std::time::Duration;

/// Maximum cooperative per-frame stepping budget (`5 ms`) keeping WASM 60 FPS responsive.
const FRAME_STEP_BUDGET: Duration = Duration::from_millis(5);

#[derive(Clone)]
struct ValidFieldSnapshot {
    u: Vec<f64>,
    v: Vec<f64>,
    state: VortexToolState,
}

/// Interactive laboratory coupling D2Q9 Lattice Boltzmann wakes with 2D reaction-diffusion morphogenesis.
pub struct VortexMorphogenesisTool {
    /// Underlying zero-allocation coupled fluid-morphogenesis engine.
    pub engine: CoupledVortexMorphogenesis,
    /// Active preset badge tag (`Custom` after manual edits).
    pub preset_tag: VortexPresetTag,
    /// Selected canvas painting brush mode.
    pub brush_mode: VortexBrushMode,
    /// Brush disc radius in lattice cells.
    pub brush_radius: usize,
    /// Selected visualization layer.
    pub display_mode: VortexDisplayMode,
    /// Whether live frame stepping is paused.
    pub paused: bool,
    pub(crate) gs_params: (f64, f64),
    pub(crate) sn_params: (f64, f64),
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) pixels: Vec<Color32>,
    pub(crate) dirty: bool,
    pub(crate) last_high_contrast: Option<bool>,
    pub(crate) cursor_cell: (usize, usize),
    pub(crate) last_canvas_id: Option<egui::Id>,
    pub(crate) diverged: bool,
    pub(crate) diagnostic_report: Option<DiagnosticReport>,
    pub(crate) step_count: usize,
    revision: u64,
    last_valid: Option<ValidFieldSnapshot>,
    theory: VortexMorphogenesisTheory,
}

impl Default for VortexMorphogenesisTool {
    fn default() -> Self {
        let engine = CoupledVortexMorphogenesis::from_preset(VortexRegimePreset::KarmanVortexSpots);
        let gs_params = (0.038, 0.060);
        let sn_params = (0.10, 0.90);
        let mut tool = Self {
            cursor_cell: (engine.config.width / 2, engine.config.height / 2),
            engine,
            preset_tag: VortexPresetTag::KarmanVortexSpots,
            brush_mode: VortexBrushMode::Obstacle,
            brush_radius: 3,
            display_mode: VortexDisplayMode::CoupledWake,
            paused: false,
            gs_params,
            sn_params,
            texture: None,
            pixels: Vec::new(),
            dirty: true,
            last_high_contrast: None,
            last_canvas_id: None,
            diverged: false,
            diagnostic_report: None,
            step_count: 0,
            revision: 1,
            last_valid: None,
            theory: VortexMorphogenesisTheory::default(),
        };
        tool.capture_valid_snapshot();
        tool
    }
}

impl VortexMorphogenesisTool {
    pub(crate) fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    pub(crate) fn kinetics_kind(&self) -> VortexKineticsKind {
        match self.engine.config.kinetics {
            MorphogenKinetics::GrayScott(_) => VortexKineticsKind::GrayScott,
            MorphogenKinetics::Schnakenberg(_) => VortexKineticsKind::Schnakenberg,
        }
    }

    /// Switches active reaction kinetics between [`GrayScottKinetics`] and [`SchnakenbergKinetics`].
    pub fn set_kinetics_kind(&mut self, kind: VortexKineticsKind) {
        self.engine.config.kinetics = match kind {
            VortexKineticsKind::GrayScott => {
                MorphogenKinetics::GrayScott(GrayScottKinetics::new(self.gs_params.0, self.gs_params.1))
            }
            VortexKineticsKind::Schnakenberg => MorphogenKinetics::Schnakenberg(
                SchnakenbergKinetics::new(self.sn_params.0, self.sn_params.1),
            ),
        };
        self.engine.reset_homogeneous_morphogens();
        self.engine.transpose_lbm_to_row_major();
        self.preset_tag = VortexPresetTag::Custom;
        self.dirty = true;
        self.bump_revision();
    }

    /// Loads one of the four canonical physical-biological presets.
    pub fn select_regime_preset(&mut self, preset: VortexRegimePreset) {
        let state = VortexToolState::for_preset(preset);
        if let Ok(json) = serde_json::to_string(&state) {
            let _ = self.try_load_state(&json);
        }
    }

    /// Resets the active preset (or re-seeds homogeneous morphogens when customized).
    pub fn reset_active_regime(&mut self) {
        self.diverged = false;
        self.diagnostic_report = None;
        if let Some(regime) = self.preset_tag.as_regime() {
            self.engine.apply_preset(regime);
        } else {
            self.engine.reset_homogeneous_morphogens();
            self.engine.transpose_lbm_to_row_major();
        }
        self.capture_valid_snapshot();
        self.dirty = true;
        self.bump_revision();
    }

    /// Stamps the active obstacle/emitter/erase brush at lattice coordinates `(gx, gy)`.
    pub fn apply_brush_at(&mut self, gx: usize, gy: usize) {
        match self.brush_mode {
            VortexBrushMode::Obstacle => {
                self.engine
                    .paint_obstacle_disc(gx, gy, self.brush_radius, true);
            }
            VortexBrushMode::Emitter => {
                self.engine
                    .paint_emitter_disc(gx, gy, self.brush_radius, true);
            }
            VortexBrushMode::Erase => {
                self.engine
                    .paint_obstacle_disc(gx, gy, self.brush_radius, false);
                self.engine
                    .paint_emitter_disc(gx, gy, self.brush_radius, false);
            }
        }
        self.engine.transpose_lbm_to_row_major();
        self.preset_tag = VortexPresetTag::Custom;
        self.dirty = true;
        self.bump_revision();
    }

    pub(crate) fn screen_to_grid(&self, rect: egui::Rect, pos: egui::Pos2) -> Option<(usize, usize)> {
        if !rect.contains(pos) || rect.width() <= 1.0 || rect.height() <= 1.0 {
            return None;
        }
        let rel_x = ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 0.999_999);
        let rel_y = ((pos.y - rect.min.y) / rect.height()).clamp(0.0, 0.999_999);
        let gx = (rel_x * (self.engine.config.width as f32)) as usize;
        let py = (rel_y * (self.engine.config.height as f32)) as usize;
        let gy = self.engine.config.height.saturating_sub(1).saturating_sub(py);
        Some((gx, gy))
    }

    /// Advances one guarded coupled simulation step, halting and recording diagnostics on non-finite state.
    pub fn step_single_tick(&mut self) {
        if self.diverged {
            return;
        }
        if !self.has_finite_config() || !self.engine.is_finite() {
            self.record_divergence("morphogen_or_velocity");
            return;
        }
        self.capture_valid_snapshot();
        self.engine.step();
        self.step_count += 1;
        self.dirty = true;
        if !self.engine.is_finite() {
            self.record_divergence("coupled_field");
        }
    }

    fn step_within_frame_budget(&mut self) {
        let start = crate::web_io::PlatformInstant::now();
        for _ in 0..2 {
            self.step_single_tick();
            if self.diverged || start.elapsed() >= FRAME_STEP_BUDGET {
                break;
            }
        }
    }

    fn capture_valid_snapshot(&mut self) {
        if !self.has_finite_config() || !self.engine.is_finite() {
            return;
        }
        let state = self.to_tool_state();
        if let Some(snap) = &mut self.last_valid {
            snap.u.clear();
            snap.u.extend_from_slice(self.engine.morphogens.u());
            snap.v.clear();
            snap.v.extend_from_slice(self.engine.morphogens.v());
            snap.state = state;
        } else {
            self.last_valid = Some(ValidFieldSnapshot {
                u: self.engine.morphogens.u().to_vec(),
                v: self.engine.morphogens.v().to_vec(),
                state,
            });
        }
    }

    /// Rolls back morphogen concentrations and parameters to the last verified finite snapshot.
    pub fn rollback_to_last_valid(&mut self) {
        if let Some(snap) = self.last_valid.clone() {
            if let Ok(json) = serde_json::to_string(&snap.state) {
                let _ = self.try_load_state(&json);
            }
            if self.engine.morphogens.u().len() == snap.u.len() {
                self.engine.morphogens.u_mut().copy_from_slice(&snap.u);
                self.engine.morphogens.v_mut().copy_from_slice(&snap.v);
            }
        }
        self.diverged = false;
        self.diagnostic_report = None;
        self.paused = true;
        self.dirty = true;
        self.bump_revision();
    }

    fn record_divergence(&mut self, variable: &str) {
        self.diverged = true;
        self.paused = true;
        self.diagnostic_report = Some(DiagnosticReport::new(
            (self.step_count as f64) * self.engine.config.dt.max(0.0),
            self.step_count,
            variable,
            "Non-finite velocity or morphogen value detected; step halted with snapshot ready for rollback.",
        ));
    }

    fn has_finite_config(&self) -> bool {
        let c = &self.engine.config;
        c.dx.is_finite()
            && c.tau.is_finite()
            && c.inlet_ux.is_finite()
            && c.advection_coupling.is_finite()
            && c.d_u.is_finite()
            && c.d_v.is_finite()
            && c.dt.is_finite()
            && c.emitter_strength.is_finite()
            && self.gs_params.0.is_finite()
            && self.gs_params.1.is_finite()
            && self.sn_params.0.is_finite()
            && self.sn_params.1.is_finite()
    }

    fn to_tool_state(&self) -> VortexToolState {
        let c = &self.engine.config;
        VortexToolState {
            advection_coupling: c.advection_coupling,
            d_u: c.d_u,
            d_v: c.d_v,
            dt: c.dt,
            dx: c.dx,
            emitter_strength: c.emitter_strength,
            fresh_inlet_feed: c.fresh_inlet_feed,
            gs_f: self.gs_params.0,
            gs_k: self.gs_params.1,
            height: c.height,
            inlet_ux: c.inlet_ux,
            kinetics: self.kinetics_kind(),
            paused: self.paused,
            preset: self.preset_tag,
            shear_profile: c.shear_profile,
            sn_a: self.sn_params.0,
            sn_b: self.sn_params.1,
            tau: c.tau,
            width: c.width,
            brush_mode: self.brush_mode,
            brush_radius: self.brush_radius,
            display_mode: self.display_mode,
        }
    }
}

impl FiniteStateGuard for VortexMorphogenesisTool {
    fn is_finite_state(&self) -> bool {
        !self.diverged && self.has_finite_config() && self.engine.is_finite()
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.diagnostic_report.as_ref()
    }

    fn diverged(&self) -> bool {
        self.diverged || !FiniteStateGuard::is_finite_state(self)
    }

    fn auto_scale_step_size(&mut self) {
        self.rollback_to_last_valid();
        self.engine.config.dt = (self.engine.config.dt * 0.5).clamp(0.01, 10.0);
        self.bump_revision();
    }

    fn clamp_parameters(&mut self) {
        let c = &mut self.engine.config;
        c.dx = c.dx.clamp(0.05, 5.0);
        c.tau = c.tau.clamp(0.51, 2.0);
        c.inlet_ux = c.inlet_ux.clamp(-VELOCITY_SAFETY_CAP, VELOCITY_SAFETY_CAP);
        c.advection_coupling = c.advection_coupling.clamp(0.0, 2.0);
        c.d_u = c.d_u.clamp(0.001, 5.0);
        c.d_v = c.d_v.clamp(0.001, 200.0);
        c.dt = c.dt.clamp(0.01, 10.0);
        c.emitter_strength = c.emitter_strength.clamp(0.0, 1.0);
        self.gs_params.0 = self.gs_params.0.clamp(0.0, 0.2);
        self.gs_params.1 = self.gs_params.1.clamp(0.0, 0.2);
        self.sn_params.0 = self.sn_params.0.clamp(0.0, 1.0);
        self.sn_params.1 = self.sn_params.1.clamp(0.0, 2.0);
        self.rollback_to_last_valid();
    }

    fn reset_parameters(&mut self) {
        self.select_regime_preset(VortexRegimePreset::KarmanVortexSpots);
        self.diverged = false;
        self.diagnostic_report = None;
    }
}

impl InteractiveTool for VortexMorphogenesisTool {
    fn name(&self) -> &'static str {
        "Vortex Morphogenesis"
    }

    fn domain(&self) -> &'static str {
        "biology"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::raster_only()
    }

    fn export_raster_image(&self) -> Option<ColorImage> {
        let hc = self.last_high_contrast.unwrap_or(false);
        Some(render::build_export_image(&self.engine, self.display_mode, hc))
    }

    fn step_simulation(&mut self) {
        self.step_single_tick();
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        if !self.paused && !FiniteStateGuard::diverged(self) {
            self.step_within_frame_budget();
            ui.ctx().request_repaint();
        }
        self.draw_controls(ui);
        self.draw_canvas(ui);
    }

    fn state_revision(&self) -> u64 {
        self.revision
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.last_canvas_id
    }

    fn has_finite_parameters(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }

    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.to_tool_state()).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: VortexToolState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid Vortex Morphogenesis state: {e}"))?;
        parsed.validate()?;

        let cfg = parsed.to_engine_config();
        if let Some(regime) = parsed.preset.as_regime() {
            if parsed.width != self.engine.config.width || parsed.height != self.engine.config.height
            {
                self.engine = CoupledVortexMorphogenesis::new(cfg);
            }
            self.engine.apply_preset(regime);
            self.engine.config = cfg;
        } else if parsed.width != self.engine.config.width
            || parsed.height != self.engine.config.height
        {
            self.engine = CoupledVortexMorphogenesis::new(cfg);
        } else {
            self.engine.config = cfg;
        }
        self.engine.transpose_lbm_to_row_major();

        self.gs_params = (parsed.gs_f, parsed.gs_k);
        self.sn_params = (parsed.sn_a, parsed.sn_b);
        self.preset_tag = parsed.preset;
        self.paused = parsed.paused;
        self.brush_mode = parsed.brush_mode;
        self.brush_radius = parsed.brush_radius;
        self.display_mode = parsed.display_mode;
        self.diverged = false;
        self.diagnostic_report = None;
        self.dirty = true;
        self.capture_valid_snapshot();
        self.bump_revision();
        Ok(())
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

#[cfg(test)]
mod tests;

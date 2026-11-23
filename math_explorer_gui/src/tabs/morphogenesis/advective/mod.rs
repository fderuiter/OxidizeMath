//! Co-stepped 2D D2Q9 Lattice Boltzmann + Schnakenberg / Gray-Scott advective morphogenesis studio.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod canvas;
pub mod engine;
pub mod presets;
#[cfg(test)]
mod tests;
pub mod ui;

pub use engine::AdvectiveMorphogenesisEngine;
pub use presets::{AdvectiveKineticsPreset, AdvectiveMorphogenesisTheory, AdvectiveToolState};

use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{
    decode_mask_hex, encode_mask_hex, take_advective_morphogenesis_handoff, BrushMode,
    ExportCapabilities, FairingStamp, InteractiveTool, ToolMetadata, ToolStateBounds, LBM_CORE_X,
    LBM_CORE_Y, LBM_HEIGHT, LBM_WIDTH,
};
use eframe::egui;
use eframe::egui::ColorImage;
use math_commons::math_kernel::types::flatten_2d_index;
use scientific_metadata::theory::TheoryDescribable;
use std::hash::{Hash, Hasher};

/// Interactive studio co-stepping `LatticeBoltzmannD2Q9` with `TuringSystem` under 2D upwind advection.
pub struct AdvectiveMorphogenesisTool {
    /// Current serializable parameter and mask configuration.
    pub cfg: AdvectiveToolState,
    /// Preallocated zero-allocation coupled fluid + reaction-diffusion engine.
    pub engine: AdvectiveMorphogenesisEngine,
    theory: AdvectiveMorphogenesisTheory,
    pixels: Vec<egui::Color32>,
    texture: Option<egui::TextureHandle>,
    texture_dirty: bool,
    pub(crate) cursor: [usize; 2],
    pub(crate) last_drag_plot_pt: Option<[f64; 2]>,
    pub(crate) focused_canvas: Option<egui::Id>,
    pub(crate) high_contrast: bool,
    diagnostic: Option<DiagnosticReport>,
}

impl Default for AdvectiveMorphogenesisTool {
    fn default() -> Self {
        let cfg = AdvectiveToolState::default();
        let engine = AdvectiveMorphogenesisEngine::new(&cfg);
        let mut tool = Self {
            cfg,
            engine,
            theory: AdvectiveMorphogenesisTheory::default(),
            pixels: vec![egui::Color32::BLACK; LBM_WIDTH * LBM_HEIGHT],
            texture: None,
            texture_dirty: true,
            cursor: [LBM_CORE_X as usize, LBM_CORE_Y as usize],
            last_drag_plot_pt: None,
            focused_canvas: None,
            high_contrast: false,
            diagnostic: None,
        };
        tool.refresh_pixels_in_place();
        tool
    }
}

impl AdvectiveMorphogenesisTool {
    /// Advances a single coupled D2Q9 + reaction-diffusion step and updates the raster preview.
    pub fn step_once(&mut self) {
        self.engine.step_coupled(&self.cfg);
        if !self.fields_are_finite() {
            self.diagnostic = Some(DiagnosticReport {
                step_index: self.engine.metrics.step_count as usize,
                t_diverged: self.engine.metrics.step_count as f64,
                variable: "morphogen_or_velocity".to_string(),
                message: "Non-finite state encountered during coupled step; parameters clamped."
                    .to_string(),
            });
            FiniteStateGuard::clamp_parameters(self);
        } else {
            self.diagnostic = None;
        }
        self.refresh_pixels_in_place();
    }

    /// Switches to a built-in reaction-diffusion kinetics preset and reseeds morphogen nuclei.
    pub fn apply_kinetics_preset(&mut self, preset: AdvectiveKineticsPreset) {
        let (a, b, f, k, d_u, d_v, dt, peclet) = preset.default_parameters();
        self.cfg.preset = preset;
        self.cfg.a = a;
        self.cfg.b = b;
        self.cfg.f = f;
        self.cfg.k = k;
        self.cfg.d_u = d_u;
        self.cfg.d_v = d_v;
        self.cfg.dt = dt;
        self.cfg.peclet = peclet;
        self.engine.reseed_morphogens(&self.cfg);
        self.engine.refresh_metrics(&self.cfg);
        self.refresh_pixels_in_place();
    }

    /// Stamps a built-in aerodynamic fairing onto the `100 x 50` lattice.
    pub fn apply_fairing_stamp(&mut self, stamp: FairingStamp) {
        self.cfg.stamp = stamp;
        self.cfg.mask_hex = None;
        stamp.write_mask(&mut self.engine.mask);
        self.engine.sync_obstacle_mask(&self.cfg);
        self.engine.refresh_metrics(&self.cfg);
        self.refresh_pixels_in_place();
    }

    pub(crate) fn sync_custom_mask_mutation(&mut self) {
        self.engine.sync_obstacle_mask(&self.cfg);
        self.cfg.mask_hex = Some(encode_mask_hex(&self.engine.mask));
        self.engine.refresh_metrics(&self.cfg);
        self.refresh_pixels_in_place();
    }

    /// Applies any pending cross-tab handoff from `FluidDynamicsTab`.
    pub fn poll_incoming_fluid_handoff(&mut self, _ctx: &egui::Context) -> bool {
        let Some(handoff) = take_advective_morphogenesis_handoff() else {
            return false;
        };
        self.cfg.viscosity = handoff.viscosity.clamp(0.018, 0.25);
        if let Some(hex) = handoff.mask_hex {
            if decode_mask_hex(&hex, &mut self.engine.mask).is_ok() {
                self.cfg.mask_hex = Some(hex);
            }
        }
        self.engine.reset_fluid_wake(self.cfg.viscosity);
        self.engine.reseed_morphogens(&self.cfg);
        self.engine.refresh_metrics(&self.cfg);
        self.refresh_pixels_in_place();
        true
    }

    /// Collects streamwise centerline morphogen and shear-layer vorticity datasets for `.csv` export.
    #[must_use]
    pub fn collect_wake_profile_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let field = self.engine.active_morphogen(self.cfg.preset);
        let secondary = self.engine.secondary_morphogen(self.cfg.preset);
        let y_mid = LBM_CORE_Y as usize;
        let y_shear = (y_mid + 5).min(LBM_HEIGHT - 1);

        let mut centerline_primary = Vec::with_capacity(LBM_WIDTH);
        let mut shear_vorticity = Vec::with_capacity(LBM_WIDTH);
        let mut centerline_secondary = Vec::with_capacity(LBM_WIDTH);

        for x in 0..LBM_WIDTH {
            let idx_mid = flatten_2d_index(x, y_mid, LBM_WIDTH);
            let idx_sh = flatten_2d_index(x, y_shear, LBM_WIDTH);
            centerline_primary.push([x as f64, field[idx_mid]]);
            shear_vorticity.push([x as f64, self.engine.vorticity[idx_sh] * 10.0]);
            centerline_secondary.push([x as f64, secondary[idx_mid]]);
        }
        vec![
            ("centerline_morphogen".to_string(), centerline_primary),
            ("shear_layer_vorticity_x10".to_string(), shear_vorticity),
            (
                "centerline_secondary_species".to_string(),
                centerline_secondary,
            ),
        ]
    }

    fn fields_are_finite(&self) -> bool {
        self.engine.schnak.u().iter().all(|v| v.is_finite())
            && self.engine.schnak.v().iter().all(|v| v.is_finite())
            && self.engine.gray_scott.u().iter().all(|v| v.is_finite())
            && self.engine.gray_scott.v().iter().all(|v| v.is_finite())
            && self.engine.vorticity.iter().all(|v| v.is_finite())
    }

    #[cfg(test)]
    pub(crate) fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
    }
}

impl FiniteStateGuard for AdvectiveMorphogenesisTool {
    fn is_finite_state(&self) -> bool {
        self.cfg.validate().is_ok() && self.fields_are_finite()
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.diagnostic.as_ref()
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }

    fn auto_scale_step_size(&mut self) {
        let cfl = self.engine.metrics.cfl_limit.clamp(0.001, 1.0);
        self.cfg.dt = (self.cfg.dt * 0.5).clamp(0.001, cfl);
        self.engine.refresh_metrics(&self.cfg);
    }

    fn clamp_parameters(&mut self) {
        let sanitize = |v: f64, lo: f64, hi: f64, fallback: f64| {
            if v.is_finite() {
                v.clamp(lo, hi)
            } else {
                fallback
            }
        };
        self.cfg.viscosity = sanitize(self.cfg.viscosity, 0.018, 0.25, 0.020);
        self.cfg.peclet = sanitize(self.cfg.peclet, 0.0, 30.0, 10.0);
        self.cfg.dt = sanitize(self.cfg.dt, 0.001, 1.0, 0.03);
        self.cfg.a = sanitize(self.cfg.a, 0.01, 0.60, 0.10);
        self.cfg.b = sanitize(self.cfg.b, 0.10, 2.00, 0.90);
        self.cfg.f = sanitize(self.cfg.f, 0.01, 0.10, 0.0367);
        self.cfg.k = sanitize(self.cfg.k, 0.03, 0.08, 0.0649);
        self.cfg.d_u = sanitize(self.cfg.d_u, 0.02, 2.00, 0.15);
        self.cfg.d_v = sanitize(self.cfg.d_v, 0.02, 12.00, 3.50);
        self.cfg.brush_radius = sanitize(self.cfg.brush_radius, 0.0, 8.0, 2.0);
        if !matches!(self.cfg.brush_mode, BrushMode::Solid | BrushMode::Erase) {
            self.cfg.brush_mode = BrushMode::Solid;
        }
        if !self.fields_are_finite() {
            self.engine.reset_fluid_wake(self.cfg.viscosity);
            self.engine.reseed_morphogens(&self.cfg);
        }
        self.engine.sync_parameters(&self.cfg);
        self.engine.refresh_metrics(&self.cfg);
        self.refresh_pixels_in_place();
    }

    fn reset_parameters(&mut self) {
        self.cfg = AdvectiveToolState::default();
        self.diagnostic = None;
        self.cfg.stamp.write_mask(&mut self.engine.mask);
        self.engine.reset_fluid_wake(self.cfg.viscosity);
        self.engine.reseed_morphogens(&self.cfg);
        self.engine.refresh_metrics(&self.cfg);
        self.refresh_pixels_in_place();
    }
}

impl InteractiveTool for AdvectiveMorphogenesisTool {
    fn name(&self) -> &'static str {
        "Advective Morphogenesis Studio"
    }

    fn domain(&self) -> &'static str {
        "biology"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        Vec::new()
    }

    fn export_raster_image(&self) -> Option<ColorImage> {
        (self.pixels.len() == LBM_WIDTH * LBM_HEIGHT).then(|| ColorImage {
            size: [LBM_WIDTH, LBM_HEIGHT],
            pixels: self.pixels.clone(),
            ..Default::default()
        })
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.collect_wake_profile_datasets()
    }

    fn step_simulation(&mut self) {
        self.step_once();
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_studio(ui);
    }

    fn has_finite_parameters(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }

    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
    }

    fn state_bounds(&self) -> ToolStateBounds {
        crate::url_state::URL_JSON_BOUNDS
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.cfg.preset.hash(&mut h);
        self.cfg.viscosity.to_bits().hash(&mut h);
        self.cfg.peclet.to_bits().hash(&mut h);
        self.cfg.dt.to_bits().hash(&mut h);
        self.cfg.a.to_bits().hash(&mut h);
        self.cfg.b.to_bits().hash(&mut h);
        self.cfg.f.to_bits().hash(&mut h);
        self.cfg.k.to_bits().hash(&mut h);
        self.cfg.d_u.to_bits().hash(&mut h);
        self.cfg.d_v.to_bits().hash(&mut h);
        self.cfg.stamp.hash(&mut h);
        self.cfg.mask_hex.hash(&mut h);
        self.cfg.view_mode.hash(&mut h);
        self.cfg.brush_radius.to_bits().hash(&mut h);
        self.cfg.brush_mode.hash(&mut h);
        self.cfg.rng_seed.hash(&mut h);
        self.cfg.playing.hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.cfg).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: AdvectiveToolState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid AdvectiveMorphogenesis state: {e}"))?;
        parsed.validate()?;
        if let Some(hex) = &parsed.mask_hex {
            decode_mask_hex(hex, &mut self.engine.mask)?;
        } else {
            parsed.stamp.write_mask(&mut self.engine.mask);
        }
        self.cfg = parsed;
        self.diagnostic = None;
        self.engine.reset_fluid_wake(self.cfg.viscosity);
        self.engine.reseed_morphogens(&self.cfg);
        self.engine.refresh_metrics(&self.cfg);
        self.refresh_pixels_in_place();
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

inventory::submit! {
    ToolMetadata {
        id: "AdvectiveMorphogenesisTool",
        display_name: "Advective Morphogenesis Studio",
        domain: "biology",
        tags: &[
            "morphogenesis",
            "advection",
            "peclet",
            "lattice boltzmann",
            "vortex street",
            "turing",
            "gray-scott",
        ],
        build: || Box::new(AdvectiveMorphogenesisTool::default()),
    }
}

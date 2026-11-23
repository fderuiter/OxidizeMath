#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:quantum_mechanics]

use eframe::egui;
use egui_plot::ExportPrimitive;
use math_explorer::physics::quantum::gross_pitaevskii::{
    GpeInvariants2D, GpeParams2D, GpeTrapConfig, GpeTrapKind, GrossPitaevskiiSolver2D,
    VortexChargeSummary,
};

use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};

mod canvas;
mod presets;
mod ui;

#[cfg(test)]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

pub(crate) const GRID_DIM: usize = 128;
pub(crate) const DOMAIN_SPAN: f64 = 20.0;
pub(crate) const DENSITY_CUTOFF_RATIO: f64 = 0.025;
pub(crate) const MAX_GPE_TELEMETRY: usize = 256;

pub(crate) const COUPLING_G_MIN: f64 = 0.0;
pub(crate) const COUPLING_G_MAX: f64 = 600.0;
pub(crate) const OMEGA_MIN: f64 = 0.0;
pub(crate) const OMEGA_MAX: f64 = 3.0;
pub(crate) const OBSTACLE_H_MIN: f64 = 0.0;
pub(crate) const OBSTACLE_H_MAX: f64 = 40.0;
pub(crate) const OBSTACLE_W_MIN: f64 = 0.25;
pub(crate) const OBSTACLE_W_MAX: f64 = 4.0;
pub(crate) const HEALING_MIN: f64 = 0.15;
pub(crate) const HEALING_MAX: f64 = 1.5;
pub(crate) const GPE_DT_MIN: f64 = 2.0e-4;
pub(crate) const GPE_DT_MAX: f64 = 0.015;
pub(crate) const CURSOR_BOUND: f64 = 8.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum SuperfluidPreset {
    VortexDipole,
    VortexQuadrupole,
    CoRotatingPair,
    ObstaclePinningCooling,
}

impl SuperfluidPreset {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::VortexDipole => "Superfluid Vortex Dipole (+/-2pi)",
            Self::VortexQuadrupole => "Quantized Vortex Quadrupole",
            Self::CoRotatingPair => "Co-Rotating Vortex Pair (+2pi/+2pi)",
            Self::ObstaclePinningCooling => "Obstacle Pinning & Cooling Ground State",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum CondensateViewMode {
    SuperfluidDensity,
    CyclicPhase,
    WcagHighContrastDensity,
}

impl CondensateViewMode {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::SuperfluidDensity => "Condensate Density |ψ|²",
            Self::CyclicPhase => "Cyclic Phase arg(ψ) ∈ [-π, π]",
            Self::WcagHighContrastDensity => "WCAG 2.2 High-Contrast |ψ|²",
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SuperfluidVortexState {
    pub preset: SuperfluidPreset,
    pub trap_kind: GpeTrapKind,
    pub view_mode: CondensateViewMode,
    pub interaction_g: f64,
    pub imaginary_time_cooling: bool,
    pub omega_x: f64,
    pub omega_y: f64,
    pub obstacle_height: f64,
    pub obstacle_width: f64,
    pub dipole_separation: f64,
    pub healing_length: f64,
    pub time_step: f64,
    pub cursor_x: f64,
    pub cursor_y: f64,
    pub stamp_charge: i32,
    pub show_vortex_markers: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct GpeTelemetryPoint {
    pub time: f64,
    pub norm_error: f64,
    pub energy: f64,
}

pub struct SuperfluidVortexTool {
    pub(crate) state: SuperfluidVortexState,
    pub(crate) solver: GrossPitaevskiiSolver2D,
    pub(crate) initial_invariants: GpeInvariants2D,
    pub(crate) current_invariants: GpeInvariants2D,
    pub(crate) charge_summary: VortexChargeSummary,
    pub(crate) rgba_pixels: Vec<egui::Color32>,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) telemetry: Vec<GpeTelemetryPoint>,
    pub(crate) paused: bool,
    pub(crate) steps_per_frame: usize,
    pub(crate) last_step_ms: f64,
    pub(crate) state_rev: u64,
    pub(crate) focused_canvas: Option<egui::Id>,
}

impl Default for SuperfluidVortexTool {
    fn default() -> Self {
        let mut state = presets::default_state_for_preset(SuperfluidPreset::VortexDipole);
        state.healing_length = 0.36;
        let solver = GrossPitaevskiiSolver2D::new(GRID_DIM, GRID_DIM, DOMAIN_SPAN, DOMAIN_SPAN);
        let mut tool = Self {
            state,
            solver,
            initial_invariants: GpeInvariants2D::default(),
            current_invariants: GpeInvariants2D::default(),
            charge_summary: VortexChargeSummary::default(),
            rgba_pixels: vec![egui::Color32::BLACK; GRID_DIM * GRID_DIM],
            texture: None,
            telemetry: Vec::with_capacity(MAX_GPE_TELEMETRY),
            paused: false,
            steps_per_frame: 2,
            last_step_ms: 0.0,
            state_rev: 1,
            focused_canvas: None,
        };
        tool.reseed_condensate();
        tool
    }
}

impl SuperfluidVortexTool {
    pub(crate) fn trap_config(&self) -> GpeTrapConfig {
        GpeTrapConfig {
            kind: self.state.trap_kind,
            omega_x: self.state.omega_x,
            omega_y: self.state.omega_y,
            obstacle_height: self.state.obstacle_height,
            obstacle_width: self.state.obstacle_width,
        }
    }

    pub(crate) fn gpe_params(&self) -> GpeParams2D {
        GpeParams2D {
            interaction_g: self.state.interaction_g,
            imaginary_time_cooling: self.state.imaginary_time_cooling,
        }
    }

    pub(crate) fn apply_preset(&mut self, preset: SuperfluidPreset) {
        self.state = presets::default_state_for_preset(preset);
        self.state_rev = self.state_rev.wrapping_add(1);
        self.reseed_condensate();
    }

    pub(crate) fn reseed_condensate(&mut self) {
        self.solver.configure_trap(self.trap_config());
        self.solver.seed_condensate(self.state.interaction_g, 1.0);
        presets::imprint_preset_vortices(&mut self.solver, &self.state);
        self.telemetry.clear();
        self.initial_invariants = self.solver.compute_invariants(self.state.interaction_g);
        self.current_invariants = self.initial_invariants;
        self.charge_summary = self.solver.detect_vortices(DENSITY_CUTOFF_RATIO);
        self.record_telemetry_sample();
    }

    pub(crate) fn update_trap_in_place(&mut self) {
        self.solver.configure_trap(self.trap_config());
        self.initial_invariants = self.solver.compute_invariants(self.state.interaction_g);
        self.refresh_diagnostics();
    }

    pub(crate) fn stamp_vortex_at(&mut self, x: f64, y: f64, charge: i32) {
        self.solver
            .imprint_vortex(x, y, charge, self.state.healing_length);
        self.initial_invariants = self.solver.compute_invariants(self.state.interaction_g);
        self.refresh_diagnostics();
    }

    pub(crate) fn advance_physics_step(&mut self) {
        let start = crate::web_io::PlatformInstant::now();
        let params = self.gpe_params();
        self.solver.step(&params, self.state.time_step);
        self.last_step_ms = start.elapsed().as_secs_f64() * 1_000.0;
    }

    pub(crate) fn refresh_diagnostics(&mut self) {
        self.charge_summary = self.solver.detect_vortices(DENSITY_CUTOFF_RATIO);
        self.current_invariants = self.solver.compute_invariants(self.state.interaction_g);
        self.record_telemetry_sample();
    }

    fn record_telemetry_sample(&mut self) {
        if self.telemetry.len() >= MAX_GPE_TELEMETRY {
            self.telemetry.remove(0);
        }
        self.telemetry.push(GpeTelemetryPoint {
            time: self.solver.time(),
            norm_error: (self.current_invariants.norm - self.initial_invariants.norm).abs(),
            energy: self.current_invariants.energy,
        });
    }
}

impl InteractiveTool for SuperfluidVortexTool {
    fn name(&self) -> &'static str {
        "Superfluid Vortex & GPE Studio"
    }

    fn domain(&self) -> &'static str {
        "quantum"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn state_revision(&self) -> u64 {
        self.state_rev
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds {
            max_bytes: 8_192,
            max_depth: 6,
            max_array_len: 32,
        }
    }

    fn has_finite_parameters(&self) -> bool {
        presets::is_valid_state(&self.state)
    }

    fn clamp_parameters(&mut self) -> bool {
        presets::clamp_state_fields(&mut self.state);
        self.state_rev = self.state_rev.wrapping_add(1);
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: SuperfluidVortexState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid Superfluid Vortex state JSON: {e}"))?;
        if !presets::is_valid_state(&parsed) {
            return Err("Superfluid Vortex parameters out of finite valid bounds".into());
        }
        self.state = parsed;
        self.state_rev = self.state_rev.wrapping_add(1);
        self.reseed_condensate();
        Ok(())
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_raster_image(&self) -> Option<egui::ColorImage> {
        Some(self.build_export_color_image())
    }

    fn export_plot_primitives(&self) -> Vec<ExportPrimitive> {
        self.build_export_primitives()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.build_export_datasets()
    }

    fn step_simulation(&mut self) {
        self.advance_physics_step();
        self.refresh_diagnostics();
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        self.draw_controls_panel(parent_ui);
        if !self.paused {
            for _ in 0..self.steps_per_frame {
                self.advance_physics_step();
            }
            self.refresh_diagnostics();
            parent_ui.ctx().request_repaint();
        }
        self.draw_central_workspace(parent_ui);
    }
}

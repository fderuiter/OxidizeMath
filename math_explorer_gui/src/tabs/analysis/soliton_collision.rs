#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:advanced_linear_algebra]
// [cite:graph_parameters_rust]

use eframe::egui;
use egui_plot::ExportPrimitive;
use math_explorer::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::analysis::pde::kdv::{
    DetectedPulsePeak, KdvEquationMode, KdvParams, KdvSolver1D, SolitonInvariants, SolitonSpec,
};

use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};

mod canvas;
mod presets;
mod ui;

#[cfg(test)]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

pub(crate) const GRID_POINTS: usize = 512;
pub(crate) const WATERFALL_WIDTH: usize = 128;
pub(crate) const WATERFALL_ROWS: usize = 128;
pub(crate) const MAX_TELEMETRY_SAMPLES: usize = 256;

pub(crate) const DOMAIN_LEN_MIN: f64 = 16.0;
pub(crate) const DOMAIN_LEN_MAX: f64 = 80.0;
pub(crate) const SPEED_C_MIN: f64 = 0.5;
pub(crate) const SPEED_C_MAX: f64 = 24.0;
pub(crate) const POS_X_MIN: f64 = -36.0;
pub(crate) const POS_X_MAX: f64 = 36.0;
pub(crate) const DT_MIN: f64 = 5.0e-4;
pub(crate) const DT_MAX: f64 = 0.04;
pub(crate) const NLS_COUPLING_MIN: f64 = -4.0;
pub(crate) const NLS_COUPLING_MAX: f64 = 4.0;
pub(crate) const COSINE_AMP_MIN: f64 = 0.2;
pub(crate) const COSINE_AMP_MAX: f64 = 6.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum SolitonPreset {
    TwoSolitonOvertaking,
    LinearAiryCollapse,
    ZabuskyKruskalFission,
    CubicNlsEnvelopeCollision,
}

impl SolitonPreset {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::TwoSolitonOvertaking => "Two-Soliton Overtaking Collision",
            Self::LinearAiryCollapse => "Linear Airy Dispersion Collapse",
            Self::ZabuskyKruskalFission => "Zabusky-Kruskal Soliton Fission",
            Self::CubicNlsEnvelopeCollision => "Cubic NLS Bright Envelope Collision",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum SolitonColormap {
    PerceptualThermal,
    BipolarWave,
    WcagHighContrast,
}

impl SolitonColormap {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::PerceptualThermal => "Perceptual Thermal",
            Self::BipolarWave => "Bipolar Wave Amplitude",
            Self::WcagHighContrast => "WCAG 2.2 High Contrast",
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SolitonCollisionState {
    pub preset: SolitonPreset,
    pub mode: KdvEquationMode,
    pub colormap: SolitonColormap,
    pub nonlinear_enabled: bool,
    pub domain_length: f64,
    pub fast_speed_c1: f64,
    pub slow_speed_c2: f64,
    pub fast_x0: f64,
    pub slow_x0: f64,
    pub pulse_x0: f64,
    pub pulse_speed_c: f64,
    pub cosine_amplitude: f64,
    pub cosine_harmonic: usize,
    pub nls_coupling: f64,
    pub time_step: f64,
    pub show_initial_overlay: bool,
    pub show_ballistic_markers: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TelemetryPoint {
    pub time: f64,
    pub max_drift: f64,
    pub peak_ratio: f64,
}

pub struct SolitonCollisionTool {
    pub(crate) state: SolitonCollisionState,
    pub(crate) solver: KdvSolver1D,
    pub(crate) initial_invariants: SolitonInvariants,
    pub(crate) current_invariants: SolitonInvariants,
    pub(crate) initial_peak: f64,
    pub(crate) top_peaks: (DetectedPulsePeak, DetectedPulsePeak),
    pub(crate) measured_shifts: (f64, f64),
    pub(crate) waterfall_ring: Vec<f64>,
    pub(crate) waterfall_head: usize,
    pub(crate) rgba_pixels: Vec<egui::Color32>,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) telemetry: Vec<TelemetryPoint>,
    pub(crate) paused: bool,
    pub(crate) steps_per_frame: usize,
    pub(crate) last_substeps: usize,
    pub(crate) last_step_ms: f64,
    pub(crate) state_rev: u64,
}

impl Default for SolitonCollisionTool {
    fn default() -> Self {
        let mut state = presets::default_state_for_preset(SolitonPreset::TwoSolitonOvertaking);
        state.pulse_speed_c = 9.0;
        let solver = KdvSolver1D::new(GRID_POINTS, state.domain_length);
        let mut tool = Self {
            state,
            solver,
            initial_invariants: SolitonInvariants::default(),
            current_invariants: SolitonInvariants::default(),
            initial_peak: 1.0,
            top_peaks: (DetectedPulsePeak::default(), DetectedPulsePeak::default()),
            measured_shifts: (0.0, 0.0),
            waterfall_ring: vec![0.0; WATERFALL_WIDTH * WATERFALL_ROWS],
            waterfall_head: 0,
            rgba_pixels: vec![egui::Color32::BLACK; WATERFALL_WIDTH * WATERFALL_ROWS],
            texture: None,
            telemetry: Vec::with_capacity(MAX_TELEMETRY_SAMPLES),
            paused: false,
            steps_per_frame: 2,
            last_substeps: 1,
            last_step_ms: 0.0,
            state_rev: 1,
        };
        tool.reseed_from_state();
        tool
    }
}

impl SolitonCollisionTool {
    pub(crate) fn pde_params(&self) -> KdvParams {
        KdvParams {
            mode: self.state.mode,
            nonlinear_enabled: self.state.nonlinear_enabled,
            nls_coupling: self.state.nls_coupling,
        }
    }

    pub(crate) fn apply_preset(&mut self, preset: SolitonPreset) {
        self.state = presets::default_state_for_preset(preset);
        self.state_rev = self.state_rev.wrapping_add(1);
        self.reseed_from_state();
    }

    pub(crate) fn reseed_from_state(&mut self) {
        if (self.solver.length() - self.state.domain_length).abs() > 1.0e-9 {
            self.solver = KdvSolver1D::new(GRID_POINTS, self.state.domain_length);
        }
        match self.state.preset {
            SolitonPreset::ZabuskyKruskalFission => {
                self.solver
                    .seed_cosine_wave(self.state.cosine_amplitude, self.state.cosine_harmonic);
            }
            SolitonPreset::CubicNlsEnvelopeCollision => {
                self.seed_nls_state();
            }
            SolitonPreset::TwoSolitonOvertaking | SolitonPreset::LinearAiryCollapse => {
                if self.state.mode == KdvEquationMode::CubicNls1D {
                    self.seed_nls_state();
                } else {
                    self.solver.seed_kdv_solitons(&[
                        SolitonSpec {
                            speed_c: self.state.fast_speed_c1,
                            x0: self.state.fast_x0,
                        },
                        SolitonSpec {
                            speed_c: self.state.slow_speed_c2,
                            x0: self.state.slow_x0,
                        },
                    ]);
                }
            }
        }
        self.waterfall_ring.fill(0.0);
        self.waterfall_head = 0;
        self.telemetry.clear();
        let params = self.pde_params();
        self.initial_invariants = self.solver.compute_invariants(&params);
        self.current_invariants = self.initial_invariants;
        self.initial_peak = self
            .solver
            .field()
            .iter()
            .copied()
            .fold(1.0e-9_f64, |m, v| m.max(v.abs()));
        self.push_waterfall_row();
        self.refresh_diagnostics();
    }

    fn seed_nls_state(&mut self) {
        let eta1 = (0.5 * self.state.fast_speed_c1).sqrt().clamp(0.4, 3.5);
        let eta2 = (0.5 * self.state.slow_speed_c2).sqrt().clamp(0.4, 3.5);
        self.solver.seed_nls_bright_solitons(&[
            (eta1, 2.2, self.state.fast_x0),
            (eta2, -1.8, self.state.slow_x0),
        ]);
    }

    pub(crate) fn advance_physics_step(&mut self) {
        let start = crate::web_io::PlatformInstant::now();
        let params = self.pde_params();
        self.last_substeps = self.solver.step(&params, self.state.time_step);
        self.push_waterfall_row();
        self.last_step_ms = start.elapsed().as_secs_f64() * 1_000.0;
    }

    pub(crate) fn refresh_diagnostics(&mut self) {
        let params = self.pde_params();
        self.current_invariants = self.solver.compute_invariants(&params);
        self.top_peaks = self.solver.locate_top_two_peaks(self.state.mode);
        let (c_fast, x_fast, c_slow, x_slow) =
            if self.state.fast_speed_c1 >= self.state.slow_speed_c2 {
                (
                    self.state.fast_speed_c1,
                    self.state.fast_x0,
                    self.state.slow_speed_c2,
                    self.state.slow_x0,
                )
            } else {
                (
                    self.state.slow_speed_c2,
                    self.state.slow_x0,
                    self.state.fast_speed_c1,
                    self.state.fast_x0,
                )
            };
        self.measured_shifts = self
            .solver
            .measure_two_soliton_phase_shifts(c_fast, x_fast, c_slow, x_slow);

        let drift = self
            .current_invariants
            .relative_drift(self.initial_invariants);
        let max_drift = drift[0].max(drift[1]).max(drift[2]);
        let peak_now = self
            .solver
            .field()
            .iter()
            .copied()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        if self.telemetry.len() >= MAX_TELEMETRY_SAMPLES {
            self.telemetry.remove(0);
        }
        self.telemetry.push(TelemetryPoint {
            time: self.solver.time(),
            max_drift,
            peak_ratio: peak_now / self.initial_peak.max(1.0e-9),
        });
    }

    pub(crate) fn push_waterfall_row(&mut self) {
        let field = self.solver.field();
        let stride = GRID_POINTS / WATERFALL_WIDTH;
        let row = self.waterfall_head % WATERFALL_ROWS;
        for col in 0..WATERFALL_WIDTH {
            let sample = field[col * stride];
            let idx = flatten_2d_index(col, row, WATERFALL_WIDTH);
            self.waterfall_ring[idx] = if sample.is_finite() { sample } else { 0.0 };
        }
        self.waterfall_head = self.waterfall_head.wrapping_add(1);
    }

    pub(crate) fn reanchor_invariant_baseline(&mut self) {
        let params = self.pde_params();
        self.initial_invariants = self.solver.compute_invariants(&params);
        self.current_invariants = self.initial_invariants;
    }
}

impl InteractiveTool for SolitonCollisionTool {
    fn name(&self) -> &'static str {
        "KdV Soliton & NLS Studio"
    }

    fn domain(&self) -> &'static str {
        "analysis"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn state_revision(&self) -> u64 {
        self.state_rev
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
        let parsed: SolitonCollisionState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid KdV Soliton state JSON: {e}"))?;
        if !presets::is_valid_state(&parsed) {
            return Err("KdV Soliton parameters out of finite valid bounds".into());
        }
        self.state = parsed;
        self.state_rev = self.state_rev.wrapping_add(1);
        self.reseed_from_state();
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

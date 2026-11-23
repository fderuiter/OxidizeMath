//! Interactive 60 GHz MIMO super-resolution and respiratory gating twin for LINAC beams.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod canvas;
pub mod pipeline;
pub mod spectra;
pub mod state;
#[cfg(test)]
mod tests;
pub mod ui;

use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use eframe::egui;
use math_explorer::physics::medical::radar_gating::BeamState;
use pipeline::RadarPipeline;
use state::RadarGatingState;
use std::collections::HashMap;

/// Interactive 60 GHz MIMO FMCW super-resolution and Kalman respiratory gating twin.
pub struct RadarGatingTool {
    /// Serializable RF, spatial scatterer, Kalman, and gating configuration.
    pub state: RadarGatingState,
    /// Preallocated zero-allocation simulation and spectral pipeline.
    pub pipeline: RadarPipeline,
    /// Whether continuous `PlaybackTransport` stepping is active.
    pub playing: bool,
    /// Currently selected scatterer index (`0` = chest, `1` = clothing/clutter).
    pub selected_target: usize,
    /// Scatterer currently dragged by pointer input, if any.
    pub dragging_target: Option<usize>,
    /// Canvas widget ID when the 2D room canvas holds keyboard focus.
    pub focused_canvas: Option<egui::Id>,
    /// Last rendered frame wall-clock duration in milliseconds.
    pub last_frame_ms: f64,
}

impl Default for RadarGatingTool {
    fn default() -> Self {
        let state = RadarGatingState::default();
        let pipeline = RadarPipeline::new(&state);
        Self {
            state,
            pipeline,
            playing: false,
            selected_target: 0,
            dragging_target: None,
            focused_canvas: None,
            last_frame_ms: 0.0,
        }
    }
}

impl RadarGatingTool {
    /// Steps the respiratory, phase, Kalman, and gating loop with zero heap allocations.
    pub fn step_gating(&mut self) -> BeamState {
        self.pipeline.step_zero_alloc(&self.state)
    }

    /// Resets playback, faults, and pipeline history using current parameters.
    pub fn reset_simulation(&mut self) {
        self.playing = false;
        self.pipeline.extra_clutter_active = false;
        self.pipeline.rebuild_all(&self.state);
    }

    /// Injects a deterministic thoracic cough spike into the respiratory trajectory.
    pub fn inject_cough(&mut self) {
        self.pipeline.inject_cough_spike();
    }

    /// Injects a sensor-loss `NaN` fault into `GatingLogic::evaluate`.
    pub fn inject_nan_fault(&mut self) {
        self.pipeline.inject_nan_fault();
    }

    /// Resets a latched `SafetyInterlock` back to `BeamState::Standby`.
    pub fn reset_interlock(&mut self) -> BeamState {
        self.pipeline.reset_interlock_to_standby()
    }

    /// Arms the LINAC gating controller from `Standby` to `BeamState::Armed`.
    pub fn arm_gating(&mut self) -> BeamState {
        self.pipeline.arm_controller()
    }

    /// Returns the widget ID when the 2D room canvas holds keyboard focus.
    #[cfg(test)]
    pub(super) fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
    }
}

impl InteractiveTool for RadarGatingTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Radar Respiratory Gating"
    }

    fn domain(&self) -> &'static str {
        "medical"
    }

    fn show(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            self.show_inside(ui);
        });
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        ui::show_tool_inside(self, ui);
    }

    fn state_bounds(&self) -> ToolStateBounds {
        RadarGatingState::STATE_BOUNDS
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, raw: &str) -> Result<(), String> {
        self.state_bounds().validate_json(raw)?;
        let loaded: RadarGatingState =
            serde_json::from_str(raw).map_err(|e| format!("Invalid RadarGatingTool state: {e}"))?;
        loaded.validate_bounds()?;
        self.state = loaded;
        self.pipeline.rebuild_all(&self.state);
        Ok(())
    }

    fn clamp_parameters(&mut self) -> bool {
        self.state.samples_per_chirp = self
            .state
            .samples_per_chirp
            .clamp(state::MIN_SAMPLES_PER_CHIRP, state::MAX_SAMPLES_PER_CHIRP);
        self.state.snapshots = self
            .state
            .snapshots
            .clamp(state::MIN_SNAPSHOTS, state::MAX_SNAPSHOTS);
        self.state.sweep_bins = self
            .state
            .sweep_bins
            .clamp(state::MIN_SWEEP_BINS, state::MAX_SWEEP_BINS);
        self.pipeline.rebuild_all(&self.state);
        true
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        vec![
            (
                "Coarse FFT Range Spectrum".to_string(),
                self.pipeline.spectra.coarse_fft.clone(),
            ),
            (
                "Spatial CZT Zoom Spectrum".to_string(),
                self.pipeline.spectra.czt_zoom.clone(),
            ),
            (
                "MUSIC Super-Resolution Pseudospectrum".to_string(),
                self.pipeline.spectra.music_pseudospectrum.clone(),
            ),
            (
                "Chest Unwrapped Displacement (mm)".to_string(),
                self.pipeline.measured_trace_mm.clone(),
            ),
            (
                "Kalman Lookahead Chest Trace (mm)".to_string(),
                self.pipeline.kalman_trace_mm.clone(),
            ),
            (
                "LINAC BeamActive Window".to_string(),
                self.pipeline.beam_active_trace.clone(),
            ),
            (
                "LINAC Duty Cycle (%)".to_string(),
                self.pipeline.duty_cycle_trace_pct.clone(),
            ),
        ]
    }

    fn export_figure_options(&self) -> Option<egui_plot::export::ExportOptions> {
        Some(egui_plot::export::ExportOptions {
            line_thickness: 2.2,
            ..Default::default()
        })
    }

    fn step_simulation(&mut self) {
        let _ = self.step_gating();
    }
}

impl scientific_metadata::theory::TheoryDescribable for RadarGatingTool {
    fn theory_description(&self) -> String {
        "60 GHz FMCW range resolution is Rayleigh-limited by $\\Delta R = c / (2B)$ ($3.75\\text{ cm}$ at $B = 4\\text{ GHz}$), while slow-time phase unwrapping extracts sub-millimeter thoracic motion $d = \\lambda \\Delta\\phi / (4\\pi)$ after elliptical static-clutter centering. Orthogonal noise-subspace projection (MUSIC) resolves scatterers closer than $c / (2B)$, and a constant-velocity Kalman filter with Schmitt-trigger latency compensation $\\hat{z}(t + \\tau_{\\text{sys}}) = \\hat{z}(t) + \\hat{v}(t)\\tau_{\\text{sys}}$ gates the LINAC beam with IEEE-754 safety interlocks.".to_string()
    }

    fn phonetic_description(&self) -> String {
        "Sixty gigahertz MIMO FMCW radar super-resolution using Chirp Z-Transform, MUSIC eigenspace projection, elliptical clutter filtering, phase unwrapping, and Kalman Schmitt-trigger respiratory beam gating.".to_string()
    }

    fn theory_citation(&self) -> String {
        "Bressler et al., Med. Phys. (2024); Schmidt, IEEE Trans. Antennas Propag. 34(3), 276-280 (1986); Kalman, J. Basic Eng. 82(1), 35-45 (1960)".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "FMCW & Sub-Rayleigh MUSIC".to_string(),
            "Beat frequency $f_b = 2BR / (c T_c)$ maps slant range $R$ to Fourier bins spaced by $\\Delta R = c / (2B)$. MUSIC decomposes the multi-snapshot covariance $R_{xx} = E_s \\Lambda_s E_s^H + \\sigma^2 E_n E_n^H$ and evaluates $P_{\\text{MUSIC}}(r) = 1 / \\|E_n^H a(r)\\|^2$ to split sub-Rayleigh chest and clothing scatterers.".to_string(),
        );
        map.insert(
            "Kalman Lookahead & Schmitt Gating".to_string(),
            "After elliptical DC clutter subtraction and $2\\pi$ phase unwrapping smoothed over a bi-quadratic thoracic patch, a constant-velocity Kalman filter predicts $\\hat{z} + \\hat{v}\\tau_{\\text{sys}}$ against hysteresis bounds $T \\pm H$ while latching SafetyInterlock on non-finite inputs.".to_string(),
        );
        map
    }
}

/// Starter preset JSON resolving two scatterers `1.5 cm` apart (`< 3.75 cm` Rayleigh limit at `4 GHz`).
pub const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "60 GHz Sub-Rayleigh Chest & Gown Split (1.5 cm at 4 GHz)",
    "domain": "medical",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "primary_target": { "x_m": 0.0, "y_m": 0.700, "rcs": 1.0 },
    "secondary_target": { "x_m": 0.0, "y_m": 0.715, "rcs": 0.95 },
    "bandwidth_ghz": 4.0,
    "steering_deg": 0.0,
    "num_antennas": 8,
    "samples_per_chirp": 48,
    "snapshots": 24,
    "sweep_bins": 240,
    "noise_sigma": 0.01,
    "clutter_window": 28,
    "static_clutter_gain": 0.50,
    "breath_amplitude_mm": 6.0,
    "breath_period_s": 4.2,
    "kalman_process_var": 0.00025,
    "kalman_meas_var": 0.000015,
    "gating_threshold_mm": -1.8,
    "gating_hysteresis_mm": 0.6,
    "system_latency_ms": 60.0,
    "rng_seed": 2026
  }
}"#;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "RadarGatingTool",
        display_name: "Radar Respiratory Gating",
        domain: "medical",
        tags: &["radar", "mimo", "music", "kalman", "radiotherapy"],
        build: || Box::new(RadarGatingTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "radar_gating_sub_rayleigh_starter",
        title: "60 GHz Sub-Rayleigh Chest & Gown Split (1.5 cm at 4 GHz)",
        description: "MIMO MUSIC resolves chest wall and gown scatterers 1.5 cm apart (< 3.75 cm c/2B limit) with Kalman LINAC respiratory gating.",
        category: "Medical Physics",
        icon: "📡",
        domain: "medical",
        tool_name: Some("RadarGatingTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

// [cite:mmwave_radiotherapy_setup]

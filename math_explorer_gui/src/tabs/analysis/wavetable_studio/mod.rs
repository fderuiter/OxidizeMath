#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:graph_parameters_rust]

use std::collections::BTreeMap;

use eframe::egui;
use egui_plot::ExportPrimitive;
use math_explorer::pure_math::analysis::pde::wave_2d::{BoundaryMode2D, Wave2DMedium};

use crate::audio::wav::{WavSampleFormat, WavSampleRate};
use crate::audio::wavetable::{MAX_WAVETABLE_FRAMES, WAVETABLE_FRAME_SAMPLES};
use crate::framework::{
    ExportCapabilities, InteractiveTool, MathExpressionInputBar, ToolStateBounds,
};

pub(crate) mod engine;
mod presets;
mod ui;

#[cfg(test)]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

pub(crate) const AUDITION_FREQ_MIN_HZ: f64 = 20.0;
pub(crate) const AUDITION_FREQ_MAX_HZ: f64 = 2_000.0;
pub(crate) const PDE_STIFFNESS_MIN: f64 = 1.0e-5;
pub(crate) const PDE_STIFFNESS_MAX: f64 = 0.02;
pub(crate) const PDE_SPEED_MIN: f64 = 0.2;
pub(crate) const PDE_SPEED_MAX: f64 = 4.0;
pub(crate) const PDE_ANGLE_MIN_DEG: f64 = -90.0;
pub(crate) const PDE_ANGLE_MAX_DEG: f64 = 90.0;
pub(crate) const PARAM_ABS_LIMIT: f64 = 20.0;

pub(crate) const DEFAULT_WAVETABLE_FORMULA: &str =
    "sin(pi * x + (1.0 + 5.0 * m) * sin(2.0 * pi * x)) + 0.35 * m * sin(5.0 * pi * x)";

pub(crate) const FORMULA_PRESETS: &[(&str, &str)] = &[
    (
        "FM Phase Fold",
        "sin(pi * x + (1.0 + 5.0 * m) * sin(2.0 * pi * x)) + 0.35 * m * sin(5.0 * pi * x)",
    ),
    (
        "Chebyshev Formant",
        "sin(pi * x) * cos((1.0 + 7.0 * m) * pi * x) * exp(-a * x * x)",
    ),
    (
        "Sync Windowed Saw",
        "sin((1.0 + 9.0 * m) * pi * (x + 1.0)) * (1.0 - x * x)",
    ),
    (
        "Odd-Even Harmonic Blend",
        "(1.0 - m) * sin(pi * x) + m * tanh(3.5 * sin(pi * x) + 1.5 * sin(3.0 * pi * x))",
    ),
];

/// Synthesis generator family driving single-cycle `2048`-sample wavetable frames.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WavetableSourceMode {
    /// Compiled symbolic formula `f(x, m)` over `x in [-1, 1)` and morph `m in [0, 1]`.
    #[default]
    FormulaExpr,
    /// 2D Störmer-Verlet PDE wave/plate spatial cross-section swept across morph `m in [0, 1]`.
    PdeWaveSlice,
}

impl WavetableSourceMode {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::FormulaExpr => "Symbolic Formula f(x, m)",
            Self::PdeWaveSlice => "2D PDE Wave / Plate Slice",
        }
    }
}

/// Shareable state for [`WavetableStudioTool`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WavetableStudioState {
    pub source_mode: WavetableSourceMode,
    pub formula: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub formula_params: BTreeMap<String, f64>,
    pub frame_count: u16,
    pub morph_pos: f64,
    pub audition_freq_hz: f64,
    pub bandlimit_fundamental_hz: f64,
    pub sample_rate: WavSampleRate,
    pub sample_format: WavSampleFormat,
    pub pde_medium: Wave2DMedium,
    pub pde_boundary: BoundaryMode2D,
    pub pde_stiffness: f64,
    pub pde_wave_speed: f64,
    pub pde_slice_angle_deg: f64,
}

/// Unified PDE cross-section and symbolic `CompiledExpr` `2048`-sample wavetable studio.
pub struct WavetableStudioTool {
    pub(crate) state: WavetableStudioState,
    pub(crate) expr_bar: MathExpressionInputBar,
    pub(crate) engine: engine::WavetableBankEngine,
    pub(crate) state_rev: u64,
    pub(crate) pde_dirty: bool,
    pub(crate) preview_dirty: bool,
    pub(crate) bank_dirty: bool,
    #[cfg(test)]
    pub(crate) last_exported_wav: Option<Vec<u8>>,
}

impl Default for WavetableStudioTool {
    fn default() -> Self {
        let state = presets::default_studio_state();
        let expr_bar = build_wavetable_expr_bar(&state.formula, &state.formula_params);
        let mut tool = Self {
            state,
            expr_bar,
            engine: engine::WavetableBankEngine::new(),
            state_rev: 1,
            pde_dirty: true,
            preview_dirty: true,
            bank_dirty: true,
            #[cfg(test)]
            last_exported_wav: None,
        };
        tool.refresh_active_preview();
        tool
    }
}

pub(crate) fn build_wavetable_expr_bar(
    formula: &str,
    params: &BTreeMap<String, f64>,
) -> MathExpressionInputBar {
    let mut bar = MathExpressionInputBar::new("f(x, m)", formula, FORMULA_PRESETS.to_vec())
        .with_axes(&["x", "m"])
        .with_param_range(-8.0..=8.0, 1.4)
        .with_param_spec("a", "Envelope a", 0.0..=8.0, 1.4)
        .with_param_spec("k", "Harmonic k", 0.5..=16.0, 3.0);
    if !params.is_empty() {
        bar.set_params_map(params);
    }
    bar
}

impl WavetableStudioTool {
    pub(crate) fn sync_state_params_from_bar(&mut self) {
        self.state.formula = self.expr_bar.raw_text.clone();
        self.state.formula_params = self.expr_bar.params_map();
    }

    pub(crate) fn refresh_active_preview(&mut self) {
        if self.state.source_mode == WavetableSourceMode::PdeWaveSlice && self.pde_dirty {
            self.engine.rebuild_pde_field(&self.state);
            self.pde_dirty = false;
        }
        self.engine
            .refresh_single_preview(&self.state, &self.expr_bar);
        self.preview_dirty = false;
    }

    pub(crate) fn synthesize_and_encode_wav(&mut self) -> Vec<u8> {
        if self.state.source_mode == WavetableSourceMode::PdeWaveSlice && self.pde_dirty {
            self.engine.rebuild_pde_field(&self.state);
            self.pde_dirty = false;
        }
        let wav = self.engine.encode_bank_wav(&self.state, &self.expr_bar);
        self.bank_dirty = false;
        wav
    }
}

impl InteractiveTool for WavetableStudioTool {
    fn name(&self) -> &'static str {
        "Wavetable Studio (PDE & Formula)"
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
        presets::is_valid_studio_state(&self.state)
            && self
                .expr_bar
                .param_slice()
                .iter()
                .all(|v| v.is_finite() && v.abs() <= PARAM_ABS_LIMIT)
    }

    fn clamp_parameters(&mut self) -> bool {
        presets::clamp_studio_state(&mut self.state);
        self.expr_bar
            .clamp_active_params(-PARAM_ABS_LIMIT..=PARAM_ABS_LIMIT);
        self.sync_state_params_from_bar();
        self.pde_dirty = true;
        self.preview_dirty = true;
        self.bank_dirty = true;
        self.state_rev = self.state_rev.wrapping_add(1);
        crate::audio::notify_solver_diverged();
        true
    }

    fn save_state(&self) -> Option<String> {
        let mut snap = self.state.clone();
        snap.formula = self.expr_bar.raw_text.clone();
        snap.formula_params = self.expr_bar.params_map();
        serde_json::to_string(&snap).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: WavetableStudioState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid Wavetable Studio state JSON: {e}"))?;
        if !presets::is_valid_studio_state(&parsed) {
            return Err("Wavetable Studio parameters out of finite valid bounds".into());
        }
        let bar = build_wavetable_expr_bar(&parsed.formula, &parsed.formula_params);
        if !bar.is_valid() {
            return Err(bar
                .error_message()
                .unwrap_or_else(|| "Invalid wavetable formula".into()));
        }
        self.state = parsed;
        self.expr_bar = bar;
        self.pde_dirty = true;
        self.preview_dirty = true;
        self.bank_dirty = true;
        self.state_rev = self.state_rev.wrapping_add(1);
        self.refresh_active_preview();
        Ok(())
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_plot_primitives(&self) -> Vec<ExportPrimitive> {
        self.build_export_primitives()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let wave: Vec<[f64; 2]> = self
            .engine
            .active_frame
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let x = -1.0 + 2.0 * (i as f64) / (WAVETABLE_FRAME_SAMPLES as f64);
                [x, if s.is_finite() { s as f64 } else { 0.0 }]
            })
            .collect();
        let spec: Vec<[f64; 2]> = self
            .engine
            .harmonic_mags
            .iter()
            .enumerate()
            .take(64)
            .map(|(k, &m)| [(k + 1) as f64, if m.is_finite() { m as f64 } else { 0.0 }])
            .collect();
        vec![
            ("wavetable_single_cycle_2048".into(), wave),
            ("wavetable_harmonic_magnitudes".into(), spec),
        ]
    }

    fn step_simulation(&mut self) {
        self.state.morph_pos = (self.state.morph_pos + 0.015).fract();
        self.preview_dirty = true;
        self.refresh_active_preview();
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
        if self.preview_dirty {
            self.refresh_active_preview();
        }
        let max_frames = MAX_WAVETABLE_FRAMES as u16;
        self.state.frame_count = self.state.frame_count.clamp(1, max_frames);
        self.draw_controls_sidebar(parent_ui);
        self.draw_central_plots(parent_ui);
    }
}

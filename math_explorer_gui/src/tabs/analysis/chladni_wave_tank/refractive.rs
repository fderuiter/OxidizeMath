#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:graph_parameters_rust]

use std::collections::BTreeMap;

use eframe::egui;
use math_explorer::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::analysis::pde::wave_2d::apply_biconvex_lens;

use super::{ChladniWaveTankState, ChladniWaveTankTool, GRID_SIZE, REFRACTIVE_MAX, REFRACTIVE_MIN};
use crate::framework::{MathExpressionInputBar, bounded_drag_value};

pub(crate) const MAX_REFRACTIVE_EXPR_BYTES: usize = 256;
pub(crate) const MAX_REFRACTIVE_PARAMS: usize = 6;
pub(crate) const PARAM_BOUND_MIN: f64 = -10.0;
pub(crate) const PARAM_BOUND_MAX: f64 = 10.0;

pub(crate) const DEFAULT_LUNEBURG_EXPR: &str =
    "sqrt(1 + a * max(0, 1 - ((x - 0.45)^2 + (y - 0.5)^2) / r^2))";
pub(crate) const DEFAULT_GRIN_EXPR: &str =
    "1 + g * max(0, 1 - ((x - 0.44) / w)^2) * max(0, 1 - ((y - 0.5) / 0.34)^2)";
pub(crate) const DEFAULT_BICONVEX_EXPR: &str =
    "1 + (n0 - 1) * step(w * max(0, 1 - ((y - 0.5) / 0.30)^2) - abs(x - 0.35) - 0.001)";

const REFRACTIVE_PRESETS: &[(&str, &str)] = &[
    ("Luneburg", DEFAULT_LUNEBURG_EXPR),
    ("Parabolic GRIN", DEFAULT_GRIN_EXPR),
    ("Biconvex AST", DEFAULT_BICONVEX_EXPR),
];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum RefractiveMode {
    #[default]
    Biconvex,
    Formula,
}

impl RefractiveMode {
    pub(crate) fn is_default(&self) -> bool {
        matches!(self, Self::Biconvex)
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Biconvex => "Analytic Biconvex Lens",
            Self::Formula => "2D Formula n(x, y)",
        }
    }
}

pub(crate) struct RefractiveSandbox {
    pub(crate) input_bar: MathExpressionInputBar,
}

impl Default for RefractiveSandbox {
    fn default() -> Self {
        Self::new()
    }
}

impl RefractiveSandbox {
    pub(crate) fn new() -> Self {
        let mut input_bar = MathExpressionInputBar::new("n(x, y)", "", REFRACTIVE_PRESETS.to_vec())
            .with_axes(&["x", "y"])
            .with_param_range(0.0..=3.0, 1.0)
            .with_param_spec("a", "Luneburg Contrast a", 0.1..=2.5, 1.0)
            .with_param_spec("r", "Lens Radius r", 0.12..=0.45, 0.31)
            .with_param_spec("g", "GRIN Peak Δn (g)", 0.1..=1.8, 0.65)
            .with_param_spec("w", "Axial Half-Width w", 0.05..=0.40, 0.22)
            .with_param_spec("n0", "Core Index n₀", REFRACTIVE_MIN..=REFRACTIVE_MAX, 1.65);
        input_bar.set_expression(DEFAULT_LUNEBURG_EXPR);
        Self { input_bar }
    }

    pub(crate) fn sync_from_state(&mut self, state: &ChladniWaveTankState) {
        let target_expr = if state.refractive_expr.trim().is_empty() {
            DEFAULT_LUNEBURG_EXPR
        } else {
            state.refractive_expr.as_str()
        };
        if self.input_bar.raw_text != target_expr {
            self.input_bar.set_expression(target_expr);
        }
        if !state.refractive_params.is_empty() {
            self.input_bar.set_params_map(&state.refractive_params);
        }
    }

    pub(crate) fn sync_to_state(&self, state: &mut ChladniWaveTankState) {
        if self.input_bar.is_valid() {
            let raw = self.input_bar.raw_text.trim();
            if raw.len() <= MAX_REFRACTIVE_EXPR_BYTES {
                state.refractive_expr = raw.to_string();
            }
            state.refractive_params = self.input_bar.params_map();
        }
    }

    pub(crate) fn populate_wave_speed_mask(
        &self,
        wave_speed_mask: &mut [f64],
        width: usize,
        height: usize,
    ) {
        let Some(compiled) = self.input_bar.active_compiled() else {
            wave_speed_mask.fill(1.0);
            return;
        };
        let params = self.input_bar.param_slice();
        let denom_x = (width.saturating_sub(1)).max(1) as f64;
        let denom_y = (height.saturating_sub(1)).max(1) as f64;
        for y in 0..height {
            let yn = (y as f64) / denom_y;
            for x in 0..width {
                let xn = (x as f64) / denom_x;
                let raw_n = compiled.eval_2d(xn, yn, params);
                let safe_n = if raw_n.is_finite() {
                    raw_n.clamp(REFRACTIVE_MIN, REFRACTIVE_MAX)
                } else {
                    1.0
                };
                wave_speed_mask[flatten_2d_index(x, y, width)] = 1.0 / safe_n;
            }
        }
    }
}

pub(super) fn are_refractive_fields_valid(state: &ChladniWaveTankState) -> bool {
    if state.refractive_expr.len() > MAX_REFRACTIVE_EXPR_BYTES
        || state.refractive_params.len() > MAX_REFRACTIVE_PARAMS
    {
        return false;
    }
    let params_ok = state.refractive_params.iter().all(|(k, v)| {
        !k.is_empty()
            && k.len() <= 16
            && v.is_finite()
            && (PARAM_BOUND_MIN..=PARAM_BOUND_MAX).contains(v)
    });
    if !params_ok {
        return false;
    }
    if state.refractive_mode == RefractiveMode::Formula && !state.refractive_expr.trim().is_empty()
    {
        let probe = MathExpressionInputBar::new("n(x, y)", &state.refractive_expr, Vec::new())
            .with_axes(&["x", "y"]);
        return probe.is_valid();
    }
    true
}

pub(super) fn sanitize_refractive_fields(state: &mut ChladniWaveTankState) {
    if state.refractive_expr.len() > MAX_REFRACTIVE_EXPR_BYTES {
        state.refractive_expr = DEFAULT_LUNEBURG_EXPR.to_string();
    }
    if state.refractive_mode == RefractiveMode::Formula && !state.refractive_expr.trim().is_empty()
    {
        let probe = MathExpressionInputBar::new("n(x, y)", &state.refractive_expr, Vec::new())
            .with_axes(&["x", "y"]);
        if !probe.is_valid() {
            state.refractive_expr = DEFAULT_LUNEBURG_EXPR.to_string();
        }
    }
    let mut clean: BTreeMap<String, f64> = BTreeMap::new();
    for (k, &v) in state
        .refractive_params
        .iter()
        .filter(|(k, _)| !k.is_empty() && k.len() <= 16)
        .take(MAX_REFRACTIVE_PARAMS)
    {
        let val = if v.is_finite() {
            v.clamp(PARAM_BOUND_MIN, PARAM_BOUND_MAX)
        } else {
            1.0
        };
        clean.insert(k.clone(), val);
    }
    state.refractive_params = clean;
}

pub(crate) fn detector_focal_metrics(detector_buf: &[f64]) -> (f64, f64) {
    let slice = if detector_buf.len() >= 48 {
        &detector_buf[22..(detector_buf.len() - 22)]
    } else {
        detector_buf
    };
    let mut peak = 0.0_f64;
    let mut sum = 0.0_f64;
    let mut count = 0_usize;
    for &val in slice {
        if val.is_finite() && val >= 0.0 {
            peak = peak.max(val);
            sum += val;
            count += 1;
        }
    }
    if count == 0 || sum <= 1.0e-12 || peak <= 1.0e-12 {
        return (1.0, slice.len().max(1) as f64);
    }
    let mean = sum / (count as f64);
    let half_max = 0.5 * peak;
    let fwhm = slice
        .iter()
        .filter(|&&v| v.is_finite() && v >= half_max)
        .count()
        .max(1) as f64;
    (peak / mean, fwhm)
}

impl ChladniWaveTankTool {
    pub(super) fn apply_active_lens_mask(&mut self) {
        match self.state.refractive_mode {
            RefractiveMode::Biconvex => {
                apply_biconvex_lens(
                    &mut self.solver.wave_speed_mask,
                    GRID_SIZE,
                    GRID_SIZE,
                    44,
                    self.state.lens_aperture,
                    8.0,
                    self.state.refractive_index,
                );
            }
            RefractiveMode::Formula => {
                self.refractive.populate_wave_speed_mask(
                    &mut self.solver.wave_speed_mask,
                    GRID_SIZE,
                    GRID_SIZE,
                );
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn set_refractive_formula(&mut self, expr: &str) {
        self.state.lens_enabled = true;
        self.state.refractive_mode = RefractiveMode::Formula;
        self.refractive.input_bar.set_expression(expr);
        self.refractive.sync_to_state(&mut self.state);
        self.rebuild_masks();
        self.solver.reset_detector();
        self.state_rev = self.state_rev.wrapping_add(1);
        self.analysis_dirty = true;
    }

    pub(super) fn draw_refractive_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut mask_changed = false;
        let prev_mode = self.state.refractive_mode;
        egui::ComboBox::from_id_salt("chladni_refractive_mode")
            .selected_text(self.state.refractive_mode.label())
            .show_ui(ui, |ui| {
                for mode in [RefractiveMode::Biconvex, RefractiveMode::Formula] {
                    ui.selectable_value(&mut self.state.refractive_mode, mode, mode.label());
                }
            });
        if self.state.refractive_mode != prev_mode {
            if self.state.refractive_mode == RefractiveMode::Formula {
                self.refractive.sync_from_state(&self.state);
                self.refractive.sync_to_state(&mut self.state);
            }
            mask_changed = true;
        }
        match self.state.refractive_mode {
            RefractiveMode::Biconvex => {
                ui.horizontal(|ui| {
                    ui.label("Refractive n:");
                    mask_changed |= ui
                        .add(bounded_drag_value(
                            &mut self.state.refractive_index,
                            REFRACTIVE_MIN..=REFRACTIVE_MAX,
                            0.05,
                        ))
                        .changed();
                });
            }
            RefractiveMode::Formula => {
                if self.refractive.input_bar.ui(ui) && self.refractive.input_bar.is_valid() {
                    self.refractive.sync_to_state(&mut self.state);
                    mask_changed = true;
                }
            }
        }
        mask_changed
    }

    pub(crate) fn compute_observables(&self) -> Vec<(&'static str, f64)> {
        let sand_energy = self
            .sand
            .mean_normalized_energy(self.solver.energy_field(), GRID_SIZE);
        let (focal_gain, focal_fwhm) = detector_focal_metrics(&self.detector_buf);
        let cfl = self
            .solver
            .effective_courant(&self.pde_params(), self.state.time_step);
        vec![
            ("sand_nodal_energy", sand_energy),
            ("focal_gain", focal_gain),
            ("focal_fwhm", focal_fwhm),
            ("effective_courant", cfl),
            ("dominant_k", self.dominant_peak.k_radial),
            ("driver_frequency", self.state.driver_frequency),
            ("custom_pin_count", self.state.custom_clamps.len() as f64),
        ]
    }
}

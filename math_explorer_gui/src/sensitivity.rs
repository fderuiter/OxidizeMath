#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Parameter sensitivity and perturbation analysis panel module.
//!
//! Provides central finite difference perturbation analysis (+/- 1% delta)
//! and normalized sensitivity elasticity metrics for active parameters.

use eframe::egui;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

/// Normalizes floating-point values to canonical IEEE-754 bit patterns
/// (`-0.0` maps to `+0.0` and all `NaN` payloads map to a single quiet `NaN`).
#[must_use]
pub fn canonical_f64_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0_f64.to_bits()
    } else if value.is_nan() {
        f64::NAN.to_bits()
    } else {
        value.to_bits()
    }
}

fn hash_context<C: Hash>(context: &C) -> u64 {
    let mut hasher = DefaultHasher::new();
    context.hash(&mut hasher);
    hasher.finish()
}

fn set_map_value(map: &mut HashMap<String, f64>, key: &str, value: f64) {
    if let Some(slot) = map.get_mut(key) {
        *slot = value;
    } else {
        map.insert(key.to_string(), value);
    }
}

/// Quantitative sensitivity metrics for a single input parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct SensitivityMetric {
    /// Parameter identifier name.
    pub param_name: String,
    /// Baseline value of parameter.
    pub baseline_value: f64,
    /// Approximated partial derivative dY/dP using central finite differences.
    pub partial_derivative: f64,
    /// Normalized sensitivity index (elasticity = |(dY/dP) * (P / Y)|).
    pub elasticity: f64,
    /// Flag indicating if elasticity exceeds the high-sensitivity threshold (> 2.0).
    pub is_high_sensitivity: bool,
}

/// Returns `true` when parameters are mid-gesture (pointer held down or local `AnimatedSlider` sweep active),
/// matching continuous-edit coalescing in `WorkspaceSessionManager` while scoping sweeps to the current pass/pane.
#[must_use]
pub fn is_transient_parameter_gesture(ctx: &egui::Context, sweep_active: bool) -> bool {
    sweep_active || ctx.input(|i| i.pointer.any_down())
}

/// Maximum synchronous instruction budget per pass for sensitivity finite-difference sweeps.
pub const MAX_SENSITIVITY_SYNC_INSTRUCTIONS: usize = 2_000_000;

/// Minimum ODE integration step count preserved when scaling sensitivity perturbations.
pub const MIN_SENSITIVITY_STEPS: usize = 32;

/// Sensitivity analysis engine for parameter perturbation calculations.
#[derive(Debug, Clone, Default)]
pub struct SensitivityEngine {
    /// Cached calculated sensitivity metrics.
    pub metrics: Vec<SensitivityMetric>,
    /// Last calculated baseline model output Y.
    pub baseline_output: f64,
    /// Current batch index for round-robin parameter evaluation bounding (max 5 sample runs).
    pub batch_offset: usize,
    cached_params: Vec<(String, u64)>,
    cached_context_hash: Option<u64>,
    evaluated_count: usize,
    baseline_map: HashMap<String, f64>,
    gesture_deferred: bool,
    budget_notice: Option<String>,
    last_announced_budget: std::cell::RefCell<Option<String>>,
}

impl SensitivityEngine {
    /// Creates a new SensitivityEngine instance.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` when all active parameters have been evaluated for the current inputs.
    #[must_use]
    pub fn is_converged(&self) -> bool {
        self.evaluated_count >= self.cached_params.len()
    }

    /// Returns `true` when `params` and `context` match the memoized parameter and context state.
    #[must_use]
    pub fn matches_cache<K: AsRef<str>, C: Hash>(&self, params: &[(K, f64)], context: &C) -> bool {
        self.cached_context_hash == Some(hash_context(context))
            && self.schema_matches(params)
            && self.values_match(params)
    }

    fn schema_matches<K: AsRef<str>>(&self, params: &[(K, f64)]) -> bool {
        self.cached_params.len() == params.len()
            && self
                .cached_params
                .iter()
                .zip(params)
                .all(|((cached_key, _), (key, _))| cached_key == key.as_ref())
    }

    fn values_match<K: AsRef<str>>(&self, params: &[(K, f64)]) -> bool {
        self.cached_params
            .iter()
            .zip(params)
            .all(|((_, cached_bits), (_, val))| *cached_bits == canonical_f64_bits(*val))
    }

    /// Returns any active synchronous instruction-budget scaling notice.
    #[must_use]
    pub fn budget_notice(&self) -> Option<&str> {
        self.budget_notice.as_deref()
    }

    /// Updates the external step-scaling notice displayed in the sensitivity panel.
    pub fn set_budget_notice(&mut self, notice: Option<String>) {
        self.budget_notice = notice;
    }

    /// Scales `requested_steps` so a full sensitivity pass (`1 + 2 * min(param_count, 2)` runs)
    /// stays within [`MAX_SENSITIVITY_SYNC_INSTRUCTIONS`].
    #[must_use]
    pub fn scale_steps_for_budget(
        param_count: usize,
        requested_steps: usize,
        stages_per_step: usize,
        inst_per_rhs: usize,
    ) -> (usize, Option<String>) {
        if param_count == 0 || requested_steps <= MIN_SENSITIVITY_STEPS {
            return (requested_steps, None);
        }
        let batch_runs = 1 + 2 * param_count.min(2);
        let ops_per_step = batch_runs
            .saturating_mul(stages_per_step.max(1))
            .saturating_mul(inst_per_rhs.max(1));
        if requested_steps.saturating_mul(ops_per_step) <= MAX_SENSITIVITY_SYNC_INSTRUCTIONS {
            return (requested_steps, None);
        }
        let capped = (MAX_SENSITIVITY_SYNC_INSTRUCTIONS / ops_per_step.max(1))
            .clamp(MIN_SENSITIVITY_STEPS, requested_steps);
        if capped < requested_steps {
            let msg = format!(
                "Sensitivity budget applied: scaled perturbation steps from {requested_steps} to {capped} ({MAX_SENSITIVITY_SYNC_INSTRUCTIONS} instruction cap)."
            );
            (capped, Some(msg))
        } else {
            (requested_steps, None)
        }
    }

    fn reset_empty(&mut self) {
        self.metrics.clear();
        self.cached_params.clear();
        self.baseline_map.clear();
        self.cached_context_hash = None;
        self.evaluated_count = 0;
        self.batch_offset = 0;
        self.gesture_deferred = false;
        self.budget_notice = None;
    }

    fn sync_cache_on_miss<K: AsRef<str>>(&mut self, params: &[(K, f64)], context_hash: u64) {
        if self.schema_matches(params) {
            for (idx, (key, val)) in params.iter().enumerate() {
                self.cached_params[idx].1 = canonical_f64_bits(*val);
                set_map_value(&mut self.baseline_map, key.as_ref(), *val);
                self.metrics[idx].baseline_value = *val;
            }
        } else {
            self.batch_offset = 0;
            self.cached_params.clear();
            self.baseline_map.clear();
            self.metrics.clear();
            self.cached_params.reserve(params.len());
            self.baseline_map.reserve(params.len());
            self.metrics.reserve(params.len());
            for (key, val) in params {
                let name = key.as_ref();
                self.cached_params
                    .push((name.to_string(), canonical_f64_bits(*val)));
                self.baseline_map.insert(name.to_string(), *val);
                self.metrics.push(SensitivityMetric {
                    param_name: name.to_string(),
                    baseline_value: *val,
                    partial_derivative: 0.0,
                    elasticity: 0.0,
                    is_high_sensitivity: false,
                });
            }
        }
        self.cached_context_hash = Some(context_hash);
        self.evaluated_count = 0;
    }

    /// Evaluates normalized sensitivity indices using central finite differences (+/- 1% delta).
    ///
    /// Respects the constraint bounding perturbation evaluations to at most 5 sample runs
    /// (1 baseline + max 2 parameters * 2 perturbations per step = 5 runs max) to prevent UI slowdown.
    pub fn compute<K, F>(&mut self, params: &[(K, f64)], eval_fn: F) -> &Vec<SensitivityMetric>
    where
        K: AsRef<str>,
        F: FnMut(&HashMap<String, f64>) -> f64,
    {
        self.compute_with_context(params, &(), eval_fn)
    }

    /// Defers finite-difference perturbations during an active drag or animated sweep gesture,
    /// retaining settled `metrics` and suppressing unconverged repaint requests until release.
    pub fn defer_for_gesture(&mut self) -> &Vec<SensitivityMetric> {
        self.gesture_deferred = true;
        &self.metrics
    }

    /// Evaluates normalized sensitivity indices memoized by canonical parameter bits and `context`.
    pub fn compute_with_context<K, C, F>(
        &mut self,
        params: &[(K, f64)],
        context: &C,
        eval_fn: F,
    ) -> &Vec<SensitivityMetric>
    where
        K: AsRef<str>,
        C: Hash,
        F: FnMut(&HashMap<String, f64>) -> f64,
    {
        self.compute_with_baseline(params, context, None, eval_fn)
    }

    /// Evaluates normalized sensitivity indices memoized by canonical parameter bits and `context`,
    /// reusing `baseline_output_hint` on cache misses when finite instead of invoking `eval_fn` for the baseline.
    pub fn compute_with_baseline<K, C, F>(
        &mut self,
        params: &[(K, f64)],
        context: &C,
        baseline_output_hint: Option<f64>,
        eval_fn: F,
    ) -> &Vec<SensitivityMetric>
    where
        K: AsRef<str>,
        C: Hash,
        F: FnMut(&HashMap<String, f64>) -> f64,
    {
        self.compute_budgeted_with_baseline(params, context, baseline_output_hint, 1, eval_fn)
    }

    /// Evaluates normalized sensitivity indices while bounding per-pass synchronous instruction work
    /// (`inst_per_eval * evaluations <= MAX_SENSITIVITY_SYNC_INSTRUCTIONS`).
    pub fn compute_budgeted_with_baseline<K, C, F>(
        &mut self,
        params: &[(K, f64)],
        context: &C,
        baseline_output_hint: Option<f64>,
        inst_per_eval: usize,
        mut eval_fn: F,
    ) -> &Vec<SensitivityMetric>
    where
        K: AsRef<str>,
        C: Hash,
        F: FnMut(&HashMap<String, f64>) -> f64,
    {
        self.gesture_deferred = false;
        if params.is_empty() {
            self.reset_empty();
            return &self.metrics;
        }
        let max_batch = max_params_for_eval_cost(inst_per_eval.max(1));
        let context_hash = hash_context(context);
        let is_hit = self.cached_context_hash == Some(context_hash)
            && self.schema_matches(params)
            && self.values_match(params);
        if is_hit && self.evaluated_count >= params.len() {
            return &self.metrics;
        }
        if !is_hit {
            self.sync_cache_on_miss(params, context_hash);
            if max_batch == 0 {
                self.baseline_output = baseline_output_hint.unwrap_or(f64::NAN);
                self.evaluated_count = params.len();
                self.budget_notice = Some(format!(
                    "Sensitivity budget exceeded ({inst_per_eval} instructions/run > {} cap); simplify formula or increase dt.",
                    MAX_SENSITIVITY_SYNC_INSTRUCTIONS / 3
                ));
                return &self.metrics;
            }
            self.baseline_output = baseline_output_hint
                .filter(|v| v.is_finite())
                .unwrap_or_else(|| eval_fn(&self.baseline_map));
        }
        if max_batch == 1 && params.len() > 1 && self.budget_notice.is_none() {
            self.budget_notice = Some(format!(
                "Sensitivity budget applied: throttling perturbations to 1 parameter per pass ({MAX_SENSITIVITY_SYNC_INSTRUCTIONS} instruction cap)."
            ));
        }
        self.eval_perturbation_batch(params, max_batch, &mut eval_fn);
        &self.metrics
    }

    fn eval_perturbation_batch<K, F>(
        &mut self,
        params: &[(K, f64)],
        max_batch: usize,
        eval_fn: &mut F,
    ) where
        K: AsRef<str>,
        F: FnMut(&HashMap<String, f64>) -> f64,
    {
        let num_params = params.len();
        let batch_count = (num_params - self.evaluated_count).min(max_batch);
        let y0 = self.baseline_output;
        for step in 0..batch_count {
            let idx = (self.batch_offset + step) % num_params;
            let (ref name, p_val) = params[idx];
            self.metrics[idx] =
                eval_single_param(name.as_ref(), p_val, &mut self.baseline_map, y0, eval_fn);
        }
        self.batch_offset = (self.batch_offset + batch_count) % num_params;
        self.evaluated_count += batch_count;
    }
}

fn max_params_for_eval_cost(inst_per_eval: usize) -> usize {
    if inst_per_eval <= MAX_SENSITIVITY_SYNC_INSTRUCTIONS / 5 {
        2
    } else if inst_per_eval <= MAX_SENSITIVITY_SYNC_INSTRUCTIONS / 3 {
        1
    } else {
        0
    }
}

fn eval_single_param<F>(
    name: &str,
    p_val: f64,
    baseline_map: &mut HashMap<String, f64>,
    y0: f64,
    eval_fn: &mut F,
) -> SensitivityMetric
where
    F: FnMut(&HashMap<String, f64>) -> f64,
{
    let delta = if p_val.abs() > 1e-12 {
        0.01 * p_val.abs()
    } else {
        0.01
    };

    // Evaluate upper perturbation P+ = P + delta
    set_map_value(baseline_map, name, p_val + delta);
    let y_plus = eval_fn(baseline_map);

    // Evaluate lower perturbation P- = P - delta
    set_map_value(baseline_map, name, p_val - delta);
    let y_minus = eval_fn(baseline_map);

    // Restore baseline
    set_map_value(baseline_map, name, p_val);

    let dy_dp = if delta > 0.0 {
        (y_plus - y_minus) / (2.0 * delta)
    } else {
        0.0
    };

    let elasticity = if y0.abs() > 1e-12 && p_val.abs() > 1e-12 {
        (dy_dp * p_val / y0).abs()
    } else if y0.abs() > 1e-12 {
        (dy_dp / y0).abs()
    } else {
        dy_dp.abs()
    };

    let dy_dp_clean = if dy_dp.is_finite() { dy_dp } else { 0.0 };
    let elasticity_clean = if elasticity.is_finite() {
        elasticity
    } else {
        0.0
    };

    SensitivityMetric {
        param_name: name.to_string(),
        baseline_value: p_val,
        partial_derivative: dy_dp_clean,
        elasticity: elasticity_clean,
        is_high_sensitivity: elasticity_clean > 2.0,
    }
}

#[path = "sensitivity_ui.rs"]
mod sensitivity_ui;
pub use sensitivity_ui::{render_labeled_sensitivity_panel_ui, render_sensitivity_panel_ui};

#[cfg(test)]
#[path = "sensitivity_tests.rs"]
pub(crate) mod tests;

//! Frame-budgeted wide-range tornado parameter sweeps for [`super::SensitivityEngine`].

use super::{
    MAX_SENSITIVITY_BATCH_EVALS, SensitivityEngine, canonical_f64_bits, hash_context, set_map_value,
};
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

/// Lower and upper domain bounds for a single parameter during a wide-range tornado sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TornadoBounds {
    /// Minimum admissible value of the parameter.
    pub min: f64,
    /// Maximum admissible value of the parameter.
    pub max: f64,
}

impl TornadoBounds {
    /// Constructs ordered [`TornadoBounds`] from `a` and `b`.
    #[must_use]
    pub fn new(a: f64, b: f64) -> Self {
        if a <= b {
            Self { min: a, max: b }
        } else {
            Self { min: b, max: a }
        }
    }

    /// Constructs symmetric fractional tornado bounds (`baseline * (1 ± fraction)`) around `baseline`.
    #[must_use]
    pub fn relative(baseline: f64, fraction: f64) -> Self {
        let f = if fraction.is_finite() {
            fraction.abs().max(1e-6)
        } else {
            0.2
        };
        let span = if baseline.abs() > 1e-12 {
            baseline.abs() * f
        } else {
            f
        };
        Self::new(baseline - span, baseline + span)
    }
}

/// Wide-range output excursion interval produced by sweeping one parameter across its [`TornadoBounds`].
#[derive(Debug, Clone, PartialEq)]
pub struct TornadoInterval {
    /// Parameter identifier or display label.
    pub param_name: String,
    /// Baseline value of the parameter.
    pub baseline_value: f64,
    /// Lower parameter bound evaluated.
    pub low_param: f64,
    /// Upper parameter bound evaluated.
    pub high_param: f64,
    /// Model observable output evaluated at `low_param`.
    pub output_at_low: f64,
    /// Model observable output evaluated at `high_param`.
    pub output_at_high: f64,
    /// Minimum finite observable across `{output_at_low, baseline_output, output_at_high}`.
    pub min_output: f64,
    /// Maximum finite observable across `{output_at_low, baseline_output, output_at_high}`.
    pub max_output: f64,
    /// Total output excursion `max_output - min_output`.
    pub swing: f64,
}

impl TornadoInterval {
    pub(crate) fn empty(param_name: String, baseline_value: f64) -> Self {
        Self {
            param_name,
            baseline_value,
            low_param: baseline_value,
            high_param: baseline_value,
            output_at_low: 0.0,
            output_at_high: 0.0,
            min_output: 0.0,
            max_output: 0.0,
            swing: 0.0,
        }
    }

    fn update_in_place(
        &mut self,
        baseline_value: f64,
        bounds: TornadoBounds,
        baseline_output: f64,
        y_low: f64,
        y_high: f64,
    ) {
        let base_clean = if baseline_output.is_finite() {
            baseline_output
        } else {
            0.0
        };
        let low_clean = if y_low.is_finite() { y_low } else { base_clean };
        let high_clean = if y_high.is_finite() {
            y_high
        } else {
            base_clean
        };
        let min_out = base_clean.min(low_clean).min(high_clean);
        let max_out = base_clean.max(low_clean).max(high_clean);
        self.baseline_value = baseline_value;
        self.low_param = bounds.min;
        self.high_param = bounds.max;
        self.output_at_low = low_clean;
        self.output_at_high = high_clean;
        self.min_output = min_out;
        self.max_output = max_out;
        self.swing = (max_out - min_out).max(0.0);
    }
}

fn hash_tornado_bounds(bounds: &[TornadoBounds]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bounds.len().hash(&mut hasher);
    for b in bounds {
        canonical_f64_bits(b.min).hash(&mut hasher);
        canonical_f64_bits(b.max).hash(&mut hasher);
    }
    hasher.finish()
}

impl SensitivityEngine {
    /// Pre-initializes parameter schema and tornado buffers so subsequent hot-loop calls to
    /// [`Self::step_tornado_slice`] or [`Self::step_tornado`] perform zero heap allocations.
    pub fn prepare_tornado<K: AsRef<str>>(
        &mut self,
        params: &[(K, f64)],
        bounds: &[TornadoBounds],
    ) {
        if params.is_empty() {
            self.reset_empty();
            return;
        }
        let ctx_hash = hash_context(&());
        if self.cached_context_hash != Some(ctx_hash)
            || !self.schema_matches(params)
            || !self.values_match(params)
        {
            self.sync_cache_on_miss(params, ctx_hash);
        }
        let b_hash = hash_tornado_bounds(bounds);
        if self.tornado_bounds_hash != Some(b_hash) {
            self.tornado_bounds_hash = Some(b_hash);
            self.tornado_evaluated = 0;
            self.tornado_offset = 0;
        }
    }

    /// Advances a wide-range tornado sweep using a `HashMap`-based evaluator, bounding total
    /// model evaluations in this step to `eval_budget.clamp(2, MAX_SENSITIVITY_BATCH_EVALS)`.
    pub fn step_tornado<K, F>(
        &mut self,
        params: &[(K, f64)],
        bounds: &[TornadoBounds],
        eval_budget: usize,
        mut eval_fn: F,
    ) -> &[TornadoInterval]
    where
        K: AsRef<str>,
        F: FnMut(&HashMap<String, f64>) -> f64,
    {
        if params.is_empty() {
            self.reset_empty();
            return &self.tornado_intervals;
        }
        self.prepare_tornado(params, bounds);

        let budget = eval_budget.clamp(2, MAX_SENSITIVITY_BATCH_EVALS);
        let mut remaining_evals = budget;
        if !self.baseline_ready {
            self.baseline_output = eval_fn(&self.baseline_map);
            self.baseline_ready = true;
            remaining_evals = remaining_evals.saturating_sub(1);
        }
        if self.tornado_evaluated >= params.len() || remaining_evals < 2 {
            return &self.tornado_intervals;
        }

        let num_params = params.len();
        let max_params = (remaining_evals / 2).max(1);
        let batch_params = (num_params - self.tornado_evaluated).min(max_params);
        let y0 = self.baseline_output;

        for step in 0..batch_params {
            let idx = (self.tornado_offset + step) % num_params;
            let (ref key, p_val) = params[idx];
            let b = bounds
                .get(idx)
                .copied()
                .unwrap_or_else(|| TornadoBounds::relative(p_val, 0.2));
            set_map_value(&mut self.baseline_map, key.as_ref(), b.min);
            let y_low = eval_fn(&self.baseline_map);
            set_map_value(&mut self.baseline_map, key.as_ref(), b.max);
            let y_high = eval_fn(&self.baseline_map);
            set_map_value(&mut self.baseline_map, key.as_ref(), p_val);
            self.tornado_intervals[idx].update_in_place(p_val, b, y0, y_low, y_high);
        }

        self.tornado_offset = (self.tornado_offset + batch_params) % num_params;
        self.tornado_evaluated += batch_params;
        &self.tornado_intervals
    }

    /// Zero-allocation slice-based wide-range tornado sweep step bounded to `<= 8` evaluations.
    pub fn step_tornado_slice<K, F>(
        &mut self,
        params: &[(K, f64)],
        bounds: &[TornadoBounds],
        eval_budget: usize,
        mut eval_fn: F,
    ) -> &[TornadoInterval]
    where
        K: AsRef<str>,
        F: FnMut(&[f64]) -> f64,
    {
        if params.is_empty() {
            self.reset_empty();
            return &self.tornado_intervals;
        }
        self.prepare_tornado(params, bounds);

        let budget = eval_budget.clamp(2, MAX_SENSITIVITY_BATCH_EVALS);
        let mut remaining_evals = budget;
        if !self.baseline_ready {
            self.baseline_output = eval_fn(&self.param_scratch);
            self.baseline_ready = true;
            remaining_evals = remaining_evals.saturating_sub(1);
        }
        if self.tornado_evaluated >= params.len() || remaining_evals < 2 {
            return &self.tornado_intervals;
        }

        let num_params = params.len();
        let max_params = (remaining_evals / 2).max(1);
        let batch_params = (num_params - self.tornado_evaluated).min(max_params);
        let y0 = self.baseline_output;

        for step in 0..batch_params {
            let idx = (self.tornado_offset + step) % num_params;
            let p_val = params[idx].1;
            let b = bounds
                .get(idx)
                .copied()
                .unwrap_or_else(|| TornadoBounds::relative(p_val, 0.2));
            self.param_scratch[idx] = b.min;
            let y_low = eval_fn(&self.param_scratch);
            self.param_scratch[idx] = b.max;
            let y_high = eval_fn(&self.param_scratch);
            self.param_scratch[idx] = p_val;
            self.tornado_intervals[idx].update_in_place(p_val, b, y0, y_low, y_high);
        }

        self.tornado_offset = (self.tornado_offset + batch_params) % num_params;
        self.tornado_evaluated += batch_params;
        &self.tornado_intervals
    }
}

// [cite:math_commons]

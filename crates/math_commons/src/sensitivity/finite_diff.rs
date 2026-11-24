//! Finite-difference perturbation helpers for [`super::SensitivityEngine`].

use super::{HIGH_SENSITIVITY_THRESHOLD, SensitivityMetric};
#[cfg(feature = "std")]
use super::{MAX_SENSITIVITY_SYNC_INSTRUCTIONS, set_map_value};
#[cfg(feature = "std")]
use alloc::string::String;
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(feature = "std")]
pub(super) fn max_params_for_eval_cost(inst_per_eval: usize) -> usize {
    if inst_per_eval <= MAX_SENSITIVITY_SYNC_INSTRUCTIONS / 5 {
        2
    } else if inst_per_eval <= MAX_SENSITIVITY_SYNC_INSTRUCTIONS / 3 {
        1
    } else {
        0
    }
}

pub(super) fn perturbation_delta(p_val: f64) -> f64 {
    if p_val.abs() > 1e-12 {
        0.01 * p_val.abs()
    } else {
        0.01
    }
}

pub(super) fn finalize_metric_fields(
    metric: &mut SensitivityMetric,
    p_val: f64,
    y0: f64,
    delta: f64,
    y_plus: f64,
    y_minus: f64,
) {
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
    metric.baseline_value = p_val;
    metric.partial_derivative = dy_dp_clean;
    metric.elasticity = elasticity_clean;
    metric.is_high_sensitivity = elasticity_clean > HIGH_SENSITIVITY_THRESHOLD;
}

#[cfg(feature = "std")]
pub(super) fn update_metric_from_map<F>(
    metric: &mut SensitivityMetric,
    name: &str,
    p_val: f64,
    baseline_map: &mut HashMap<String, f64>,
    y0: f64,
    eval_fn: &mut F,
) where
    F: FnMut(&HashMap<String, f64>) -> f64,
{
    let delta = perturbation_delta(p_val);
    set_map_value(baseline_map, name, p_val + delta);
    let y_plus = eval_fn(baseline_map);
    set_map_value(baseline_map, name, p_val - delta);
    let y_minus = eval_fn(baseline_map);
    set_map_value(baseline_map, name, p_val);
    finalize_metric_fields(metric, p_val, y0, delta, y_plus, y_minus);
}

pub(super) fn update_metric_from_slice<F>(
    metric: &mut SensitivityMetric,
    idx: usize,
    p_val: f64,
    scratch: &mut [f64],
    y0: f64,
    eval_fn: &mut F,
) where
    F: FnMut(&[f64]) -> f64,
{
    let delta = perturbation_delta(p_val);
    scratch[idx] = p_val + delta;
    let y_plus = eval_fn(scratch);
    scratch[idx] = p_val - delta;
    let y_minus = eval_fn(scratch);
    scratch[idx] = p_val;
    finalize_metric_fields(metric, p_val, y0, delta, y_plus, y_minus);
}

impl super::SensitivityEngine {
    /// Evaluates normalized sensitivity indices using central finite differences (`±1%` delta).
    ///
    /// Bounds per-call evaluations to `<= 5` (`<= MAX_SENSITIVITY_BATCH_EVALS`) model runs:
    /// 1 baseline run on cache miss plus at most 2 parameters (`4` perturbations) per call.
    #[cfg(feature = "std")]
    pub fn compute<K, F>(
        &mut self,
        params: &[(K, f64)],
        eval_fn: F,
    ) -> &alloc::vec::Vec<SensitivityMetric>
    where
        K: AsRef<str>,
        F: FnMut(&HashMap<String, f64>) -> f64,
    {
        self.compute_with_context(params, &(), eval_fn)
    }

    /// Evaluates normalized sensitivity indices memoized by canonical parameter bits and `context`.
    #[cfg(feature = "std")]
    pub fn compute_with_context<K, C, F>(
        &mut self,
        params: &[(K, f64)],
        context: &C,
        eval_fn: F,
    ) -> &alloc::vec::Vec<SensitivityMetric>
    where
        K: AsRef<str>,
        C: core::hash::Hash,
        F: FnMut(&HashMap<String, f64>) -> f64,
    {
        self.compute_with_baseline(params, context, None, eval_fn)
    }

    /// Evaluates normalized sensitivity indices memoized by canonical parameter bits and `context`,
    /// reusing `baseline_output_hint` on cache misses when finite instead of invoking `eval_fn` for the baseline.
    #[cfg(feature = "std")]
    pub fn compute_with_baseline<K, C, F>(
        &mut self,
        params: &[(K, f64)],
        context: &C,
        baseline_output_hint: Option<f64>,
        eval_fn: F,
    ) -> &alloc::vec::Vec<SensitivityMetric>
    where
        K: AsRef<str>,
        C: core::hash::Hash,
        F: FnMut(&HashMap<String, f64>) -> f64,
    {
        self.compute_budgeted_with_baseline(params, context, baseline_output_hint, 1, eval_fn)
    }

    /// Evaluates normalized sensitivity indices while bounding per-pass synchronous instruction work
    /// (`inst_per_eval * evaluations <= MAX_SENSITIVITY_SYNC_INSTRUCTIONS`).
    #[cfg(feature = "std")]
    pub fn compute_budgeted_with_baseline<K, C, F>(
        &mut self,
        params: &[(K, f64)],
        context: &C,
        baseline_output_hint: Option<f64>,
        inst_per_eval: usize,
        mut eval_fn: F,
    ) -> &alloc::vec::Vec<SensitivityMetric>
    where
        K: AsRef<str>,
        C: core::hash::Hash,
        F: FnMut(&HashMap<String, f64>) -> f64,
    {
        self.gesture_deferred = false;
        if params.is_empty() {
            self.reset_empty();
            self.fd_started = true;
            return &self.metrics;
        }
        let max_batch = max_params_for_eval_cost(inst_per_eval.max(1));
        let context_hash = super::hash_context(context);
        let is_hit = self.cached_context_hash == Some(context_hash)
            && self.schema_matches(params)
            && self.values_match(params);
        if is_hit && self.fd_started && self.evaluated_count >= params.len() {
            return &self.metrics;
        }
        if !is_hit {
            self.sync_cache_on_miss(params, context_hash);
        }
        self.fd_started = true;
        if max_batch == 0 {
            self.baseline_output = baseline_output_hint.unwrap_or(f64::NAN);
            self.baseline_ready = true;
            self.evaluated_count = params.len();
            self.budget_notice = Some(alloc::format!(
                "Sensitivity budget exceeded ({inst_per_eval} instructions/run > {} cap); simplify formula or increase dt.",
                MAX_SENSITIVITY_SYNC_INSTRUCTIONS / 3
            ));
            return &self.metrics;
        }
        if !self.baseline_ready {
            self.baseline_output = baseline_output_hint
                .filter(|v| v.is_finite())
                .unwrap_or_else(|| eval_fn(&self.baseline_map));
            self.baseline_ready = true;
        }
        if max_batch == 1 && params.len() > 1 && self.budget_notice.is_none() {
            self.budget_notice = Some(alloc::format!(
                "Sensitivity budget applied: throttling perturbations to 1 parameter per pass ({MAX_SENSITIVITY_SYNC_INSTRUCTIONS} instruction cap)."
            ));
        }
        self.eval_map_perturbation_batch(params, max_batch, &mut eval_fn);
        &self.metrics
    }

    #[cfg(feature = "std")]
    fn eval_map_perturbation_batch<K, F>(
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
            update_metric_from_map(
                &mut self.metrics[idx],
                name.as_ref(),
                p_val,
                &mut self.baseline_map,
                y0,
                eval_fn,
            );
        }
        self.batch_offset = (self.batch_offset + batch_count) % num_params;
        self.evaluated_count += batch_count;
    }

    /// Allocation-free slice variant of `compute` accepting `&[f64]` parameter vectors.
    pub fn compute_slice<K, F>(
        &mut self,
        params: &[(K, f64)],
        mut eval_fn: F,
    ) -> &[SensitivityMetric]
    where
        K: AsRef<str>,
        F: FnMut(&[f64]) -> f64,
    {
        self.gesture_deferred = false;
        if params.is_empty() {
            self.reset_empty();
            self.fd_started = true;
            return &self.metrics;
        }
        let context_hash = super::hash_context(&());
        if self.cached_context_hash != Some(context_hash)
            || !self.schema_matches(params)
            || !self.values_match(params)
        {
            self.sync_cache_on_miss(params, context_hash);
        }
        self.fd_started = true;
        if !self.baseline_ready {
            self.baseline_output = eval_fn(&self.param_scratch);
            self.baseline_ready = true;
        }
        if self.evaluated_count >= params.len() {
            return &self.metrics;
        }

        let num_params = params.len();
        let batch_count = (num_params - self.evaluated_count).min(2);
        let y0 = self.baseline_output;
        for step in 0..batch_count {
            let idx = (self.batch_offset + step) % num_params;
            let p_val = params[idx].1;
            update_metric_from_slice(
                &mut self.metrics[idx],
                idx,
                p_val,
                &mut self.param_scratch,
                y0,
                &mut eval_fn,
            );
        }
        self.batch_offset = (self.batch_offset + batch_count) % num_params;
        self.evaluated_count += batch_count;
        &self.metrics
    }
}

// [cite:math_commons]

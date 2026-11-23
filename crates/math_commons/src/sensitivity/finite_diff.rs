//! Finite-difference perturbation helpers for [`super::SensitivityEngine`].

use super::{HIGH_SENSITIVITY_THRESHOLD, SensitivityMetric, set_map_value};
use std::collections::HashMap;

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

// [cite:math_commons]

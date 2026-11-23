//! Parameter sensitivity, wide-range tornado sweeps, and seeded Monte Carlo uncertainty propagation.
//!
//! Provides deterministic central finite-difference elasticity (`compute` / `compute_slice`),
//! frame-budgeted wide-range tornado sweeps (`step_tornado` / `step_tornado_slice`), and
//! explicit-seed `SplitMix64` Monte Carlo propagation with streaming Welford moments and
//! empirical quantiles (`step_monte_carlo` / `step_monte_carlo_slice`).
//!
//! Hot evaluation loops mutate preallocated scratch buffers in place without heap allocations
//! once parameter schemas and sample capacities are initialized, and bound per-frame GUI
//! batches to at most [`MAX_SENSITIVITY_BATCH_EVALS`] (`8`) model evaluations.

mod finite_diff;
mod monte_carlo;
mod tornado;

use finite_diff::{update_metric_from_map, update_metric_from_slice};

pub use monte_carlo::{
    DEFAULT_MONTE_CARLO_SAMPLES, DEFAULT_MONTE_CARLO_SEED, MonteCarloSummary, ParameterUncertainty,
    SplitMix64,
};
pub use tornado::{TornadoBounds, TornadoInterval};

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

/// Maximum number of model evaluations permitted in a single cooperative GUI frame batch.
pub const MAX_SENSITIVITY_BATCH_EVALS: usize = 8;

/// Normalized elasticity threshold above which a parameter is flagged as high sensitivity.
pub const HIGH_SENSITIVITY_THRESHOLD: f64 = 2.0;

/// Normalizes floating-point values to canonical IEEE-754 bit patterns
/// (`-0.0` maps to `+0.0` and all `NaN` payloads map to a single quiet `NaN`).
///
/// # Examples
/// ```rust
/// use math_commons::sensitivity::canonical_f64_bits;
///
/// assert_eq!(canonical_f64_bits(-0.0), canonical_f64_bits(0.0));
/// assert_eq!(canonical_f64_bits(f64::NAN), canonical_f64_bits(-f64::NAN));
/// ```
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

pub(crate) fn hash_context<C: Hash>(context: &C) -> u64 {
    let mut hasher = DefaultHasher::new();
    context.hash(&mut hasher);
    hasher.finish()
}

pub(crate) fn set_map_value(map: &mut HashMap<String, f64>, key: &str, value: f64) {
    if let Some(slot) = map.get_mut(key) {
        *slot = value;
    } else {
        map.insert(key.to_string(), value);
    }
}

/// Quantitative local finite-difference sensitivity metrics for a single input parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct SensitivityMetric {
    /// Parameter identifier or display label.
    pub param_name: String,
    /// Baseline value of the parameter.
    pub baseline_value: f64,
    /// Approximated partial derivative `dY/dP` using central finite differences (`±1%` delta).
    pub partial_derivative: f64,
    /// Normalized sensitivity index (`|(dY/dP) * (P / Y)|`).
    pub elasticity: f64,
    /// Flag indicating if elasticity exceeds [`HIGH_SENSITIVITY_THRESHOLD`] (`> 2.0`).
    pub is_high_sensitivity: bool,
}

/// Cooperative sensitivity analysis engine supporting local finite-difference elasticity,
/// wide-range tornado sweeps, and seeded Monte Carlo input-uncertainty propagation.
///
/// # Examples
/// ```rust
/// use math_commons::sensitivity::{
///     ParameterUncertainty, SensitivityEngine, TornadoBounds,
/// };
///
/// let params = [("gain", 2.0_f64), ("offset", 1.0_f64)];
/// let bounds = [TornadoBounds::new(1.0, 3.0), TornadoBounds::new(0.5, 1.5)];
/// let uncertainties = [
///     ParameterUncertainty::new(0.1, 1.0, 3.0),
///     ParameterUncertainty::new(0.05, 0.5, 1.5),
/// ];
///
/// let mut engine = SensitivityEngine::new();
/// engine.compute_slice(&params, |vals| vals[0] * vals[0] + vals[1]);
/// while !engine.is_tornado_converged() {
///     engine.step_tornado_slice(&params, &bounds, 8, |vals| vals[0] * vals[0] + vals[1]);
/// }
/// while !engine.is_monte_carlo_converged() {
///     engine.step_monte_carlo_slice(&params, &uncertainties, 42, 32, 8, |vals| {
///         vals[0] * vals[0] + vals[1]
///     });
/// }
/// assert_eq!(engine.tornado_intervals().len(), 2);
/// assert_eq!(engine.monte_carlo_summary().unwrap().sample_count, 32);
/// ```
#[derive(Debug, Clone, Default)]
pub struct SensitivityEngine {
    /// Cached local finite-difference sensitivity metrics per active parameter.
    pub metrics: Vec<SensitivityMetric>,
    /// Last calculated baseline model output `Y(P_0)`.
    pub baseline_output: f64,
    /// Current round-robin parameter index for finite-difference batching.
    pub batch_offset: usize,
    pub(crate) cached_params: Vec<(String, u64)>,
    pub(crate) cached_context_hash: Option<u64>,
    pub(crate) evaluated_count: usize,
    pub(crate) baseline_map: HashMap<String, f64>,
    pub(crate) param_scratch: Vec<f64>,
    pub(crate) baseline_ready: bool,
    pub(crate) fd_started: bool,
    pub(crate) gesture_deferred: bool,

    // Wide-range tornado sweep state
    pub(crate) tornado_intervals: Vec<TornadoInterval>,
    pub(crate) tornado_bounds_hash: Option<u64>,
    pub(crate) tornado_evaluated: usize,
    pub(crate) tornado_offset: usize,

    // Seeded Monte Carlo uncertainty propagation state
    pub(crate) mc_state: monte_carlo::MonteCarloState,
}

impl SensitivityEngine {
    /// Creates a new empty [`SensitivityEngine`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` when all active parameters have converged for finite-difference elasticity.
    #[must_use]
    pub fn is_converged(&self) -> bool {
        self.fd_started && self.evaluated_count >= self.cached_params.len()
    }

    /// Returns `true` when `params` and `context` match the memoized parameter and context state.
    #[must_use]
    pub fn matches_cache<K: AsRef<str>, C: Hash>(&self, params: &[(K, f64)], context: &C) -> bool {
        self.cached_context_hash == Some(hash_context(context))
            && self.schema_matches(params)
            && self.values_match(params)
    }

    /// Returns wide-range tornado intervals evaluated for the active parameter vector.
    #[must_use]
    pub fn tornado_intervals(&self) -> &[TornadoInterval] {
        &self.tornado_intervals
    }

    /// Returns `true` when wide-range tornado sweeps have finished for all active parameters.
    #[must_use]
    pub fn is_tornado_converged(&self) -> bool {
        !self.cached_params.is_empty()
            && self.tornado_bounds_hash.is_some()
            && self.tornado_evaluated >= self.cached_params.len()
    }

    /// Returns the streaming Monte Carlo uncertainty summary if any samples have been accumulated.
    #[must_use]
    pub fn monte_carlo_summary(&self) -> Option<&MonteCarloSummary> {
        self.mc_state.summary()
    }

    /// Returns `true` when Monte Carlo uncertainty propagation has reached its target sample count.
    #[must_use]
    pub fn is_monte_carlo_converged(&self) -> bool {
        self.mc_state.is_converged()
    }

    /// Defers finite-difference and uncertainty perturbations during an active drag or animated
    /// sweep gesture, retaining settled `metrics` and suppressing repaint requests until release.
    pub fn defer_for_gesture(&mut self) -> &Vec<SensitivityMetric> {
        self.gesture_deferred = true;
        &self.metrics
    }

    /// Returns `true` when any initiated sensitivity, tornado, or Monte Carlo batch still has remaining steps.
    #[must_use]
    pub fn has_pending_work(&self) -> bool {
        !self.gesture_deferred
            && ((self.fd_started && self.evaluated_count < self.cached_params.len())
                || (self.tornado_bounds_hash.is_some() && !self.is_tornado_converged())
                || (self.mc_state.is_active() && !self.is_monte_carlo_converged()))
    }

    pub(crate) fn schema_matches<K: AsRef<str>>(&self, params: &[(K, f64)]) -> bool {
        self.cached_params.len() == params.len()
            && self
                .cached_params
                .iter()
                .zip(params)
                .all(|((cached_key, _), (key, _))| cached_key == key.as_ref())
    }

    pub(crate) fn values_match<K: AsRef<str>>(&self, params: &[(K, f64)]) -> bool {
        self.cached_params
            .iter()
            .zip(params)
            .all(|((_, cached_bits), (_, val))| *cached_bits == canonical_f64_bits(*val))
    }

    pub(crate) fn reset_empty(&mut self) {
        self.metrics.clear();
        self.cached_params.clear();
        self.baseline_map.clear();
        self.param_scratch.clear();
        self.tornado_intervals.clear();
        self.cached_context_hash = None;
        self.evaluated_count = 0;
        self.batch_offset = 0;
        self.baseline_ready = false;
        self.fd_started = false;
        self.gesture_deferred = false;
        self.tornado_bounds_hash = None;
        self.tornado_evaluated = 0;
        self.tornado_offset = 0;
        self.mc_state.reset();
    }

    pub(crate) fn sync_cache_on_miss<K: AsRef<str>>(
        &mut self,
        params: &[(K, f64)],
        context_hash: u64,
    ) {
        if self.schema_matches(params) {
            for (idx, (key, val)) in params.iter().enumerate() {
                self.cached_params[idx].1 = canonical_f64_bits(*val);
                set_map_value(&mut self.baseline_map, key.as_ref(), *val);
                self.param_scratch[idx] = *val;
                self.metrics[idx].baseline_value = *val;
                self.tornado_intervals[idx].baseline_value = *val;
            }
        } else {
            self.rebuild_schema_buffers(params);
        }
        self.cached_context_hash = Some(context_hash);
        self.evaluated_count = 0;
        self.baseline_ready = false;
        self.fd_started = false;
        self.tornado_bounds_hash = None;
        self.tornado_evaluated = 0;
        self.tornado_offset = 0;
        self.mc_state.invalidate_config();
    }

    fn rebuild_schema_buffers<K: AsRef<str>>(&mut self, params: &[(K, f64)]) {
        self.batch_offset = 0;
        self.cached_params.clear();
        self.baseline_map.clear();
        self.param_scratch.clear();
        self.metrics.clear();
        self.tornado_intervals.clear();

        let len = params.len();
        self.cached_params.reserve(len);
        self.baseline_map.reserve(len);
        self.param_scratch.reserve(len);
        self.metrics.reserve(len);
        self.tornado_intervals.reserve(len);

        for (key, val) in params {
            let name = key.as_ref();
            self.cached_params
                .push((name.to_string(), canonical_f64_bits(*val)));
            self.baseline_map.insert(name.to_string(), *val);
            self.param_scratch.push(*val);
            self.metrics.push(SensitivityMetric {
                param_name: name.to_string(),
                baseline_value: *val,
                partial_derivative: 0.0,
                elasticity: 0.0,
                is_high_sensitivity: false,
            });
            self.tornado_intervals
                .push(TornadoInterval::empty(name.to_string(), *val));
        }
    }

    /// Evaluates normalized sensitivity indices using central finite differences (`±1%` delta).
    ///
    /// Bounds per-call evaluations to `<= 5` (`<= MAX_SENSITIVITY_BATCH_EVALS`) model runs:
    /// 1 baseline run on cache miss plus at most 2 parameters (`4` perturbations) per call.
    pub fn compute<K, F>(&mut self, params: &[(K, f64)], eval_fn: F) -> &Vec<SensitivityMetric>
    where
        K: AsRef<str>,
        F: FnMut(&HashMap<String, f64>) -> f64,
    {
        self.compute_with_context(params, &(), eval_fn)
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
            self.fd_started = true;
            return &self.metrics;
        }
        let context_hash = hash_context(context);
        if self.cached_context_hash != Some(context_hash)
            || !self.schema_matches(params)
            || !self.values_match(params)
        {
            self.sync_cache_on_miss(params, context_hash);
        }
        self.fd_started = true;
        if !self.baseline_ready {
            self.baseline_output = baseline_output_hint
                .filter(|v| v.is_finite())
                .unwrap_or_else(|| eval_fn(&self.baseline_map));
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
            let (ref name, p_val) = params[idx];
            update_metric_from_map(
                &mut self.metrics[idx],
                name.as_ref(),
                p_val,
                &mut self.baseline_map,
                y0,
                &mut eval_fn,
            );
        }
        self.batch_offset = (self.batch_offset + batch_count) % num_params;
        self.evaluated_count += batch_count;
        &self.metrics
    }

    /// Allocation-free slice variant of [`Self::compute`] accepting `&[f64]` parameter vectors.
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
        let context_hash = hash_context(&());
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

#[cfg(test)]
mod tests;

// [cite:math_commons]

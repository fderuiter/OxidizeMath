//! Seeded `SplitMix64` Monte Carlo input-uncertainty propagation with streaming Welford moments
//! and zero-allocation empirical quantile estimation.

use super::{
    MAX_SENSITIVITY_BATCH_EVALS, SensitivityEngine, canonical_f64_bits, hash_context, set_map_value,
};
use std::collections::HashMap;
use std::f64::consts::TAU;
use std::hash::{DefaultHasher, Hash, Hasher};

/// Default total Monte Carlo sample count for interactive GUI uncertainty panels.
pub const DEFAULT_MONTE_CARLO_SAMPLES: usize = 64;

/// Default deterministic 64-bit seed for reproducible Monte Carlo propagation.
pub const DEFAULT_MONTE_CARLO_SEED: u64 = 0x5EED_CAFE_0042;

/// Maximum retained Monte Carlo sample capacity to bound memory in long headless runs.
const MAX_MONTE_CARLO_SAMPLES: usize = 16_384;

/// Deterministic, zero-allocation `SplitMix64` pseudo-random generator seeded by an explicit `u64`.
///
/// Uses Sebastiano Vigna's canonical `SplitMix64` constants without touching OS entropy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Initializes a deterministic `SplitMix64` stream from `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advances the state and returns the next 64-bit pseudo-random integer.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Draws a uniform `f64` strictly inside `(0.0, 1.0)` with 53-bit mantissa resolution.
    pub fn next_open_unit_f64(&mut self) -> f64 {
        let bits = (self.next_u64() >> 11) | 1;
        (bits as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    /// Draws a deterministic standard normal `N(0, 1)` variate via Box-Muller transform.
    pub fn next_standard_normal(&mut self) -> f64 {
        let u1 = self.next_open_unit_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_open_unit_f64();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }
}

/// Bounded Gaussian input uncertainty specification for a single model parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParameterUncertainty {
    /// Standard deviation `σ >= 0` of input perturbations around the baseline value.
    pub std_dev: f64,
    /// Lower clamp bound enforced on sampled parameter realizations.
    pub min: f64,
    /// Upper clamp bound enforced on sampled parameter realizations.
    pub max: f64,
}

impl ParameterUncertainty {
    /// Creates bounded Gaussian uncertainty `(std_dev, [min, max])`.
    #[must_use]
    pub fn new(std_dev: f64, a: f64, b: f64) -> Self {
        let (min, max) = if a <= b { (a, b) } else { (b, a) };
        let clean_sigma = if std_dev.is_finite() && std_dev > 0.0 {
            std_dev
        } else {
            0.0
        };
        Self {
            std_dev: clean_sigma,
            min,
            max,
        }
    }

    /// Creates bounded relative uncertainty (`std_dev = |baseline| * rel_sigma`, clamped to `[min, max]`).
    #[must_use]
    pub fn relative(baseline: f64, rel_sigma: f64, min: f64, max: f64) -> Self {
        let scale = if baseline.abs() > 1e-12 {
            baseline.abs()
        } else {
            (max - min).abs().max(1.0)
        };
        Self::new(scale * rel_sigma.abs(), min, max)
    }

    #[inline]
    fn sample(&self, baseline: f64, rng: &mut SplitMix64) -> f64 {
        if self.std_dev <= 0.0 {
            return baseline.clamp(self.min, self.max);
        }
        let raw = baseline + self.std_dev * rng.next_standard_normal();
        if raw.is_finite() {
            raw.clamp(self.min, self.max)
        } else {
            baseline.clamp(self.min, self.max)
        }
    }
}

/// Aggregated Monte Carlo output distribution summary computed via Welford's online algorithm
/// and empirical order statistics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MonteCarloSummary {
    /// Number of finite model evaluations accumulated so far.
    pub sample_count: usize,
    /// Target total sample count configured for convergence.
    pub target_samples: usize,
    /// Welford streaming sample mean.
    pub mean: f64,
    /// Unbiased sample variance (`M2 / (n - 1)` when `n > 1`, else `0.0`).
    pub variance: f64,
    /// Sample standard deviation (`variance.sqrt()`).
    pub std_dev: f64,
    /// Empirical 5th percentile (`P5`).
    pub p05: f64,
    /// Empirical 50th percentile / median (`P50`).
    pub p50: f64,
    /// Empirical 95th percentile (`P95`).
    pub p95: f64,
    /// Minimum observed finite output.
    pub min: f64,
    /// Maximum observed finite output.
    pub max: f64,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct MonteCarloState {
    config_hash: Option<u64>,
    rng: Option<SplitMix64>,
    target_samples: usize,
    evaluated_draws: usize,
    finite_count: usize,
    welford_mean: f64,
    welford_m2: f64,
    min_val: f64,
    max_val: f64,
    samples: Vec<f64>,
    quantile_scratch: Vec<f64>,
    cached_summary: Option<MonteCarloSummary>,
}

impl MonteCarloState {
    pub(crate) fn reset(&mut self) {
        self.config_hash = None;
        self.rng = None;
        self.target_samples = 0;
        self.evaluated_draws = 0;
        self.finite_count = 0;
        self.welford_mean = 0.0;
        self.welford_m2 = 0.0;
        self.min_val = f64::INFINITY;
        self.max_val = f64::NEG_INFINITY;
        self.samples.clear();
        self.quantile_scratch.clear();
        self.cached_summary = None;
    }

    pub(crate) fn invalidate_config(&mut self) {
        self.config_hash = None;
        self.evaluated_draws = 0;
        self.finite_count = 0;
        self.cached_summary = None;
    }

    pub(crate) fn summary(&self) -> Option<&MonteCarloSummary> {
        self.cached_summary.as_ref()
    }

    pub(crate) fn is_active(&self) -> bool {
        self.config_hash.is_some() && self.target_samples > 0
    }

    pub(crate) fn is_converged(&self) -> bool {
        self.config_hash.is_some()
            && self.target_samples > 0
            && self.evaluated_draws >= self.target_samples
    }

    fn prepare(&mut self, cfg_hash: u64, seed: u64, target: usize) {
        if self.config_hash == Some(cfg_hash) {
            return;
        }
        self.config_hash = Some(cfg_hash);
        self.rng = Some(SplitMix64::new(seed));
        self.target_samples = target;
        self.evaluated_draws = 0;
        self.finite_count = 0;
        self.welford_mean = 0.0;
        self.welford_m2 = 0.0;
        self.min_val = f64::INFINITY;
        self.max_val = f64::NEG_INFINITY;
        self.samples.clear();
        if self.samples.capacity() < target {
            self.samples.reserve(target - self.samples.len());
        }
        self.quantile_scratch.resize(target, 0.0);
        self.cached_summary = None;
    }

    fn push_observation(&mut self, y: f64) {
        self.evaluated_draws += 1;
        if !y.is_finite() {
            return;
        }
        self.finite_count += 1;
        let n = self.finite_count as f64;
        let delta = y - self.welford_mean;
        self.welford_mean += delta / n;
        let delta2 = y - self.welford_mean;
        self.welford_m2 += delta * delta2;
        self.min_val = self.min_val.min(y);
        self.max_val = self.max_val.max(y);
        if self.samples.len() < self.target_samples {
            self.samples.push(y);
        }
    }

    fn recompute_summary(&mut self) {
        let n = self.samples.len();
        if n == 0 {
            self.cached_summary = None;
            return;
        }
        let work = &mut self.quantile_scratch[..n];
        work.copy_from_slice(&self.samples);
        let p05 = select_quantile_in_place(work, 0.05);
        let p50 = select_quantile_in_place(work, 0.50);
        let p95 = select_quantile_in_place(work, 0.95);
        let variance = if self.finite_count > 1 {
            (self.welford_m2 / ((self.finite_count - 1) as f64)).max(0.0)
        } else {
            0.0
        };
        self.cached_summary = Some(MonteCarloSummary {
            sample_count: self.finite_count,
            target_samples: self.target_samples,
            mean: self.welford_mean,
            variance,
            std_dev: variance.sqrt(),
            p05,
            p50,
            p95,
            min: self.min_val,
            max: self.max_val,
        });
    }
}

fn select_quantile_in_place(work: &mut [f64], q: f64) -> f64 {
    let last = work.len() - 1;
    let rank_f = (q.clamp(0.0, 1.0) * (last as f64)).round();
    let idx = (rank_f as usize).min(last);
    let (_, val, _) = work.select_nth_unstable_by(idx, f64::total_cmp);
    *val
}

fn hash_mc_config(uncertainties: &[ParameterUncertainty], seed: u64, target_samples: usize) -> u64 {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    target_samples.hash(&mut hasher);
    uncertainties.len().hash(&mut hasher);
    for u in uncertainties {
        canonical_f64_bits(u.std_dev).hash(&mut hasher);
        canonical_f64_bits(u.min).hash(&mut hasher);
        canonical_f64_bits(u.max).hash(&mut hasher);
    }
    hasher.finish()
}

impl SensitivityEngine {
    /// Pre-allocates parameter and Monte Carlo quantile buffers so subsequent calls to
    /// [`Self::step_monte_carlo_slice`] or [`Self::step_monte_carlo`] run with zero heap allocations.
    pub fn prepare_monte_carlo<K: AsRef<str>>(
        &mut self,
        params: &[(K, f64)],
        uncertainties: &[ParameterUncertainty],
        seed: u64,
        target_samples: usize,
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
        let target = target_samples.clamp(1, MAX_MONTE_CARLO_SAMPLES);
        let cfg_hash = hash_mc_config(uncertainties, seed, target);
        self.mc_state.prepare(cfg_hash, seed, target);
    }

    /// Advances seeded `SplitMix64` Monte Carlo uncertainty propagation using a `HashMap`-based
    /// evaluator, bounding evaluations per call to `eval_budget.clamp(1, MAX_SENSITIVITY_BATCH_EVALS)`.
    pub fn step_monte_carlo<K, F>(
        &mut self,
        params: &[(K, f64)],
        uncertainties: &[ParameterUncertainty],
        seed: u64,
        target_samples: usize,
        eval_budget: usize,
        mut eval_fn: F,
    ) -> Option<&MonteCarloSummary>
    where
        K: AsRef<str>,
        F: FnMut(&HashMap<String, f64>) -> f64,
    {
        if params.is_empty() {
            self.reset_empty();
            return None;
        }
        self.prepare_monte_carlo(params, uncertainties, seed, target_samples);

        let budget = eval_budget.clamp(1, MAX_SENSITIVITY_BATCH_EVALS);
        let remaining = self
            .mc_state
            .target_samples
            .saturating_sub(self.mc_state.evaluated_draws);
        let batch = remaining.min(budget);
        if batch == 0 {
            return self.mc_state.summary();
        }

        let mut rng = self.mc_state.rng.unwrap_or_else(|| SplitMix64::new(seed));
        for _ in 0..batch {
            for (idx, (key, base)) in params.iter().enumerate() {
                let sampled = uncertainties
                    .get(idx)
                    .map_or(*base, |u| u.sample(*base, &mut rng));
                set_map_value(&mut self.baseline_map, key.as_ref(), sampled);
            }
            let y = eval_fn(&self.baseline_map);
            self.mc_state.push_observation(y);
        }
        self.mc_state.rng = Some(rng);
        for (key, base) in params {
            set_map_value(&mut self.baseline_map, key.as_ref(), *base);
        }
        self.mc_state.recompute_summary();
        self.mc_state.summary()
    }

    /// Zero-allocation slice-based seeded Monte Carlo propagation step bounded to `<= 8` evaluations.
    pub fn step_monte_carlo_slice<K, F>(
        &mut self,
        params: &[(K, f64)],
        uncertainties: &[ParameterUncertainty],
        seed: u64,
        target_samples: usize,
        eval_budget: usize,
        mut eval_fn: F,
    ) -> Option<&MonteCarloSummary>
    where
        K: AsRef<str>,
        F: FnMut(&[f64]) -> f64,
    {
        if params.is_empty() {
            self.reset_empty();
            return None;
        }
        self.prepare_monte_carlo(params, uncertainties, seed, target_samples);

        let budget = eval_budget.clamp(1, MAX_SENSITIVITY_BATCH_EVALS);
        let remaining = self
            .mc_state
            .target_samples
            .saturating_sub(self.mc_state.evaluated_draws);
        let batch = remaining.min(budget);
        if batch == 0 {
            return self.mc_state.summary();
        }

        let mut rng = self.mc_state.rng.unwrap_or_else(|| SplitMix64::new(seed));
        for _ in 0..batch {
            for (idx, &(_, base)) in params.iter().enumerate() {
                let sampled = uncertainties
                    .get(idx)
                    .map_or(base, |u| u.sample(base, &mut rng));
                self.param_scratch[idx] = sampled;
            }
            let y = eval_fn(&self.param_scratch);
            self.mc_state.push_observation(y);
        }
        self.mc_state.rng = Some(rng);
        for (idx, &(_, base)) in params.iter().enumerate() {
            self.param_scratch[idx] = base;
        }
        self.mc_state.recompute_summary();
        self.mc_state.summary()
    }
}

// [cite:math_commons]

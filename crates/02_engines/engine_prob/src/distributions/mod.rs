//! Continuous and discrete `#![no_std]` probability distributions, traits, deterministic RNG
//! primitives, safeguarded quantile root-finding, and zero-allocation discrete Bayes normalization.

pub mod continuous;
pub mod discrete;
pub mod gamma_beta;
pub mod t_chi2;

pub use continuous::{Exponential, Normal, Uniform};
pub use discrete::{Bernoulli, Binomial, Categorical, Poisson};
pub use gamma_beta::{Beta, Gamma};
pub use t_chi2::{ChiSquared, StudentsT};

use crate::error::ProbError;
use rand_core::RngCore;

const INV_TWO_POW_53: f64 = 1.0 / 9_007_199_254_740_992.0;

/// Traits shared by univariate continuous probability distributions.
pub trait ContinuousDist {
    /// Evaluates the probability density function $f(x)$.
    fn pdf(&self, x: f64) -> f64;
    /// Evaluates the log-density $\ln f(x)$.
    fn ln_pdf(&self, x: f64) -> f64;
    /// Evaluates the cumulative distribution function $F(x) = P(X \le x)$.
    fn cdf(&self, x: f64) -> f64;
    /// Evaluates the quantile (inverse CDF) $F^{-1}(p)$ for $p \in [0, 1]$.
    fn quantile(&self, p: f64) -> Result<f64, ProbError>;
    /// Analytical expectation $\mathbb{E}\[X\]$.
    fn mean(&self) -> f64;
    /// Analytical variance $\operatorname{Var}(X)$.
    fn variance(&self) -> f64;
    /// Draws a deterministic sample from `rng` without OS entropy or heap allocations.
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> f64;
}

/// Traits shared by univariate discrete probability distributions over `u64` / indices.
pub trait DiscreteDist {
    /// Evaluates the probability mass function $P(X = k)$.
    fn pmf(&self, k: u64) -> f64;
    /// Evaluates the log-probability mass $\ln P(X = k)$.
    fn ln_pmf(&self, k: u64) -> f64;
    /// Evaluates the cumulative distribution function $P(X \le k)$.
    fn cdf(&self, k: u64) -> f64;
    /// Evaluates the discrete quantile $\inf \{k \in \mathbb{N}_0 : P(X \le k) \ge p\}$.
    fn quantile(&self, p: f64) -> Result<u64, ProbError>;
    /// Analytical expectation $\mathbb{E}\[X\]$.
    fn mean(&self) -> f64;
    /// Analytical variance $\operatorname{Var}(X)$.
    fn variance(&self) -> f64;
    /// Draws a deterministic integer sample from `rng` without OS entropy or heap allocations.
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> u64;
}

/// Draws a deterministic 53-bit uniform scalar in `[0, 1)` from `rng`.
#[inline]
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn next_f64_unit<R: RngCore + ?Sized>(rng: &mut R) -> f64 {
    let val = ((rng.next_u64() >> 11) as f64) * INV_TWO_POW_53;
    debug_assert!((0.0..1.0).contains(&val));
    val
}

/// Draws a deterministic 53-bit uniform scalar strictly inside `(0, 1)` from `rng`.
#[inline]
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn next_f64_open01<R: RngCore + ?Sized>(rng: &mut R) -> f64 {
    let val = (((rng.next_u64() >> 11) as f64) + 0.5) * INV_TWO_POW_53;
    debug_assert!(val > 0.0 && val < 1.0);
    val
}

/// Draws a deterministic standard normal variate via Box-Muller over `(0, 1)` uniforms.
#[inline]
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub(crate) fn sample_standard_normal<R: RngCore + ?Sized>(rng: &mut R) -> f64 {
    let u1 = next_f64_open01(rng);
    let u2 = next_f64_unit(rng);
    let radius = libm::sqrt(-2.0 * libm::log(u1));
    let z = radius * libm::cos(core::f64::consts::TAU * u2);
    debug_assert!(z.is_finite());
    z
}

/// Safeguarded Newton-bisection solver for strictly increasing continuous CDFs.
pub(crate) fn safeguarded_quantile<F, D>(
    target_p: f64,
    mut lo: f64,
    mut hi: f64,
    seed: f64,
    cdf_fn: F,
    pdf_fn: D,
) -> Result<f64, ProbError>
where
    F: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    let mut x = seed.clamp(lo, hi);
    for _ in 0..80 {
        let fx = cdf_fn(x) - target_p;
        if fx.abs() <= 1.0e-14 || (hi - lo) <= 1.0e-14 * (1.0 + x.abs()) {
            return Ok(x);
        }
        if fx < 0.0 {
            lo = x;
        } else {
            hi = x;
        }
        let dfx = pdf_fn(x);
        let candidate = if dfx > 1.0e-300 {
            x - fx / dfx
        } else {
            f64::NAN
        };
        x = if candidate.is_finite() && candidate > lo && candidate < hi {
            candidate
        } else {
            0.5 * (lo + hi)
        };
    }
    Ok(0.5 * (lo + hi))
}

/// Normalizes unnormalized log-weights in-place into probabilities in `out` via log-sum-exp,
/// returning $\ln Z = \ln \sum_i \exp(\ell_i)$ with zero heap allocations.
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn normalize_log_weights_into(log_weights: &[f64], out: &mut [f64]) -> Result<f64, ProbError> {
    let n = log_weights.len();
    if n == 0 {
        return Err(ProbError::EmptyInput {
            context: "normalize_log_weights_into",
        });
    }
    if out.len() < n {
        return Err(ProbError::BufferTooSmall {
            required: n,
            actual: out.len(),
        });
    }
    let mut max_lw = f64::NEG_INFINITY;
    for &lw in log_weights {
        if lw.is_nan() {
            return Err(ProbError::NumericalFailure {
                reason: "NaN log-weight encountered",
            });
        }
        if lw > max_lw {
            max_lw = lw;
        }
    }
    if !max_lw.is_finite() {
        let uniform = 1.0 / (n as f64);
        out[..n].fill(uniform);
        return Ok(f64::NEG_INFINITY);
    }
    let mut sum = 0.0;
    for (slot, &lw) in out[..n].iter_mut().zip(log_weights.iter()) {
        let w = libm::exp(lw - max_lw);
        *slot = w;
        sum += w;
    }
    for slot in &mut out[..n] {
        *slot /= sum;
    }
    debug_assert!(sum > 0.0);
    Ok(max_lw + libm::log(sum))
}

/// Computes normalized discrete posteriors $P(Z = k \mid x) \propto L_k \pi_k$ into `out` with
/// zero heap allocations, falling back to normalized `priors` (or uniform) if total evidence is zero.
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn discrete_posterior_into(
    likelihoods: &[f64],
    priors: &[f64],
    out: &mut [f64],
) -> Result<f64, ProbError> {
    let n = likelihoods.len();
    if n == 0 {
        return Err(ProbError::EmptyInput {
            context: "discrete_posterior_into",
        });
    }
    if priors.len() != n {
        return Err(ProbError::DimensionMismatch {
            expected: n,
            actual: priors.len(),
        });
    }
    if out.len() < n {
        return Err(ProbError::BufferTooSmall {
            required: n,
            actual: out.len(),
        });
    }
    let mut evidence = 0.0;
    for i in 0..n {
        let l = likelihoods[i].max(0.0);
        let p = priors[i].max(0.0);
        if !l.is_finite() || !p.is_finite() {
            return Err(ProbError::NumericalFailure {
                reason: "non-finite likelihood or prior in discrete_posterior_into",
            });
        }
        let prod = l * p;
        out[i] = prod;
        evidence += prod;
    }
    if evidence > 0.0 {
        for slot in &mut out[..n] {
            *slot /= evidence;
        }
        debug_assert!(out[..n].iter().all(|&v| v >= 0.0));
        return Ok(evidence);
    }
    let prior_sum: f64 = priors.iter().map(|&p| p.max(0.0)).sum();
    if prior_sum > 0.0 {
        for (slot, &p) in out[..n].iter_mut().zip(priors.iter()) {
            *slot = p.max(0.0) / prior_sum;
        }
    } else {
        out[..n].fill(1.0 / (n as f64));
    }
    Ok(0.0)
}

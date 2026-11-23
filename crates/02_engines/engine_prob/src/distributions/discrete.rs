//! Univariate discrete distributions: [`Bernoulli`], [`Binomial`], [`Poisson`], and slice-backed [`Categorical`].

use super::{DiscreteDist, next_f64_open01, next_f64_unit};
use crate::error::ProbError;
use crate::special::{
    ln_factorial, ln_gamma, regularized_incomplete_beta, regularized_upper_gamma,
    standard_normal_quantile,
};
use rand_core::RngCore;

/// Bernoulli distribution $\operatorname{Bernoulli}(p)$ over $\{0, 1\}$ with $p \in [0, 1]$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bernoulli {
    p: f64,
}

impl Bernoulli {
    /// Constructs $\operatorname{Bernoulli}(p)$ with $p \in [0, 1]$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn new(p: f64) -> Result<Self, ProbError> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(ProbError::InvalidProbability { value: p });
        }
        debug_assert!((0.0..=1.0).contains(&p));
        Ok(Self { p })
    }

    /// Success probability $p$.
    #[must_use]
    #[inline]
    pub const fn p(&self) -> f64 {
        self.p
    }
}

impl DiscreteDist for Bernoulli {
    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn pmf(&self, k: u64) -> f64 {
        match k {
            0 => 1.0 - self.p,
            1 => self.p,
            _ => 0.0,
        }
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn ln_pmf(&self, k: u64) -> f64 {
        let m = self.pmf(k);
        if m <= 0.0 {
            f64::NEG_INFINITY
        } else {
            libm::log(m)
        }
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn cdf(&self, k: u64) -> f64 {
        if k == 0 { 1.0 - self.p } else { 1.0 }
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn quantile(&self, q: f64) -> Result<u64, ProbError> {
        if !q.is_finite() || !(0.0..=1.0).contains(&q) {
            return Err(ProbError::InvalidProbability { value: q });
        }
        Ok(u64::from(q > 1.0 - self.p))
    }

    #[inline]
    fn mean(&self) -> f64 {
        self.p
    }

    #[inline]
    fn variance(&self) -> f64 {
        self.p * (1.0 - self.p)
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> u64 {
        u64::from(next_f64_unit(rng) < self.p)
    }
}

/// Binomial distribution $\operatorname{Binomial}(n, p)$ over $\{0, \dots, n\}$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Binomial {
    n: u64,
    p: f64,
}

impl Binomial {
    /// Constructs $\operatorname{Binomial}(n, p)$ with $p \in [0, 1]$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn new(n: u64, p: f64) -> Result<Self, ProbError> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(ProbError::InvalidProbability { value: p });
        }
        debug_assert!((0.0..=1.0).contains(&p));
        Ok(Self { n, p })
    }

    /// Number of trials $n$.
    #[must_use]
    #[inline]
    pub const fn n(&self) -> u64 {
        self.n
    }

    /// Trial success probability $p$.
    #[must_use]
    #[inline]
    pub const fn p(&self) -> f64 {
        self.p
    }
}

impl DiscreteDist for Binomial {
    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn pmf(&self, k: u64) -> f64 {
        let lp = self.ln_pmf(k);
        if lp.is_infinite() { 0.0 } else { libm::exp(lp) }
    }

    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn ln_pmf(&self, k: u64) -> f64 {
        if k > self.n {
            return f64::NEG_INFINITY;
        }
        if self.p == 0.0 {
            return if k == 0 { 0.0 } else { f64::NEG_INFINITY };
        }
        if self.p == 1.0 {
            return if k == self.n { 0.0 } else { f64::NEG_INFINITY };
        }
        let log_choose = ln_factorial(self.n) - ln_factorial(k) - ln_factorial(self.n - k);
        log_choose + (k as f64) * libm::log(self.p) + ((self.n - k) as f64) * libm::log1p(-self.p)
    }

    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn cdf(&self, k: u64) -> f64 {
        if k >= self.n {
            return 1.0;
        }
        if self.p == 0.0 {
            return 1.0;
        }
        if self.p == 1.0 {
            return 0.0;
        }
        regularized_incomplete_beta((self.n - k) as f64, (k + 1) as f64, 1.0 - self.p)
            .unwrap_or(f64::NAN)
    }

    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn quantile(&self, q: f64) -> Result<u64, ProbError> {
        if !q.is_finite() || !(0.0..=1.0).contains(&q) {
            return Err(ProbError::InvalidProbability { value: q });
        }
        if q == 0.0 || self.n == 0 || self.p == 0.0 {
            return Ok(0);
        }
        if q == 1.0 || self.p == 1.0 {
            return Ok(self.n);
        }
        let mut lo = 0_u64;
        let mut hi = self.n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.cdf(mid) >= q {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        Ok(lo)
    }

    #[inline]
    fn mean(&self) -> f64 {
        (self.n as f64) * self.p
    }

    #[inline]
    fn variance(&self) -> f64 {
        (self.n as f64) * self.p * (1.0 - self.p)
    }

    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> u64 {
        if self.n <= 64 {
            let mut hits = 0_u64;
            for _ in 0..self.n {
                if next_f64_unit(rng) < self.p {
                    hits += 1;
                }
            }
            return hits;
        }
        let u = next_f64_unit(rng);
        self.quantile(u).unwrap_or(0)
    }
}

/// Poisson distribution $\operatorname{Poisson}(\lambda)$ over $\mathbb{N}_0$ with rate $\lambda > 0$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Poisson {
    lambda: f64,
}

impl Poisson {
    /// Constructs $\operatorname{Poisson}(\lambda)$ with $\lambda > 0$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn new(lambda: f64) -> Result<Self, ProbError> {
        if !lambda.is_finite() || lambda <= 0.0 {
            return Err(ProbError::InvalidParameter {
                name: "lambda",
                value: lambda,
            });
        }
        debug_assert!(lambda > 0.0);
        Ok(Self { lambda })
    }

    /// Expected count rate $\lambda$.
    #[must_use]
    #[inline]
    pub const fn lambda(&self) -> f64 {
        self.lambda
    }

    fn sample_ptrs<R: RngCore + ?Sized>(lambda: f64, rng: &mut R) -> u64 {
        let slam = libm::sqrt(lambda);
        let b = 0.931 + 2.53 * slam;
        let a = -0.059 + 0.02483 * b;
        let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
        let vr = 0.9277 - 3.6224 / (b - 2.0);
        loop {
            let u = next_f64_unit(rng) - 0.5;
            let v = next_f64_open01(rng);
            let us = 0.5 - u.abs();
            let k_f = libm::floor((2.0 * a / us + b) * u + lambda + 0.43);
            if k_f < 0.0 {
                continue;
            }
            let k = k_f as u64;
            if us >= 0.07 && v <= vr {
                return k;
            }
            if us < 0.013 && v > us {
                continue;
            }
            let lhs = libm::log(v * inv_alpha / (a / (us * us) + b));
            let rhs = -lambda + (k as f64) * libm::log(lambda) - ln_gamma((k as f64) + 1.0);
            if lhs <= rhs {
                return k;
            }
        }
    }
}

impl DiscreteDist for Poisson {
    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn pmf(&self, k: u64) -> f64 {
        libm::exp(self.ln_pmf(k))
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn ln_pmf(&self, k: u64) -> f64 {
        (k as f64) * libm::log(self.lambda) - self.lambda - ln_factorial(k)
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn cdf(&self, k: u64) -> f64 {
        regularized_upper_gamma((k as f64) + 1.0, self.lambda).unwrap_or(f64::NAN)
    }

    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn quantile(&self, q: f64) -> Result<u64, ProbError> {
        if !q.is_finite() || !(0.0..=1.0).contains(&q) {
            return Err(ProbError::InvalidProbability { value: q });
        }
        if q == 0.0 {
            return Ok(0);
        }
        if q == 1.0 {
            return Err(ProbError::InvalidProbability { value: q });
        }
        let z = standard_normal_quantile(q)?.max(0.0);
        let mut hi = ((self.lambda + z * libm::sqrt(self.lambda) + 8.0) as u64).max(16);
        while self.cdf(hi) < q {
            hi = hi.saturating_mul(2);
        }
        let mut lo = 0_u64;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.cdf(mid) >= q {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        Ok(lo)
    }

    #[inline]
    fn mean(&self) -> f64 {
        self.lambda
    }

    #[inline]
    fn variance(&self) -> f64 {
        self.lambda
    }

    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> u64 {
        if self.lambda < 30.0 {
            let threshold = libm::exp(-self.lambda);
            let mut k = 0_u64;
            let mut prod = next_f64_open01(rng);
            while prod > threshold {
                k += 1;
                prod *= next_f64_open01(rng);
            }
            k
        } else {
            Self::sample_ptrs(self.lambda, rng)
        }
    }
}

/// Zero-allocation slice-backed categorical / weighted-index distribution over `{0, ..., K - 1}`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Categorical<'a> {
    weights: &'a [f64],
    total_weight: f64,
}

impl<'a> Categorical<'a> {
    /// Validates non-negative weights with strictly positive finite sum without heap allocations.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn new(weights: &'a [f64]) -> Result<Self, ProbError> {
        if weights.is_empty() {
            return Err(ProbError::EmptyInput {
                context: "Categorical::new",
            });
        }
        let mut total = 0.0;
        for &w in weights {
            if !w.is_finite() || w < 0.0 {
                return Err(ProbError::InvalidParameter {
                    name: "weight",
                    value: w,
                });
            }
            total += w;
        }
        if total <= 0.0 || !total.is_finite() {
            return Err(ProbError::InvalidParameter {
                name: "total_weight",
                value: total,
            });
        }
        debug_assert!(total > 0.0);
        Ok(Self {
            weights,
            total_weight: total,
        })
    }

    /// Number of support categories $K$.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.weights.len()
    }

    /// Whether the categorical support is empty (always `false` for validated instances).
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Draws a category index in `0..self.len()` deterministically with zero heap allocations.
    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn sample_index<R: RngCore + ?Sized>(&self, rng: &mut R) -> usize {
        let target = next_f64_unit(rng) * self.total_weight;
        let mut acc = 0.0;
        let mut last_nonzero = 0usize;
        for (idx, &w) in self.weights.iter().enumerate() {
            if w > 0.0 {
                last_nonzero = idx;
                acc += w;
                if target < acc {
                    return idx;
                }
            }
        }
        last_nonzero
    }
}

impl DiscreteDist for Categorical<'_> {
    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn pmf(&self, k: u64) -> f64 {
        let Ok(idx) = usize::try_from(k) else {
            return 0.0;
        };
        self.weights
            .get(idx)
            .map_or(0.0, |&w| w / self.total_weight)
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn ln_pmf(&self, k: u64) -> f64 {
        let p = self.pmf(k);
        if p <= 0.0 {
            f64::NEG_INFINITY
        } else {
            libm::log(p)
        }
    }

    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn cdf(&self, k: u64) -> f64 {
        let Ok(idx) = usize::try_from(k) else {
            return 1.0;
        };
        if idx + 1 >= self.weights.len() {
            return 1.0;
        }
        let prefix: f64 = self.weights[..=idx].iter().sum();
        (prefix / self.total_weight).clamp(0.0, 1.0)
    }

    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn quantile(&self, q: f64) -> Result<u64, ProbError> {
        if !q.is_finite() || !(0.0..=1.0).contains(&q) {
            return Err(ProbError::InvalidProbability { value: q });
        }
        let target = q * self.total_weight;
        let mut acc = 0.0;
        let mut last_nonzero = 0usize;
        for (idx, &w) in self.weights.iter().enumerate() {
            if w > 0.0 {
                last_nonzero = idx;
                acc += w;
                if acc >= target {
                    return Ok(idx as u64);
                }
            }
        }
        Ok(last_nonzero as u64)
    }

    fn mean(&self) -> f64 {
        let mut sum = 0.0;
        for (idx, &w) in self.weights.iter().enumerate() {
            sum += (idx as f64) * w;
        }
        sum / self.total_weight
    }

    fn variance(&self) -> f64 {
        let mu = self.mean();
        let mut sum = 0.0;
        for (idx, &w) in self.weights.iter().enumerate() {
            let d = (idx as f64) - mu;
            sum += d * d * w;
        }
        sum / self.total_weight
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> u64 {
        self.sample_index(rng) as u64
    }
}

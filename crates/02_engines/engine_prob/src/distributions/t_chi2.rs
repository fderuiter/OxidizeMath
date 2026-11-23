//! Student's $t$ ([`StudentsT`]) and Chi-Squared ([`ChiSquared`]) `#![no_std]` distributions.

use super::gamma_beta::Gamma;
use super::{ContinuousDist, safeguarded_quantile, sample_standard_normal};
use crate::error::ProbError;
use crate::special::{
    ln_gamma, regularized_incomplete_beta, regularized_lower_gamma, regularized_upper_gamma,
    standard_normal_quantile,
};
use core::f64::consts::PI;
use rand_core::RngCore;

/// Student's $t$-distribution parameterized by degrees of freedom $\nu > 0$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentsT {
    dof: f64,
}

impl StudentsT {
    /// Constructs a Student's $t$-distribution with $\nu > 0$ degrees of freedom.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn new(dof: f64) -> Result<Self, ProbError> {
        if !dof.is_finite() || dof <= 0.0 {
            return Err(ProbError::InvalidParameter {
                name: "dof",
                value: dof,
            });
        }
        debug_assert!(dof > 0.0);
        Ok(Self { dof })
    }

    /// Degrees of freedom $\nu$.
    #[must_use]
    #[inline]
    pub const fn dof(&self) -> f64 {
        self.dof
    }
}

impl ContinuousDist for StudentsT {
    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn pdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            f64::NAN
        } else if x.is_infinite() {
            0.0
        } else {
            libm::exp(self.ln_pdf(x))
        }
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn ln_pdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x.is_infinite() {
            return f64::NEG_INFINITY;
        }
        let half_v = 0.5 * self.dof;
        let half_vp1 = 0.5 * (self.dof + 1.0);
        ln_gamma(half_vp1)
            - ln_gamma(half_v)
            - 0.5 * libm::log(self.dof * PI)
            - half_vp1 * libm::log1p((x * x) / self.dof)
    }

    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x.is_infinite() {
            return if x.is_sign_positive() { 1.0 } else { 0.0 };
        }
        if x == 0.0 {
            return 0.5;
        }
        let z = self.dof / (self.dof + x * x);
        let ib = regularized_incomplete_beta(0.5 * self.dof, 0.5, z).unwrap_or(f64::NAN);
        if x > 0.0 { 1.0 - 0.5 * ib } else { 0.5 * ib }
    }

    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn quantile(&self, p: f64) -> Result<f64, ProbError> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(ProbError::InvalidProbability { value: p });
        }
        if p == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        if p == 0.5 {
            return Ok(0.0);
        }
        if (self.dof - 1.0).abs() <= 1.0e-15 {
            return Ok(libm::tan(PI * (p - 0.5)));
        }
        if (self.dof - 2.0).abs() <= 1.0e-15 {
            return Ok((2.0 * p - 1.0) / libm::sqrt(2.0 * p * (1.0 - p)));
        }
        let sign = if p > 0.5 { 1.0 } else { -1.0 };
        let p_upper = if p > 0.5 { p } else { 1.0 - p };
        let z = standard_normal_quantile(p_upper)?;
        let seed = z + (z * z * z + z) / (4.0 * self.dof);
        let mut hi = (seed * 2.0 + 4.0).max(4.0);
        while self.cdf(hi) < p_upper {
            hi *= 2.0;
        }
        let pos_q = safeguarded_quantile(
            p_upper,
            0.0,
            hi,
            seed.max(0.0),
            |t| self.cdf(t),
            |t| self.pdf(t),
        )?;
        Ok(sign * pos_q)
    }

    #[inline]
    fn mean(&self) -> f64 {
        if self.dof > 1.0 { 0.0 } else { f64::NAN }
    }

    #[inline]
    fn variance(&self) -> f64 {
        if self.dof > 2.0 {
            self.dof / (self.dof - 2.0)
        } else if self.dof > 1.0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    }

    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> f64 {
        let z = sample_standard_normal(rng);
        let chi2 = Gamma::new(0.5 * self.dof, 0.5)
            .map(|g| g.sample(rng))
            .unwrap_or(self.dof);
        z / libm::sqrt((chi2 / self.dof).max(1.0e-300))
    }
}

/// Chi-squared distribution $\chi^2(\nu)$ with $\nu > 0$ degrees of freedom.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChiSquared {
    dof: f64,
}

impl ChiSquared {
    /// Constructs $\chi^2(\nu)$ with $\nu > 0$ degrees of freedom.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn new(dof: f64) -> Result<Self, ProbError> {
        if !dof.is_finite() || dof <= 0.0 {
            return Err(ProbError::InvalidParameter {
                name: "dof",
                value: dof,
            });
        }
        debug_assert!(dof > 0.0);
        Ok(Self { dof })
    }

    /// Degrees of freedom $\nu$.
    #[must_use]
    #[inline]
    pub const fn dof(&self) -> f64 {
        self.dof
    }

    /// Upper-tail survival probability $P(X \ge x) = Q(\nu / 2, x / 2)$ avoiding catastrophic cancellation.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn survival(&self, x: f64) -> f64 {
        if x.is_nan() {
            f64::NAN
        } else if x <= 0.0 {
            1.0
        } else {
            regularized_upper_gamma(0.5 * self.dof, 0.5 * x).unwrap_or(f64::NAN)
        }
    }
}

impl ContinuousDist for ChiSquared {
    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn pdf(&self, x: f64) -> f64 {
        Gamma::new(0.5 * self.dof, 0.5)
            .map(|g| g.pdf(x))
            .unwrap_or(f64::NAN)
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn ln_pdf(&self, x: f64) -> f64 {
        Gamma::new(0.5 * self.dof, 0.5)
            .map(|g| g.ln_pdf(x))
            .unwrap_or(f64::NAN)
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            regularized_lower_gamma(0.5 * self.dof, 0.5 * x).unwrap_or(f64::NAN)
        }
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn quantile(&self, p: f64) -> Result<f64, ProbError> {
        Gamma::new(0.5 * self.dof, 0.5)?.quantile(p)
    }

    #[inline]
    fn mean(&self) -> f64 {
        self.dof
    }

    #[inline]
    fn variance(&self) -> f64 {
        2.0 * self.dof
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> f64 {
        Gamma::new(0.5 * self.dof, 0.5)
            .map(|g| g.sample(rng))
            .unwrap_or(self.dof)
    }
}

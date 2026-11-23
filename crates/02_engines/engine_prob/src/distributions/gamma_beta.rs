//! Gamma ([`Gamma`]) and Beta ([`Beta`]) `#![no_std]` univariate continuous distributions.

use super::{ContinuousDist, next_f64_open01, safeguarded_quantile, sample_standard_normal};
use crate::error::ProbError;
use crate::special::{
    ln_beta, ln_gamma, regularized_incomplete_beta, regularized_lower_gamma,
    standard_normal_quantile,
};
use rand_core::RngCore;

/// Gamma distribution $\operatorname{Gamma}(\alpha, \beta)$ parameterized by shape $\alpha > 0$ and rate $\beta > 0$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gamma {
    shape: f64,
    rate: f64,
}

impl Gamma {
    /// Constructs $\operatorname{Gamma}(\alpha, \beta)$ with shape $\alpha > 0$ and rate $\beta > 0$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn new(shape: f64, rate: f64) -> Result<Self, ProbError> {
        if !shape.is_finite() || shape <= 0.0 {
            return Err(ProbError::InvalidParameter {
                name: "shape",
                value: shape,
            });
        }
        if !rate.is_finite() || rate <= 0.0 {
            return Err(ProbError::InvalidParameter {
                name: "rate",
                value: rate,
            });
        }
        debug_assert!(shape > 0.0 && rate > 0.0);
        Ok(Self { shape, rate })
    }

    /// Shape parameter $\alpha$.
    #[must_use]
    #[inline]
    pub const fn shape(&self) -> f64 {
        self.shape
    }

    /// Rate parameter $\beta$.
    #[must_use]
    #[inline]
    pub const fn rate(&self) -> f64 {
        self.rate
    }

    pub(super) fn sample_unit_rate<R: RngCore + ?Sized>(shape: f64, rng: &mut R) -> f64 {
        if shape < 1.0 {
            let g_boost = Self::sample_marsaglia_tsang(shape + 1.0, rng);
            let u = next_f64_open01(rng);
            return g_boost * libm::pow(u, 1.0 / shape);
        }
        Self::sample_marsaglia_tsang(shape, rng)
    }

    fn sample_marsaglia_tsang<R: RngCore + ?Sized>(shape: f64, rng: &mut R) -> f64 {
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / libm::sqrt(9.0 * d);
        loop {
            let z = sample_standard_normal(rng);
            let v_root = 1.0 + c * z;
            if v_root <= 0.0 {
                continue;
            }
            let v = v_root * v_root * v_root;
            let u = next_f64_open01(rng);
            let z2 = z * z;
            if u < 1.0 - 0.0331 * z2 * z2 {
                return d * v;
            }
            if libm::log(u) < 0.5 * z2 + d * (1.0 - v + libm::log(v)) {
                return d * v;
            }
        }
    }
}

impl ContinuousDist for Gamma {
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else if x == 0.0 {
            if self.shape < 1.0 {
                f64::INFINITY
            } else if self.shape == 1.0 {
                self.rate
            } else {
                0.0
            }
        } else {
            libm::exp(self.ln_pdf(x))
        }
    }

    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn ln_pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            f64::NEG_INFINITY
        } else if x == 0.0 {
            if self.shape < 1.0 {
                f64::INFINITY
            } else if self.shape == 1.0 {
                libm::log(self.rate)
            } else {
                f64::NEG_INFINITY
            }
        } else {
            self.shape * libm::log(self.rate) + (self.shape - 1.0) * libm::log(x)
                - self.rate * x
                - ln_gamma(self.shape)
        }
    }

    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            regularized_lower_gamma(self.shape, self.rate * x).unwrap_or(f64::NAN)
        }
    }

    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn quantile(&self, p: f64) -> Result<f64, ProbError> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(ProbError::InvalidProbability { value: p });
        }
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        let z = standard_normal_quantile(p)?;
        let wh = 1.0 - 1.0 / (9.0 * self.shape) + z / libm::sqrt(9.0 * self.shape);
        let seed_unit = if wh > 0.05 {
            self.shape * wh * wh * wh
        } else {
            libm::pow(p * libm::exp(ln_gamma(self.shape + 1.0)), 1.0 / self.shape).max(1.0e-300)
        };
        let mut hi = (seed_unit * 2.0 + 4.0).max(self.shape + 10.0);
        while regularized_lower_gamma(self.shape, hi)? < p {
            hi *= 2.0;
        }
        let unit_q = safeguarded_quantile(
            p,
            0.0,
            hi,
            seed_unit,
            |u| regularized_lower_gamma(self.shape, u).unwrap_or(0.0),
            |u| {
                if u <= 0.0 {
                    0.0
                } else {
                    libm::exp((self.shape - 1.0) * libm::log(u) - u - ln_gamma(self.shape))
                }
            },
        )?;
        Ok(unit_q / self.rate)
    }

    #[inline]
    fn mean(&self) -> f64 {
        self.shape / self.rate
    }

    #[inline]
    fn variance(&self) -> f64 {
        self.shape / (self.rate * self.rate)
    }

    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> f64 {
        Self::sample_unit_rate(self.shape, rng) / self.rate
    }
}

/// Beta distribution $\operatorname{Beta}(\alpha, \beta)$ over $[0, 1]$ with $\alpha > 0, \beta > 0$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Beta {
    alpha: f64,
    beta: f64,
}

impl Beta {
    /// Constructs $\operatorname{Beta}(\alpha, \beta)$ with $\alpha > 0, \beta > 0$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn new(alpha: f64, beta: f64) -> Result<Self, ProbError> {
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(ProbError::InvalidParameter {
                name: "alpha",
                value: alpha,
            });
        }
        if !beta.is_finite() || beta <= 0.0 {
            return Err(ProbError::InvalidParameter {
                name: "beta",
                value: beta,
            });
        }
        debug_assert!(alpha > 0.0 && beta > 0.0);
        Ok(Self { alpha, beta })
    }

    /// Shape parameter $\alpha$.
    #[must_use]
    #[inline]
    pub const fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Shape parameter $\beta$.
    #[must_use]
    #[inline]
    pub const fn beta(&self) -> f64 {
        self.beta
    }
}

impl ContinuousDist for Beta {
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn pdf(&self, x: f64) -> f64 {
        if !(0.0..=1.0).contains(&x) {
            return 0.0;
        }
        if x == 0.0 || x == 1.0 {
            let s = self.ln_pdf(x);
            return if s.is_infinite() && s.is_sign_positive() {
                f64::INFINITY
            } else if s.is_finite() {
                libm::exp(s)
            } else {
                0.0
            };
        }
        libm::exp(self.ln_pdf(x))
    }

    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn ln_pdf(&self, x: f64) -> f64 {
        if !(0.0..=1.0).contains(&x) {
            return f64::NEG_INFINITY;
        }
        if x == 0.0 {
            return if self.alpha < 1.0 {
                f64::INFINITY
            } else if self.alpha == 1.0 {
                -ln_beta(1.0, self.beta)
            } else {
                f64::NEG_INFINITY
            };
        }
        if x == 1.0 {
            return if self.beta < 1.0 {
                f64::INFINITY
            } else if self.beta == 1.0 {
                -ln_beta(self.alpha, 1.0)
            } else {
                f64::NEG_INFINITY
            };
        }
        (self.alpha - 1.0) * libm::log(x) + (self.beta - 1.0) * libm::log1p(-x)
            - ln_beta(self.alpha, self.beta)
    }

    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else if x >= 1.0 {
            1.0
        } else {
            regularized_incomplete_beta(self.alpha, self.beta, x).unwrap_or(f64::NAN)
        }
    }

    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn quantile(&self, p: f64) -> Result<f64, ProbError> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(ProbError::InvalidProbability { value: p });
        }
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(1.0);
        }
        let seed = (self.alpha / (self.alpha + self.beta)).clamp(1.0e-12, 1.0 - 1.0e-12);
        safeguarded_quantile(
            p,
            0.0,
            1.0,
            seed,
            |u| regularized_incomplete_beta(self.alpha, self.beta, u).unwrap_or(0.0),
            |u| self.pdf(u),
        )
    }

    #[inline]
    fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    #[inline]
    fn variance(&self) -> f64 {
        let s = self.alpha + self.beta;
        (self.alpha * self.beta) / (s * s * (s + 1.0))
    }

    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> f64 {
        let gx = Gamma::sample_unit_rate(self.alpha, rng);
        let gy = Gamma::sample_unit_rate(self.beta, rng);
        let denom = gx + gy;
        if denom > 0.0 { gx / denom } else { self.mean() }
    }
}

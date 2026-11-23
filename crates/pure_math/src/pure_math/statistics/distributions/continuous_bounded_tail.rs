use super::special::{
    beta_quantile, gamma_unit_quantile, sample_gamma_unit_scale, sample_mean_and_var,
    validate_positive, validate_probability,
};
use super::traits::ContinuousDistribution;
use crate::error::DistributionError;
use oxidize_core::rng::OxidizeRng;
use statrs::function::{beta, gamma};
use std::f64::consts::LN_2;

/// Beta distribution $\mathrm{Beta}(\alpha, \beta)$ on $[0, 1]$ with $\alpha > 0, \beta > 0$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BetaDist {
    alpha: f64,
    beta: f64,
}

impl BetaDist {
    /// Constructs $\mathrm{Beta}(\alpha, \beta)$ verifying `alpha > 0` and `beta > 0`.
    #[verified_engine::verified]
    pub fn new(alpha: f64, beta_param: f64) -> Result<Self, DistributionError> {
        validate_positive("alpha", alpha)?;
        validate_positive("beta", beta_param)?;
        Ok(Self {
            alpha,
            beta: beta_param,
        })
    }

    /// Returns first shape parameter $\alpha$.
    #[verified_engine::verified]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns second shape parameter $\beta$.
    #[verified_engine::verified]
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Method-of-moments estimator on open unit-interval observations $x_i \in (0, 1)$.
    #[verified_engine::verified]
    pub fn fit_mom(data: &[f64]) -> Result<Self, DistributionError> {
        for &x in data {
            if !x.is_finite() || x <= 0.0 || x >= 1.0 {
                return Err(DistributionError::FittingFailed {
                    reason: "Beta observations must lie strictly in (0.0, 1.0)",
                });
            }
        }
        let (mean, _, sample_var) = sample_mean_and_var(data)?;
        let max_var = mean * (1.0 - mean);
        if sample_var <= 0.0 || sample_var >= max_var {
            return Err(DistributionError::FittingFailed {
                reason: "sample variance incompatible with Beta moment constraints",
            });
        }
        let concentration = max_var / sample_var - 1.0;
        Self::new(mean * concentration, (1.0 - mean) * concentration)
    }

    /// Maximum likelihood estimator via damped 2D digamma/trigamma Newton-Raphson initialized at MoM.
    #[verified_engine::verified]
    pub fn fit_mle(data: &[f64]) -> Result<Self, DistributionError> {
        let mom = Self::fit_mom(data)?;
        let n = data.len() as f64;
        let mean_ln_x = data.iter().map(|&x| x.ln()).sum::<f64>() / n;
        let mean_ln_1mx = data.iter().map(|&x| (-x).ln_1p()).sum::<f64>() / n;
        let mut a = mom.alpha;
        let mut b = mom.beta;
        for _ in 0..48 {
            let psi_ab = gamma::digamma(a + b);
            let g1 = gamma::digamma(a) - psi_ab - mean_ln_x;
            let g2 = gamma::digamma(b) - psi_ab - mean_ln_1mx;
            if g1.abs().max(g2.abs()) < 1e-11 {
                break;
            }
            let tri_a = trigamma_approx(a);
            let tri_b = trigamma_approx(b);
            let tri_ab = trigamma_approx(a + b);
            let det = (tri_a - tri_ab) * (tri_b - tri_ab) - tri_ab * tri_ab;
            if det.abs() <= 1e-24 || !det.is_finite() {
                break;
            }
            let da = ((tri_b - tri_ab) * g1 + tri_ab * g2) / det;
            let db = (tri_ab * g1 + (tri_a - tri_ab) * g2) / det;
            let mut step = 1.0;
            while (a - step * da <= 1e-8 || b - step * db <= 1e-8) && step > 1e-6 {
                step *= 0.5;
            }
            a = (a - step * da).max(1e-8);
            b = (b - step * db).max(1e-8);
        }
        Self::new(a, b)
    }
}

fn trigamma_approx(x: f64) -> f64 {
    let h = (x * 1e-6).max(1e-8);
    (gamma::digamma(x + h) - gamma::digamma((x - h).max(1e-12))) / (2.0 * h)
}

impl ContinuousDistribution for BetaDist {
    #[verified_engine::verified]
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 || x > 1.0 {
            0.0
        } else if x == 0.0 {
            if (self.alpha - 1.0).abs() < 1e-15 {
                self.beta
            } else if self.alpha > 1.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else if x == 1.0 {
            if (self.beta - 1.0).abs() < 1e-15 {
                self.alpha
            } else if self.beta > 1.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            self.ln_pdf(x).exp()
        }
    }

    #[verified_engine::verified]
    fn ln_pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 {
            if x == 0.0 && (self.alpha - 1.0).abs() < 1e-15 {
                self.beta.ln()
            } else if x == 1.0 && (self.beta - 1.0).abs() < 1e-15 {
                self.alpha.ln()
            } else {
                f64::NEG_INFINITY
            }
        } else {
            (self.alpha - 1.0) * x.ln() + (self.beta - 1.0) * (-x).ln_1p()
                - beta::ln_beta(self.alpha, self.beta)
        }
    }

    #[verified_engine::verified]
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else if x >= 1.0 {
            1.0
        } else {
            beta::beta_reg(self.alpha, self.beta, x)
        }
    }

    #[verified_engine::verified]
    fn quantile(&self, p: f64) -> Result<f64, DistributionError> {
        validate_probability(p)?;
        Ok(beta_quantile(self.alpha, self.beta, p))
    }

    #[verified_engine::verified]
    fn mean(&self) -> Option<f64> {
        Some(self.alpha / (self.alpha + self.beta))
    }

    #[verified_engine::verified]
    fn variance(&self) -> Option<f64> {
        let s = self.alpha + self.beta;
        Some((self.alpha * self.beta) / (s * s * (s + 1.0)))
    }

    #[verified_engine::verified]
    fn support(&self) -> (f64, f64) {
        (0.0, 1.0)
    }

    #[verified_engine::verified]
    fn sample(&self, rng: &mut OxidizeRng) -> f64 {
        let ga = sample_gamma_unit_scale(self.alpha, rng);
        let gb = sample_gamma_unit_scale(self.beta, rng);
        let denom = ga + gb;
        if denom > 0.0 {
            (ga / denom).clamp(0.0, 1.0)
        } else {
            self.alpha / (self.alpha + self.beta)
        }
    }
}

/// Chi-squared distribution $\chi^2_k$ on $[0, \infty)$ with degrees of freedom $k > 0$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChiSquaredDist {
    df: f64,
}

impl ChiSquaredDist {
    /// Constructs $\chi^2_k$ with `df > 0`.
    #[verified_engine::verified]
    pub fn new(df: f64) -> Result<Self, DistributionError> {
        validate_positive("df", df)?;
        Ok(Self { df })
    }

    /// Returns degrees of freedom $k$.
    #[verified_engine::verified]
    pub fn df(&self) -> f64 {
        self.df
    }

    /// Method-of-moments estimator $\hat{k} = \bar{x}$ on strictly positive data.
    #[verified_engine::verified]
    pub fn fit_mom(data: &[f64]) -> Result<Self, DistributionError> {
        if data.is_empty() {
            return Err(DistributionError::InsufficientData {
                required: 1,
                actual: 0,
            });
        }
        let mut sum = 0.0;
        for &x in data {
            if !x.is_finite() || x <= 0.0 {
                return Err(DistributionError::FittingFailed {
                    reason: "chi-squared observations must be strictly positive",
                });
            }
            sum += x;
        }
        Self::new(sum / (data.len() as f64))
    }

    /// Maximum likelihood estimator solving $\psi(k / 2) + \ln 2 = \overline{\ln x}$ via bisection.
    #[verified_engine::verified]
    pub fn fit_mle(data: &[f64]) -> Result<Self, DistributionError> {
        let mom = Self::fit_mom(data)?;
        let target = data.iter().map(|&x| x.ln()).sum::<f64>() / (data.len() as f64) - LN_2;
        let mut lo = (mom.df * 0.05).max(1e-4);
        let mut hi = (mom.df * 20.0).max(10.0);
        while gamma::digamma(0.5 * lo) > target && lo > 1e-8 {
            lo *= 0.5;
        }
        while gamma::digamma(0.5 * hi) < target && hi < 1e8 {
            hi *= 2.0;
        }
        for _ in 0..64 {
            let mid = 0.5 * (lo + hi);
            if gamma::digamma(0.5 * mid) < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Self::new(0.5 * (lo + hi))
    }
}

impl ContinuousDistribution for ChiSquaredDist {
    #[verified_engine::verified]
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            if x == 0.0 && (self.df - 2.0).abs() < 1e-15 {
                0.5
            } else {
                0.0
            }
        } else {
            self.ln_pdf(x).exp()
        }
    }

    #[verified_engine::verified]
    fn ln_pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            if x == 0.0 && (self.df - 2.0).abs() < 1e-15 {
                -LN_2
            } else {
                f64::NEG_INFINITY
            }
        } else {
            let half_k = 0.5 * self.df;
            -half_k * LN_2 - gamma::ln_gamma(half_k) + (half_k - 1.0) * x.ln() - 0.5 * x
        }
    }

    #[verified_engine::verified]
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            gamma::gamma_lr(0.5 * self.df, 0.5 * x)
        }
    }

    #[verified_engine::verified]
    fn quantile(&self, p: f64) -> Result<f64, DistributionError> {
        validate_probability(p)?;
        Ok(2.0 * gamma_unit_quantile(0.5 * self.df, p))
    }

    #[verified_engine::verified]
    fn mean(&self) -> Option<f64> {
        Some(self.df)
    }

    #[verified_engine::verified]
    fn variance(&self) -> Option<f64> {
        Some(2.0 * self.df)
    }

    #[verified_engine::verified]
    fn support(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }

    #[verified_engine::verified]
    fn sample(&self, rng: &mut OxidizeRng) -> f64 {
        2.0 * sample_gamma_unit_scale(0.5 * self.df, rng)
    }
}

// [cite:clinical_trials_statistics]

use super::special::{
    beta_quantile, sample_gamma_unit_scale, sample_mean_and_var, sample_standard_normal,
    validate_positive, validate_probability,
};
use super::traits::ContinuousDistribution;
use crate::error::DistributionError;
use oxidize_core::rng::OxidizeRng;
use statrs::function::{beta, gamma};
use std::f64::consts::PI;

/// Location-scale Student-$t$ distribution $t_\nu(\mu, \sigma^2)$ with $\nu > 0$ degrees of freedom.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentTDist {
    df: f64,
    location: f64,
    scale: f64,
}

impl StudentTDist {
    /// Constructs a general location-scale Student-$t$ distribution with `df > 0` and `scale > 0`.
    #[verified_engine::verified]
    pub fn new(df: f64, location: f64, scale: f64) -> Result<Self, DistributionError> {
        validate_positive("df", df)?;
        if !location.is_finite() {
            return Err(DistributionError::InvalidParameter {
                name: "location",
                value: location,
            });
        }
        validate_positive("scale", scale)?;
        Ok(Self {
            df,
            location,
            scale,
        })
    }

    /// Constructs the standard Student-$t$ distribution $t_\nu(0, 1)$.
    #[verified_engine::verified]
    pub fn standard(df: f64) -> Result<Self, DistributionError> {
        Self::new(df, 0.0, 1.0)
    }

    /// Returns degrees of freedom $\nu$.
    #[verified_engine::verified]
    pub fn df(&self) -> f64 {
        self.df
    }

    /// Returns location parameter $\mu$.
    #[verified_engine::verified]
    pub fn location(&self) -> f64 {
        self.location
    }

    /// Returns scale parameter $\sigma$.
    #[verified_engine::verified]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Method-of-moments estimator matching mean, variance, and excess kurtosis.
    #[verified_engine::verified]
    pub fn fit_mom(data: &[f64]) -> Result<Self, DistributionError> {
        if data.len() < 4 {
            return Err(DistributionError::InsufficientData {
                required: 4,
                actual: data.len(),
            });
        }
        let (mean, pop_var, sample_var) = sample_mean_and_var(data)?;
        if sample_var <= 0.0 {
            return Err(DistributionError::FittingFailed {
                reason: "zero sample variance",
            });
        }
        let m4 = data
            .iter()
            .map(|&x| {
                let d = x - mean;
                d * d * d * d
            })
            .sum::<f64>()
            / (data.len() as f64);
        let excess_kurt = m4 / (pop_var * pop_var) - 3.0;
        let df = if excess_kurt > 0.05 {
            (4.0 + 6.0 / excess_kurt).clamp(2.1, 200.0)
        } else {
            100.0
        };
        let scale = (sample_var * (df - 2.0) / df).sqrt();
        Self::new(df, mean, scale)
    }

    /// Zero-allocation EM / coordinate-ascent maximum likelihood estimator for $(\nu, \mu, \sigma)$.
    #[verified_engine::verified]
    pub fn fit_mle(data: &[f64]) -> Result<Self, DistributionError> {
        let init = Self::fit_mom(data)?;
        let n = data.len() as f64;
        let mut mu = init.location;
        let mut sigma2 = init.scale * init.scale;
        let mut nu = init.df;
        for _ in 0..32 {
            let mut sum_w = 0.0;
            let mut sum_wx = 0.0;
            let mut sum_wd2 = 0.0;
            let mut sum_ln_w = 0.0;
            for &x in data {
                let d2 = (x - mu) * (x - mu) / sigma2;
                let w = (nu + 1.0) / (nu + d2);
                sum_w += w;
                sum_wx += w * x;
                sum_wd2 += w * d2;
                sum_ln_w += w.ln() - w;
            }
            if sum_w > 0.0 {
                mu = sum_wx / sum_w;
            }
            sigma2 = (sum_wd2 / n).max(1e-16);
            let rhs =
                1.0 + sum_ln_w / n + gamma::digamma(0.5 * (nu + 1.0)) - (0.5 * (nu + 1.0)).ln();
            nu = update_student_df(rhs, nu);
        }
        Self::new(nu, mu, sigma2.sqrt())
    }
}

fn update_student_df(rhs: f64, fallback: f64) -> f64 {
    let mut lo = 0.5_f64;
    let mut hi = 200.0_f64;
    let score = |v: f64| -gamma::digamma(0.5 * v) + (0.5 * v).ln() + rhs;
    if !rhs.is_finite() || score(lo) * score(hi) > 0.0 {
        return fallback;
    }
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if score(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

impl ContinuousDistribution for StudentTDist {
    #[verified_engine::verified]
    fn pdf(&self, x: f64) -> f64 {
        self.ln_pdf(x).exp()
    }

    #[verified_engine::verified]
    fn ln_pdf(&self, x: f64) -> f64 {
        let z = (x - self.location) / self.scale;
        gamma::ln_gamma(0.5 * (self.df + 1.0))
            - gamma::ln_gamma(0.5 * self.df)
            - 0.5 * (self.df * PI).ln()
            - self.scale.ln()
            - 0.5 * (self.df + 1.0) * (z * z / self.df).ln_1p()
    }

    #[verified_engine::verified]
    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x == f64::NEG_INFINITY {
            return 0.0;
        }
        if x == f64::INFINITY {
            return 1.0;
        }
        let z = (x - self.location) / self.scale;
        let t = self.df / (self.df + z * z);
        let ib = beta::beta_reg(0.5 * self.df, 0.5, t);
        if z >= 0.0 { 1.0 - 0.5 * ib } else { 0.5 * ib }
    }

    #[verified_engine::verified]
    fn quantile(&self, p: f64) -> Result<f64, DistributionError> {
        validate_probability(p)?;
        if p == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        if (p - 0.5).abs() < 1e-16 {
            return Ok(self.location);
        }
        let tail = 2.0 * p.min(1.0 - p);
        let u = beta_quantile(0.5 * self.df, 0.5, tail).clamp(1e-15, 1.0);
        let abs_z = (self.df * (1.0 - u) / u).sqrt();
        let z = if p >= 0.5 { abs_z } else { -abs_z };
        Ok(self.location + self.scale * z)
    }

    #[verified_engine::verified]
    fn mean(&self) -> Option<f64> {
        if self.df > 1.0 {
            Some(self.location)
        } else {
            None
        }
    }

    #[verified_engine::verified]
    fn variance(&self) -> Option<f64> {
        if self.df > 2.0 {
            Some(self.scale * self.scale * self.df / (self.df - 2.0))
        } else if self.df > 1.0 {
            Some(f64::INFINITY)
        } else {
            None
        }
    }

    #[verified_engine::verified]
    fn support(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }

    #[verified_engine::verified]
    fn sample(&self, rng: &mut OxidizeRng) -> f64 {
        let z = sample_standard_normal(rng);
        let chi2_over_df = (2.0 * sample_gamma_unit_scale(0.5 * self.df, rng)) / self.df;
        self.location + self.scale * z / chi2_over_df.max(1e-30).sqrt()
    }
}

// [cite:clinical_trials_statistics]

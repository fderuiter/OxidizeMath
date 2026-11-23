use crate::error::BayesianError;
use crate::pure_math::statistics::distributions::ContinuousDistribution;

/// Posterior credible interval $[L, U]$ at probability mass `level` $\in (0, 1)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CredibleInterval {
    /// Lower endpoint $L$.
    pub lower: f64,
    /// Upper endpoint $U$.
    pub upper: f64,
    /// Enclosed posterior probability mass $1 - \alpha \in (0, 1)$.
    pub level: f64,
}

impl CredibleInterval {
    /// Returns the interval width $U - L$.
    #[verified_engine::verified]
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    /// Returns `true` if `value` lies inside `[lower, upper]`.
    #[verified_engine::verified]
    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }
}

#[verified_engine::verified]
pub(crate) fn validate_credible_level(level: f64) -> Result<f64, BayesianError> {
    if !level.is_finite() || level <= 0.0 || level >= 1.0 {
        return Err(BayesianError::InvalidCredibleLevel { level });
    }
    Ok(level)
}

#[verified_engine::verified]
pub(crate) fn validate_hyperparam(name: &'static str, value: f64) -> Result<f64, BayesianError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(BayesianError::InvalidHyperparameter { name, value });
    }
    Ok(value)
}

/// Computes the symmetric equal-tailed credible interval $[Q(\alpha/2), Q(1 - \alpha/2)]$.
#[verified_engine::verified]
pub fn equal_tailed_credible_interval<D: ContinuousDistribution>(
    dist: &D,
    level: f64,
) -> Result<CredibleInterval, BayesianError> {
    validate_credible_level(level)?;
    let tail = 0.5 * (1.0 - level);
    let lower = dist
        .quantile(tail)
        .map_err(|_| BayesianError::NumericalInstability {
            reason: "failed to invert lower quantile",
        })?;
    let upper = dist
        .quantile(1.0 - tail)
        .map_err(|_| BayesianError::NumericalInstability {
            reason: "failed to invert upper quantile",
        })?;
    Ok(CredibleInterval {
        lower,
        upper,
        level,
    })
}

/// Computes the shortest Highest Posterior Density (HPD) credible interval for unimodal posteriors
/// via golden-section minimization of $Q(p + \text{level}) - Q(p)$ over $p \in [0, 1 - \text{level}]$.
#[verified_engine::verified]
pub fn hpd_credible_interval<D: ContinuousDistribution>(
    dist: &D,
    level: f64,
) -> Result<CredibleInterval, BayesianError> {
    validate_credible_level(level)?;
    let alpha = 1.0 - level;
    let inv_phi = 0.618_033_988_749_894_9;
    let eps = (alpha * 1e-9).max(1e-13);
    let mut a = eps;
    let mut b = (alpha - eps).max(eps);
    if a >= b {
        return equal_tailed_credible_interval(dist, level);
    }

    let mut c = b - inv_phi * (b - a);
    let mut d = a + inv_phi * (b - a);
    let mut fc = interval_span(dist, c, level)?;
    let mut fd = interval_span(dist, d, level)?;

    for _ in 0..56 {
        if (b - a).abs() <= 1e-12 {
            break;
        }
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - inv_phi * (b - a);
            fc = interval_span(dist, c, level)?;
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + inv_phi * (b - a);
            fd = interval_span(dist, d, level)?;
        }
    }
    let p_opt = 0.5 * (a + b);
    let lower = dist
        .quantile(p_opt)
        .map_err(|_| BayesianError::NumericalInstability {
            reason: "HPD lower quantile failed",
        })?;
    let upper = dist
        .quantile(p_opt + level)
        .map_err(|_| BayesianError::NumericalInstability {
            reason: "HPD upper quantile failed",
        })?;
    Ok(CredibleInterval {
        lower,
        upper,
        level,
    })
}

fn interval_span<D: ContinuousDistribution>(
    dist: &D,
    p_low: f64,
    level: f64,
) -> Result<f64, BayesianError> {
    let lo = dist
        .quantile(p_low)
        .map_err(|_| BayesianError::NumericalInstability {
            reason: "HPD candidate lower quantile failed",
        })?;
    let hi = dist
        .quantile((p_low + level).min(1.0 - 1e-15))
        .map_err(|_| BayesianError::NumericalInstability {
            reason: "HPD candidate upper quantile failed",
        })?;
    Ok(hi - lo)
}

/// Qualitative strength of evidence for $M_1$ over $M_0$ on Jeffreys' (1961) scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JeffreysEvidence {
    /// $BF_{10} < 1$: evidence favors $M_0$.
    SupportsNull,
    /// $1 \le BF_{10} < \sqrt{10} \approx 3.162$: anecdotal / barely worth mentioning.
    Anecdotal,
    /// $\sqrt{10} \le BF_{10} < 10$: substantial evidence for $M_1$.
    Substantial,
    /// $10 \le BF_{10} < 31.62$: strong evidence for $M_1$.
    Strong,
    /// $31.62 \le BF_{10} < 100$: very strong evidence for $M_1$.
    VeryStrong,
    /// $BF_{10} \ge 100$: decisive evidence for $M_1$.
    Decisive,
}

/// Bayes factor $BF_{10} = \frac{P(D \mid M_1)}{P(D \mid M_0)}$ comparing $M_1$ against $M_0$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BayesFactor {
    /// Ratio $BF_{10} \ge 0$ (saturating at `f64::MAX` on extreme log-evidence).
    pub bf10: f64,
    /// Natural log Bayes factor $\ln BF_{10} = \ln P(D \mid M_1) - \ln P(D \mid M_0)$.
    pub ln_bf10: f64,
    /// Qualitative Jeffreys interpretation.
    pub evidence: JeffreysEvidence,
}

impl BayesFactor {
    /// Constructs a [`BayesFactor`] from log marginal likelihoods $\ln P(D \mid M_1)$ and $\ln P(D \mid M_0)$.
    #[verified_engine::verified]
    pub fn from_ln_marginals(ln_m1: f64, ln_m0: f64) -> Result<Self, BayesianError> {
        if !ln_m1.is_finite() || !ln_m0.is_finite() {
            return Err(BayesianError::NumericalInstability {
                reason: "non-finite log marginal likelihood",
            });
        }
        let ln_bf10 = ln_m1 - ln_m0;
        let bf10 = if ln_bf10 > 700.0 {
            f64::MAX
        } else if ln_bf10 < -745.0 {
            0.0
        } else {
            ln_bf10.exp()
        };
        Ok(Self {
            bf10,
            ln_bf10,
            evidence: classify_jeffreys(ln_bf10),
        })
    }
}

#[verified_engine::verified]
fn classify_jeffreys(ln_bf10: f64) -> JeffreysEvidence {
    let ln10 = std::f64::consts::LN_10;
    if ln_bf10 < 0.0 {
        JeffreysEvidence::SupportsNull
    } else if ln_bf10 < 0.5 * ln10 {
        JeffreysEvidence::Anecdotal
    } else if ln_bf10 < ln10 {
        JeffreysEvidence::Substantial
    } else if ln_bf10 < 1.5 * ln10 {
        JeffreysEvidence::Strong
    } else if ln_bf10 < 2.0 * ln10 {
        JeffreysEvidence::VeryStrong
    } else {
        JeffreysEvidence::Decisive
    }
}

// [cite:clinical_trials_statistics]

//! Expected utility models (`Neutral`, `CARA`, `CRRA`), certainty equivalents, and risk premia.

use super::DecisionTheoryError;
use math_commons::registry::{TOLERANCE_FAST, TOLERANCE_HIGH};

/// Parametric von Neumann-Morgenstern risk attitude for wealth lotteries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RiskAttitude {
    /// Risk-neutral linear utility $u(w) = w$.
    Neutral,
    /// Constant Absolute Risk Aversion $u(w) = \frac{1 - e^{-\alpha w}}{\alpha}$ with $\alpha > 0$.
    Cara {
        /// Arrow-Pratt absolute risk-aversion coefficient $\alpha > 0$.
        alpha: f64,
    },
    /// Constant Relative Risk Aversion $u(w) = \frac{w^{1-\gamma} - 1}{1 - \gamma}$ ($\ln w$ at $\gamma = 1$) for $\gamma \ge 0, w > 0$.
    Crra {
        /// Arrow-Pratt relative risk-aversion coefficient $\gamma \ge 0$.
        gamma: f64,
    },
}

#[verified_engine::verified]
pub(crate) fn validate_lottery(payoffs: &[f64], probs: &[f64]) -> Result<(), DecisionTheoryError> {
    if payoffs.is_empty() {
        return Err(DecisionTheoryError::EmptyOutcomes);
    }
    if payoffs.len() != probs.len() {
        return Err(DecisionTheoryError::DimensionMismatch {
            expected: payoffs.len(),
            actual: probs.len(),
        });
    }
    let mut sum = 0.0;
    for (&w, &p) in payoffs.iter().zip(probs.iter()) {
        if !w.is_finite() {
            return Err(DecisionTheoryError::InvalidRiskParameter { value: w });
        }
        if !p.is_finite() || p < -TOLERANCE_HIGH || p > 1.0 + TOLERANCE_FAST {
            return Err(DecisionTheoryError::InvalidPrior { sum: f64::NAN });
        }
        sum += p.max(0.0);
    }
    if (sum - 1.0).abs() > TOLERANCE_FAST {
        return Err(DecisionTheoryError::InvalidPrior { sum });
    }
    Ok(())
}

impl RiskAttitude {
    /// Validates risk-attitude parameters.
    #[verified_engine::verified]
    pub fn validate(self) -> Result<(), DecisionTheoryError> {
        match self {
            Self::Neutral => Ok(()),
            Self::Cara { alpha } => {
                if !alpha.is_finite() || alpha <= 0.0 {
                    Err(DecisionTheoryError::InvalidRiskParameter { value: alpha })
                } else {
                    Ok(())
                }
            }
            Self::Crra { gamma } => {
                if !gamma.is_finite() || gamma < 0.0 {
                    Err(DecisionTheoryError::InvalidRiskParameter { value: gamma })
                } else {
                    Ok(())
                }
            }
        }
    }

    /// Evaluates $u(w)$ at deterministic wealth `wealth`.
    #[verified_engine::verified]
    pub fn utility(self, wealth: f64) -> Result<f64, DecisionTheoryError> {
        self.validate()?;
        if !wealth.is_finite() {
            return Err(DecisionTheoryError::InvalidRiskParameter { value: wealth });
        }
        match self {
            Self::Neutral => Ok(wealth),
            Self::Cara { alpha } => Ok((1.0 - (-alpha * wealth).exp()) / alpha),
            Self::Crra { gamma } => {
                if wealth <= 0.0 {
                    return Err(DecisionTheoryError::NonPositiveWealth { value: wealth });
                }
                if (gamma - 1.0).abs() <= 1e-8 {
                    Ok(wealth.ln())
                } else {
                    Ok((wealth.powf(1.0 - gamma) - 1.0) / (1.0 - gamma))
                }
            }
        }
    }

    /// Evaluates expected utility $\mathbb{E}[u(W)] = \sum_i p_i u(w_i)$.
    #[verified_engine::verified]
    pub fn expected_utility(
        self,
        payoffs: &[f64],
        probs: &[f64],
    ) -> Result<f64, DecisionTheoryError> {
        self.validate()?;
        validate_lottery(payoffs, probs)?;
        let mut eu = 0.0;
        for (&w, &raw_p) in payoffs.iter().zip(probs.iter()) {
            let p = raw_p.max(0.0);
            if p > 0.0 {
                eu += p * self.utility(w)?;
            } else if matches!(self, Self::Crra { .. }) && w <= 0.0 {
                return Err(DecisionTheoryError::NonPositiveWealth { value: w });
            }
        }
        Ok(eu)
    }

    /// Computes certainty equivalent $CE = u^{-1}(\mathbb{E}[u(W)])$ using overflow-stabilized formulas.
    #[verified_engine::verified]
    pub fn certainty_equivalent(
        self,
        payoffs: &[f64],
        probs: &[f64],
    ) -> Result<f64, DecisionTheoryError> {
        self.validate()?;
        validate_lottery(payoffs, probs)?;
        match self {
            Self::Neutral => {
                let mut ev = 0.0;
                for (&w, &p) in payoffs.iter().zip(probs.iter()) {
                    ev += p.max(0.0) * w;
                }
                Ok(ev)
            }
            Self::Cara { alpha } => {
                let mut max_z = f64::NEG_INFINITY;
                for (&w, &p) in payoffs.iter().zip(probs.iter()) {
                    if p > 0.0 {
                        max_z = max_z.max(-alpha * w);
                    }
                }
                let mut scaled = 0.0;
                for (&w, &raw_p) in payoffs.iter().zip(probs.iter()) {
                    let p = raw_p.max(0.0);
                    if p > 0.0 {
                        scaled += p * (-alpha * w - max_z).exp();
                    }
                }
                Ok(-(max_z + scaled.ln()) / alpha)
            }
            Self::Crra { gamma } => {
                let mut max_w = 0.0_f64;
                for &w in payoffs {
                    if w <= 0.0 {
                        return Err(DecisionTheoryError::NonPositiveWealth { value: w });
                    }
                    max_w = max_w.max(w);
                }
                if (gamma - 1.0).abs() <= 1e-8 {
                    let mut log_ev = 0.0;
                    for (&w, &p) in payoffs.iter().zip(probs.iter()) {
                        log_ev += p.max(0.0) * w.ln();
                    }
                    Ok(log_ev.exp())
                } else {
                    let power = 1.0 - gamma;
                    let mut scaled_sum = 0.0;
                    for (&w, &raw_p) in payoffs.iter().zip(probs.iter()) {
                        let p = raw_p.max(0.0);
                        if p > 0.0 {
                            scaled_sum += p * (w / max_w).powf(power);
                        }
                    }
                    Ok(max_w * scaled_sum.powf(1.0 / power))
                }
            }
        }
    }

    /// Computes Arrow-Pratt risk premium $\pi = \mathbb{E}\[W\] - CE(W)$.
    #[verified_engine::verified]
    pub fn risk_premium(self, payoffs: &[f64], probs: &[f64]) -> Result<f64, DecisionTheoryError> {
        let ev = Self::Neutral.certainty_equivalent(payoffs, probs)?;
        let ce = self.certainty_equivalent(payoffs, probs)?;
        Ok(ev - ce)
    }
}

// [cite:graph_parameters_rust]

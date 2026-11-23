//! `#![no_std]` split conformal prediction quantiles, prediction-set masks, conformal p-values,
//! and anytime-valid [`EValue`] calibrators.

use crate::error::ProbError;

pub(crate) fn sort_f64_in_place(slice: &mut [f64]) -> Result<(), ProbError> {
    for &x in slice.iter() {
        if !x.is_finite() {
            return Err(ProbError::InvalidParameter {
                name: "score",
                value: x,
            });
        }
    }
    let len = slice.len();
    if len <= 1 {
        return Ok(());
    }
    for start in (0..len / 2).rev() {
        sift_down(slice, start, len);
    }
    for end in (1..len).rev() {
        slice.swap(0, end);
        sift_down(slice, 0, end);
    }
    Ok(())
}

#[inline]
fn sift_down(slice: &mut [f64], mut root: usize, end: usize) {
    loop {
        let left = 2 * root + 1;
        if left >= end {
            break;
        }
        let right = left + 1;
        let mut child = left;
        if right < end && slice[right] > slice[left] {
            child = right;
        }
        if slice[root] >= slice[child] {
            break;
        }
        slice.swap(root, child);
        root = child;
    }
}

/// Evaluates the exact finite-sample split conformal quantile over an ascending `sorted_scores` slice
/// at miscoverage rate $\alpha \in (0, 1)$:
///
/// $$\hat{q}_{1 - \alpha} = s_{(\lceil (n + 1)(1 - \alpha) \rceil)}$$
///
/// Returns `f64::INFINITY` when $\lceil (n + 1)(1 - \alpha) \rceil > n$, guaranteeing marginal
/// coverage $P(Y_{n+1} \in \hat{C}(X_{n+1})) \ge 1 - \alpha$ under exchangeability.
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn conformal_quantile_sorted(sorted_scores: &[f64], alpha: f64) -> Result<f64, ProbError> {
    if sorted_scores.is_empty() {
        return Err(ProbError::EmptyInput {
            context: "conformal_quantile_sorted",
        });
    }
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(ProbError::InvalidProbability { value: alpha });
    }
    let n = sorted_scores.len();
    let rank_f = libm::ceil(((n + 1) as f64) * (1.0 - alpha) - 1.0e-12);
    let rank = rank_f.max(1.0) as usize;
    if rank > n {
        return Ok(f64::INFINITY);
    }
    let val = sorted_scores[rank - 1];
    debug_assert!(val.is_finite());
    Ok(val)
}

/// Sorts `scores` in-place via deterministic $O(n \log n)$ heapsort (zero heap allocations) and
/// returns the finite-sample split conformal quantile at miscoverage rate $\alpha \in (0, 1)$.
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn conformal_quantile_into(scores: &mut [f64], alpha: f64) -> Result<f64, ProbError> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(ProbError::InvalidProbability { value: alpha });
    }
    sort_f64_in_place(scores)?;
    conformal_quantile_sorted(scores, alpha)
}

/// Convenience `alloc`-enabled wrapper computing [`conformal_quantile_into`] without mutating `scores`.
#[cfg(feature = "alloc")]
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn conformal_quantile(scores: &[f64], alpha: f64) -> Result<f64, ProbError> {
    let mut buf = alloc::vec::Vec::from(scores);
    conformal_quantile_into(&mut buf, alpha)
}

/// Populates `mask_out[i] = (candidate_scores[i] <= threshold)` and returns the cardinality of
/// the conformal prediction set $\hat{C}(x) = \{y : s(x, y) \le \hat{q}_{1 - \alpha}\}$.
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn conformal_prediction_mask_into(
    candidate_scores: &[f64],
    threshold: f64,
    mask_out: &mut [bool],
) -> Result<usize, ProbError> {
    if threshold.is_nan() {
        return Err(ProbError::InvalidParameter {
            name: "threshold",
            value: threshold,
        });
    }
    if mask_out.len() < candidate_scores.len() {
        return Err(ProbError::BufferTooSmall {
            required: candidate_scores.len(),
            actual: mask_out.len(),
        });
    }
    let mut included = 0usize;
    for (slot, &s) in mask_out[..candidate_scores.len()]
        .iter_mut()
        .zip(candidate_scores.iter())
    {
        if !s.is_finite() {
            return Err(ProbError::InvalidParameter {
                name: "candidate_score",
                value: s,
            });
        }
        let keep = s <= threshold;
        *slot = keep;
        if keep {
            included += 1;
        }
    }
    debug_assert!(included <= candidate_scores.len());
    Ok(included)
}

/// Computes the valid split conformal p-value
/// $p(s_{\text{test}}) = \frac{1 + \sum_{i=1}^n \mathbb{I}(s_i \ge s_{\text{test}})}{n + 1}$.
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn conformal_p_value(calibration_scores: &[f64], test_score: f64) -> Result<f64, ProbError> {
    if calibration_scores.is_empty() {
        return Err(ProbError::EmptyInput {
            context: "conformal_p_value",
        });
    }
    if !test_score.is_finite() {
        return Err(ProbError::InvalidParameter {
            name: "test_score",
            value: test_score,
        });
    }
    let mut ge_count = 1usize;
    for &s in calibration_scores {
        if !s.is_finite() {
            return Err(ProbError::InvalidParameter {
                name: "calibration_score",
                value: s,
            });
        }
        if s >= test_score {
            ge_count += 1;
        }
    }
    let p = (ge_count as f64) / ((calibration_scores.len() + 1) as f64);
    debug_assert!(p > 0.0 && p <= 1.0);
    Ok(p)
}

/// Non-negative anytime-valid evidence scalar $E \ge 0$ satisfying $\mathbb{E}_{H_0}\[E\] \le 1$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EValue(f64);

impl EValue {
    /// Validates and wraps a non-negative e-value scalar ($E \ge 0$, finite or $+\infty$).
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn new(value: f64) -> Result<Self, ProbError> {
        if value.is_nan() || value < 0.0 {
            return Err(ProbError::InvalidParameter {
                name: "e_value",
                value,
            });
        }
        debug_assert!(value >= 0.0);
        Ok(Self(value))
    }

    /// Neutral evidence $E = 1$.
    #[must_use]
    #[inline]
    pub const fn neutral() -> Self {
        Self(1.0)
    }

    /// Vovk-Wang admissible p-to-e calibrator $e_\kappa(p) = \kappa p^{\kappa - 1}$ for $\kappa \in (0, 1)$ and $p \in [0, 1]$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn from_p_value_power(p_value: f64, kappa: f64) -> Result<Self, ProbError> {
        if !p_value.is_finite() || !(0.0..=1.0).contains(&p_value) {
            return Err(ProbError::InvalidProbability { value: p_value });
        }
        if !kappa.is_finite() || kappa <= 0.0 || kappa >= 1.0 {
            return Err(ProbError::InvalidParameter {
                name: "kappa",
                value: kappa,
            });
        }
        if p_value == 0.0 {
            return Ok(Self(f64::INFINITY));
        }
        Ok(Self(kappa * libm::pow(p_value, kappa - 1.0)))
    }

    /// Bounded betting e-value $E = 1 + \lambda (x - \mu_0)$ for $x, \mu_0 \in [0, 1]$ and
    /// valid stake fraction $\lambda \in [-1 / (1 - \mu_0), 1 / \mu_0]$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn from_bounded_bet(obs: f64, null_mean: f64, bet_lambda: f64) -> Result<Self, ProbError> {
        if !obs.is_finite() || !(0.0..=1.0).contains(&obs) {
            return Err(ProbError::InvalidProbability { value: obs });
        }
        if !null_mean.is_finite() || null_mean <= 0.0 || null_mean >= 1.0 {
            return Err(ProbError::InvalidProbability { value: null_mean });
        }
        let min_l = -1.0 / (1.0 - null_mean);
        let max_l = 1.0 / null_mean;
        if !bet_lambda.is_finite() || bet_lambda < min_l || bet_lambda > max_l {
            return Err(ProbError::InvalidParameter {
                name: "bet_lambda",
                value: bet_lambda,
            });
        }
        Self::new((1.0 + bet_lambda * (obs - null_mean)).max(0.0))
    }

    /// Underlying numeric e-value $E$.
    #[must_use]
    #[inline]
    pub const fn value(&self) -> f64 {
        self.0
    }

    /// Combines independent or sequentially conditional e-values by multiplication ($E_1 \cdot E_2$).
    #[must_use]
    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn multiply(self, next: Self) -> Self {
        let prod = self.0 * next.0;
        debug_assert!(prod >= 0.0);
        Self(prod)
    }

    /// Averages arbitrary (possibly dependent) e-values into an admissible e-value $\frac{1}{m}\sum_{j=1}^m E_j$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn average(e_values: &[Self]) -> Result<Self, ProbError> {
        if e_values.is_empty() {
            return Err(ProbError::EmptyInput {
                context: "EValue::average",
            });
        }
        let sum: f64 = e_values.iter().map(|e| e.0).sum();
        Ok(Self(sum / (e_values.len() as f64)))
    }

    /// Ville's / Markov's inequality level-$\alpha$ rejection test: returns `true` iff $E \ge 1 / \alpha$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn rejects_at(self, alpha: f64) -> Result<bool, ProbError> {
        if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
            return Err(ProbError::InvalidProbability { value: alpha });
        }
        Ok(self.0 >= 1.0 / alpha)
    }

    /// Anytime-valid capped p-value $\min(1, 1 / E)$.
    #[must_use]
    #[inline]
    pub fn implied_p_value(self) -> f64 {
        if self.0 <= 1.0 {
            1.0
        } else {
            (1.0 / self.0).clamp(0.0, 1.0)
        }
    }
}

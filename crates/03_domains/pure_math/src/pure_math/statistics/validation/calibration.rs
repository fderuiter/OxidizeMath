//! Proper scoring rules (Brier score and domain-guarded binary log-loss) for model calibration.

use super::clv::SettledBet;
use crate::error::KellyError;

/// Comparative calibration report benchmarking model probabilities against devigged closing lines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationComparison {
    /// Number of settled bets evaluated.
    pub num_bets: usize,
    /// Mean squared error (Brier score) of the bettor's model probabilities.
    pub model_brier: f64,
    /// Mean squared error (Brier score) of the devigged closing probabilities.
    pub closing_brier: f64,
    /// Brier Skill Score $1 - \text{BS}_{\text{model}} / \text{BS}_{\text{closing}}$ (positive beats closing).
    pub brier_skill_score: f64,
    /// Binary cross-entropy (log-loss) of the bettor's model probabilities.
    pub model_log_loss: f64,
    /// Binary cross-entropy (log-loss) of the devigged closing probabilities.
    pub closing_log_loss: f64,
    /// Log-loss edge $\mathcal{L}_{\text{closing}} - \mathcal{L}_{\text{model}}$ in nats per bet (positive beats closing).
    pub log_loss_edge: f64,
}

fn single_binary_log_loss(prob: f64, won: bool) -> Result<f64, KellyError> {
    if !prob.is_finite() || prob <= 0.0 || prob >= 1.0 {
        return Err(KellyError::InvalidProbability { value: prob });
    }
    let loss = if won { -prob.ln() } else { -(-prob).ln_1p() };
    Ok(loss)
}

/// Computes the Brier score $\frac{1}{n} \sum_{i=1}^n (p_i - y_i)^2$ across settled binary outcomes.
///
/// Performs zero heap allocations.
///
/// # Errors
///
/// Returns [`KellyError::InsufficientBets`] if `probabilities` is empty,
/// [`KellyError::DimensionMismatch`] if slice lengths differ, or
/// [`KellyError::InvalidProbability`] if any $p_i \notin [0, 1]$.
///
/// # Example
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::validation::brier_score;
///
/// let probs = [0.60, 0.40, 0.75, 0.25];
/// let outcomes = [true, false, true, false];
/// let bs = brier_score(&probs, &outcomes).unwrap();
/// assert!((bs - 0.11125).abs() < 1e-12);
/// ```
#[verified_engine::verified]
pub fn brier_score(probabilities: &[f64], outcomes: &[bool]) -> Result<f64, KellyError> {
    if probabilities.len() != outcomes.len() {
        return Err(KellyError::DimensionMismatch {
            expected: probabilities.len(),
            actual: outcomes.len(),
        });
    }
    if probabilities.is_empty() {
        return Err(KellyError::InsufficientBets {
            required: 1,
            actual: 0,
        });
    }
    let mut sum_sq = 0.0;
    for (&p, &won) in probabilities.iter().zip(outcomes.iter()) {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(KellyError::InvalidProbability { value: p });
        }
        let y = if won { 1.0 } else { 0.0 };
        let diff = p - y;
        sum_sq += diff * diff;
    }
    Ok(sum_sq / (probabilities.len() as f64))
}

/// Computes domain-guarded binary log-loss $-\frac{1}{n} \sum [y_i \ln p_i + (1 - y_i) \ln(1 - p_i)]$.
///
/// Strictly guards the open domain $p_i \in (0, 1)$ and evaluates $\ln(1 - p_i)$ via `ln_1p(-p)`
/// with zero heap allocations.
///
/// # Errors
///
/// Returns [`KellyError::InsufficientBets`] if empty, [`KellyError::DimensionMismatch`] on
/// mismatched lengths, or [`KellyError::InvalidProbability`] if any $p_i \notin (0, 1)$.
#[verified_engine::verified]
pub fn binary_log_loss(probabilities: &[f64], outcomes: &[bool]) -> Result<f64, KellyError> {
    if probabilities.len() != outcomes.len() {
        return Err(KellyError::DimensionMismatch {
            expected: probabilities.len(),
            actual: outcomes.len(),
        });
    }
    if probabilities.is_empty() {
        return Err(KellyError::InsufficientBets {
            required: 1,
            actual: 0,
        });
    }
    let mut total_loss = 0.0;
    for (&p, &won) in probabilities.iter().zip(outcomes.iter()) {
        total_loss += single_binary_log_loss(p, won)?;
    }
    Ok(total_loss / (probabilities.len() as f64))
}

/// Evaluates closing-line `(brier_score, binary_log_loss)` directly over a [`SettledBet`] slice.
#[verified_engine::verified]
pub fn settled_closing_calibration(bets: &[SettledBet]) -> Result<(f64, f64), KellyError> {
    if bets.is_empty() {
        return Err(KellyError::InsufficientBets {
            required: 1,
            actual: 0,
        });
    }
    let mut sum_brier = 0.0;
    let mut sum_loss = 0.0;
    for bet in bets {
        let p = bet.closing_prob.value();
        sum_loss += single_binary_log_loss(p, bet.won)?;
        let y = if bet.won { 1.0 } else { 0.0 };
        let d = p - y;
        sum_brier += d * d;
    }
    let n = bets.len() as f64;
    Ok((sum_brier / n, sum_loss / n))
}

fn validate_calibration_slices(
    model_probs: &[f64],
    closing_probs: &[f64],
    outcomes: &[bool],
) -> Result<usize, KellyError> {
    if model_probs.len() != closing_probs.len() {
        return Err(KellyError::DimensionMismatch {
            expected: model_probs.len(),
            actual: closing_probs.len(),
        });
    }
    if model_probs.len() != outcomes.len() {
        return Err(KellyError::DimensionMismatch {
            expected: model_probs.len(),
            actual: outcomes.len(),
        });
    }
    if model_probs.is_empty() {
        return Err(KellyError::InsufficientBets {
            required: 1,
            actual: 0,
        });
    }
    Ok(model_probs.len())
}

/// Compares model probabilities against devigged closing probabilities in one zero-allocation pass.
#[verified_engine::verified]
pub fn compare_calibration(
    model_probs: &[f64],
    closing_probs: &[f64],
    outcomes: &[bool],
) -> Result<CalibrationComparison, KellyError> {
    let len = validate_calibration_slices(model_probs, closing_probs, outcomes)?;
    let mut sum_model_bs = 0.0;
    let mut sum_close_bs = 0.0;
    let mut sum_model_ll = 0.0;
    let mut sum_close_ll = 0.0;

    for i in 0..len {
        let pm = model_probs[i];
        let pc = closing_probs[i];
        let won = outcomes[i];
        sum_model_ll += single_binary_log_loss(pm, won)?;
        sum_close_ll += single_binary_log_loss(pc, won)?;
        let y = if won { 1.0 } else { 0.0 };
        let dm = pm - y;
        let dc = pc - y;
        sum_model_bs += dm * dm;
        sum_close_bs += dc * dc;
    }

    let n = len as f64;
    let model_brier = sum_model_bs / n;
    let closing_brier = sum_close_bs / n;
    let brier_skill_score = if closing_brier > 0.0 {
        1.0 - (model_brier / closing_brier)
    } else if model_brier == 0.0 {
        0.0
    } else {
        f64::NEG_INFINITY
    };
    let model_log_loss = sum_model_ll / n;
    let closing_log_loss = sum_close_ll / n;

    Ok(CalibrationComparison {
        num_bets: len,
        model_brier,
        closing_brier,
        brier_skill_score,
        model_log_loss,
        closing_log_loss,
        log_loss_edge: closing_log_loss - model_log_loss,
    })
}

// [cite:clinical_trials_statistics]

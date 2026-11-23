use super::types::{Answer, Response};
use crate::error::AIError;
pub use engine_prob::EValue;
use engine_prob::{conformal_p_value, conformal_prediction_mask_into, conformal_quantile};
use std::collections::HashMap;

fn map_prob_error(err: engine_prob::ProbError) -> AIError {
    AIError::Math(math_commons::error::MathError::NumericalError {
        reason: err.to_string(),
    })
}

/// Calculates the soft self-consistency scores for a set of responses.
///
/// # Arguments
///
/// * `responses` - A slice of responses.
///
/// # Returns
///
/// A vector of scores corresponding to each response.
#[verified_engine::verified]
pub fn calculate_soft_self_consistency_scores(responses: &[Response]) -> Vec<f64> {
    let mut answer_probabilities: HashMap<&Answer, f64> = HashMap::new();
    for response in responses {
        *answer_probabilities.entry(&response.answer).or_insert(0.0) += response.probability;
    }
    responses
        .iter()
        .map(|response| {
            answer_probabilities
                .get(&response.answer)
                .copied()
                .unwrap_or(0.0)
        })
        .collect()
}

/// Computes nonconformity scores $s_i = 1 - \operatorname{clamp}(p_i, 0, 1) \in [0, 1]$ for `responses`.
#[verified_engine::verified]
pub fn response_nonconformity_scores(responses: &[Response]) -> Vec<f64> {
    responses
        .iter()
        .map(|r| 1.0 - r.probability.clamp(0.0, 1.0))
        .collect()
}

/// Calibrates the split-conformal nonconformity threshold $\hat{q}_{1-\alpha}$ guaranteeing
/// finite-sample marginal coverage $P(Y_{n+1} \in \mathcal{C}_{1-\alpha}(X_{n+1})) \ge 1 - \alpha$
/// via [`engine_prob::conformal_quantile`].
#[verified_engine::verified]
pub fn calibrate_response_threshold(
    calibration_responses: &[Response],
    alpha: f64,
) -> Result<f64, AIError> {
    let scores = response_nonconformity_scores(calibration_responses);
    conformal_quantile(&scores, alpha).map_err(map_prob_error)
}

/// Evaluates boolean inclusion mask $s(r_j) \le \hat{q}_{1-\alpha}$ across `candidates` via
/// [`engine_prob::conformal_prediction_mask_into`].
#[verified_engine::verified]
pub fn conformal_response_mask(
    calibration_responses: &[Response],
    candidates: &[Response],
    alpha: f64,
) -> Result<Vec<bool>, AIError> {
    let threshold = calibrate_response_threshold(calibration_responses, alpha)?;
    let cand_scores = response_nonconformity_scores(candidates);
    let mut mask = vec![false; candidates.len()];
    conformal_prediction_mask_into(&cand_scores, threshold, &mut mask).map_err(map_prob_error)?;
    Ok(mask)
}

/// Returns the calibrated split-conformal prediction set $\mathcal{C}_{1-\alpha}$ of candidate
/// responses whose nonconformity score does not exceed $\hat{q}_{1-\alpha}$.
#[verified_engine::verified]
pub fn conformal_response_set<'a>(
    calibration_responses: &[Response],
    candidates: &'a [Response],
    alpha: f64,
) -> Result<Vec<&'a Response>, AIError> {
    let mask = conformal_response_mask(calibration_responses, candidates, alpha)?;
    Ok(candidates
        .iter()
        .zip(mask.into_iter())
        .filter_map(|(resp, keep)| if keep { Some(resp) } else { None })
        .collect())
}

/// Evaluates finite-sample conformal p-values for each candidate response against `calibration_responses`
/// via [`engine_prob::conformal_p_value`].
#[verified_engine::verified]
pub fn response_conformal_p_values(
    calibration_responses: &[Response],
    candidates: &[Response],
) -> Result<Vec<f64>, AIError> {
    let cal_scores = response_nonconformity_scores(calibration_responses);
    candidates
        .iter()
        .map(|cand| {
            let s = 1.0 - cand.probability.clamp(0.0, 1.0);
            conformal_p_value(&cal_scores, s).map_err(map_prob_error)
        })
        .collect()
}

/// Accumulates an anytime-valid sequential e-value process testing exchangeability / calibration of
/// `stream_responses` against `calibration_responses` using bounded conformal p-value betting
/// $E_T = \prod_{t=1}^T (1 + \lambda (0.5 - p_t))$ via [`engine_prob::EValue`].
#[verified_engine::verified]
pub fn sequential_calibration_e_value(
    calibration_responses: &[Response],
    stream_responses: &[Response],
    bet_fraction: f64,
) -> Result<EValue, AIError> {
    if !bet_fraction.is_finite() || !(-2.0..=2.0).contains(&bet_fraction) {
        return Err(AIError::InvalidParameter {
            name: "bet_fraction".to_string(),
            value: bet_fraction,
        });
    }
    let p_vals = response_conformal_p_values(calibration_responses, stream_responses)?;
    let mut e_prod = EValue::neutral();
    for p in p_vals {
        // Under exchangeability, conformal p-values satisfy E[1 - p_t] <= 0.5.
        let step = EValue::from_bounded_bet(1.0 - p, 0.5, bet_fraction).map_err(map_prob_error)?;
        e_prod = e_prod.multiply(step);
    }
    Ok(e_prod)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::self_calibration::types::Response;

    fn resp(label: &str, prob: f64) -> Response {
        Response {
            text: label.to_string(),
            probability: prob,
            answer: label.to_string(),
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_calculate_soft_self_consistency_scores() {
        let responses = vec![
            resp("A", 0.8),
            resp("B", 0.1),
            Response {
                text: "A".to_string(),
                probability: 0.05,
                answer: "A".to_string(),
            },
            Response {
                text: "B".to_string(),
                probability: 0.05,
                answer: "B".to_string(),
            },
        ];
        let scores = calculate_soft_self_consistency_scores(&responses);
        let expected_scores = [0.85, 0.15, 0.85, 0.15];
        assert_eq!(scores.len(), expected_scores.len());
        for (score, expected) in scores.iter().zip(expected_scores.iter()) {
            assert!((score - expected).abs() < math_commons::registry::TOLERANCE_STANDARD);
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_conformal_prediction_set_and_e_value_drift_detection() {
        let cal: Vec<Response> = (0..19)
            .map(|i| resp("cal", 0.95 - (i as f64) * 0.02))
            .collect();
        let q90 = calibrate_response_threshold(&cal, 0.10).unwrap();
        assert!(q90 > 0.0 && q90 < 1.0);

        let candidates = vec![resp("high", 0.90), resp("med", 0.68), resp("low", 0.15)];
        let set = conformal_response_set(&cal, &candidates, 0.10).unwrap();
        assert_eq!(set.len(), 2);
        assert_eq!(set[0].answer, "high");
        assert_eq!(set[1].answer, "med");

        // Exchangeable stream keeps e-value modest; severely degraded stream triggers Ville rejection.
        let in_dist = vec![resp("ok1", 0.88), resp("ok2", 0.82), resp("ok3", 0.91)];
        let e_null = sequential_calibration_e_value(&cal, &in_dist, 1.2).unwrap();
        assert!(!e_null.rejects_at(0.05).unwrap());

        let shifted: Vec<Response> = (0..12).map(|_| resp("bad", 0.02)).collect();
        let e_shift = sequential_calibration_e_value(&cal, &shifted, 1.5).unwrap();
        assert!(e_shift.rejects_at(0.05).unwrap());
    }
}

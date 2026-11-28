use super::scoring::calculate_soft_self_consistency_scores;
use super::temperature::{calculate_answer_entropy, map_entropy_to_temperature};
use super::types::Response;
use crate::error::AIError;

const LOG_SCORE_FLOOR: f64 = 1e-24;
const SIMPLEX_SUM_TOLERANCE: f64 = 1e-6;

#[verified_engine::verified]
fn calculate_target_distribution(responses: &[Response]) -> Result<Vec<f64>, AIError> {
    let scores = calculate_soft_self_consistency_scores(responses);
    let entropy = calculate_answer_entropy(responses);
    let temperature = map_entropy_to_temperature(entropy);
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(AIError::Math(
            math_commons::error::MathError::NumericalError {
                reason: "KL divergence temperature must be finite and positive",
            },
        ));
    }
    if temperature < math_commons::registry::TOLERANCE_STANDARD {
        let mut max_score = f64::NEG_INFINITY;
        let mut max_idx = 0;
        for (i, &score) in scores.iter().enumerate() {
            if score > max_score {
                max_score = score;
                max_idx = i;
            }
        }
        let mut p = vec![0.0; scores.len()];
        if !p.is_empty() {
            p[max_idx] = 1.0;
        }
        return Ok(p);
    }

    // Log-space temperature-scaled softmax: P_i \propto \exp(\ln(\max(S_i, \epsilon)) / \tau).
    let log_logits: Vec<f64> = scores
        .iter()
        .map(|&s| s.max(LOG_SCORE_FLOOR).ln() / temperature)
        .collect();
    let max_logit = log_logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = log_logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum_exps: f64 = exps.iter().sum();
    if !sum_exps.is_finite() || sum_exps <= 0.0 {
        return Err(AIError::Math(
            math_commons::error::MathError::NumericalError {
                reason: "KL divergence response probabilities must have positive sum",
            },
        ));
    }
    Ok(exps.into_iter().map(|e| e / sum_exps).collect())
}

fn validate_kl_inputs(responses: &[Response], predicted_dist: &[f64]) -> Result<(), AIError> {
    if responses.len() != predicted_dist.len() {
        return Err(AIError::Math(
            math_commons::error::MathError::DimensionMismatch {
                expected: math_commons::math_kernel::types::Dimension(predicted_dist.len()),
                actual: math_commons::math_kernel::types::Dimension(responses.len()),
            },
        ));
    }
    if responses.is_empty() {
        return Err(AIError::Math(
            math_commons::error::MathError::NumericalError {
                reason: "KL divergence requires non-empty distributions",
            },
        ));
    }
    let mut response_mass = 0.0_f64;
    for r in responses {
        if !r.probability.is_finite() || r.probability < 0.0 {
            return Err(AIError::Math(
                math_commons::error::MathError::NumericalError {
                    reason: "KL divergence inputs must be finite and non-negative",
                },
            ));
        }
        response_mass += r.probability;
    }
    if !response_mass.is_finite() || response_mass <= 0.0 {
        return Err(AIError::Math(
            math_commons::error::MathError::NumericalError {
                reason: "KL divergence response probabilities must have positive sum",
            },
        ));
    }

    let mut sum_q = 0.0_f64;
    for &q in predicted_dist {
        if !q.is_finite() || !(0.0..=1.0).contains(&q) {
            return Err(AIError::Math(
                math_commons::error::MathError::NumericalError {
                    reason: "KL divergence inputs must be finite and non-negative",
                },
            ));
        }
        sum_q += q;
    }
    if !sum_q.is_finite() || (sum_q - 1.0).abs() > SIMPLEX_SUM_TOLERANCE {
        return Err(AIError::Math(
            math_commons::error::MathError::NumericalError {
                reason: "KL divergence predicted distribution must sum to 1.0",
            },
        ));
    }
    Ok(())
}

/// Calculates the KL divergence loss $D_{\mathrm{KL}}(Q \parallel P) = \sum_i Q_i \ln(Q_i / P_i)$
/// between the predicted simplex distribution $Q$ and the log-space self-consistency target $P$.
#[verified_engine::verified]
pub fn calculate_kl_divergence_loss(
    responses: &[Response],
    predicted_dist: &[f64],
) -> Result<f64, AIError> {
    validate_kl_inputs(responses, predicted_dist)?;
    let target_dist = calculate_target_distribution(responses)?;
    let mut kl_divergence = 0.0_f64;
    for (&q, &p) in predicted_dist.iter().zip(target_dist.iter()) {
        if q > 0.0 {
            let p_safe = p.max(f64::MIN_POSITIVE);
            kl_divergence += q * (q / p_safe).ln();
        }
    }
    Ok(kl_divergence.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::self_calibration::types::Response;

    #[test]
    #[verified_engine::verified]
    fn test_kl_divergence_calculation() -> Result<(), crate::error::AIError> {
        let responses = vec![
            Response {
                text: "A".to_string(),
                probability: 0.8,
                answer: "A".to_string(),
            },
            Response {
                text: "B".to_string(),
                probability: 0.1,
                answer: "B".to_string(),
            },
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
        let predicted_dist_q = vec![0.4, 0.1, 0.4, 0.1];
        let loss = calculate_kl_divergence_loss(&responses, &predicted_dist_q)?;
        assert!((loss - 0.0007855814).abs() < math_commons::registry::TOLERANCE_FAST);
        Ok(())
    }

    #[test]
    #[verified_engine::verified]
    fn test_kl_divergence_rejects_invalid_simplex_and_suppresses_zero_prob_answers() {
        let valid_responses = vec![
            Response {
                text: "correct".to_string(),
                probability: 0.9,
                answer: "42".to_string(),
            },
            Response {
                text: "zero-prob".to_string(),
                probability: 0.0,
                answer: "wrong".to_string(),
            },
        ];

        // Target distribution must assign < 1e-6 mass to 0.0-probability answers.
        let target = calculate_target_distribution(&valid_responses).expect("valid target");
        assert!(
            target[1] < 1e-6,
            "0.0-probability answer received excessive mass {}",
            target[1]
        );
        assert!((target[0] + target[1] - 1.0).abs() < 1e-12);

        // [0.0, 0.0] and [NaN, NaN] must return Err on both predicted_dist and responses.
        assert!(calculate_kl_divergence_loss(&valid_responses, &[0.0, 0.0]).is_err());
        assert!(calculate_kl_divergence_loss(&valid_responses, &[f64::NAN, f64::NAN]).is_err());
        assert!(calculate_kl_divergence_loss(&valid_responses, &[0.7, 0.7]).is_err());
        assert!(calculate_kl_divergence_loss(&[], &[]).is_err());

        let zero_responses = vec![
            Response {
                text: "a".to_string(),
                probability: 0.0,
                answer: "a".to_string(),
            },
            Response {
                text: "b".to_string(),
                probability: 0.0,
                answer: "b".to_string(),
            },
        ];
        assert!(calculate_kl_divergence_loss(&zero_responses, &[0.5, 0.5]).is_err());

        let nan_responses = vec![
            Response {
                text: "a".to_string(),
                probability: f64::NAN,
                answer: "a".to_string(),
            },
            Response {
                text: "b".to_string(),
                probability: f64::NAN,
                answer: "b".to_string(),
            },
        ];
        assert!(calculate_kl_divergence_loss(&nan_responses, &[0.5, 0.5]).is_err());
    }
}

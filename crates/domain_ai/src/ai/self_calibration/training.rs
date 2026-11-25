use super::scoring::calculate_soft_self_consistency_scores;
use super::temperature::{calculate_answer_entropy, map_entropy_to_temperature};
use super::types::Response;

#[verified_engine::verified]
fn calculate_target_distribution(responses: &[Response]) -> Vec<f64> {
    let scores = calculate_soft_self_consistency_scores(responses);
    let entropy = calculate_answer_entropy(responses);
    let temperature = map_entropy_to_temperature(entropy);
    if temperature.abs() < math_commons::registry::TOLERANCE_STANDARD {
        let mut max_score = f64::NEG_INFINITY;
        let mut max_idx = 0;
        for (i, score) in scores.iter().enumerate() {
            if *score > max_score {
                max_score = *score;
                max_idx = i;
            }
        }
        let mut p = vec![0.0; scores.len()];
        if !p.is_empty() {
            p[max_idx] = 1.0;
        }
        return p;
    }
    let temp_scores: Vec<f64> = scores.iter().map(|s| s / temperature).collect();
    let max_temp_score = temp_scores
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = temp_scores
        .iter()
        .map(|s| (s - max_temp_score).exp())
        .collect();
    let sum_exps: f64 = exps.iter().sum();
    if sum_exps.abs() < math_commons::registry::TOLERANCE_STANDARD {
        let n = responses.len();
        if n > 0 {
            vec![1.0 / n as f64; n]
        } else {
            vec![]
        }
    } else {
        exps.iter().map(|e| e / sum_exps).collect()
    }
}

use crate::error::AIError;

/// Calculates the KL divergence loss between the predicted distribution and the target distribution.
///
/// # Arguments
///
/// * `responses` - A slice of responses (used to calculate target distribution).
/// * `predicted_dist` - The predicted probability distribution.
///
/// # Returns
///
/// The KL divergence loss.
#[verified_engine::verified]
pub fn calculate_kl_divergence_loss(
    responses: &[Response],
    predicted_dist: &[f64],
) -> Result<f64, AIError> {
    if responses.len() != predicted_dist.len() {
        return Err(AIError::Math(
            math_commons::error::MathError::DimensionMismatch {
                expected: math_commons::math_kernel::types::Dimension(predicted_dist.len()),
                actual: math_commons::math_kernel::types::Dimension(responses.len()),
            },
        ));
    }
    if responses.is_empty() {
        return Ok(0.0);
    }
    let target_dist = calculate_target_distribution(responses);
    let mut kl_divergence = 0.0;
    for (q, p) in predicted_dist.iter().zip(target_dist.iter()) {
        if *q > 0.0 && *p > 0.0 {
            kl_divergence += q * (q / p).ln();
        }
    }
    Ok(kl_divergence)
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
        assert!((loss - 0.07019869).abs() < math_commons::registry::TOLERANCE_FAST);
        Ok(())
    }
}

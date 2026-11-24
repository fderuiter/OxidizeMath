// [cite:self_calibration]
// [cite:self_calibration_paper]

use super::scoring::calculate_soft_self_consistency_scores;
use super::temperature::{calculate_answer_entropy, map_entropy_to_temperature};
use super::types::Response;

use crate::error::AIError;

/// Computes the dynamically temperature-scaled target probability distribution $P(y_i \mid x)$
/// over sampled candidate responses using soft self-consistency scores and answer-level Shannon entropy.
///
/// First, candidate responses sharing equivalent normalized answers pool their generation
/// probabilities via [`calculate_soft_self_consistency_scores`]. Next, the answer-group Shannon
/// entropy $H(A)$ is mapped onto an adaptive softmax temperature $T(H)$ via
/// [`map_entropy_to_temperature`]. When $T \approx 0$, probability mass concentrates on the
/// highest-scoring response; otherwise a numerically stabilized Boltzmann distribution
/// $P_i \propto \exp(S_i / T)$ is returned.
///
/// # Arguments
///
/// * `responses` - Slice of sampled [`Response`] candidates with model probabilities and extracted answers.
///
/// # Returns
///
/// A vector of target probabilities summing to `1.0` (or an empty vector if `responses` is empty).
///
/// # Example
///
/// ```
/// # use domain_ai::__doctest_facade as math_explorer;
/// use math_explorer::ai::self_calibration::training::calculate_target_distribution;
/// use math_explorer::ai::self_calibration::types::Response;
///
/// let responses = vec![
///     Response { text: "Step 1 -> 42".into(), probability: 0.7, answer: "42".into() },
///     Response { text: "Alt -> 42".into(), probability: 0.2, answer: "42".into() },
///     Response { text: "Guess -> 7".into(), probability: 0.1, answer: "7".into() },
/// ];
/// let target = calculate_target_distribution(&responses);
/// assert_eq!(target.len(), 3);
/// assert!((target.iter().sum::<f64>() - 1.0).abs() < 1e-9);
/// assert!(target[0] > target[2]);
/// ```
#[verified_engine::verified]
pub fn calculate_target_distribution(responses: &[Response]) -> Vec<f64> {
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
        debug_assert!(
            p.len() == responses.len(),
            "Target length matches responses"
        );
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

/// Calculates the KL divergence loss $D_{\mathrm{KL}}(Q \parallel P)$ between a predicted response
/// distribution $Q$ and the entropy-calibrated target distribution $P$ from
/// [`calculate_target_distribution`].
///
/// # Arguments
///
/// * `responses` - A slice of responses (used to calculate target distribution).
/// * `predicted_dist` - The predicted probability distribution.
///
/// # Returns
///
/// The KL divergence loss, or [`AIError::Math`] if slice lengths mismatch.
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
    debug_assert!(kl_divergence.is_finite(), "KL divergence must be finite");
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

    #[test]
    #[verified_engine::verified]
    fn test_target_distribution_simplex_and_consensus_concentration() {
        assert!(calculate_target_distribution(&[]).is_empty());

        let unanimous = vec![
            Response {
                text: "r1".to_string(),
                probability: 0.6,
                answer: "42".to_string(),
            },
            Response {
                text: "r2".to_string(),
                probability: 0.4,
                answer: "42".to_string(),
            },
        ];
        let p_unanimous = calculate_target_distribution(&unanimous);
        assert_eq!(p_unanimous.len(), 2);
        assert!((p_unanimous.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((p_unanimous[0] - p_unanimous[1]).abs() < 1e-12);

        let mixed = vec![
            Response {
                text: "a1".to_string(),
                probability: 0.55,
                answer: "yes".to_string(),
            },
            Response {
                text: "a2".to_string(),
                probability: 0.30,
                answer: "yes".to_string(),
            },
            Response {
                text: "b1".to_string(),
                probability: 0.15,
                answer: "no".to_string(),
            },
        ];
        let p_mixed = calculate_target_distribution(&mixed);
        assert_eq!(p_mixed.len(), 3);
        assert!((p_mixed.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(p_mixed[0] > p_mixed[2]);
        assert!((p_mixed[0] - p_mixed[1]).abs() < 1e-12);
    }
}

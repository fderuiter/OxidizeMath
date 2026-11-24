// [cite:self_calibration_paper]
use super::types::{Answer, Response};
use crate::error::AIError;
use std::collections::HashMap;

pub use pure_math::pure_math::statistics::calibration::TemperatureScaler;

/// Fits a supervised post-hoc [`TemperatureScaler`] minimizing held-out binary `log_loss`.
///
/// Unlike [`map_entropy_to_temperature`]—which computes a label-free self-consistency heuristic
/// from clustered response strings—this function requires ground-truth binary `outcomes`.
#[verified_engine::verified]
pub fn fit_binary_temperature_scaling(
    logits: &[f64],
    outcomes: &[bool],
) -> Result<TemperatureScaler, AIError> {
    Ok(TemperatureScaler::fit_binary(logits, outcomes)?)
}

/// Fits a supervised post-hoc [`TemperatureScaler`] minimizing held-out multiclass `log_loss`
/// over row-major `logits_flat` (`targets.len() * num_classes`).
#[verified_engine::verified]
pub fn fit_multiclass_temperature_scaling(
    logits_flat: &[f64],
    num_classes: usize,
    targets: &[usize],
) -> Result<TemperatureScaler, AIError> {
    Ok(TemperatureScaler::fit_multiclass(
        logits_flat,
        num_classes,
        targets,
    )?)
}

/// Calculates the entropy of the answer distribution.
///
/// # Arguments
///
/// * `responses` - A slice of responses.
///
/// # Returns
///
/// The entropy value.
#[verified_engine::verified]
pub fn calculate_answer_entropy(responses: &[Response]) -> f64 {
    if responses.is_empty() {
        return 0.0;
    }
    let mut answer_counts: HashMap<&Answer, usize> = HashMap::new();
    for response in responses {
        *answer_counts.entry(&response.answer).or_insert(0) += 1;
    }
    let total_responses = responses.len() as f64;
    let mut entropy = 0.0;
    for count in answer_counts.values() {
        let p_a = (*count as f64) / total_responses;
        if p_a > 0.0 {
            entropy -= p_a * p_a.ln();
        }
    }
    entropy
}

/// Maps entropy to a temperature value.
///
/// # Arguments
///
/// * `entropy` - The entropy value.
///
/// # Returns
///
/// The temperature value.
#[verified_engine::verified]
pub fn map_entropy_to_temperature(entropy: f64) -> f64 {
    1.0 + 0.5 * entropy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::self_calibration::types::Response;
    #[test]
    #[verified_engine::verified]
    fn test_calculate_answer_entropy() {
        let responses = vec![
            Response {
                text: "A".to_string(),
                probability: 0.7,
                answer: "A".to_string(),
            },
            Response {
                text: "A".to_string(),
                probability: 0.1,
                answer: "A".to_string(),
            },
            Response {
                text: "B".to_string(),
                probability: 0.1,
                answer: "B".to_string(),
            },
            Response {
                text: "C".to_string(),
                probability: 0.1,
                answer: "C".to_string(),
            },
        ];
        let entropy = calculate_answer_entropy(&responses);
        assert!((entropy - 1.03972077).abs() < 1e-8);
    }
}

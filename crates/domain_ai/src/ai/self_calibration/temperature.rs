// [cite:self_calibration_paper]
use super::types::{Answer, Response};
use std::collections::HashMap;

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

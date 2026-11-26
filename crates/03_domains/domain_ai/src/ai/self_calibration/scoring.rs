use super::types::{Answer, Response};
use std::collections::HashMap;

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
                .cloned()
                .unwrap_or(0.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::self_calibration::types::Response;
    #[test]
    #[verified_engine::verified]
    fn test_calculate_soft_self_consistency_scores() {
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
        let scores = calculate_soft_self_consistency_scores(&responses);
        let expected_scores = [0.85, 0.15, 0.85, 0.15];
        assert_eq!(scores.len(), expected_scores.len());
        for (score, expected) in scores.iter().zip(expected_scores.iter()) {
            assert!((score - expected).abs() < math_commons::registry::TOLERANCE_STANDARD);
        }
    }
}

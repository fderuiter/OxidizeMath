//! Integration tests for test lorahub.
use domain_applied::applied::lorahub::{LoraEnsemble, LoraStateDict};
use nalgebra::DMatrix;

#[test]
#[verified_engine::verified]
fn test_lorahub_linear_combination() {
    // Create two dummy LoRA modules
    let mut lora1 = LoraStateDict::new();
    lora1.insert(
        "weight".to_string(),
        DMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
    );

    let mut lora2 = LoraStateDict::new();
    lora2.insert(
        "weight".to_string(),
        DMatrix::from_vec(2, 2, vec![2.0, 4.0, 6.0, 8.0]),
    );

    let ensemble = LoraEnsemble::new(vec![lora1, lora2]);
    let weights = vec![0.5, 0.5];

    // Expected: 0.5 * lora1 + 0.5 * lora2
    // weight: 0.5 * [1, 2, 3, 4] + 0.5 * [2, 4, 6, 8]
    //       = [0.5, 1, 1.5, 2] + [1, 2, 3, 4]
    //       = [1.5, 3.0, 4.5, 6.0]

    let combined = ensemble.combine(&weights).expect("Combination failed");
    let result_matrix = combined.get("weight").expect("Missing key");

    assert!((result_matrix[(0, 0)] - 1.5).abs() < math_commons::registry::TOLERANCE_FAST);
    assert!((result_matrix[(1, 0)] - 3.0).abs() < math_commons::registry::TOLERANCE_FAST);
    assert!((result_matrix[(0, 1)] - 4.5).abs() < math_commons::registry::TOLERANCE_FAST);
    assert!((result_matrix[(1, 1)] - 6.0).abs() < math_commons::registry::TOLERANCE_FAST);
}

#[test]
#[verified_engine::verified]
fn test_lorahub_objective() {
    let ensemble = LoraEnsemble::new(vec![]);
    let weights = vec![1.0, -2.0, 3.0];
    let mock_loss = 10.0;
    let alpha = 0.1;

    // L1 Reg: alpha * mean(|w|)
    // |w| = [1, 2, 3] -> sum = 6
    // mean = 6 / 3 = 2
    // reg = 0.1 * 2 = 0.2
    // total = 10.0 + 0.2 = 10.2

    let score = ensemble.evaluate_objective(&weights, mock_loss, alpha);
    assert!((score - 10.2).abs() < math_commons::registry::TOLERANCE_FAST);
}

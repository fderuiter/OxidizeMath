//! Integration tests for test lorahub.
use domain_applied::applied::lorahub::{
    LoraEnsemble, LoraStateDict, NelderMeadConfig, NelderMeadSimplex,
};
use nalgebra::DMatrix;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

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

#[test]
#[verified_engine::verified]
fn test_lorahub_nelder_mead_zero_allocations_and_sparsity() {
    let mut m1 = LoraStateDict::new();
    m1.insert(
        "proj".to_string(),
        DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
    );
    let mut m2 = LoraStateDict::new();
    m2.insert(
        "proj".to_string(),
        DMatrix::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]),
    );
    let mut m3 = LoraStateDict::new();
    m3.insert(
        "proj".to_string(),
        DMatrix::from_vec(2, 2, vec![0.5, 0.5, 0.5, 0.5]),
    );

    let ensemble = LoraEnsemble::new(vec![m1, m2, m3]);
    let mut simplex = NelderMeadSimplex::new(3);
    let config = NelderMeadConfig {
        tolerance: 1e-8,
        max_steps: 240,
        ..NelderMeadConfig::default()
    };
    let initial = [0.25, -0.35, 0.45];
    let mut dense_weights = [0.0; 3];
    let mut sparse_weights = [0.0; 3];

    // Task loss strongly depends on w[0] -> 0.8, w[1] -> -0.5, and only weakly on w[2] -> 0.08
    let mut task_loss = |w: &[f64]| {
        4.0 * (w[0] - 0.8).powi(2) + 4.0 * (w[1] + 0.5).powi(2) + 0.15 * (w[2] - 0.08).powi(2)
    };

    let ((summary_unreg, summary_sparse), _metrics) = VerifiedEngine::run_verified(|| {
        let unreg = ensemble.optimize_weights_in_place(
            &mut simplex,
            &initial,
            &mut dense_weights,
            &config,
            0.0,
            &mut task_loss,
        );
        let sparse = ensemble.optimize_weights_in_place(
            &mut simplex,
            &initial,
            &mut sparse_weights,
            &config,
            0.6,
            &mut task_loss,
        );
        (unreg, sparse)
    });

    assert!(summary_unreg.steps > 0);
    assert!(summary_sparse.steps > 0);
    assert!((dense_weights[0] - 0.8).abs() < 0.03);
    assert!((dense_weights[1] + 0.5).abs() < 0.03);
    // Under L1 penalty alpha = 0.6, the weakly-contributing 3rd adapter shrinks substantially
    assert!(sparse_weights[2].abs() < dense_weights[2].abs());
    assert!(sparse_weights[2].abs() < 0.02);
}

//! Integration tests for grpo.
use domain_applied::applied::grpo::formulas::*;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_response_level_advantage() {
    let rewards = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let advantage = response_level_advantage(&rewards, 3.0);
    assert!((advantage - 0.0).abs() < math_commons::registry::TOLERANCE_STANDARD);

    let advantage_high = response_level_advantage(&rewards, 5.0);
    assert!(advantage_high > 0.0);

    let advantage_low = response_level_advantage(&rewards, 1.0);
    assert!(advantage_low < 0.0);
}

#[test]
#[verified_engine::verified]
fn test_response_level_advantage_zero_allocations() {
    let rewards = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mut out = [0.0; 5];

    let ((), _metrics) = VerifiedEngine::run_verified(|| {
        for &r in &rewards {
            let _adv = response_level_advantage(&rewards, r);
        }
        normalize_group_advantages(&rewards, &mut out);
    });
}

#[test]
#[verified_engine::verified]
fn test_advantage_empty_group() {
    let empty: [f64; 0] = [];
    assert_eq!(response_level_advantage(&empty, 1.0), 0.0);

    let mut out = [99.0; 3];
    normalize_group_advantages(&empty, &mut out);
    assert_eq!(out, [0.0, 0.0, 0.0]);

    let batch = batch_response_level_advantages(&empty);
    assert!(batch.is_empty());
}

#[test]
#[verified_engine::verified]
fn test_advantage_single_response_group() {
    let single = [42.0];
    let adv = response_level_advantage(&single, 42.0);
    assert!((adv - 0.0).abs() < math_commons::registry::TOLERANCE_STANDARD);

    let mut out = [99.0];
    normalize_group_advantages(&single, &mut out);
    assert!((out[0] - 0.0).abs() < math_commons::registry::TOLERANCE_STANDARD);

    let batch = batch_response_level_advantages(&single);
    assert_eq!(batch.len(), 1);
    assert!((batch[0] - 0.0).abs() < math_commons::registry::TOLERANCE_STANDARD);
}

#[test]
#[verified_engine::verified]
fn test_advantage_constant_rewards() {
    let constant = [3.5, 3.5, 3.5, 3.5];
    for &r in &constant {
        let adv = response_level_advantage(&constant, r);
        assert!((adv - 0.0).abs() < math_commons::registry::TOLERANCE_STANDARD);
    }

    let mut out = [99.0; 4];
    normalize_group_advantages(&constant, &mut out);
    for &val in &out {
        assert!((val - 0.0).abs() < math_commons::registry::TOLERANCE_STANDARD);
    }

    let batch = batch_group_advantages(&constant);
    for &val in &batch {
        assert!((val - 0.0).abs() < math_commons::registry::TOLERANCE_STANDARD);
    }
}

#[test]
#[verified_engine::verified]
fn test_advantage_numerical_parity() {
    let rewards = [1.0, 0.5, 0.2];
    let adv_0 = response_level_advantage(&rewards, rewards[0]);
    let adv_1 = response_level_advantage(&rewards, rewards[1]);
    let adv_2 = response_level_advantage(&rewards, rewards[2]);

    assert!((adv_0 - 1.0722219).abs() < 1e-4);
    assert!((adv_1 - (-0.1649572)).abs() < 1e-4);
    assert!((adv_2 - (-0.9072647)).abs() < 1e-4);

    let mut out = [0.0; 3];
    normalize_group_advantages(&rewards, &mut out);
    assert!((out[0] - adv_0).abs() < math_commons::registry::TOLERANCE_STANDARD);
    assert!((out[1] - adv_1).abs() < math_commons::registry::TOLERANCE_STANDARD);
    assert!((out[2] - adv_2).abs() < math_commons::registry::TOLERANCE_STANDARD);

    let batch = batch_response_level_advantages(&rewards);
    assert_eq!(batch.len(), 3);
    for i in 0..3 {
        assert!((batch[i] - out[i]).abs() < math_commons::registry::TOLERANCE_STANDARD);
    }
}

#[test]
#[verified_engine::verified]
fn test_clipped_surrogate_objective() {
    let pi_thetas = vec![1.1, 1.3];
    let pi_theta_olds = vec![1.0, 1.0];
    let advantages = vec![2.0, 2.0];
    let epsilon = 0.2;
    let beta = 0.1;
    let kl_divergence = 0.5;

    // Calculation for first element: min(1.1 * 2.0, 1.1 * 2.0) = 2.2
    // Calculation for second element: min(1.3 * 2.0, 1.2 * 2.0) = 2.4
    // sum = 2.2 + 2.4 = 4.6
    // G = 2
    // L_GRPO = -(1/2) * 4.6 + 0.1 * 0.5 = -2.3 + 0.05 = -2.25
    let objective = clipped_surrogate_objective(
        &pi_thetas,
        &pi_theta_olds,
        &advantages,
        epsilon,
        beta,
        kl_divergence,
    );
    assert!((objective - (-2.25)).abs() < math_commons::registry::TOLERANCE_STANDARD);
}

#[test]
#[verified_engine::verified]
fn test_uncertainty_reward() {
    let reward_max = uncertainty_reward(0.5);
    assert!((reward_max - 1.0).abs() < math_commons::registry::TOLERANCE_STANDARD);

    let reward_min = uncertainty_reward(0.0);
    assert!((reward_min - 0.0).abs() < math_commons::registry::TOLERANCE_STANDARD);

    let reward_min_2 = uncertainty_reward(1.0);
    assert!((reward_min_2 - 0.0).abs() < math_commons::registry::TOLERANCE_STANDARD);
}

#[test]
#[verified_engine::verified]
fn test_pairwise_distance_bleu() {
    let dist = pairwise_distance_bleu("hello world", "hello world");
    assert!((dist - 0.0).abs() < math_commons::registry::TOLERANCE_STANDARD);

    let dist_diff = pairwise_distance_bleu("hello world", "goodbye world");
    assert!(dist_diff > 0.0);
}

#[test]
#[verified_engine::verified]
fn test_repetition_penalty() {
    let penalty = repetition_penalty(10, 100, 1.0);
    assert!((penalty - 0.1).abs() < math_commons::registry::TOLERANCE_STANDARD);
}

#[test]
#[verified_engine::verified]
fn test_composite_reward() {
    let reward = composite_reward(0.8, 0.1);
    assert!((reward - 0.7).abs() < math_commons::registry::TOLERANCE_STANDARD);

    let reward_zero = composite_reward(0.1, 0.8);
    assert!((reward_zero - 0.0).abs() < math_commons::registry::TOLERANCE_STANDARD);
}

#[test]
#[verified_engine::verified]
fn test_binary_reward() {
    assert_eq!(binary_reward(true), 1);
    assert_eq!(binary_reward(false), 0);
}

#[test]
#[verified_engine::verified]
fn test_solver_empirical_accuracy() {
    let responses = vec![
        "apple".to_string(),
        "banana".to_string(),
        "apple".to_string(),
    ];
    let accuracy = solver_empirical_accuracy(&responses, "apple");
    assert!((accuracy - 2.0 / 3.0).abs() < math_commons::registry::TOLERANCE_STANDARD);

    let accuracy_no_match = solver_empirical_accuracy(&responses, "orange");
    assert!((accuracy_no_match - 0.0).abs() < math_commons::registry::TOLERANCE_STANDARD);
}

#[test]
#[verified_engine::verified]
fn test_kl_divergence_lower_bound() {
    let bound = kl_divergence_lower_bound(0.5, 1.0);
    assert!((bound - 0.125).abs() < math_commons::registry::TOLERANCE_STANDARD);

    let bound_zero_p = kl_divergence_lower_bound(0.0, 1.0);
    assert!((bound_zero_p - 0.0).abs() < math_commons::registry::TOLERANCE_STANDARD);
}

//! GRPO (Group Relative Policy Optimization) Demo
//!
//! This example demonstrates how to use the GRPO formulas provided by
//! `math_explorer::applied::grpo::formulas` to compute response-level
//! advantages and a clipped surrogate objective.

use math_explorer::applied::grpo::formulas::{
    clipped_surrogate_objective, normalize_group_advantages,
};

fn main() {
    println!("=== GRPO (Group Relative Policy Optimization) Demo ===");

    // 1. Simulate a group of 3 outputs with raw rewards
    let rewards = vec![1.0, 0.5, 0.2];
    println!("Raw Rewards: {:?}", rewards);

    // 2. Calculate Advantages (normalized z-scores)
    // Mean = 0.566, StdDev ≈ 0.404
    // Demonstrates zero-allocation group advantage normalization into a pre-allocated buffer
    let mut advantages = vec![0.0; rewards.len()];
    normalize_group_advantages(&rewards, &mut advantages);

    println!(
        "Advantages: [{:.4}, {:.4}, {:.4}]",
        advantages[0], advantages[1], advantages[2]
    );

    // 3. Compute Objective
    // Assume current policy probabilities (pi) and old policy (pi_old)
    // If pi > pi_old for a good advantage (adv_0), objective increases.
    let pi_thetas = vec![0.6, 0.3, 0.1];
    let pi_olds = vec![0.5, 0.3, 0.2];

    let loss = clipped_surrogate_objective(
        &pi_thetas,
        &pi_olds,
        &advantages,
        0.2,  // Epsilon (clipping range 0.8 - 1.2)
        0.01, // Beta (KL penalty)
        0.05, // KL Divergence
    );

    println!("GRPO Loss: {:.4}", loss);
}

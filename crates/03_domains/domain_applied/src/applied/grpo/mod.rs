#![doc = include_str!("README.md")]

/// Zero-allocation critic-free Gaussian GRPO policy controller.
pub mod controller;
/// Group-relative advantage, clipped surrogate objective, and KL penalty formulas.
pub mod formulas;
/// Policy evaluation and rollout aggregation metrics for GRPO.
pub mod metrics;
/// Verifiable reward shaping and rule-based scoring utilities.
pub mod rewards;

pub use controller::{
    GRPO_GROUP_SIZE, GrpoController, GrpoUpdateParams, POLICY_DIM, ROLLOUT_HORIZON,
};

// [cite:grpo]

use pure_math::theory_verification;

theory_verification!(
    module = "grpo",
    constants = {
        BETA = 0.01;
    },
    test = {
        use approx::assert_relative_eq;
        assert_relative_eq!(BETA, 0.01, epsilon = math_commons::registry::TOLERANCE_FAST);
    }
);

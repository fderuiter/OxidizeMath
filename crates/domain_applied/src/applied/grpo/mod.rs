#![doc = include_str!("README.md")]

#[allow(missing_docs)]
pub mod formulas;
#[allow(missing_docs)]
pub mod metrics;
#[allow(missing_docs)]
pub mod rewards;

// [cite:grpo]

use pure_math::theory_verification;

theory_verification!(
    module = "grpo",
    constants = {
        BETA = 0.01;
    },
    test = {
        assert_relative_eq!(BETA, 0.01, epsilon = math_commons::registry::TOLERANCE_FAST);
    }
);

#![doc = include_str!("README.md")]

/// Gradient-free LoRA adapter ensemble optimization and inference orchestration.
pub mod ensemble;
/// Zero-allocation bounded Nelder-Mead simplex optimizer for LoRA weight composition.
pub mod simplex;
/// Adapter weight combination and regularized objective evaluation strategies.
pub mod strategies;
/// Tensor state-dictionary types representing low-rank LoRA modules.
pub mod types;

pub use ensemble::LoraEnsemble;
pub use simplex::{
    NelderMeadConfig, NelderMeadSimplex, SimplexOptimizationSummary, SimplexStepOutcome,
};
pub use types::LoraStateDict;

// [cite:lorahub]

use pure_math::theory_verification;

theory_verification!(
    module = "lorahub",
    constants = {
        ALPHA = 32.0;
    },
    test = {
        use approx::assert_relative_eq;
        assert_relative_eq!(
            ALPHA,
            32.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }
);

#![doc = include_str!("README.md")]

#[allow(missing_docs)]
pub mod ensemble;
#[allow(missing_docs)]
pub mod strategies;
#[allow(missing_docs)]
pub mod types;

pub use ensemble::LoraEnsemble;
pub use types::LoraStateDict;

// [cite:lorahub]

use pure_math::theory_verification;

theory_verification!(
    module = "lorahub",
    constants = {
        ALPHA = 32.0;
    },
    test = {
        assert_relative_eq!(
            ALPHA,
            32.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }
);

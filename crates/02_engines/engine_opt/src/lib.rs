#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]
// [cite:engine_opt]

#[cfg(feature = "alloc")]
extern crate alloc;

/// Typed `#![no_std]` curve-fitting error diagnostics.
pub mod error;
/// Stack-allocated polynomial and damped Levenberg-Marquardt curve-fitting solvers.
pub mod fit;

pub use error::{FitError, LinalgError};
pub use math_scalars::series::{PointSeries, Sample2D};

#[cfg(test)]
mod serde_tests;

#[cfg(all(test, feature = "std"))]
#[global_allocator]
static TEST_ALLOCATOR: verified_engine::allocator::VerifiedAllocator =
    verified_engine::allocator::VerifiedAllocator;

#[cfg(all(test, feature = "std"))]
macro_rules! theory_verification {
    (
        module = $module_name:expr,
        constants = { $( $const_name:ident = $const_val:expr; )* },
        test = $test_body:block
    ) => {
        mod theory_verification {
            $( const $const_name: f64 = $const_val; )*

            #[test]
            #[verified_engine::verified]
            fn test_theory_verification() {
                let _ = $module_name;
                $test_body
            }
        }
    };
}

#[cfg(all(test, feature = "std"))]
theory_verification!(
    module = "engine_opt",
    constants = {
        PARABOLIC_PEAK_Y = 4.0;
    },
    test = {
        use crate::fit::{FitModel, fit_model};
        use scientific_metadata::theory::{MaturityTier, TheoryDescribable};

        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = [0.0, 3.0, 4.0, 3.0, 0.0];
        let fit = fit_model(&(&xs[..], &ys[..]), FitModel::Quadratic, false).unwrap();
        assert!((fit.eval(2.0) - PARABOLIC_PEAK_Y).abs() < 1e-9);
        assert_eq!(FitModel::Quadratic.maturity_tier(), MaturityTier::Validated);
    }
);

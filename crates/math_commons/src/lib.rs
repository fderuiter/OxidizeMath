#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]
// [cite:math_commons]

extern crate alloc;

/// Symbolic mathematical expression AST parser, evaluator, and GLSL emitter.
#[cfg(feature = "std")]
pub mod ast;
pub mod constants;
/// Shared numerical, bounds, and curve-fitting diagnostic error types.
pub mod error;
/// Stack-allocated polynomial and damped Levenberg-Marquardt curve-fitting solvers.
pub mod fit;
/// Colormap gradients and strongly typed grid/step primitives.
pub mod math_kernel;
/// Zero-allocation point-of-interest detection (extrema, roots, inflections, intersections).
pub mod poi;
/// Validated bounded floating-point wrapper types (`UnitInterval`, `PositiveFloat`, `NonNegativeFloat`).
pub mod primitives;
pub mod registry;
/// Contiguous and strided 2D `(x, y)` series traits for zero-allocation numerical kernels.
pub mod series;
pub mod stochastic;
/// Quantum angular-momentum coupling and Wigner $3j$ / Clebsch-Gordan coefficients.
pub mod wigner;

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
            use super::*;
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
    module = "math_commons",
    constants = {
        SPEED_OF_LIGHT_M_S = 299_792_458.0;
        PARABOLIC_PEAK_Y = 4.0;
    },
    test = {
        use crate::fit::{FitModel, fit_model};
        use crate::poi::parabolic_vertex;
        use scientific_metadata::theory::{MaturityTier, TheoryDescribable};

        assert_eq!(constants::C, SPEED_OF_LIGHT_M_S);
        let vertex = parabolic_vertex([1.0, 3.0], [2.0, 4.0], [3.0, 3.0]);
        assert!((vertex[1] - PARABOLIC_PEAK_Y).abs() < 1e-12);
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = [0.0, 3.0, 4.0, 3.0, 0.0];
        let fit = fit_model(&(&xs[..], &ys[..]), FitModel::Quadratic, false).unwrap();
        assert!((fit.eval(2.0) - PARABOLIC_PEAK_Y).abs() < 1e-9);
        assert_eq!(FitModel::Quadratic.maturity_tier(), MaturityTier::Certified);
    }
);

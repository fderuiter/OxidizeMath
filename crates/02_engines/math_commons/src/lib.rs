#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]
// [cite:math_commons]

#[cfg(feature = "alloc")]
extern crate alloc;

/// Symbolic mathematical expression AST parser, evaluator, and GLSL emitter.
#[cfg(feature = "std")]
pub mod ast;
pub mod constants;
/// Shared numerical and bounds diagnostic error types.
pub mod error;
/// Colormap gradients and strongly typed grid/step primitives.
pub mod math_kernel;
/// Zero-allocation point-of-interest detection (extrema, roots, inflections, intersections).
pub mod poi;
/// Validated bounded floating-point wrapper types (`UnitInterval`, `PositiveFloat`, `NonNegativeFloat`).
pub mod primitives;
pub mod registry;
/// Contiguous and strided 2D `(x, y)` series traits for zero-allocation numerical kernels.
pub mod series;
/// Zero-allocation `#![no_std]` Markov SCC, Householder QR stationary, and OU calibration kernels.
pub mod stochastic;
/// Quantum angular-momentum coupling and Wigner $3j$ / Clebsch-Gordan coefficients.
pub mod wigner;

#[doc(hidden)]
pub use math_scalars::advance_sample2d_plateau;

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
        use crate::poi::{closest_threshold_crossing, parabolic_vertex};
        use crate::wigner::ClebschGordan;

        assert_eq!(constants::C, SPEED_OF_LIGHT_M_S);
        let vertex = parabolic_vertex([1.0, 3.0], [2.0, 4.0], [3.0, 3.0]);
        assert!((vertex[1] - PARABOLIC_PEAK_Y).abs() < 1e-12);
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = [0.0, 3.0, 4.0, 3.0, 0.0];
        let root = closest_threshold_crossing(&(&xs[..], &ys[..]), 2.0, 0.5).unwrap();
        assert!(root > 0.0 && root < 1.0);
        let cg = ClebschGordan {
            tj1: 2,
            tm1: 2,
            tj2: 2,
            tm2: -2,
            tj12: 4,
            tm12: 0,
        };
        assert!((cg.value() - (1.0 / 6.0_f64).sqrt()).abs() < 1e-12);
    }
);

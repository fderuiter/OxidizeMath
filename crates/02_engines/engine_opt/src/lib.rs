#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]
// [cite:engine_opt]

#[cfg(feature = "alloc")]
extern crate alloc;

/// Multi-period Almgren-Chriss optimal market execution over projected `MPC`.
pub mod almgren_chriss;
/// Finite-horizon backward induction and infinite-horizon Howard policy iteration (`lu_solve`) solvers.
pub mod dp;
/// Typed `#![no_std]` curve-fitting, optimal-control, dynamic-programming, MFG, and Nash error diagnostics.
pub mod error;
/// Stack-allocated polynomial and damped Levenberg-Marquardt curve-fitting solvers.
pub mod fit;
/// Single-shooting box-projected iterative LQR (`iLQR`) solvers.
pub mod ilqr;
/// Monotone 1D Mean-Field Game (Engquist-Osher HJB / donor-cell Fokker-Planck) Mann solvers.
pub mod mfg;
/// Constrained receding-horizon projected nonlinear Model Predictive Control (`MPC`) solvers.
pub mod mpc;
/// Lemke-Howson bimatrix complementary pivoting and polymatrix regret-matching Nash solvers.
pub mod nash;

pub use almgren_chriss::{
    AlmgrenChrissConfig, almgren_chriss_analytical_trajectory_into, almgren_chriss_workspace_len,
    solve_almgren_chriss_mpc_into,
};
pub use dp::{dp_finite_workspace_len, dp_infinite_workspace_len};
pub use error::{FitError, LinalgError, OptError, PdeError};
pub use ilqr::ilqr_workspace_len;
pub use math_scalars::series::{PointSeries, Sample2D};
pub use math_scalars::workspace::{ScalarWorkspace, SolverWorkspace};
pub use mfg::mfg_1d_workspace_len;
pub use mpc::mpc_workspace_len;
pub use nash::{nash_bimatrix_workspace_len, nash_polymatrix_workspace_len};

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

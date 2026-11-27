//! Optional `nalgebra` (`DMatrix` / `DVector`) high-level linear system and decomposition wrappers
//! over `#![no_std]` [`crate`] solvers.
// [cite:engine_linalg]
// [cite:advanced_linear_algebra]

pub mod decomposition;
pub mod numerical;

pub use decomposition::{effective_rank, low_rank_approximation, try_low_rank_approximation};
pub use numerical::{
    LinearAlgebraError, moore_penrose_pseudoinverse, solve_cholesky_system,
    solve_conjugate_gradient, solve_gmres_restarted, solve_linear_system, solve_normal_equation,
    solve_tikhonov_system, try_moore_penrose_pseudoinverse,
};

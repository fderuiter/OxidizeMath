#![no_std]
#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
#![doc = include_str!("../README.md")]
// [cite:engine_linalg]
// [cite:advanced_linear_algebra]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod cholesky;
pub mod krylov;
pub mod lu;
pub mod norms;
pub mod qr;
pub mod scalar;
pub mod sparse;
pub mod svd_bidiag;
pub mod svd_jacobi;
pub mod svd_ops;
pub mod tikhonov;
pub mod types;

pub use cholesky::{
    cholesky_factor, cholesky_semidefinite, cholesky_semidefinite_workspace_len, cholesky_solve,
    cholesky_solve_factored, cholesky_workspace_len,
};
pub use krylov::{KrylovReport, cg_solve, cg_workspace_len, gmres_solve, gmres_workspace_len};
pub use lu::{
    lu_determinant, lu_factor_in_place, lu_inverse, lu_inverse_workspace_len, lu_solve,
    lu_solve_factored, lu_solve_transpose_factored, lu_workspace_len,
};
pub use norms::{
    cond_l1, cond_l1_workspace_len, cond_spectral, norm_frobenius, norm_l1, norm_linf,
    norm_spectral, spectral_workspace_len,
};
pub use qr::{
    StreamingQr, qr_apply_qt, qr_factor_pivoted, qr_factor_workspace_len,
    qr_least_squares_workspace_len, qr_solve_least_squares, qr_solve_r_pivoted,
    qr_solve_rt_permuted,
};
pub use scalar::{Scalar, hypot_f64, sqrt_f64};
pub use sparse::CsrMatrixView;
pub use svd_bidiag::{svd_bidiagonal, svd_bidiagonal_workspace_len};
pub use svd_jacobi::{
    svd_jacobi, svd_jacobi_workspace_len, symmetric_eigen_jacobi, symmetric_eigen_workspace_len,
};
pub use svd_ops::{
    qr_or_svd_solve, qr_or_svd_workspace_len, svd_effective_rank, svd_low_rank_approx,
    svd_ops_workspace_len, svd_pseudoinverse, svd_solve_least_squares,
};
pub use tikhonov::{
    DampingStyle, damped_normal_workspace_len, solve_damped_normal, tikhonov_solve,
    tikhonov_workspace_len,
};
#[cfg(feature = "alloc")]
pub use types::OwnedWorkspace;
pub use types::{
    DenseView, LinalgError, SolverWorkspace, add_scaled_vec_array, check_all_finite,
    identity_array, identity_plus_scaled_array, mat_idx, mat_mul, mat_mul_array,
    mat_mul_transpose_right, mat_mul_transpose_right_array, mat_vec_mul,
};

#[cfg(test)]
mod serde_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_conditioning;

//! Linear algebra decompositions, canonical forms, eigenvalue diagnostics, and numerical solvers.

pub mod canonical;
pub mod decomposition;
pub mod eigen;
pub mod numerical;

pub use decomposition::{
    MatrixNorm, QrFactors, SvdFactors, cholesky_decomposition, condition_number, effective_rank,
    low_rank_approximation, matrix_norm, qr_decomposition, svd_decomposition,
    try_low_rank_approximation,
};
pub use numerical::{
    LinearAlgebraError, moore_penrose_pseudoinverse, solve_cholesky_system,
    solve_conjugate_gradient, solve_gmres_restarted, solve_linear_system, solve_nnls,
    solve_normal_equation, solve_ridge_regression, solve_tikhonov_system,
    try_moore_penrose_pseudoinverse,
};

// [cite:advanced_linear_algebra]

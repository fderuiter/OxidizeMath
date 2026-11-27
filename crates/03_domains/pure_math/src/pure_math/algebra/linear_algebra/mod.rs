pub mod canonical;
pub mod decomposition;
pub mod eigen;
pub mod numerical;

pub use decomposition::{
    effective_rank, effective_rank_with_workspace, low_rank_approximation,
    svd_decomposition_workspace_bytes, try_low_rank_approximation,
    try_low_rank_approximation_with_workspace,
};
pub use numerical::{
    LinearAlgebraError, linalg_solve_workspace_bytes, moore_penrose_pseudoinverse,
    solve_cholesky_system, solve_cholesky_system_with_workspace, solve_conjugate_gradient,
    solve_conjugate_gradient_with_workspace, solve_gmres_restarted,
    solve_gmres_restarted_with_workspace, solve_linear_system, solve_linear_system_with_workspace,
    solve_normal_equation, solve_normal_equation_with_workspace, solve_tikhonov_system,
    solve_tikhonov_system_with_workspace, try_moore_penrose_pseudoinverse,
    try_moore_penrose_pseudoinverse_with_workspace,
};

// [cite:advanced_linear_algebra]

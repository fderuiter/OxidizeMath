pub mod canonical;
pub mod decomposition;
pub mod eigen;
pub mod numerical;
pub mod numerical_workspace;

pub use decomposition::{effective_rank, low_rank_approximation, try_low_rank_approximation};
pub use numerical::{
    LinearAlgebraError, moore_penrose_pseudoinverse, solve_cholesky_system,
    solve_conjugate_gradient, solve_gmres_restarted, solve_linear_system, solve_normal_equation,
    solve_tikhonov_system, try_moore_penrose_pseudoinverse,
};

// [cite:advanced_linear_algebra]

pub use decomposition::{effective_rank_with_workspace, try_low_rank_approximation_with_workspace};
pub use numerical_workspace::*;

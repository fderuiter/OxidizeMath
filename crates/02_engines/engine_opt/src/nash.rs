// [cite:engine_opt]
//! Zero-allocation Lemke-Howson bimatrix pivoting and polymatrix regret-matching Nash equilibrium solvers.

mod lemke_howson;
mod regret;

pub use lemke_howson::{
    BimatrixNashStats, nash_bimatrix_workspace_len, solve_bimatrix_lemke_howson,
};
pub use regret::{
    PolymatrixConfig, PolymatrixNashStats, nash_polymatrix_workspace_len,
    solve_polymatrix_regret_matching,
};

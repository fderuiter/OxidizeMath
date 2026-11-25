//! Verifies zero heap allocations across all 10 `pure_math` `*_with_workspace` entry points.

use math_memory::OwnedByteArena;
use nalgebra::{DMatrix, DVector};
use pure_math::pure_math::algebra::linear_algebra::decomposition::{
    effective_rank_with_workspace, try_low_rank_approximation_with_workspace,
};
use pure_math::pure_math::algebra::linear_algebra::numerical::{
    solve_cholesky_system_with_workspace, solve_conjugate_gradient_with_workspace,
    solve_gmres_restarted_with_workspace, solve_linear_system_with_workspace,
    solve_normal_equation_with_workspace, solve_tikhonov_system_with_workspace,
    try_moore_penrose_pseudoinverse_with_workspace,
};
use pure_math::pure_math::statistics::copula::{
    Correlation, CorrelationMatrix, core::cholesky_factor_with_workspace,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn all_ten_pure_math_with_workspace_entry_points_zero_heap_allocs() {
    let spd = DMatrix::from_row_slice(3, 3, &[4.0, 1.0, 0.2, 1.0, 3.0, 0.5, 0.2, 0.5, 2.5]);
    let rect = DMatrix::from_row_slice(4, 2, &[1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
    let b3 = DVector::from_column_slice(&[1.0, 2.0, 3.0]);
    let b4 = DVector::from_column_slice(&[1.0, 2.0, 2.9, 4.1]);
    let x0_3 = DVector::zeros(3);
    let mut x3 = DVector::zeros(3);
    let mut x2 = DVector::zeros(2);
    let mut pinv = DMatrix::zeros(2, 4);
    let mut low_rank = DMatrix::zeros(4, 2);
    let corr = [1.0, 0.4, 0.2, 0.4, 1.0, 0.3, 0.2, 0.3, 1.0];
    let corr_mat = DMatrix::from_row_slice(3, 3, &corr);
    let mut corr_obj =
        CorrelationMatrix::equicorrelated(3, Correlation::new(0.2).unwrap()).unwrap();
    let mut chol = [0.0_f64; 9];
    let mut arena = OwnedByteArena::with_capacity(65_536);

    let (rk, _metrics) = VerifiedEngine::run_verified(|| {
        solve_linear_system_with_workspace(&spd, &b3, &mut x3, &mut arena).unwrap();
        solve_normal_equation_with_workspace(&rect, &b4, &mut x2, &mut arena).unwrap();
        try_moore_penrose_pseudoinverse_with_workspace(&rect, 1e-10, &mut pinv, &mut arena)
            .unwrap();
        solve_cholesky_system_with_workspace(&spd, &b3, &mut x3, &mut arena).unwrap();
        solve_tikhonov_system_with_workspace(&rect, &b4, 0.05, &mut x2, &mut arena).unwrap();
        solve_conjugate_gradient_with_workspace(&spd, &b3, &x0_3, 1e-10, 64, &mut x3, &mut arena)
            .unwrap();
        solve_gmres_restarted_with_workspace(&spd, &b3, &x0_3, 3, 1e-10, 8, &mut x3, &mut arena)
            .unwrap();
        let rk = effective_rank_with_workspace(&rect, 1e-8, &mut arena).unwrap();
        try_low_rank_approximation_with_workspace(&rect, 1, &mut low_rank, &mut arena).unwrap();
        cholesky_factor_with_workspace(&corr, 3, 1e-12, &mut chol, &mut arena).unwrap();
        corr_obj
            .update_with_workspace(&corr_mat, &mut arena)
            .unwrap();
        rk
    });
    assert_eq!(rk, 2);
    assert!((corr_obj.get_correlation(0, 1).unwrap() - 0.4).abs() < 1e-12);
}

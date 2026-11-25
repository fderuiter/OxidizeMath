// [cite:engine_linalg]
//! OS-allocator-level zero-heap-allocation verification for `engine_linalg` solvers.

use engine_linalg::{
    ByteArena, CsrMatrixView, cg_solve, cholesky_solve, gmres_solve, lu_solve, qr_or_svd_solve,
    svd_bidiagonal, tikhonov_solve,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOC: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_engine_linalg_solvers_zero_heap_allocations_under_verified_allocator() {
    let a = [4.0_f64, 1.0, 1.0, 3.0];
    let b = [5.0_f64, 4.0];
    let row_ptr = [0_usize, 2, 4];
    let col_ind = [0_usize, 1, 0, 1];
    let csr = CsrMatrixView::new(2, 2, &row_ptr, &col_ind, &a).unwrap();

    let ((), _receipt) = VerifiedEngine::run_verified(|| {
        let mut scratch = [0.0_f64; 256];
        let mut ws = ByteArena::from_slice(&mut scratch);
        let mut x = [0.0_f64; 2];
        let mut perm = [0_usize; 2];
        let mut u = [0.0_f64; 4];
        let mut s = [0.0_f64; 2];
        let mut vt = [0.0_f64; 4];

        lu_solve(&a, &b, &mut x, 2, &mut perm, &mut ws).unwrap();
        cholesky_solve(&a, &b, &mut x, 2, &mut ws).unwrap();
        qr_or_svd_solve(&a, 2, 2, &b, &mut x, &mut perm, &mut ws).unwrap();
        svd_bidiagonal(&a, 2, 2, &mut u, &mut s, &mut vt, &mut ws).unwrap();
        tikhonov_solve(&a, 2, 2, &b, 0.1, &mut x, &mut ws).unwrap();
        cg_solve(&csr, &b, &mut x, 1e-12, 20, &mut ws).unwrap();
        x = [0.0; 2];
        gmres_solve(&csr, &b, &mut x, 2, 1e-12, 10, &mut ws).unwrap();
    });
}

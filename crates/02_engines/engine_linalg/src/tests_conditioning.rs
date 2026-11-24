#![cfg_attr(any(), verified(opt_out = "tests"))]

use crate::*;
use verified_engine::engine::VerifiedEngine;

#[test]
fn test_lauchli_and_near_collinear_glm_conditioning_at_eps_1e8() {
    let eps = 1e-8_f64;
    // 4x3 Lauchli matrix: top row ones, bottom 3x3 = eps * I_3
    let lauchli = [1.0, 1.0, 1.0, eps, 0.0, 0.0, 0.0, eps, 0.0, 0.0, 0.0, eps];
    let beta_true = [1.2_f64, -0.7, 0.4];
    let mut y = [0.0_f64; 4];
    mat_vec_mul(&lauchli, 4, 3, &beta_true, &mut y).unwrap();

    let mut beta_est = [0.0_f64; 3];
    let mut perm = [0_usize; 3];
    let mut scratch = [0.0_f64; 256];
    let rank = qr_or_svd_solve(
        &lauchli,
        4,
        3,
        &y,
        &mut beta_est,
        &mut perm,
        &mut SolverWorkspace::new(&mut scratch),
    )
    .unwrap();
    assert_eq!(rank, 3);

    let mut err_sq = 0.0;
    let mut ref_sq = 0.0;
    for i in 0..3 {
        let d = beta_est[i] - beta_true[i];
        err_sq += d * d;
        ref_sq += beta_true[i] * beta_true[i];
    }
    let rel_l2 = (err_sq / ref_sq).sqrt();
    assert!(
        rel_l2 <= 1.7e-8,
        "Lauchli relative L2 error {rel_l2:e} exceeded 1.7e-8"
    );
    assert!(
        (beta_est[2] - 0.4).abs() <= 2e-8,
        "Lauchli beta_2={} drifted from 0.4",
        beta_est[2]
    );
}

#[test]
fn test_vandermonde_streaming_qr_and_cohort_glm_contrast_at_eps_1e8() {
    let eps = 1e-8_f64;
    let coeffs = [0.5_f64, -1.1, 0.4];
    let mut stream_qr = StreamingQr::<f64, 5>::new(3).unwrap();
    for i in 0..8 {
        let t = (i as f64) - 3.5;
        let u = eps * t;
        let row = [1.0, u, u * u];
        let rhs = coeffs[0] + coeffs[1] * u + coeffs[2] * u * u;
        stream_qr.push_row(&row, rhs).unwrap();
    }
    // At u = eps * t, u^2 ~ 1e-15; test scaled/perturbed Vandermonde at conditioning 1/eps:
    let mut vander = [0.0_f64; 18];
    let mut rhs_v = [0.0_f64; 6];
    for r in 0..6 {
        let x1 = -1.0 + 0.4 * (r as f64);
        let x2 = x1 + eps * ((r * r + 1) as f64 * 0.3 - 0.7);
        vander[mat_idx(r, 0, 3)] = 1.0;
        vander[mat_idx(r, 1, 3)] = x1;
        vander[mat_idx(r, 2, 3)] = x2;
        rhs_v[r] = 1.5 - 0.8 * x1 + 0.4 * x2;
    }
    let mut stream_collinear = StreamingQr::<f64, 5>::new(3).unwrap();
    for (r, &rhs_r) in rhs_v.iter().enumerate() {
        let start = r * 3;
        stream_collinear
            .push_row(&vander[start..(start + 3)], rhs_r)
            .unwrap();
    }
    let sol = stream_collinear.solve(0.0).unwrap();
    assert!(
        (sol[2] - 0.4).abs() <= 1.7e-8,
        "StreamingQr beta_2={} error exceeded 1.7e-8",
        sol[2]
    );

    // Verify pre-factored QR contrast variance c^T (X^T X)^-1 c on near-collinear cohort GLM
    let mut qr_buf = vander;
    let mut tau = [0.0_f64; 3];
    let mut perm = [0_usize; 3];
    let mut scratch = [0.0_f64; 64];
    let rank = qr_factor_pivoted(
        &mut qr_buf,
        6,
        3,
        &mut tau,
        &mut perm,
        &mut SolverWorkspace::new(&mut scratch),
    )
    .unwrap();
    assert_eq!(rank, 3);
    let contrast = [0.0_f64, 0.0, 1.0];
    let mut w_work = [0.0_f64; 3];
    let var_factor =
        qr_solve_rt_permuted(&qr_buf, 6, 3, rank, &perm, &contrast, &mut w_work).unwrap();
    assert!(var_factor.is_finite() && var_factor > 1e14);
}

#[test]
fn test_nan_and_non_convergence_errors() {
    let bad = [1.0_f64, f64::NAN, 0.0, 1.0];
    let mut pinv = [0.0_f64; 4];
    let mut approx = [0.0_f64; 4];
    let mut scratch = [0.0_f64; 128];

    assert_eq!(
        svd_pseudoinverse(
            &bad,
            2,
            2,
            1e-10,
            &mut pinv,
            &mut SolverWorkspace::new(&mut scratch),
        ),
        Err(LinalgError::InvalidInput)
    );
    assert_eq!(
        svd_low_rank_approx(
            &bad,
            2,
            2,
            1,
            &mut approx,
            &mut SolverWorkspace::new(&mut scratch),
        ),
        Err(LinalgError::InvalidInput)
    );

    // Non-convergence when Jacobi max_sweeps = 0 on non-diagonal matrix
    let non_diag = [2.0_f64, 1.0, 1.0, 3.0];
    let mut u = [0.0_f64; 4];
    let mut s = [0.0_f64; 2];
    let mut vt = [0.0_f64; 4];
    assert_eq!(
        svd_jacobi(
            &non_diag,
            (2, 2),
            &mut u,
            &mut s,
            &mut vt,
            0,
            &mut SolverWorkspace::new(&mut scratch),
        ),
        Err(LinalgError::NonConvergence)
    );
}

#[test]
fn test_zero_heap_allocations_dense_solvers() {
    let a = [4.0_f64, 1.0, 1.0, 3.0];
    let b = [5.0_f64, 4.0];

    let ((), _metrics) = VerifiedEngine::run_verified(|| {
        let mut scratch = [0.0_f64; 256];
        let mut x = [0.0_f64; 2];
        let mut perm = [0_usize; 2];
        let mut u = [0.0_f64; 4];
        let mut s = [0.0_f64; 2];
        let mut vt = [0.0_f64; 4];

        qr_solve_least_squares(
            &a,
            2,
            2,
            &b,
            &mut x,
            &mut perm,
            &mut SolverWorkspace::new(&mut scratch),
        )
        .unwrap();
        cholesky_solve(&a, &b, &mut x, 2, &mut SolverWorkspace::new(&mut scratch)).unwrap();
        svd_jacobi(
            &a,
            (2, 2),
            &mut u,
            &mut s,
            &mut vt,
            32,
            &mut SolverWorkspace::new(&mut scratch),
        )
        .unwrap();
        svd_bidiagonal(
            &a,
            2,
            2,
            &mut u,
            &mut s,
            &mut vt,
            &mut SolverWorkspace::new(&mut scratch),
        )
        .unwrap();
    });
}

#[test]
fn test_zero_heap_allocations_krylov_solvers() {
    let a = [4.0_f64, 1.0, 1.0, 3.0];
    let b = [5.0_f64, 4.0];
    let row_ptr = [0_usize, 2, 4];
    let col_ind = [0_usize, 1, 0, 1];
    let csr = CsrMatrixView::new(2, 2, &row_ptr, &col_ind, &a).unwrap();

    let ((), _metrics) = VerifiedEngine::run_verified(|| {
        let mut scratch = [0.0_f64; 256];
        let mut x = [0.0_f64; 2];
        cg_solve(
            &csr,
            &b,
            &mut x,
            1e-12,
            20,
            &mut SolverWorkspace::new(&mut scratch),
        )
        .unwrap();
        x = [0.0; 2];
        gmres_solve(
            &csr,
            &b,
            &mut x,
            2,
            1e-12,
            10,
            &mut SolverWorkspace::new(&mut scratch),
        )
        .unwrap();
    });
}

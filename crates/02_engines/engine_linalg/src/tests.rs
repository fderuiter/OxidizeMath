#![cfg_attr(any(), verified(opt_out = "tests"))]

use crate::*;

#[test]
fn test_scalar_sqrt_and_hypot_f32_f64() {
    for &val in &[0.0_f64, 1e-312, 2.0, 9.0, 12345.6789, 1e200] {
        let s = sqrt_f64(val);
        let rel = if val == 0.0 {
            s.abs()
        } else {
            ((s * s - val) / val).abs()
        };
        assert!(rel < 2e-15, "sqrt_f64 failed at {val}: rel={rel}");
    }
    assert!(sqrt_f64(-1.0).is_nan());
    assert!(sqrt_f64(f64::NAN).is_nan());
    assert!(
        ((hypot_f64(3.0e200, 4.0e200) - 5.0e200) / 5.0e200).abs() < 1e-15,
        "hypot_f64 overflow scaling inaccurate"
    );
    assert!((<f32 as ComplexField>::sqrt(25.0_f32) - 5.0_f32).abs() < 1e-6);
}

#[test]
fn test_lu_and_cholesky_f32_and_f64() {
    let a = [4.0_f64, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0];
    let b = [7.0_f64, 10.0, 10.0];
    let mut x_lu = [0.0_f64; 3];
    let mut x_chol = [0.0_f64; 3];
    let mut piv = [0_usize; 3];
    let mut buf = [0.0_f64; 64];

    lu_solve(
        &a,
        &b,
        &mut x_lu,
        3,
        &mut piv,
        &mut SolverWorkspace::new(&mut buf),
    )
    .unwrap();
    cholesky_solve(&a, &b, &mut x_chol, 3, &mut SolverWorkspace::new(&mut buf)).unwrap();
    for i in 0..3 {
        assert!((x_lu[i] - 1.0).abs() < 1e-12);
        assert!((x_chol[i] - 1.0).abs() < 1e-12);
    }

    let a32 = [4.0_f32, 1.0, 1.0, 3.0];
    let b32 = [1.0_f32, 2.0];
    let mut x32 = [0.0_f32; 2];
    let mut buf32 = [0.0_f32; 16];
    cholesky_solve(
        &a32,
        &b32,
        &mut x32,
        2,
        &mut SolverWorkspace::new(&mut buf32),
    )
    .unwrap();
    assert!((x32[0] - 1.0 / 11.0).abs() < 1e-6);
    assert!((x32[1] - 7.0 / 11.0).abs() < 1e-6);
}

#[test]
fn test_semidefinite_cholesky_rank_deficient_and_indefinite_rejection() {
    let psd = [1.0_f64, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut l = [0.0_f64; 9];
    let mut recon = [0.0_f64; 9];
    let mut scratch = [0.0_f64; 128];
    cholesky_semidefinite(
        &psd,
        &mut l,
        3,
        1e-10,
        1e-10,
        &mut SolverWorkspace::new(&mut scratch),
    )
    .unwrap();
    mat_mul_transpose_right(&l, 3, 3, &l, 3, &mut recon).unwrap();
    for i in 0..9 {
        assert!((recon[i] - psd[i]).abs() < 1e-10);
    }

    let indef = [1.0_f64, 0.9, 0.9, 0.9, 1.0, -0.9, 0.9, -0.9, 1.0];
    assert_eq!(
        cholesky_semidefinite(
            &indef,
            &mut l,
            3,
            1e-10,
            1e-10,
            &mut SolverWorkspace::new(&mut scratch),
        ),
        Err(LinalgError::NotPositiveDefinite)
    );
}

#[test]
fn test_jacobi_and_bidiagonal_svd_tall() {
    let tall = [
        3.0_f64, 1.0, 1.0, -1.0, 3.0, 1.0, 1.0, 1.0, 3.0, 0.5, -0.5, 2.0,
    ];
    let mut u_j = [0.0_f64; 12];
    let mut s_j = [0.0_f64; 3];
    let mut vt_j = [0.0_f64; 9];
    let mut u_b = [0.0_f64; 12];
    let mut s_b = [0.0_f64; 3];
    let mut vt_b = [0.0_f64; 9];
    let mut scratch = [0.0_f64; 256];

    svd_jacobi(
        &tall,
        (4, 3),
        &mut u_j,
        &mut s_j,
        &mut vt_j,
        64,
        &mut SolverWorkspace::new(&mut scratch),
    )
    .unwrap();
    svd_bidiagonal(
        &tall,
        4,
        3,
        &mut u_b,
        &mut s_b,
        &mut vt_b,
        &mut SolverWorkspace::new(&mut scratch),
    )
    .unwrap();

    for i in 0..3 {
        assert!(
            (s_j[i] - s_b[i]).abs() < 1e-11,
            "singular value mismatch at {i}: {} vs {}",
            s_j[i],
            s_b[i]
        );
    }
    for r in 0..4 {
        for c in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                sum += u_b[mat_idx(r, k, 3)] * s_b[k] * vt_b[mat_idx(k, c, 3)];
            }
            assert!((sum - tall[mat_idx(r, c, 3)]).abs() < 1e-11);
        }
    }
}

#[test]
fn test_bidiagonal_svd_wide() {
    let wide = [1.0_f64, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0];
    let mut uw = [0.0_f64; 4];
    let mut sw = [0.0_f64; 2];
    let mut vtw = [0.0_f64; 8];
    let mut scratch = [0.0_f64; 256];
    svd_bidiagonal(
        &wide,
        2,
        4,
        &mut uw,
        &mut sw,
        &mut vtw,
        &mut SolverWorkspace::new(&mut scratch),
    )
    .unwrap();
    for r in 0..2 {
        for c in 0..4 {
            let mut sum = 0.0;
            for k in 0..2 {
                sum += uw[mat_idx(r, k, 2)] * sw[k] * vtw[mat_idx(k, c, 4)];
            }
            assert!((sum - wide[mat_idx(r, c, 4)]).abs() < 1e-11);
        }
    }
}

#[test]
fn test_csr_cg_and_gmres_solvers() {
    // 4x4 SPD tridiagonal matrix for CG
    let row_ptr = [0_usize, 2, 5, 8, 10];
    let col_ind = [0_usize, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let vals = [4.0_f64, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
    let csr_spd = CsrMatrixView::new(4, 4, &row_ptr, &col_ind, &vals).unwrap();
    let b = [3.0_f64, 2.0, 2.0, 3.0];
    let mut x = [0.0_f64; 4];
    let mut scratch = [0.0_f64; 256];

    let rep_cg = cg_solve(
        &csr_spd,
        &b,
        &mut x,
        1e-12,
        50,
        &mut SolverWorkspace::new(&mut scratch[..cg_workspace_len(4)]),
    )
    .unwrap();
    assert!(rep_cg.residual_norm < 1e-11);
    for &xi in &x {
        assert!((xi - 1.0).abs() < 1e-11);
    }

    // Nonsymmetric convection-diffusion CSR matrix for restarted GMRES
    let ns_vals = [5.0_f64, -2.0, -0.5, 5.0, -2.0, -0.5, 5.0, -2.0, -0.5, 5.0];
    let csr_ns = CsrMatrixView::new(4, 4, &row_ptr, &col_ind, &ns_vals).unwrap();
    let b_ns = [3.0_f64, 2.5, 2.5, 4.5];
    let mut x_gmres = [0.0_f64; 4];
    let rep_gm = gmres_solve(
        &csr_ns,
        &b_ns,
        &mut x_gmres,
        2,
        1e-12,
        20,
        &mut SolverWorkspace::new(&mut scratch[..gmres_workspace_len(4, 2)]),
    )
    .unwrap();
    assert!(rep_gm.residual_norm < 1e-11);
    for &xi in &x_gmres {
        assert!((xi - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_norms_condition_estimators_and_tikhonov() {
    let a = [3.0_f64, 0.0, 0.0, 1.5];
    let mut piv = [0_usize; 2];
    let mut scratch = [0.0_f64; 128];

    assert!((norm_l1(&a, 2, 2).unwrap() - 3.0).abs() < 1e-12);
    assert!((norm_linf(&a, 2, 2).unwrap() - 3.0).abs() < 1e-12);
    assert!((norm_frobenius(&a, 2, 2).unwrap() - (11.25_f64).sqrt()).abs() < 1e-12);
    assert!(
        (norm_spectral(&a, 2, 2, &mut SolverWorkspace::new(&mut scratch)).unwrap() - 3.0).abs()
            < 1e-12
    );
    assert!(
        (cond_spectral(&a, 2, 2, &mut SolverWorkspace::new(&mut scratch)).unwrap() - 2.0).abs()
            < 1e-12
    );
    assert!(
        (cond_l1(&a, 2, &mut piv, &mut SolverWorkspace::new(&mut scratch)).unwrap() - 2.0).abs()
            < 1e-12
    );

    let b = [6.0_f64, 3.0];
    let mut x_ridge = [0.0_f64; 2];
    tikhonov_solve(
        &a,
        2,
        2,
        &b,
        0.5,
        &mut x_ridge,
        &mut SolverWorkspace::new(&mut scratch),
    )
    .unwrap();
    let expected_0 = (3.0 * 6.0) / (9.0 + 0.25);
    let expected_1 = (1.5 * 3.0) / (2.25 + 0.25);
    assert!((x_ridge[0] - expected_0).abs() < 1e-12);
    assert!((x_ridge[1] - expected_1).abs() < 1e-12);
}

//! Zero-allocation verification tests for `Complex<f64>` and `Complex<f32>` solvers in `engine_linalg`.
#![cfg_attr(any(), verified(opt_out = "tests"))]
// [cite:engine_linalg]

use engine_linalg::{
    Complex, Complex32, Complex64, CsrMatrixView, FieldScalar, SolverWorkspace, gmres_solve,
    lu_determinant, lu_factor_in_place, lu_inverse, lu_solve, mat_mul, mat_vec_mul, norm_frobenius,
    qr_solve_least_squares,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_complex64_lu_solve_inverse_and_determinant_zero_alloc() {
    let a = [
        Complex64::new(3.0, 1.0),
        Complex64::new(-1.0, 2.0),
        Complex64::new(0.5, -0.5),
        Complex64::new(1.0, -1.0),
        Complex64::new(4.0, -2.0),
        Complex64::new(1.5, 0.5),
        Complex64::new(-0.5, 1.0),
        Complex64::new(0.5, -1.5),
        Complex64::new(5.0, 1.0),
    ];
    let x_true = [
        Complex64::new(1.0, -2.0),
        Complex64::new(-0.5, 1.5),
        Complex64::new(2.0, 0.25),
    ];
    let mut b = [Complex64::ZERO; 3];
    mat_vec_mul(&a, 3, 3, &x_true, &mut b).unwrap();

    VerifiedEngine::run_verified(|| {
        let mut x = [Complex64::ZERO; 3];
        let mut piv = [0_usize; 3];
        let mut scratch = [Complex64::ZERO; 32];
        lu_solve(
            &a,
            &b,
            &mut x,
            3,
            &mut piv,
            &mut SolverWorkspace::new(&mut scratch),
        )
        .unwrap();
        for i in 0..3 {
            assert!(
                (x[i] - x_true[i]).abs() < 1e-12,
                "Complex64 LU mismatch at {i}: got {}, expected {}",
                x[i],
                x_true[i]
            );
        }
    });
}

#[test]
fn test_complex64_lu_inverse_and_determinant_zero_alloc() {
    let a = [
        Complex64::new(3.0, 1.0),
        Complex64::new(-1.0, 2.0),
        Complex64::new(0.5, -0.5),
        Complex64::new(1.0, -1.0),
        Complex64::new(4.0, -2.0),
        Complex64::new(1.5, 0.5),
        Complex64::new(-0.5, 1.0),
        Complex64::new(0.5, -1.5),
        Complex64::new(5.0, 1.0),
    ];
    VerifiedEngine::run_verified(|| {
        let mut piv = [0_usize; 3];
        let mut scratch = [Complex64::ZERO; 32];
        let mut inv = [Complex64::ZERO; 9];
        lu_inverse(
            &a,
            &mut inv,
            3,
            &mut piv,
            &mut SolverWorkspace::new(&mut scratch),
        )
        .unwrap();
        let mut prod = [Complex64::ZERO; 9];
        mat_mul(&a, 3, 3, &inv, 3, &mut prod).unwrap();
        for r in 0..3 {
            for c in 0..3 {
                let expected = if r == c {
                    Complex64::ONE
                } else {
                    Complex64::ZERO
                };
                assert!((prod[r * 3 + c] - expected).abs() < 1e-12);
            }
        }
        let mut lu_buf = a;
        let parity = lu_factor_in_place(&mut lu_buf, 3, &mut piv, 0.0).unwrap();
        let det = lu_determinant(&lu_buf, 3, parity).unwrap();
        assert!(det.abs() > 1.0);
    });
}

#[test]
fn test_complex_qr_least_squares_f64_and_f32_zero_alloc() {
    let a = [
        Complex64::new(2.0, 1.0),
        Complex64::new(0.5, -1.0),
        Complex64::new(-1.0, 1.5),
        Complex64::new(3.0, 0.5),
        Complex64::new(0.0, 2.0),
        Complex64::new(1.0, 1.0),
        Complex64::new(1.5, -0.5),
        Complex64::new(-0.5, 2.5),
    ];
    let x_true = [Complex64::new(1.25, -0.75), Complex64::new(-0.5, 2.0)];
    let mut b = [Complex64::ZERO; 4];
    mat_vec_mul(&a, 4, 2, &x_true, &mut b).unwrap();

    VerifiedEngine::run_verified(|| {
        let mut x = [Complex64::ZERO; 2];
        let mut perm = [0_usize; 2];
        let mut scratch = [Complex64::ZERO; 64];
        let rank = qr_solve_least_squares(
            &a,
            4,
            2,
            &b,
            &mut x,
            &mut perm,
            &mut SolverWorkspace::new(&mut scratch),
        )
        .unwrap();
        assert_eq!(rank, 2);
        for j in 0..2 {
            assert!(
                (x[j] - x_true[j]).abs() < 1e-12,
                "Complex64 QR mismatch at {j}: got {}, expected {}",
                x[j],
                x_true[j]
            );
        }

        let a32 = a.map(|z| Complex32::new(z.re as f32, z.im as f32));
        let b32 = b.map(|z| Complex32::new(z.re as f32, z.im as f32));
        let mut x32 = [Complex32::ZERO; 2];
        let mut perm32 = [0_usize; 2];
        let mut scratch32 = [Complex32::ZERO; 64];
        let rank32 = qr_solve_least_squares(
            &a32,
            4,
            2,
            &b32,
            &mut x32,
            &mut perm32,
            &mut SolverWorkspace::new(&mut scratch32),
        )
        .unwrap();
        assert_eq!(rank32, 2);
        assert!((x32[0] - Complex::new(1.25_f32, -0.75_f32)).abs() < 1e-5);
        assert!((x32[1] - Complex::new(-0.5_f32, 2.0_f32)).abs() < 1e-5);
    });
}

#[test]
fn test_complex_gmres_and_frobenius_norm_zero_alloc() {
    let values = [
        Complex64::new(5.0, 2.0),
        Complex64::new(-1.0, 0.5),
        Complex64::new(0.5, -1.0),
        Complex64::new(4.0, -1.5),
        Complex64::new(1.0, 1.0),
        Complex64::new(-0.5, 0.25),
        Complex64::new(6.0, 1.0),
    ];
    let row_ptr = [0_usize, 2, 5, 7];
    let col_ind = [0_usize, 1, 0, 1, 2, 1, 2];
    let csr = CsrMatrixView::new(3, 3, &row_ptr, &col_ind, &values).unwrap();
    let x_true = [
        Complex64::new(1.0, 1.0),
        Complex64::new(-2.0, 0.5),
        Complex64::new(0.5, -1.5),
    ];
    let mut b = [Complex64::ZERO; 3];
    csr.spmv(&x_true, &mut b).unwrap();

    VerifiedEngine::run_verified(|| {
        let mut x = [Complex64::ZERO; 3];
        let mut scratch = [Complex64::ZERO; 128];
        let report = gmres_solve(
            &csr,
            &b,
            &mut x,
            3,
            1e-12,
            5,
            &mut SolverWorkspace::new(&mut scratch),
        )
        .unwrap();
        assert!(report.residual_norm < 1e-11);
        for i in 0..3 {
            assert!((x[i] - x_true[i]).abs() < 1e-11);
        }
        let frob = norm_frobenius(&values, 1, 7).unwrap();
        assert!(frob > 8.0);
    });
}

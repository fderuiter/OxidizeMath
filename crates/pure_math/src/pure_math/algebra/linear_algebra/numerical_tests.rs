use super::*;

fn lauchli_matrix(n: usize, mu: f64) -> DMatrix<f64> {
    DMatrix::from_fn(n + 1, n, |i, j| {
        if i == 0 {
            1.0
        } else if i - 1 == j {
            mu
        } else {
            0.0
        }
    })
}

fn vandermonde_matrix(rows: usize, cols: usize) -> DMatrix<f64> {
    DMatrix::from_fn(rows, cols, |i, j| {
        let t = (i as f64) / ((rows.saturating_sub(1)).max(1) as f64);
        t.powi(j as i32)
    })
}

#[test]
#[verified_engine::verified]
fn test_normal_equation() {
    let a = DMatrix::from_row_slice(3, 2, &[1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
    let b = DVector::from_column_slice(&[1.0, 2.0, 3.0]);

    let c_vec = solve_normal_equation(&a, &b).unwrap();
    assert!(c_vec[0].abs() < math_commons::registry::TOLERANCE_FAST);
    assert!((c_vec[1] - 1.0).abs() < math_commons::registry::TOLERANCE_FAST);
}

#[test]
#[verified_engine::verified]
fn test_lauchli_system_avoids_condition_squaring() {
    // For mu = 1e-8, 1.0 + mu^2 == 1.0 in IEEE-754 f64 so explicit A^T A is strictly rank-1,
    // whereas Householder QR preserves mu on R's diagonal and recovers x_true accurately.
    let a = lauchli_matrix(5, 1e-8);
    let x_true = DVector::from_column_slice(&[1.0, -2.0, 0.5, 3.0, -1.5]);
    let b = &a * &x_true;

    let x_hat = solve_normal_equation(&a, &b).unwrap();
    let rel_err = (&x_hat - &x_true).norm() / x_true.norm();
    assert!(
        rel_err < 1e-7,
        "Expected accurate QR solve on Lauchli(1e-8), got rel_err={rel_err:e}"
    );
}

#[test]
#[verified_engine::verified]
fn test_vandermonde_and_rank_deficient_svd_fallback() {
    let v = vandermonde_matrix(14, 9);
    let coeffs = DVector::from_fn(9, |j, _| if j % 2 == 0 { 1.0 } else { -0.5 });
    let y = &v * &coeffs;
    let fit = solve_normal_equation(&v, &y).unwrap();
    assert!((&fit - &coeffs).norm() / coeffs.norm() < 1e-6);

    // Duplicate column triggers R_{1,1} == 0 -> thin SVD minimum-norm fallback
    let collinear = DMatrix::from_row_slice(3, 2, &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    let rhs = DVector::from_column_slice(&[2.0, 4.0, 6.0]);
    let min_norm = solve_normal_equation(&collinear, &rhs).unwrap();
    assert!((min_norm[0] - 1.0).abs() < 1e-10);
    assert!((min_norm[1] - 1.0).abs() < 1e-10);

    // All-zero matrix must fail with SingularMatrix
    let zero_a = DMatrix::zeros(3, 2);
    assert_eq!(
        solve_normal_equation(&zero_a, &rhs),
        Err(LinearAlgebraError::SingularMatrix)
    );
}

#[test]
#[verified_engine::verified]
fn test_pseudoinverse_and_error_edge_cases() {
    let rect = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
    let pinv = moore_penrose_pseudoinverse(&rect, 1e-12).unwrap();
    let reconstructed = &rect * &pinv * &rect;
    assert!((&reconstructed - &rect).norm() < 1e-12);

    let nan_mat = DMatrix::from_row_slice(2, 2, &[1.0, f64::NAN, 0.0, 1.0]);
    let inf_vec = DVector::from_column_slice(&[1.0, f64::INFINITY]);
    let ok_vec = DVector::from_column_slice(&[1.0, 2.0]);
    let empty_mat = DMatrix::<f64>::zeros(0, 0);

    assert_eq!(
        moore_penrose_pseudoinverse(&nan_mat, 1e-9),
        Err(LinearAlgebraError::NonFiniteInput)
    );
    assert_eq!(
        moore_penrose_pseudoinverse(&rect, -1e-6),
        Err(LinearAlgebraError::NotPositiveDefinite)
    );
    assert_eq!(
        moore_penrose_pseudoinverse(&empty_mat, 1e-9),
        Err(LinearAlgebraError::DimensionMismatch)
    );
    assert_eq!(
        solve_normal_equation(&nan_mat, &ok_vec),
        Err(LinearAlgebraError::NonFiniteInput)
    );
    assert_eq!(
        solve_normal_equation(&DMatrix::identity(2, 2), &inf_vec),
        Err(LinearAlgebraError::NonFiniteInput)
    );
    assert_eq!(
        solve_normal_equation(&rect, &ok_vec),
        Err(LinearAlgebraError::DimensionMismatch)
    );
}

#[test]
#[verified_engine::verified]
fn test_ridge_regression_augmented_qr() {
    let a = lauchli_matrix(4, 1e-8);
    let b = DVector::from_column_slice(&[4.0, 1e-8, 2e-8, -1e-8, 0.5e-8]);

    let x_small = solve_ridge_regression(&a, &b, 1e-6).unwrap();
    let x_large = solve_ridge_regression(&a, &b, 10.0).unwrap();
    assert!(x_large.norm() < x_small.norm());

    // Compare against closed-form (A^T A + lambda I)x = A^T b on well-conditioned A
    let well = DMatrix::from_row_slice(3, 2, &[2.0, 1.0, 1.0, 3.0, 0.5, -1.0]);
    let rhs = DVector::from_column_slice(&[1.0, 2.0, -0.5]);
    let lam = 0.25;
    let x_ridge = solve_ridge_regression(&well, &rhs, lam).unwrap();
    let normal_mat = well.transpose() * &well + DMatrix::identity(2, 2) * lam;
    let expected = normal_mat.lu().solve(&(well.transpose() * &rhs)).unwrap();
    assert!((&x_ridge - &expected).norm() < 1e-12);

    assert_eq!(
        solve_ridge_regression(&well, &rhs, -0.1),
        Err(LinearAlgebraError::NotPositiveDefinite)
    );
    assert_eq!(
        solve_ridge_regression(&well, &rhs, f64::NAN),
        Err(LinearAlgebraError::NonFiniteInput)
    );
}

#[test]
#[verified_engine::verified]
fn test_nnls_active_set_kkt_conditions() {
    // Unconstrained OLS solution has negative second component, so NNLS must clamp x_1 = 0
    // and satisfy KKT conditions: x >= 0, w = A^T(b - Ax) <= 0, and x_j * w_j == 0.
    let a = DMatrix::from_row_slice(3, 2, &[1.0, 0.2, 1.0, 1.0, 0.5, 2.0]);
    let b = DVector::from_column_slice(&[2.0, 0.5, -1.0]);

    let x = solve_nnls(&a, &b).unwrap();
    assert!(x.iter().all(|&v| v >= 0.0));
    assert!(x[0] > 0.0);
    assert!(x[1].abs() < 1e-12);

    let w = a.transpose() * (&b - &a * &x);
    for j in 0..x.len() {
        assert!(w[j] <= 1e-10, "Dual KKT violation at {j}: w={}", w[j]);
        assert!(
            (x[j] * w[j]).abs() <= 1e-10,
            "Complementary slackness violation at {j}"
        );
    }
}

use super::*;
use nalgebra::DMatrix;

#[test]
#[verified_engine::verified]
fn test_effective_rank() {
    let m = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);
    assert_eq!(effective_rank(&m, None), 1);

    let m2 = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    assert_eq!(effective_rank(&m2, None), 2);

    let nan_m = DMatrix::from_row_slice(2, 2, &[f64::NAN, 0.0, 0.0, 1.0]);
    assert_eq!(effective_rank(&nan_m, None), 0);
}

#[test]
#[verified_engine::verified]
fn test_low_rank_approx() {
    let m = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 1.0]);
    let approx = low_rank_approximation(&m, 1).unwrap();

    let diff = &m - &approx;
    let norm = diff.norm();
    assert!((norm - 1.0).abs() < 1e-5);

    let nan_m = DMatrix::from_row_slice(2, 2, &[2.0, f64::NAN, 0.0, 1.0]);
    assert_eq!(
        low_rank_approximation(&nan_m, 1),
        Err(LinearAlgebraError::NonFiniteInput)
    );
}

#[test]
#[verified_engine::verified]
fn test_qr_cholesky_and_svd_decompositions() {
    let rect = DMatrix::from_row_slice(
        4,
        3,
        &[
            1.0, 2.0, 0.5, 2.0, -1.0, 1.0, 0.5, 3.0, -2.0, -1.0, 1.0, 4.0,
        ],
    );
    let (q, r) = qr_decomposition(&rect).unwrap();
    assert!((&q * &r - &rect).norm() < 1e-12);
    assert!((q.transpose() * &q - DMatrix::identity(3, 3)).norm() < 1e-12);

    let (u, s, v_t) = svd_decomposition(&rect).unwrap();
    assert!(s[0] >= s[1] && s[1] >= s[2]);
    let recon = &u * DMatrix::from_diagonal(&s) * &v_t;
    assert!((recon - &rect).norm() < 1e-12);

    // Symmetric positive definite Cholesky
    let spd = DMatrix::from_row_slice(2, 2, &[4.0, 2.0, 2.0, 5.0]);
    let l = cholesky_decomposition(&spd).unwrap();
    assert!((&l * l.transpose() - &spd).norm() < 1e-12);

    // Indefinite and asymmetric matrices must return NotPositiveDefinite
    let indef = DMatrix::from_row_slice(2, 2, &[1.0, 3.0, 3.0, 1.0]);
    assert_eq!(
        cholesky_decomposition(&indef),
        Err(LinearAlgebraError::NotPositiveDefinite)
    );
    let asym = DMatrix::from_row_slice(2, 2, &[4.0, 10.0, 0.0, 4.0]);
    assert_eq!(
        cholesky_decomposition(&asym),
        Err(LinearAlgebraError::NotPositiveDefinite)
    );
}

#[test]
#[verified_engine::verified]
fn test_condition_number_and_matrix_norms() {
    // Lauchli(3, 1e-6) has sigma_max = sqrt(3 + 1e-12) and sigma_min = 1e-6
    let mu = 1e-6;
    let lauchli = DMatrix::from_row_slice(
        4,
        3,
        &[1.0, 1.0, 1.0, mu, 0.0, 0.0, 0.0, mu, 0.0, 0.0, 0.0, mu],
    );
    let cond = condition_number(&lauchli).unwrap();
    let expected_cond = (3.0 + mu * mu).sqrt() / mu;
    assert!((cond - expected_cond).abs() / expected_cond < 1e-9);

    let singular = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 0.0, 0.0]);
    assert_eq!(condition_number(&singular).unwrap(), f64::INFINITY);

    let m = DMatrix::from_row_slice(2, 2, &[1.0, -3.0, 2.0, 4.0]);
    assert!((matrix_norm(&m, MatrixNorm::One).unwrap() - 7.0).abs() < 1e-12);
    assert!((matrix_norm(&m, MatrixNorm::Infinity).unwrap() - 6.0).abs() < 1e-12);
    assert!((matrix_norm(&m, MatrixNorm::Frobenius).unwrap() - 30.0_f64.sqrt()).abs() < 1e-12);
    let spec = matrix_norm(&m, MatrixNorm::Spectral).unwrap();
    assert!(spec > 5.0 && spec <= 30.0_f64.sqrt());
}

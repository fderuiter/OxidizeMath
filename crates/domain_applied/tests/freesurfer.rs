//! Integration tests for freesurfer.
use domain_applied::applied::freesurfer::*;
use nalgebra::{DMatrix, DVector};

#[test]
#[verified_engine::verified]
fn test_surface_reconstruction() {
    let mut surface = Surface {
        vertices: vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    };
    let internal = internal_energy(&surface, 0.5, 0.2);
    assert!(internal > 0.0);

    let external = external_energy(&surface, 10.0);
    assert!(external < 0.0);

    let initial_vertex = surface.vertices[0][0];
    evolve_surface(&mut surface, 0.1, 1.0);
    assert_ne!(initial_vertex, surface.vertices[0][0]);
}

#[test]
#[verified_engine::verified]
fn test_bayesian_classification() {
    let likelihood = 0.8;
    let prior = 0.5;
    let posterior = bayesian_classification(likelihood, prior);
    assert_eq!(posterior, 0.4);
}

#[test]
#[verified_engine::verified]
fn test_cortical_thickness() {
    // Add the test points as vertices to the surfaces to ensure the simplified
    // vertex-only distance calculation works as expected for this test.
    let white_surface = Surface {
        vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
    };
    let pial_surface = Surface {
        vertices: vec![[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [0.5, 2.0, 0.0]],
    };
    let v_w = [0.5, 0.0, 0.0];
    let v_p = [0.5, 2.0, 0.0];

    let thickness = cortical_thickness(&v_w, &v_p, &white_surface, &pial_surface);
    assert!((thickness - 2.0).abs() < math_commons::registry::TOLERANCE_STANDARD);
    assert_eq!(thickness, 2.0);
}

#[test]
#[verified_engine::verified]
fn test_cortical_thickness_exact_distance_equivalence() {
    let mut white_verts = Vec::new();
    let mut pial_verts = Vec::new();
    for i in 0..50 {
        let x = (i as f64) * 0.2;
        let y = ((i * 7) % 13) as f64 * 0.3;
        white_verts.push([x, y, 0.0]);
        pial_verts.push([x, y, 2.5]);
    }
    let white_surface = Surface {
        vertices: white_verts,
    };
    let pial_surface = Surface {
        vertices: pial_verts,
    };

    let naive_dist = |pt: &[f64; 3], surf: &Surface| -> f64 {
        surf.vertices
            .iter()
            .map(|v| {
                let dx = pt[0] - v[0];
                let dy = pt[1] - v[1];
                let dz = pt[2] - v[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .fold(f64::INFINITY, f64::min)
    };

    for i in (0..50).step_by(5) {
        let v_w = white_surface.vertices[i];
        let v_p = pial_surface.vertices[i];
        let expected = 0.5 * (naive_dist(&v_w, &pial_surface) + naive_dist(&v_p, &white_surface));
        let actual = cortical_thickness(&v_w, &v_p, &white_surface, &pial_surface);
        assert_eq!(expected, actual);
    }
}

#[test]
#[verified_engine::verified]
fn test_cortical_thickness_sparse_expansion() {
    // Surfaces with sparse vertex distributions that require expanding neighboring cell search radii
    let white_surface = Surface {
        vertices: vec![[0.0, 0.0, 0.0], [50.0, 50.0, 50.0]],
    };
    let pial_surface = Surface {
        vertices: vec![[0.0, 3.0, 0.0], [50.0, 53.0, 50.0]],
    };
    let v_w = [25.0, 25.0, 25.0];
    let v_p = [25.0, 28.0, 25.0];

    let thickness = cortical_thickness(&v_w, &v_p, &white_surface, &pial_surface);
    let naive_dist = |pt: &[f64; 3], surf: &Surface| -> f64 {
        surf.vertices
            .iter()
            .map(|v| {
                let dx = pt[0] - v[0];
                let dy = pt[1] - v[1];
                let dz = pt[2] - v[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .fold(f64::INFINITY, f64::min)
    };
    let expected = 0.5 * (naive_dist(&v_w, &pial_surface) + naive_dist(&v_p, &white_surface));
    assert_eq!(thickness, expected);
}

#[test]
#[verified_engine::verified]
fn test_glm_beta_estimation() {
    let x: DMatrix<f64> = DMatrix::from_row_slice(4, 2, &[1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let y: DVector<f64> = DVector::from_vec(vec![6.0, 5.0, 7.0, 10.0]);

    let beta = estimate_beta(&x, &y).unwrap();
    assert!((beta[0] - 3.5).abs() < math_commons::registry::TOLERANCE_STANDARD);
    assert!((beta[1] - 1.4).abs() < math_commons::registry::TOLERANCE_STANDARD);
}

#[test]
#[verified_engine::verified]
fn test_glm_t_statistic() {
    let x: DMatrix<f64> = DMatrix::from_row_slice(4, 2, &[1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let y: DVector<f64> = DVector::from_vec(vec![6.0, 5.0, 7.0, 10.0]);
    let beta = estimate_beta(&x, &y).unwrap();

    let c: DVector<f64> = DVector::from_vec(vec![0.0, 1.0]);
    let residuals = y - &x * &beta;
    let residual_variance = residuals.norm_squared() / (4.0 - 2.0);

    let t = t_statistic(&c, &beta, &x, residual_variance).unwrap();
    // The expected value was re-calculated to be 2.1602...
    assert!((t - 2.1602).abs() < 1e-4);
}

#[test]
#[verified_engine::verified]
fn test_glm_lauchli_and_vandermonde_qr_stability() {
    // Lauchli design matrix with mu = 1e-8 makes X^T X rank-1 in f64,
    // whereas QR-based estimate_beta and t_statistic remain well-behaved.
    let mu = 1e-8_f64;
    let x = DMatrix::from_row_slice(
        4,
        3,
        &[1.0, 1.0, 1.0, mu, 0.0, 0.0, 0.0, mu, 0.0, 0.0, 0.0, mu],
    );
    let beta_true = DVector::from_vec(vec![2.0, -1.0, 0.5]);
    let noise = DVector::from_vec(vec![1e-10, -1e-10, 0.5e-10, -0.5e-10]);
    let y = &x * &beta_true + &noise;

    let beta_hat = estimate_beta(&x, &y).unwrap();
    assert!((&beta_hat - &beta_true).norm() < 1e-2 * beta_true.norm());

    let residuals = &y - &x * &beta_hat;
    let sigma2 = (residuals.norm_squared() / 1.0).max(1e-24);
    let contrast = DVector::from_vec(vec![1.0, -1.0, 0.0]);
    let t_val = t_statistic(&contrast, &beta_hat, &x, sigma2).unwrap();
    assert!(t_val.is_finite() && t_val > 0.0);

    // Ill-conditioned Vandermonde polynomial design matrix
    let vand = DMatrix::from_fn(12, 6, |i, j| ((i as f64) / 11.0).powi(j as i32));
    let poly_beta = DVector::from_vec(vec![1.0, -0.5, 0.25, 0.1, -0.2, 0.05]);
    let vand_y = &vand * &poly_beta;
    let vand_est = estimate_beta(&vand, &vand_y).unwrap();
    assert!((&vand_est - &poly_beta).norm() < 1e-8);

    // Reject NaN and singular design matrices cleanly
    let nan_x = DMatrix::from_row_slice(2, 2, &[1.0, f64::NAN, 0.0, 1.0]);
    assert!(estimate_beta(&nan_x, &DVector::from_vec(vec![1.0, 2.0])).is_err());
    let sing_x = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    assert!(estimate_beta(&sing_x, &DVector::from_vec(vec![1.0, 2.0, 3.0])).is_err());
}

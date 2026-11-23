//! Integration and zero-allocation verification tests for `domain_applied::applied::freesurfer`.
use domain_applied::applied::freesurfer::*;
use nalgebra::{DMatrix, DVector};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

fn build_octahedron_surface(radius: f64) -> Surface {
    let vertices = vec![
        [radius, 0.0, 0.0],
        [-radius, 0.0, 0.0],
        [0.0, radius, 0.0],
        [0.0, -radius, 0.0],
        [0.0, 0.0, radius],
        [0.0, 0.0, -radius],
    ];
    let faces = vec![
        [4, 0, 2],
        [4, 2, 1],
        [4, 1, 3],
        [4, 3, 0],
        [5, 2, 0],
        [5, 1, 2],
        [5, 3, 1],
        [5, 0, 3],
    ];
    Surface::new(vertices, faces)
}

fn build_spherical_shell_mri(target_radius: f64) -> MriVolume {
    let dims = [17, 17, 17];
    let origin = [-8.0, -8.0, -8.0];
    let voxel_size = 1.0;
    let mut intensities = Vec::with_capacity(17 * 17 * 17);
    for iz in 0..dims[2] {
        let z = origin[2] + (iz as f64) * voxel_size;
        for iy in 0..dims[1] {
            let y = origin[1] + (iy as f64) * voxel_size;
            for ix in 0..dims[0] {
                let x = origin[0] + (ix as f64) * voxel_size;
                let r = (x * x + y * y + z * z).sqrt();
                intensities.push(if r <= target_radius { 110.0 } else { 35.0 });
            }
        }
    }
    MriVolume::new(dims, origin, voxel_size, intensities).expect("valid spherical MRI phantom")
}

#[test]
#[verified_engine::verified]
fn test_surface_reconstruction() {
    let mut surface = Surface::from_vertices(vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]);
    let internal = internal_energy(&surface, 0.5, 0.2);
    assert!(internal > 0.0);

    let external = external_energy(&surface, 10.0);
    assert!(external < 0.0);

    let initial_vertex = surface.vertices[0][0];
    evolve_surface(&mut surface, 0.1, 1.0);
    assert_ne!(initial_vertex, surface.vertices[0][0]);
}

#[test]
fn test_zero_alloc_internal_energy_and_deformable_evolution() {
    let mut surface = build_octahedron_surface(3.6);
    surface.vertices[4][2] = 4.2; // perturb north pole to activate bending rigidity
    let volume = build_spherical_shell_mri(4.5);

    let (e_tension_only, _) = VerifiedEngine::run_verified(|| internal_energy(&surface, 1.0, 0.0));
    let (e_bending_only, _) = VerifiedEngine::run_verified(|| internal_energy(&surface, 0.0, 1.0));
    assert!(e_tension_only > 0.0);
    assert!(e_bending_only > 0.0);
    assert!((e_tension_only - e_bending_only).abs() > 1e-3);

    let ((e_before, e_after), metrics) = VerifiedEngine::run_verified(|| {
        let before =
            internal_energy(&surface, 0.15, 0.1) + external_energy_volume(&surface, &volume, 1.2);
        for _ in 0..4 {
            evolve_surface_deformable(&mut surface, &volume, 0.15, 0.1, 1.2, 0.008);
        }
        let after =
            internal_energy(&surface, 0.15, 0.1) + external_energy_volume(&surface, &volume, 1.2);
        (before, after)
    });

    assert!(
        e_after < e_before,
        "Expected deformable gradient descent to decrease total energy: {e_after} >= {e_before}"
    );
    assert!(metrics.function_calls > 0);
}

#[test]
fn test_surface_empty_faces_and_degenerate_normals_edge_cases() {
    let empty = Surface::from_vertices(Vec::new());
    assert_eq!(internal_energy(&empty, 1.0, 1.0), 0.0);

    let mut collinear = Surface::new(
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        vec![[0, 1, 2]],
    );
    evolve_surface(&mut collinear, 0.25, 2.0);
    assert!(collinear.vertices.iter().flatten().all(|c| c.is_finite()));
    assert_eq!(collinear.vertices[0], [0.0, 0.0, 0.0]);
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
fn test_zero_alloc_mrf_segmentation_and_zero_sum_uniform_fallback() {
    let classes = [
        TissueClass::new(25.0, 8.0, 0.2),  // CSF
        TissueClass::new(70.0, 10.0, 0.4), // Gray matter
        TissueClass::new(110.0, 9.0, 0.4), // White matter
    ];
    let mut intensities = vec![110.0; 5 * 5 * 5];
    let center = (2 * 5 + 2) * 5 + 2;
    intensities[center] = 74.0; // noisy voxel surrounded by 6 white-matter neighbors
    let volume = MriVolume::new([5, 5, 5], [0.0, 0.0, 0.0], 1.0, intensities).unwrap();

    let mut ml_labels = vec![0_u8; 125];
    let mut mrf_labels = vec![0_u8; 125];
    let mut zero_post = [0.0_f64; 3];

    let (sweeps, metrics) = VerifiedEngine::run_verified(|| {
        let _ = mrf_segment_volume(&volume, &classes, 0.0, 5, &mut ml_labels);
        let iters = mrf_segment_volume(&volume, &classes, 1.4, 8, &mut mrf_labels);
        let _ = normalized_posteriors_into(&[0.0, 0.0, 0.0], &[0.2, 0.4, 0.4], &mut zero_post);
        iters
    });

    assert!(sweeps >= 1);
    assert!(metrics.function_calls > 0);
    assert_eq!(ml_labels[center], 1, "Unregularized ML picks gray matter");
    assert_eq!(
        mrf_labels[center], 2,
        "6-connected Potts MRF restores white matter coherence"
    );
    for p in zero_post {
        assert!((p - (1.0 / 3.0)).abs() < 1e-12);
    }
}

#[test]
#[verified_engine::verified]
fn test_cortical_thickness() {
    let white_surface =
        Surface::from_vertices(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]]);
    let pial_surface =
        Surface::from_vertices(vec![[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [0.5, 2.0, 0.0]]);
    let v_w = [0.5, 0.0, 0.0];
    let v_p = [0.5, 2.0, 0.0];

    let thickness = cortical_thickness(&v_w, &v_p, &white_surface, &pial_surface);
    assert!((thickness - 2.0).abs() < math_commons::registry::TOLERANCE_STANDARD);
    assert_eq!(thickness, 2.0);
}

#[test]
fn test_cortical_thickness_exact_distance_equivalence() {
    let mut white_verts = Vec::with_capacity(50);
    let mut pial_verts = Vec::with_capacity(50);
    for i in 0..50 {
        let x = (i as f64) * 0.2;
        let y = ((i * 7) % 13) as f64 * 0.3;
        let z = ((i * 5) % 9) as f64 * 0.15;
        white_verts.push([x, y, z]);
        pial_verts.push([x, y, z + 2.5]);
    }
    let white_surface = Surface::from_vertices(white_verts);
    let pial_surface = Surface::from_vertices(pial_verts);

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

    let batch_map = cortical_thickness_map(&white_surface, &pial_surface);
    assert_eq!(batch_map.len(), 50);

    let mut workspace = ThicknessWorkspace::with_max_vertices(50);
    let mut map_into = vec![0.0_f64; 50];
    let mut direct_samples = [0.0_f64; 10];
    let mut indexed_samples = [0.0_f64; 10];

    let (written, metrics) = VerifiedEngine::run_verified(|| {
        let count = cortical_thickness_map_into(
            &white_surface,
            &pial_surface,
            &mut workspace,
            &mut map_into,
        );
        let (white_grid, pial_grid) = workspace
            .build_grids(&white_surface, &pial_surface)
            .expect("preallocated workspace has sufficient capacity");
        for (slot, i) in (0..50).step_by(5).enumerate() {
            let v_w = white_surface.vertices[i];
            let v_p = pial_surface.vertices[i];
            direct_samples[slot] = cortical_thickness(&v_w, &v_p, &white_surface, &pial_surface);
            indexed_samples[slot] = cortical_thickness_indexed(&v_w, &v_p, &white_grid, &pial_grid);
        }
        count
    });

    assert_eq!(written, 50);
    assert!(metrics.function_calls > 0);
    assert_eq!(batch_map, map_into);

    for (slot, i) in (0..50).step_by(5).enumerate() {
        let v_w = white_surface.vertices[i];
        let v_p = pial_surface.vertices[i];
        let expected = 0.5 * (naive_dist(&v_w, &pial_surface) + naive_dist(&v_p, &white_surface));
        assert_eq!(expected, direct_samples[slot]);
        assert_eq!(expected, indexed_samples[slot]);
        assert_eq!(expected, map_into[i]);
    }
}

#[test]
fn test_zero_alloc_cortical_thickness_edge_cases() {
    let empty = Surface::from_vertices(Vec::new());
    let single_w = Surface::from_vertices(vec![[1.0, 2.0, 3.0]]);
    let single_p = Surface::from_vertices(vec![[1.0, 2.0, 6.0]]);
    let mut empty_ws = ThicknessWorkspace::new();
    let mut ws = ThicknessWorkspace::with_max_vertices(1);
    let mut out = [0.0_f64; 1];

    let ((empty_t, nan_t, undercap, ok_count), metrics) = VerifiedEngine::run_verified(|| {
        let d_empty = cortical_thickness(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0], &empty, &empty);
        let (wg, pg) = ws.build_grids(&single_w, &single_p).unwrap();
        let d_nan = cortical_thickness_indexed(&[f64::NAN, 0.0, 0.0], &[1.0, 2.0, 6.0], &wg, &pg);
        let bad = cortical_thickness_map_into(&single_w, &single_p, &mut empty_ws, &mut out);
        let good = cortical_thickness_map_into(&single_w, &single_p, &mut ws, &mut out);
        (d_empty, d_nan, bad, good)
    });

    assert_eq!(empty_t, f64::INFINITY);
    assert!(nan_t.is_nan());
    assert_eq!(undercap, 0);
    assert_eq!(ok_count, 1);
    assert_eq!(out[0], 3.0);
    assert!(metrics.function_calls > 0);
}

#[test]
fn test_zero_alloc_10k_vertex_mesh_thickness_map_into() {
    let n = 10_000_usize;
    let mut white_verts = Vec::with_capacity(n);
    let mut pial_verts = Vec::with_capacity(n);
    for i in 0..n {
        let u = (i % 100) as f64 * 0.0628;
        let v = (i / 100) as f64 * 0.0314;
        let (su, cu) = u.sin_cos();
        let (sv, cv) = v.sin_cos();
        let r_w = 20.0 + 1.5 * (3.0 * u).cos() * sv;
        let r_p = r_w + 2.5;
        white_verts.push([r_w * sv * cu, r_w * sv * su, r_w * cv]);
        pial_verts.push([r_p * sv * cu, r_p * sv * su, r_p * cv]);
    }
    let white = Surface::from_vertices(white_verts);
    let pial = Surface::from_vertices(pial_verts);
    let mut workspace = ThicknessWorkspace::with_max_vertices(n);
    let mut thicknesses = vec![0.0_f64; n];

    let (written, metrics) = VerifiedEngine::run_verified(|| {
        cortical_thickness_map_into(&white, &pial, &mut workspace, &mut thicknesses)
    });

    assert_eq!(written, n);
    assert!(metrics.function_calls > 0);
    assert!(thicknesses.iter().all(|&t| t > 1.5 && t <= 2.6));
}

#[test]
#[verified_engine::verified]
fn test_cortical_thickness_sparse_expansion() {
    let white_surface = Surface::from_vertices(vec![[0.0, 0.0, 0.0], [50.0, 50.0, 50.0]]);
    let pial_surface = Surface::from_vertices(vec![[0.0, 3.0, 0.0], [50.0, 53.0, 50.0]]);
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

    // Reject NaN and singular design matrices cleanly with typed EngineeringError
    let nan_x = DMatrix::from_row_slice(2, 2, &[1.0, f64::NAN, 0.0, 1.0]);
    assert!(matches!(
        estimate_beta(&nan_x, &DVector::from_vec(vec![1.0, 2.0])),
        Err(domain_applied::error::EngineeringError::Math(_))
    ));
    let sing_x = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    assert_eq!(
        estimate_beta(&sing_x, &DVector::from_vec(vec![1.0, 2.0, 3.0])),
        Err(domain_applied::error::EngineeringError::SingularDesignMatrix)
    );
}

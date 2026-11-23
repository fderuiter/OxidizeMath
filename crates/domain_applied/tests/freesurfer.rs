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

    for i in (0..50).step_by(5) {
        let v_w = white_surface.vertices[i];
        let v_p = pial_surface.vertices[i];
        let expected = 0.5 * (naive_dist(&v_w, &pial_surface) + naive_dist(&v_p, &white_surface));
        let actual = cortical_thickness(&v_w, &v_p, &white_surface, &pial_surface);
        assert_eq!(expected, actual);
        assert_eq!(expected, batch_map[i]);
    }
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

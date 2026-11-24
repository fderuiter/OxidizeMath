//! Integration and zero-allocation verification tests for `domain_applied::applied::freesurfer`.
use domain_applied::applied::freesurfer::*;
use nalgebra::{DMatrix, DVector};
use std::sync::Mutex;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;
use verified_engine::metrics::ComplexityMetrics;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

static VERIFIED_RUN_MUTEX: Mutex<()> = Mutex::new(());

fn run_verified_locked<F, R>(f: F) -> (R, ComplexityMetrics)
where
    F: FnOnce() -> R,
{
    let _guard = VERIFIED_RUN_MUTEX
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    VerifiedEngine::run_verified(f)
}

fn naive_surface_distance(pt: &[f64; 3], surf: &Surface) -> f64 {
    if surf.vertices.is_empty() {
        return f64::INFINITY;
    }
    if pt[0].is_nan() || pt[1].is_nan() || pt[2].is_nan() {
        return f64::NAN;
    }
    surf.vertices
        .iter()
        .map(|v| {
            let (dx, dy, dz) = (pt[0] - v[0], pt[1] - v[1], pt[2] - v[2]);
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

fn build_wavy_surfaces(count: usize) -> (Surface, Surface) {
    let mut white_verts = Vec::with_capacity(count);
    let mut pial_verts = Vec::with_capacity(count);
    for i in 0..count {
        let x = ((i * 19) % 97) as f64 * 0.15;
        let y = ((i * 37) % 89) as f64 * 0.15;
        let z = ((i * 53) % 83) as f64 * 0.05;
        white_verts.push([x, y, z]);
        pial_verts.push([x + 0.1, y - 0.05, z + 2.4]);
    }
    (
        Surface::from_vertices(white_verts),
        Surface::from_vertices(pial_verts),
    )
}

fn build_octahedron_surface(radius: f64) -> Surface {
    let r = radius;
    let vertices = vec![
        [r, 0.0, 0.0],
        [-r, 0.0, 0.0],
        [0.0, r, 0.0],
        [0.0, -r, 0.0],
        [0.0, 0.0, r],
        [0.0, 0.0, -r],
    ];
    let faces = [
        [4, 0, 2],
        [4, 2, 1],
        [4, 1, 3],
        [4, 3, 0],
        [5, 2, 0],
        [5, 1, 2],
        [5, 3, 1],
        [5, 0, 3],
    ]
    .to_vec();
    Surface::new(vertices, faces)
}

fn build_spherical_shell_mri(target_radius: f64) -> MriVolume {
    let dims = [17, 17, 17];
    let origin = [-8.0, -8.0, -8.0];
    let mut intensities = Vec::with_capacity(17 * 17 * 17);
    for iz in 0..dims[2] {
        let z = origin[2] + (iz as f64);
        for iy in 0..dims[1] {
            let y = origin[1] + (iy as f64);
            for ix in 0..dims[0] {
                let x = origin[0] + (ix as f64);
                let r = (x * x + y * y + z * z).sqrt();
                intensities.push(if r <= target_radius { 110.0 } else { 35.0 });
            }
        }
    }
    MriVolume::new(dims, origin, 1.0, intensities).expect("valid spherical MRI phantom")
}

#[test]
#[verified_engine::verified]
fn test_surface_reconstruction() {
    let mut surface = Surface::from_vertices(vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]);
    assert!(internal_energy(&surface, 0.5, 0.2) > 0.0);
    assert!(external_energy(&surface, 10.0) < 0.0);
    let initial_vertex = surface.vertices[0][0];
    evolve_surface(&mut surface, 0.1, 1.0);
    assert_ne!(initial_vertex, surface.vertices[0][0]);
}

#[test]
fn test_zero_alloc_internal_energy_and_deformable_evolution() {
    let mut surface = build_octahedron_surface(3.6);
    surface.vertices[4][2] = 4.2;
    let volume = build_spherical_shell_mri(4.5);

    let (e_tension_only, _) = run_verified_locked(|| internal_energy(&surface, 1.0, 0.0));
    let (e_bending_only, _) = run_verified_locked(|| internal_energy(&surface, 0.0, 1.0));
    assert!(e_tension_only > 0.0 && e_bending_only > 0.0);
    assert!((e_tension_only - e_bending_only).abs() > 1e-3);

    let ((e_before, e_after), metrics) = run_verified_locked(|| {
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
        "Expected descent: {e_after} >= {e_before}"
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
    assert_eq!(bayesian_classification(0.8, 0.5), 0.4);
}

#[test]
fn test_zero_alloc_mrf_segmentation_and_zero_sum_uniform_fallback() {
    let classes = [
        TissueClass::new(25.0, 8.0, 0.2),
        TissueClass::new(70.0, 10.0, 0.4),
        TissueClass::new(110.0, 9.0, 0.4),
    ];
    let mut intensities = vec![110.0; 125];
    let center = (2 * 5 + 2) * 5 + 2;
    intensities[center] = 74.0;
    let volume = MriVolume::new([5, 5, 5], [0.0, 0.0, 0.0], 1.0, intensities).unwrap();

    let mut ml_labels = vec![0_u8; 125];
    let mut mrf_labels = vec![0_u8; 125];
    let mut zero_post = [0.0_f64; 3];

    let (sweeps, metrics) = run_verified_locked(|| {
        let _ = mrf_segment_volume(&volume, &classes, 0.0, 5, &mut ml_labels);
        let iters = mrf_segment_volume(&volume, &classes, 1.4, 8, &mut mrf_labels);
        let _ = normalized_posteriors_into(&[0.0, 0.0, 0.0], &[0.2, 0.4, 0.4], &mut zero_post);
        iters
    });

    assert!(sweeps >= 1 && metrics.function_calls > 0);
    assert_eq!((ml_labels[center], mrf_labels[center]), (1, 2));
    for p in zero_post {
        assert!((p - (1.0 / 3.0)).abs() < 1e-12);
    }
}

#[test]
#[verified_engine::verified]
fn test_cortical_thickness() {
    let white = Surface::from_vertices(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]]);
    let pial = Surface::from_vertices(vec![[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [0.5, 2.0, 0.0]]);
    let thickness = cortical_thickness(&[0.5, 0.0, 0.0], &[0.5, 2.0, 0.0], &white, &pial);
    assert_eq!(thickness, 2.0);
}

#[test]
fn test_cortical_thickness_exact_distance_equivalence() {
    let mut white_verts = Vec::with_capacity(50);
    let mut pial_verts = Vec::with_capacity(50);
    for i in 0..50 {
        let (x, y, z) = (
            (i as f64) * 0.2,
            ((i * 7) % 13) as f64 * 0.3,
            ((i * 5) % 9) as f64 * 0.15,
        );
        white_verts.push([x, y, z]);
        pial_verts.push([x, y, z + 2.5]);
    }
    let white_surface = Surface::from_vertices(white_verts);
    let pial_surface = Surface::from_vertices(pial_verts);

    let batch_map = cortical_thickness_map(&white_surface, &pial_surface);
    assert_eq!(batch_map.len(), 50);

    let mut workspace = ThicknessWorkspace::with_max_vertices(50);
    let mut map_into = vec![0.0_f64; 50];
    let mut direct_samples = [0.0_f64; 10];
    let mut indexed_samples = [0.0_f64; 10];

    let (written, metrics) = run_verified_locked(|| {
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
            let (v_w, v_p) = (white_surface.vertices[i], pial_surface.vertices[i]);
            direct_samples[slot] = cortical_thickness(&v_w, &v_p, &white_surface, &pial_surface);
            indexed_samples[slot] = cortical_thickness_indexed(&v_w, &v_p, &white_grid, &pial_grid);
        }
        count
    });

    assert_eq!(written, 50);
    assert!(metrics.function_calls > 0);
    assert_eq!(batch_map, map_into);

    for (slot, i) in (0..50).step_by(5).enumerate() {
        let (v_w, v_p) = (white_surface.vertices[i], pial_surface.vertices[i]);
        let expected = 0.5
            * (naive_surface_distance(&v_w, &pial_surface)
                + naive_surface_distance(&v_p, &white_surface));
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

    let ((empty_t, nan_t, undercap, ok_count), metrics) = run_verified_locked(|| {
        let d_empty = cortical_thickness(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0], &empty, &empty);
        let (wg, pg) = ws.build_grids(&single_w, &single_p).unwrap();
        let d_nan = cortical_thickness_indexed(&[f64::NAN, 0.0, 0.0], &[1.0, 2.0, 6.0], &wg, &pg);
        let bad = cortical_thickness_map_into(&single_w, &single_p, &mut empty_ws, &mut out);
        let good = cortical_thickness_map_into(&single_w, &single_p, &mut ws, &mut out);
        (d_empty, d_nan, bad, good)
    });

    assert_eq!(empty_t, f64::INFINITY);
    assert!(nan_t.is_nan());
    assert_eq!((undercap, ok_count, out[0]), (0, 1, 3.0));
    assert!(metrics.function_calls > 0);
}

#[test]
fn test_zero_alloc_10k_vertex_mesh_thickness_map_into() {
    let n = 10_000_usize;
    let mut white_verts = Vec::with_capacity(n);
    let mut pial_verts = Vec::with_capacity(n);
    for i in 0..n {
        let (su, cu) = ((i % 100) as f64 * 0.0628).sin_cos();
        let (sv, cv) = ((i / 100) as f64 * 0.0314).sin_cos();
        let r_w = 20.0 + 1.5 * ((i % 100) as f64 * 0.1884).cos() * sv;
        let r_p = r_w + 2.5;
        white_verts.push([r_w * sv * cu, r_w * sv * su, r_w * cv]);
        pial_verts.push([r_p * sv * cu, r_p * sv * su, r_p * cv]);
    }
    let white = Surface::from_vertices(white_verts);
    let pial = Surface::from_vertices(pial_verts);
    let mut workspace = ThicknessWorkspace::with_max_vertices(n);
    let mut thicknesses = vec![0.0_f64; n];

    let (written, metrics) = run_verified_locked(|| {
        cortical_thickness_map_into(&white, &pial, &mut workspace, &mut thicknesses)
    });

    assert_eq!(written, n);
    assert!(metrics.function_calls > 0);
    assert!(thicknesses.iter().all(|&t| t > 1.5 && t <= 2.6));
}

#[test]
fn test_cortical_thickness_sparse_expansion() {
    let w = Surface::from_vertices(vec![[0.0, 0.0, 0.0], [50.0, 50.0, 50.0]]);
    let p = Surface::from_vertices(vec![[0.0, 3.0, 0.0], [50.0, 53.0, 50.0]]);
    let (vw, vp) = ([25.0, 25.0, 25.0], [25.0, 28.0, 25.0]);
    let expected = 0.5 * (naive_surface_distance(&vw, &p) + naive_surface_distance(&vp, &w));
    assert_eq!(cortical_thickness(&vw, &vp, &w, &p), expected);
    let (wg, pg) = (SpatialGrid::from_surface(&w), SpatialGrid::from_surface(&p));
    assert_eq!(cortical_thickness_with_grid(&vw, &vp, &wg, &pg), expected);
}

#[test]
fn test_verified_zero_allocation_single_and_indexed_queries() {
    let (w, p) = build_wavy_surfaces(64);
    let (wg, pg) = (SpatialGrid::from_surface(&w), SpatialGrid::from_surface(&p));
    let mut out = vec![-1.0; 64];
    let (vw, vp) = (&w.vertices[7], &p.vertices[7]);

    let ((single_t, indexed_t, grid_d), metrics) = run_verified_locked(|| {
        let single_t = cortical_thickness(vw, vp, &w, &p);
        let indexed_t = cortical_thickness_with_grid(vw, vp, &wg, &pg);
        let grid_d = pg.shortest_distance(vw);
        cortical_thickness_into(&w, &p, &wg, &pg, &mut out);
        (single_t, indexed_t, grid_d)
    });

    assert_eq!(
        (single_t, single_t, grid_d),
        (indexed_t, out[7], naive_surface_distance(vw, &p))
    );
    assert!(metrics.function_calls > 0 && metrics.arithmetic_ops > 0 && metrics.memory_loads > 0);
}

#[test]
fn test_cortical_thickness_mesh_and_grid_random_parity() {
    let mut white_verts = Vec::with_capacity(200);
    let mut pial_verts = Vec::with_capacity(200);
    for i in 0_usize..200 {
        let k = i.saturating_sub(usize::from(i % 3 == 0));
        let x = ((k * 29) % 71) as f64 * 0.12 - 3.0;
        let y = ((k * 43) % 67) as f64 * 0.12 - 2.0;
        let z = ((k * 61) % 59) as f64 * 0.08;
        white_verts.push([x, y, z]);
        pial_verts.push([x - 0.05, y + 0.07, z + 2.1]);
    }
    let white = Surface::from_vertices(white_verts);
    let pial = Surface::from_vertices(pial_verts);
    let white_grid = SpatialGrid::from_surface(&white);
    let pial_grid = SpatialGrid::from_surface(&pial);
    let mesh_out = cortical_thickness_mesh(&white, &pial);
    assert_eq!(mesh_out.len(), 200);
    assert_eq!(cortical_thickness_map(&white, &pial), mesh_out);

    for (i, &mesh_val) in mesh_out.iter().enumerate() {
        let (v_w, v_p) = (white.vertices[i], pial.vertices[i]);
        let expected =
            0.5 * (naive_surface_distance(&v_w, &pial) + naive_surface_distance(&v_p, &white));
        assert_eq!(mesh_val, expected);
        assert_eq!(cortical_thickness(&v_w, &v_p, &white, &pial), expected);
        assert_eq!(
            cortical_thickness_with_grid(&v_w, &v_p, &white_grid, &pial_grid),
            expected
        );
    }
}

#[test]
fn test_cortical_thickness_empty_nan_and_zero_extent_edges() {
    let empty = Surface::from_vertices(vec![]);
    let empty_grid = SpatialGrid::from_surface(&empty);
    let point_surf = Surface::from_vertices(vec![[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]);
    let point_grid = SpatialGrid::from_surface(&point_surf);
    assert!(cortical_thickness_mesh(&empty, &point_surf).is_empty());

    let ((empty_t, nan_single, nan_grid, zero_ext_t), _) = run_verified_locked(|| {
        let q = [2.0, 5.0, 6.0];
        let nan_q = [0.0, f64::NAN, 0.0];
        let empty_t = cortical_thickness(&q, &q, &empty, &empty);
        let nan_single = cortical_thickness(&nan_q, &q, &point_surf, &point_surf);
        let nan_grid = cortical_thickness_with_grid(&q, &nan_q, &point_grid, &point_grid);
        let zero_ext_t = cortical_thickness_with_grid(&q, &q, &point_grid, &point_grid);
        assert_eq!(empty_grid.shortest_distance(&q), f64::INFINITY);
        (empty_t, nan_single, nan_grid, zero_ext_t)
    });

    assert_eq!(empty_t, f64::INFINITY);
    assert!(nan_single.is_nan() && nan_grid.is_nan());
    assert_eq!(zero_ext_t, 5.0);
}

#[test]
fn test_cortical_thickness_into_mismatched_lengths() {
    let (white, _) = build_wavy_surfaces(5);
    let (_, pial) = build_wavy_surfaces(3);
    let white_grid = SpatialGrid::from_surface(&white);
    let pial_grid = SpatialGrid::from_surface(&pial);

    let mut oversized = [99.0; 7];
    let mut undersized = [99.0; 2];
    let mut empty_out: [f64; 0] = [];

    run_verified_locked(|| {
        cortical_thickness_into(&white, &pial, &white_grid, &pial_grid, &mut oversized);
        cortical_thickness_into(&white, &pial, &white_grid, &pial_grid, &mut undersized);
        cortical_thickness_into(&white, &pial, &white_grid, &pial_grid, &mut empty_out);
    });

    for (i, &val) in oversized[..3].iter().enumerate() {
        let expected = cortical_thickness(&white.vertices[i], &pial.vertices[i], &white, &pial);
        assert_eq!(val, expected);
    }
    assert_eq!(&oversized[3..], &[0.0, 0.0, 0.0, 0.0]);
    assert_eq!(&undersized, &oversized[..2]);
}

#[test]
fn test_reusable_spatial_grid_subquadratic_mesh_batch() {
    let count = 2_500;
    let (white, pial) = build_wavy_surfaces(count);
    let white_grid = SpatialGrid::from_surface(&white);
    let pial_grid = SpatialGrid::from_surface(&pial);
    let mut out = vec![0.0; count];

    run_verified_locked(|| {
        cortical_thickness_into(&white, &pial, &white_grid, &pial_grid, &mut out);
    });

    for idx in (0..count).step_by(250) {
        let expected = cortical_thickness(&white.vertices[idx], &pial.vertices[idx], &white, &pial);
        assert_eq!(out[idx], expected);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_verified_allocator_aborts_if_grid_built_inside_run_verified() {
    const PROBE_ENV: &str = "OXIDIZE_FREESURFER_ALLOC_ABORT_PROBE";
    if std::env::var_os(PROBE_ENV).is_some() {
        let (white, _) = build_wavy_surfaces(16);
        VerifiedEngine::run_verified(|| {
            let grid = SpatialGrid::from_surface(&white);
            std::hint::black_box(grid.shortest_distance(&[0.0, 0.0, 0.0]));
        });
        return;
    }
    let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .env(PROBE_ENV, "1")
        .args([
            "--exact",
            "test_verified_allocator_aborts_if_grid_built_inside_run_verified",
        ])
        .output()
        .expect("spawn freesurfer allocation abort probe");
    assert!(
        !output.status.success(),
        "VerifiedAllocator must abort when SpatialGrid::from_surface allocates inside run_verified"
    );
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

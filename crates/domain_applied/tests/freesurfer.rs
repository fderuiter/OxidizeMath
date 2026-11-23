//! Integration tests for freesurfer.
use domain_applied::applied::freesurfer::*;
use nalgebra::{DMatrix, DVector};
use std::sync::Mutex;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

static VERIFIED_RUN_MUTEX: Mutex<()> = Mutex::new(());

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
            let dx = pt[0] - v[0];
            let dy = pt[1] - v[1];
            let dz = pt[2] - v[2];
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
        Surface {
            vertices: white_verts,
        },
        Surface {
            vertices: pial_verts,
        },
    )
}

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
    let white_grid = SpatialGrid::from_surface(&white_surface);
    let pial_grid = SpatialGrid::from_surface(&pial_surface);
    assert_eq!(
        cortical_thickness_with_grid(&v_w, &v_p, &white_grid, &pial_grid),
        expected
    );
}

#[test]
fn test_verified_zero_allocation_single_and_indexed_queries() {
    let (white_surface, pial_surface) = build_wavy_surfaces(64);
    let white_grid = SpatialGrid::from_surface(&white_surface);
    let pial_grid = SpatialGrid::from_surface(&pial_surface);
    let mut out = vec![-1.0; 64];

    let _guard = VERIFIED_RUN_MUTEX
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let ((single_t, indexed_t, grid_d), metrics) = VerifiedEngine::run_verified(|| {
        let single_t = cortical_thickness(
            &white_surface.vertices[7],
            &pial_surface.vertices[7],
            &white_surface,
            &pial_surface,
        );
        let indexed_t = cortical_thickness_with_grid(
            &white_surface.vertices[7],
            &pial_surface.vertices[7],
            &white_grid,
            &pial_grid,
        );
        let grid_d = pial_grid.shortest_distance(&white_surface.vertices[7]);
        cortical_thickness_into(
            &white_surface,
            &pial_surface,
            &white_grid,
            &pial_grid,
            &mut out,
        );
        (single_t, indexed_t, grid_d)
    });
    drop(_guard);

    assert_eq!(single_t, indexed_t);
    assert_eq!(single_t, out[7]);
    assert_eq!(
        grid_d,
        naive_surface_distance(&white_surface.vertices[7], &pial_surface)
    );
    assert!(metrics.function_calls > 0);
    assert!(metrics.arithmetic_ops > 0);
    assert!(metrics.memory_loads > 0);
}

#[test]
fn test_cortical_thickness_mesh_and_grid_random_parity() {
    let mut white_verts = Vec::with_capacity(200);
    let mut pial_verts = Vec::with_capacity(200);
    for i in 0_usize..200 {
        // Include repeated coordinates to exercise multi-vertex CSR buckets
        let k = i.saturating_sub(usize::from(i % 3 == 0));
        let x = ((k * 29) % 71) as f64 * 0.12 - 3.0;
        let y = ((k * 43) % 67) as f64 * 0.12 - 2.0;
        let z = ((k * 61) % 59) as f64 * 0.08;
        white_verts.push([x, y, z]);
        pial_verts.push([x - 0.05, y + 0.07, z + 2.1]);
    }
    let white = Surface {
        vertices: white_verts,
    };
    let pial = Surface {
        vertices: pial_verts,
    };
    let white_grid = SpatialGrid::from_surface(&white);
    let pial_grid = SpatialGrid::from_surface(&pial);
    let mesh_out = cortical_thickness_mesh(&white, &pial);
    assert_eq!(mesh_out.len(), 200);

    for (i, &mesh_val) in mesh_out.iter().enumerate() {
        let v_w = white.vertices[i];
        let v_p = pial.vertices[i];
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
    let empty = Surface { vertices: vec![] };
    let empty_grid = SpatialGrid::from_surface(&empty);
    let point_surf = Surface {
        vertices: vec![[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
    };
    let point_grid = SpatialGrid::from_surface(&point_surf);
    assert!(cortical_thickness_mesh(&empty, &point_surf).is_empty());

    let _guard = VERIFIED_RUN_MUTEX
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let ((empty_t, nan_single, nan_grid, zero_ext_t), _) = VerifiedEngine::run_verified(|| {
        let q = [2.0, 5.0, 6.0];
        let nan_q = [0.0, f64::NAN, 0.0];
        let empty_t = cortical_thickness(&q, &q, &empty, &empty);
        let nan_single = cortical_thickness(&nan_q, &q, &point_surf, &point_surf);
        let nan_grid = cortical_thickness_with_grid(&q, &nan_q, &point_grid, &point_grid);
        let zero_ext_t = cortical_thickness_with_grid(&q, &q, &point_grid, &point_grid);
        assert_eq!(empty_grid.shortest_distance(&q), f64::INFINITY);
        (empty_t, nan_single, nan_grid, zero_ext_t)
    });
    drop(_guard);

    assert_eq!(empty_t, f64::INFINITY);
    assert!(nan_single.is_nan());
    assert!(nan_grid.is_nan());
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

    let _guard = VERIFIED_RUN_MUTEX
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    VerifiedEngine::run_verified(|| {
        cortical_thickness_into(&white, &pial, &white_grid, &pial_grid, &mut oversized);
        cortical_thickness_into(&white, &pial, &white_grid, &pial_grid, &mut undersized);
        cortical_thickness_into(&white, &pial, &white_grid, &pial_grid, &mut empty_out);
    });
    drop(_guard);

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

    let _guard = VERIFIED_RUN_MUTEX
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    VerifiedEngine::run_verified(|| {
        cortical_thickness_into(&white, &pial, &white_grid, &pial_grid, &mut out);
    });
    drop(_guard);

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
    // The expected value was re-calculated to be 2.1602...
    assert!((t - 2.1602).abs() < 1e-4);
}

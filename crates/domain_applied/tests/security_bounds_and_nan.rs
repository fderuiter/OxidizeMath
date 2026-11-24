//! Tier 5 security tests for `domain_applied`: capped isosurface bounds,
//! clinical-trial `BlockRandomizer` overflow/zero/odd guards, `NaN`/infinity stability,
//! and zero-allocation workspace extraction.

use domain_applied::applied::clinical_trials::design::{AllocationStrategy, BlockRandomizer};
use domain_applied::applied::isosurface::capped::{
    CutPlaneCap, extract_capped_isosurface, extract_capped_isosurface_into,
};
use domain_applied::applied::isosurface::workspace::IsosurfaceWorkspace;
use domain_applied::applied::isosurface::{Mesh, Point3D, VoxelGrid};
use oxidize_core::rng::OxidizeRng;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static TEST_ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn block_randomizer_rejects_zero_odd_and_capacity_overflow() {
    assert!(BlockRandomizer::new(0).is_err());
    assert!(BlockRandomizer::new(1).is_err());
    assert!(BlockRandomizer::new(3).is_err());

    let randomizer = BlockRandomizer::new(4).expect("even block size");
    let mut rng = OxidizeRng::from_seed(7);
    assert!(randomizer.assign(&mut rng, usize::MAX).is_err());

    let assigned = randomizer.assign(&mut rng, 6).expect("valid assignment");
    assert_eq!(assigned.len(), 6);
}

#[test]
fn capped_isosurface_handles_degenerate_grids_non_finite_samples_subnormals_and_zero_alloc() {
    // 1x2x2 is below the 2x2x2 minimum for central-difference capped extraction.
    let tiny = VoxelGrid::builder()
        .dimensions(1, 2, 2)
        .data(vec![0.0_f32; 4])
        .voxel_size(Point3D::new(1.0, 1.0, 1.0))
        .origin(Point3D::new(0.0, 0.0, 0.0))
        .build()
        .expect("1x2x2 VoxelGrid");
    assert!(extract_capped_isosurface(&tiny, 0.0, true, None).is_err());

    let sub = f32::MIN_POSITIVE * 0.5;
    let samples = vec![
        -1.0,
        sub,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -sub,
        0.5,
        -0.5,
    ];
    let grid = VoxelGrid::builder()
        .dimensions(2, 2, 2)
        .data(samples)
        .voxel_size(Point3D::new(1.0, 1.0, 1.0))
        .origin(Point3D::new(0.0, 0.0, 0.0))
        .build()
        .expect("2x2x2 VoxelGrid");

    let mesh = extract_capped_isosurface(
        &grid,
        0.0,
        true,
        Some(CutPlaneCap {
            axis: 0,
            max_coord: f32::NAN,
        }),
    )
    .expect("NaN cut plane safely clips without panic");
    assert!(mesh.vertices.is_empty());

    // Pre-warm padded grid, workspace, and mesh, then verify 0-byte hot path under run_verified.
    let mut padded = VoxelGrid::builder()
        .dimensions(4, 4, 4)
        .data(vec![1.0_f32; 64])
        .voxel_size(Point3D::new(1.0, 1.0, 1.0))
        .origin(Point3D::new(-1.0, -1.0, -1.0))
        .build()
        .expect("4x4x4 padded VoxelGrid");
    let mut ws = IsosurfaceWorkspace::new_lazy(4, 4);
    let mut out_mesh = Mesh::new();
    extract_capped_isosurface_into(&grid, 0.0, true, None, &mut padded, &mut ws, &mut out_mesh)
        .expect("warm-up extraction");

    let (res, metrics) = VerifiedEngine::run_verified(|| {
        extract_capped_isosurface_into(&grid, 0.0, true, None, &mut padded, &mut ws, &mut out_mesh)
    });
    res.expect("zero-alloc capped extraction");
    assert_eq!(metrics.allocations, 0);
    assert_eq!(metrics.allocated_bytes, 0);
    assert!(out_mesh.vertices.iter().all(|v| v.is_finite()));
    assert!(out_mesh.normals.iter().all(|n| n.is_finite()));
}

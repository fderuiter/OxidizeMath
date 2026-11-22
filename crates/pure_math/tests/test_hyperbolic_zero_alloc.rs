//! Zero-allocation verification tests for the SU(1,1) hyperbolic geometry engine.
#![cfg_attr(any(), verified(opt_out = "tests"))]

use num_complex::Complex64;
use pure_math::pure_math::differential_geometry::hyperbolic::{
    GeodesicArc, HyperbolicBilliardConfig, HyperbolicBilliardSystem, HyperbolicModel,
    HyperbolicTilingBuffer, POINCARE_DISK_RADIUS_MAX, gauss_bonnet_triangle, poincare_to_klein,
};
use verified_engine::allocator::{VerifiedAllocator, lock_allocations, unlock_allocations};
use verified_engine::metrics::{disable_verification, enable_verification};

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_zero_heap_allocations_across_1000_mobius_pan_and_billiard_steps() {
    let mut tiling = HyperbolicTilingBuffer::with_capacity(256);
    let tile_count = tiling
        .configure(7, 3, 3)
        .expect("pre-allocated {7, 3} tiling");
    assert!(tile_count > 20);

    let mut billiards = HyperbolicBilliardSystem::with_history_capacity(
        HyperbolicBilliardConfig {
            p: 7,
            q: 3,
            initial_pos: Complex64::new(0.06, 0.03),
            initial_angle: 0.62,
            epsilon: 1e-6,
        },
        512,
    )
    .expect("pre-allocated {7, 3} billiard system");

    let mut arc_scratch = [Complex64::new(0.0, 0.0); 17];

    enable_verification();
    lock_allocations();

    for step_idx in 0..1_000_usize {
        let phase = (step_idx as f64) * 0.07;
        let pan_delta = Complex64::new(0.045 * phase.cos(), 0.045 * phase.sin());
        let _ = tiling.apply_pan(pan_delta);

        let central = &tiling.polygons()[0];
        let arc = GeodesicArc::through_points(central.vertices[0], central.vertices[1]);
        arc.sample_into(HyperbolicModel::Poincare, &mut arc_scratch);
        let _ = poincare_to_klein(arc_scratch[8]);
        let _ = gauss_bonnet_triangle(central.center, central.vertices[0], central.vertices[1]);

        billiards.step(0.035);
    }

    unlock_allocations();
    disable_verification();

    assert_eq!(tiling.polygons().len(), tile_count);
    assert!((tiling.view_matrix().determinant() - 1.0).abs() < 1e-11);
    for poly in tiling.polygons() {
        for &v in poly.active_vertices() {
            assert!(v.norm() <= POINCARE_DISK_RADIUS_MAX);
        }
    }
    assert!(billiards.primary_bounces() >= 10);
    assert!(billiards.primary().pos.norm() <= POINCARE_DISK_RADIUS_MAX);
    assert!(billiards.twin().pos.norm() <= POINCARE_DISK_RADIUS_MAX);
    assert!(billiards.latest_sample().separation.is_finite());
}

// [cite:modular_polynomials_review]

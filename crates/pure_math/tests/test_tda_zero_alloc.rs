//! Integration tests verifying zero-allocation hot paths for 3D TDA filtration and Z₂ homology.
#![cfg_attr(any(), verified(opt_out = "tests"))]

use pure_math::pure_math::statistics::tda::{
    PersistenceInterval, Point3D, TdaFiltrationWorkspace, farthest_point_landmarks_into,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_tda_3d_filtration_and_landmarking_zero_allocation_hot_path() {
    let raw_points: Vec<Point3D> = (0..128)
        .map(|i| {
            let t = i as f64 * 0.12;
            Point3D::new(8.0 * t.cos(), 8.0 * t.sin(), (0.5 * t).sin() * 4.0)
        })
        .collect();
    let radii: [f64; 24] = std::array::from_fn(|i| i as f64 * 0.75);

    let mut min_dist_sq = vec![f64::INFINITY; raw_points.len()];
    let mut landmarks = vec![Point3D::new(0.0, 0.0, 0.0); 64];
    let mut workspace = TdaFiltrationWorkspace::with_capacity(64);
    let mut intervals: Vec<PersistenceInterval> = Vec::with_capacity(128);

    let ((b0, b1, interval_count), _metrics) = VerifiedEngine::run_verified(|| {
        let count =
            farthest_point_landmarks_into(&raw_points, 64, &mut min_dist_sq, &mut landmarks)
                .expect("landmark subsampling");
        workspace
            .load_points_3d(&landmarks[..count])
            .expect("load 3d distances");
        let step = workspace
            .evaluate_radius(count, 4.5)
            .expect("single radius evaluation");
        workspace
            .compute_persistence_into(count, &radii, &mut intervals)
            .expect("24-step persistence sweep");
        (step.beta0, step.beta1, intervals.len())
    });

    assert_eq!(b0, 1);
    assert!(b1 >= 1);
    assert!(interval_count >= 64);
}

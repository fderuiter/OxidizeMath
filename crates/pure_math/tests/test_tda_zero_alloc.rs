//! Integration tests verifying zero-allocation hot paths for 3D TDA filtration and Z₂ homology.
#![cfg_attr(any(), verified(opt_out = "tests"))]

use pure_math::pure_math::statistics::tda::{
    PersistenceInterval, Point2D, Point3D, PointCloud, TdaFiltrationWorkspace, betti_numbers,
    farthest_point_landmarks_into, vietoris_rips_from_distances,
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

#[test]
fn test_tda_n100_50_step_monotonic_backwards_and_unsorted_zero_alloc_parity() {
    let points: Vec<Point2D> = (0..100)
        .map(|i| {
            let ring_r = if i < 60 { 5.0 } else { 2.2 };
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / 50.0;
            Point2D::new(ring_r * angle.cos(), ring_r * angle.sin())
        })
        .collect();
    let cloud = PointCloud::new(points.clone()).unwrap();
    let dist = cloud.distance_matrix();
    let radii: [f64; 50] = std::array::from_fn(|k| k as f64 * 0.14);
    let mut unsorted_radii = radii;
    unsorted_radii.reverse();

    let mut workspace = TdaFiltrationWorkspace::with_capacity(100);
    let mut sorted_intervals: Vec<PersistenceInterval> = Vec::with_capacity(256);
    let mut unsorted_intervals: Vec<PersistenceInterval> = Vec::with_capacity(256);
    let mut step_bettis = [(0_usize, 0_usize); 8];
    let probe_radii = [0.0, 0.6, 1.4, 2.5, 3.8, 1.4, 0.6, 4.5];

    let ((), _metrics) = VerifiedEngine::run_verified(|| {
        workspace
            .load_points_2d(&points)
            .expect("load 100 2D points without allocation");
        for &r in &radii {
            let _ = workspace.evaluate_radius(100, r).expect("monotonic step");
        }
        for (slot, &r) in step_bettis.iter_mut().zip(probe_radii.iter()) {
            let step = workspace
                .evaluate_radius(100, r)
                .expect("forward and backward radius evaluation");
            *slot = (step.beta0, step.beta1);
        }
        workspace
            .compute_persistence_into(100, &radii, &mut sorted_intervals)
            .expect("50-step sorted persistence");
        workspace
            .compute_persistence_into(100, &unsorted_radii, &mut unsorted_intervals)
            .expect("50-step unsorted persistence");
    });

    assert_eq!(sorted_intervals, unsorted_intervals);
    for (&actual, &r) in step_bettis.iter().zip(probe_radii.iter()) {
        let reference = vietoris_rips_from_distances(&dist, r).unwrap();
        assert_eq!(
            actual,
            betti_numbers(&reference).unwrap(),
            "mismatch at r={r}"
        );
    }
}

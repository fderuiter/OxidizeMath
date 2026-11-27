//! Integration test verifying zero heap allocations across `math_commons`
//! critical-point (POI) solvers under `VerifiedAllocator::lock_allocations()`.

use math_commons::poi::{
    LocalExtremum, closest_threshold_crossing, detect_local_extrema_into, intersect_polylines_into,
    parabolic_vertex, peak_extremum, threshold_crossings_into, y_at_x_into,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_poi_solvers_zero_heap_allocations() {
    let mut xs = [0.0_f64; 64];
    let mut ys_sin = [0.0_f64; 64];
    let mut poly_pts = [[0.0_f64; 2]; 32];
    let mut logistic_pts = [[0.0_f64; 2]; 30];

    for i in 0..64 {
        let x = 0.1 + (i as f64) * 0.12;
        xs[i] = x;
        ys_sin[i] = 2.4 * (1.3 * x + 0.25).sin() + 0.5;
    }
    for (i, slot) in poly_pts.iter_mut().enumerate() {
        let x = 0.2 + (i as f64) * 0.2;
        *slot = [
            x,
            1.0 - 0.5 * x + 0.3 * x * x - 0.04 * x.powi(3) + 0.005 * x.powi(4),
        ];
    }
    for (i, slot) in logistic_pts.iter_mut().enumerate() {
        let x = (i as f64) * 0.4;
        *slot = [x, 10.0 / (1.0 + (-0.9 * (x - 5.0)).exp())];
    }

    VerifiedEngine::run_verified(|| {
        let paired = (&xs[..], &ys_sin[..]);
        let mut x_hits = [0.0_f64; 16];
        let n_cross = threshold_crossings_into(&paired, 0.5, &mut x_hits);
        assert!(n_cross >= 2);
        assert!(closest_threshold_crossing(&paired, 0.5, 2.5).is_some());

        let mut y_hits = [0.0_f64; 4];
        assert_eq!(y_at_x_into(&paired, 2.0, &mut y_hits), 1);

        let mut extrema = [LocalExtremum {
            point: [0.0, 0.0],
            is_max: true,
        }; 16];
        let n_ext = detect_local_extrema_into(&paired, &mut extrema);
        assert!(n_ext >= 2);
        assert!(peak_extremum(&paired).is_some());

        let v = parabolic_vertex([0.0, 0.0], [1.0, 2.0], [2.0, 0.0]);
        assert!((v[0] - 1.0).abs() < 1e-12);

        let mut intersections = [[0.0_f64; 2]; 16];
        let n_int =
            intersect_polylines_into(&poly_pts, &logistic_pts, [0.0, 10.0], &mut intersections);
        assert!(n_int >= 1);
    });
}

// [cite:math_commons]
use super::*;
use alloc::{vec, vec::Vec};

#[test]
#[verified_engine::verified]
fn test_hermite_threshold_crossing_sub_segment_accuracy() {
    let xs: Vec<f64> = (0..=20).map(|i| i as f64 * 0.25).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| x.sin()).collect();
    let paired = (xs.as_slice(), ys.as_slice());

    let mut crossings = [0.0_f64; 8];
    let n = threshold_crossings_into(&paired, 0.5, &mut crossings);
    assert!(n >= 2);
    let exact_first = core::f64::consts::FRAC_PI_6;
    assert!((crossings[0] - exact_first).abs() < 5e-4);

    let closest = closest_threshold_crossing(&paired, 0.0, 3.0).expect("pi crossing");
    assert!((closest - core::f64::consts::PI).abs() < 5e-4);
}

#[test]
#[verified_engine::verified]
fn test_parabolic_vertex_and_peak_extremum_accuracy() {
    let pts: Vec<(f64, f64)> = (0..=15)
        .map(|i| {
            let x = i as f64 * 0.23;
            (x, -(x - 1.37).powi(2) + 4.25)
        })
        .collect();

    let extrema = detect_local_extrema(&pts);
    assert_eq!(extrema.len(), 1);
    assert!(extrema[0].is_max);
    assert!((extrema[0].point[0] - 1.37).abs() < 1e-9);
    assert!((extrema[0].point[1] - 4.25).abs() < 1e-9);

    let peak = peak_extremum(&pts).expect("peak");
    assert!((peak[0] - 1.37).abs() < 1e-9);
    assert!((peak[1] - 4.25).abs() < 1e-9);
}

#[test]
#[verified_engine::verified]
fn test_monotone_and_parametric_polyline_intersections() {
    let line_a: Vec<[f64; 2]> = (0..=20)
        .map(|i| {
            let x = i as f64 * 0.2;
            [x, 0.5 * x + 0.5]
        })
        .collect();
    let line_b: Vec<[f64; 2]> = (0..=20)
        .map(|i| {
            let x = i as f64 * 0.2;
            [x, 2.5 - 0.5 * x]
        })
        .collect();
    let hits = intersect_polylines(&line_a, &line_b, [0.0, 4.0]);
    assert_eq!(hits.len(), 1);
    assert!((hits[0][0] - 2.0).abs() < 1e-9);
    assert!((hits[0][1] - 1.5).abs() < 1e-9);

    let circle: Vec<[f64; 2]> = (0..=64)
        .map(|i| {
            let theta = (i as f64 / 64.0) * core::f64::consts::TAU;
            [theta.cos(), theta.sin()]
        })
        .collect();
    let diag = [[-1.5, -1.5], [1.5, 1.5]];
    let circle_hits = intersect_polylines(&circle, &diag, [-2.0, 2.0]);
    assert_eq!(circle_hits.len(), 2);
}

#[test]
#[verified_engine::verified]
fn test_vertical_line_crossings_with_tuple_series() {
    let pts = [(0.0, 1.0), (2.0, 5.0), (4.0, 1.0)];
    let ys = interpolate_y_at_x(&pts, 1.0);
    assert_eq!(ys.len(), 1);
    assert!((ys[0] - 3.0).abs() < 1e-12);
}

#[test]
#[verified_engine::verified]
fn test_plateau_extrema_and_fused_line_scan() {
    let pts = [
        [-1.0, -1.0],
        [1.0, 4.0],
        [2.0, 4.0],
        [3.0, 4.0],
        [4.0, 1.0],
        [5.0, -2.0],
        [6.0, -2.0],
        [7.0, 1.0],
    ];
    let scan = scan_line_pois(&pts, true, true);
    assert_eq!(scan.roots.len(), 3);
    assert_eq!(scan.y_intercepts.len(), 1);
    assert_eq!(scan.extrema.len(), 2);
    assert!(scan.extrema[0].is_max);
    assert!((scan.extrema[0].point[0] - 2.0).abs() < 1e-12);
    assert!((scan.extrema[0].point[1] - 4.0).abs() < 1e-12);
    assert!(!scan.extrema[1].is_max);
    assert!((scan.extrema[1].point[0] - 5.5).abs() < 1e-12);
    assert!((scan.extrema[1].point[1] + 2.0).abs() < 1e-12);
}

#[test]
#[verified_engine::verified]
fn test_plateau_crossing_y_intercept_and_nan_break() {
    let mut cross_pts = vec![[-30.0, 1.0]];
    for x in -29..=29 {
        if x != 0 {
            cross_pts.push([x as f64, 4.0]);
        }
    }
    cross_pts.push([30.0, 1.0]);
    let scan = scan_line_pois(&cross_pts, true, true);
    assert_eq!(scan.y_intercepts.len(), 1);
    assert!((scan.y_intercepts[0] - 4.0).abs() < 1e-9);
    assert_eq!(scan.extrema.len(), 1);
    assert!(scan.extrema[0].is_max);
    assert!(scan.extrema[0].point[0].abs() < 1e-9);

    let mut nan_plateau: Vec<[f64; 2]> = (-25..=-1).map(|x| [x as f64, 3.0]).collect();
    nan_plateau.insert(0, [-26.0, 0.0]);
    nan_plateau[12] = [-14.0, f64::NAN];
    nan_plateau.push([0.0, 3.0]);
    let nan_scan = scan_line_pois(&nan_plateau, true, true);
    assert!(nan_scan.extrema.is_empty());
    assert_eq!(nan_scan.y_intercepts.len(), 1);
    assert!((nan_scan.y_intercepts[0] - 3.0).abs() < 1e-9);
}

#[test]
#[verified_engine::verified]
fn test_advance_plateau_run_across_series_impls_and_nan_x() {
    use crate::series::PointSeries;

    #[derive(Clone, Copy)]
    struct CustomSample(f64, f64);
    impl crate::series::Sample2D for CustomSample {
        fn sample_x(&self) -> f64 {
            self.0
        }
        fn sample_y(&self) -> f64 {
            self.1
        }
    }

    let arr = [[0.0, 2.0], [1.0, 2.0], [f64::NAN, 2.0], [3.0, 2.0]];
    assert_eq!(arr.advance_plateau_run(0, 2.0, -10.0, 10.0), 2);
    assert_eq!(arr.as_slice().advance_plateau_run(0, 2.0, -10.0, 10.0), 2);
    assert_eq!(arr.as_slice().advance_plateau_run(99, 2.0, -10.0, 10.0), 4);

    let tuples = vec![(0.0, 5.0), (1.0, 5.0), (2.0, 5.0), (3.0, 1.0)];
    assert_eq!(tuples.advance_plateau_run(0, 5.0, 0.0, 1.5), 2);

    let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let ys = vec![0.0, 7.0, 7.0, 7.0, 0.0];
    let paired = (&xs, &ys);
    assert_eq!(paired.advance_plateau_run(1, 7.0, -10.0, 10.0), 4);
    let mut out = [LocalExtremum {
        point: [0.0; 2],
        is_max: false,
    }; 4];
    assert_eq!(detect_local_extrema_into(&paired, &mut out), 1);
    assert!((out[0].point[0] - 2.0).abs() < 1e-12);

    let custom = [
        CustomSample(0.0, 1.0),
        CustomSample(1.0, 4.0),
        CustomSample(2.0, 4.0),
        CustomSample(3.0, 1.0),
    ];
    assert_eq!(custom.advance_plateau_run(1, 4.0, -10.0, 10.0), 3);
}

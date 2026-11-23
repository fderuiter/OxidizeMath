// [cite:math_commons]
use super::*;

#[test]
#[verified_engine::verified]
fn test_hermite_threshold_crossing_sub_segment_accuracy() {
    let xs: Vec<f64> = (0..=20).map(|i| i as f64 * 0.25).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| x.sin()).collect();
    let paired = (xs.as_slice(), ys.as_slice());

    let mut crossings = [0.0_f64; 8];
    let n = threshold_crossings_into(&paired, 0.5, &mut crossings);
    assert!(n >= 2);
    let exact_first = std::f64::consts::FRAC_PI_6;
    assert!((crossings[0] - exact_first).abs() < 5e-4);

    let closest = closest_threshold_crossing(&paired, 0.0, 3.0).expect("pi crossing");
    assert!((closest - std::f64::consts::PI).abs() < 5e-4);
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
            let theta = (i as f64 / 64.0) * std::f64::consts::TAU;
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

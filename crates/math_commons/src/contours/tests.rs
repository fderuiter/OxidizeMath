#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::registry::MAX_GRID_SIZE;
use std::f64::consts::PI;

#[test]
fn test_closed_circular_contour_extraction() {
    let n = 41;
    let bounds = GridBounds2D::new(-2.0, 2.0, -2.0, 2.0).unwrap();
    let mut grid = Vec::with_capacity(n * n);
    for iy in 0..n {
        let y = -2.0 + 4.0 * (iy as f64) / ((n - 1) as f64);
        for ix in 0..n {
            let x = -2.0 + 4.0 * (ix as f64) / ((n - 1) as f64);
            grid.push(x * x + y * y);
        }
    }

    let paths = extract_contours(&grid, n, n, &bounds, 1.0).unwrap();
    assert_eq!(
        paths.len(),
        1,
        "r^2 = 1 inside [-2, 2]^2 must form one loop"
    );
    assert!(paths[0].closed, "circle contour must be closed");
    assert!(paths[0].len() >= 16);

    for pt in &paths[0].points {
        let r = pt[0].hypot(pt[1]);
        assert!(
            (r - 1.0).abs() < 0.03,
            "interpolated point {pt:?} deviated from unit circle (r={r})"
        );
    }
    let circumference = paths[0].arc_length();
    assert!(
        (circumference - 2.0 * PI).abs() < 0.1,
        "expected circumference ~2*pi, got {circumference}"
    );
}

#[test]
fn test_saddle_disambiguation_cases_5_and_10() {
    let bounds = GridBounds2D::new(0.0, 1.0, 0.0, 1.0).unwrap();

    // Case 5 with positive center average (bridge across v00 and v11): 0.25*(4 - 0.5 + 4 - 0.5) = 1.75 >= 0
    let connected_case5 = [4.0, -0.5, -0.5, 4.0]; // row0: [v00=4, v10=-0.5], row1: [v01=-0.5, v11=4]
    let paths_pos = extract_contours(&connected_case5, 2, 2, &bounds, 0.0).unwrap();
    assert_eq!(paths_pos.len(), 2);

    // Case 5 with negative center average (separated positive corners): 0.25*(0.5 - 4 + 0.5 - 4) = -1.75 < 0
    let separated_case5 = [0.5, -4.0, -4.0, 0.5];
    let paths_neg = extract_contours(&separated_case5, 2, 2, &bounds, 0.0).unwrap();
    assert_eq!(paths_neg.len(), 2);
    assert_ne!(
        paths_pos[0].points, paths_neg[0].points,
        "saddle center sign must switch topological pairing"
    );

    // Case 10 with positive vs negative center
    let case10_pos = [-0.5, 4.0, 4.0, -0.5];
    let case10_neg = [-4.0, 0.5, 0.5, -4.0];
    let p10_pos = extract_contours(&case10_pos, 2, 2, &bounds, 0.0).unwrap();
    let p10_neg = extract_contours(&case10_neg, 2, 2, &bounds, 0.0).unwrap();
    assert_eq!(p10_pos.len(), 2);
    assert_eq!(p10_neg.len(), 2);
    assert_ne!(p10_pos[0].points, p10_neg[0].points);
}

#[test]
fn test_deduplicate_segments_and_nearest_neighbor_cuts_pen_travel() {
    let segs = [
        [[0.0, 0.0], [1.0, 0.0]],
        [[1.0, 0.0], [0.0, 0.0]], // reversed duplicate
        [[0.0, 0.0], [1.0, 0.0]], // exact duplicate
        [[1.0, 0.0], [1.0, 1.0]],
        [[2.0, 2.0], [2.0, 2.0]], // degenerate zero-length
    ];
    let deduped = deduplicate_segments(&segs, 1e-6);
    assert_eq!(deduped.len(), 2);

    // Multi-contour scrambled paths + open paths oriented backwards relative to travel
    let scrambled = vec![
        ContourPath::new(vec![[90.0, 90.0], [95.0, 90.0]], false),
        ContourPath::new(vec![[8.0, 2.0], [2.0, 2.0]], false), // backwards from origin
        ContourPath::new(vec![[85.0, 90.0], [80.0, 90.0]], false),
        ContourPath::new(vec![[12.0, 2.0], [18.0, 2.0]], false),
        ContourPath::new(
            vec![[50.0, 50.0], [52.0, 50.0], [52.0, 52.0], [50.0, 52.0]],
            true,
        ),
        ContourPath::new(vec![[22.0, 2.0], [19.0, 2.0]], false), // backwards from [18, 2]
    ];

    let raw_travel = pen_lift_travel_distance(&scrambled);
    let ordered = order_paths_nearest_neighbor(&scrambled);
    let optimized_travel = pen_lift_travel_distance(&ordered);
    assert!(
        optimized_travel < raw_travel * 0.55,
        "expected nearest-neighbor + reversal to cut travel substantially: {optimized_travel} vs {raw_travel}"
    );
    let jumps = pen_lift_segments(&ordered);
    let jump_sum: f64 = jumps.iter().map(|s| euclidean_distance(s[0], s[1])).sum();
    assert!((jump_sum - optimized_travel).abs() < 1e-9);
}

#[test]
fn test_nearest_neighbor_cuts_travel_on_multi_contour_scalar_field() {
    let n = 64;
    let bounds = GridBounds2D::new(-10.0, 10.0, -10.0, 10.0).unwrap();
    let centers = [
        (-7.0, -7.0),
        (7.0, 7.0),
        (-7.0, 7.0),
        (7.0, -7.0),
        (-3.0, -6.5),
        (6.5, 3.0),
    ];
    let mut grid = vec![0.0; n * n];
    for iy in 0..n {
        let y = -10.0 + 20.0 * (iy as f64) / ((n - 1) as f64);
        for ix in 0..n {
            let x = -10.0 + 20.0 * (ix as f64) / ((n - 1) as f64);
            let mut val = 0.0;
            for &(cx, cy) in &centers {
                let r2 = (x - cx).powi(2) + (y - cy).powi(2);
                val += (-r2 * 0.8).exp();
            }
            grid[crate::math_kernel::types::flatten_2d_index(ix, iy, n)] = val;
        }
    }
    let contours = extract_contours(&grid, n, n, &bounds, 0.45).unwrap();
    assert!(contours.len() >= 6);
    let row_major_travel = pen_lift_travel_distance(&contours);
    let ordered = order_paths_nearest_neighbor(&contours);
    let nn_travel = pen_lift_travel_distance(&ordered);
    assert!(
        nn_travel < row_major_travel,
        "nearest-neighbor travel ({nn_travel}) must beat row-major travel ({row_major_travel})"
    );
}

#[test]
fn test_rk4_streamlines_around_cylinder_and_grid_guardrails() {
    let bounds = GridBounds2D::new(-8.0, 8.0, -6.0, 6.0).unwrap();
    let cfg = StreamlineConfig::new(bounds, 0.1, 400, 1e-4).unwrap();
    let u_inf = 10.0;
    let r_cyl = 2.0;

    // Potential flow past a cylinder of radius R = 2 at origin:
    // u = U * (1 - R^2 (x^2 - y^2) / r^4), v = -U * R^2 * 2 x y / r^4
    let seeds = [[-7.5, 0.25], [-7.5, -0.25], [-7.5, 1.5], [-7.5, -1.5]];
    let lines = trace_streamlines_rk4(&seeds, &cfg, |x, y| {
        let r2 = x * x + y * y;
        if r2 < r_cyl * r_cyl * 0.98 {
            return [0.0, 0.0];
        }
        let r4 = r2 * r2;
        let r_sq = r_cyl * r_cyl;
        let u = u_inf * (1.0 - r_sq * (x * x - y * y) / r4);
        let v = -u_inf * r_sq * (2.0 * x * y) / r4;
        [u, v]
    })
    .unwrap();

    assert_eq!(lines.len(), 4);
    for (seed, line) in seeds.iter().zip(&lines) {
        let psi_seed =
            u_inf * seed[1] * (1.0 - (r_cyl * r_cyl) / (seed[0] * seed[0] + seed[1] * seed[1]));
        for pt in &line.points {
            let r = pt[0].hypot(pt[1]);
            assert!(
                r >= r_cyl * 0.99,
                "streamline penetrated cylinder obstacle at {pt:?} (r={r})"
            );
            let psi_pt = u_inf * pt[1] * (1.0 - (r_cyl * r_cyl) / (r * r));
            assert!(
                (psi_pt - psi_seed).abs() < 0.05,
                "RK4 streamline drifted off stream function psi: {psi_pt} vs {psi_seed}"
            );
        }
    }

    // Verify MAX_GRID_SIZE guardrail
    let oversized = MAX_GRID_SIZE + 1;
    assert!(extract_contours(&[], oversized, 2, &bounds, 0.5).is_err());
}

// [cite:math_commons]

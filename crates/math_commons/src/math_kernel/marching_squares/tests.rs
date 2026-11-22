use super::*;
use crate::math_kernel::types::flatten_2d_index;

#[test]
#[verified_engine::verified]
fn test_paraboloid_extracts_closed_concentric_loops() {
    let (width, height) = (25_usize, 25_usize);
    let mut values = vec![0.0_f64; width * height];
    let cx = 12.0_f64;
    let cy = 12.0_f64;
    for y in 0..height {
        for x in 0..width {
            let dx = (x as f64) - cx;
            let dy = (y as f64) - cy;
            values[flatten_2d_index(x, y, width)] = dx * dx + dy * dy;
        }
    }

    let field = ScalarField2D::new(width, height, values, false, 4);
    let levels = MarchingSquares::extract_at_thresholds(&field, &[9.0, 25.0, 49.0]);
    assert_eq!(levels.len(), 3);

    for (level, expected_radius) in levels.iter().zip([3.0_f64, 5.0_f64, 7.0_f64]) {
        assert_eq!(
            level.paths.len(),
            1,
            "Paraboloid isoline r={expected_radius} must chain into 1 closed loop"
        );
        let loop_path = &level.paths[0];
        assert!(loop_path.closed);
        assert!(loop_path.points.len() >= 12);
        for &[px, py] in &loop_path.points {
            assert!(px.is_finite() && py.is_finite());
            let r = (px - cx).hypot(py - cy);
            assert!(
                (r - expected_radius).abs() < 0.25,
                "Expected radius ~{expected_radius}, got {r}"
            );
        }
    }
}

#[test]
#[verified_engine::verified]
fn test_saddle_field_disambiguates_cases_5_and_10_with_finite_coords() {
    // Classic hyperbolic saddle f(x, y) = x^2 - y^2 centered on grid
    let (width, height) = (21_usize, 21_usize);
    let mut values = vec![0.0_f64; width * height];
    let center = 10.0_f64;
    for y in 0..height {
        for x in 0..width {
            let dx = ((x as f64) - center) * 0.25;
            let dy = ((y as f64) - center) * 0.25;
            values[flatten_2d_index(x, y, width)] = dx * dx - dy * dy;
        }
    }

    let field = ScalarField2D::new(width, height, values, false, 6);
    let levels = MarchingSquares::extract_at_thresholds(&field, &[-1.0, 1.0]);
    assert_eq!(levels.len(), 2);

    // Each non-zero hyperbola x^2 - y^2 = +/-1 splits into 2 open branches across the box
    for level in &levels {
        assert_eq!(level.paths.len(), 2);
        for path in &level.paths {
            assert!(!path.closed);
            assert!(path.points.len() >= 5);
            for &[px, py] in &path.points {
                assert!(px.is_finite() && py.is_finite());
            }
        }
    }

    // Direct 2x2 cell test exercising both case 5 (0b0101) and case 10 (0b1010) asymptotic deciders
    // Case 5: v00 = 1.0, v10 = -0.2, v11 = 1.0, v01 = -0.2 -> saddle = (1.0 - 0.04) / 2.4 = +0.4
    let case5 = ScalarField2D::new(2, 2, vec![1.0, -0.2, -0.2, 1.0], false, 2);
    let connected = MarchingSquares::extract_at_thresholds(&case5, &[0.2]);
    let separated = MarchingSquares::extract_at_thresholds(&case5, &[0.6]);
    assert_eq!(connected[0].paths.len(), 2);
    assert_eq!(separated[0].paths.len(), 2);
    // At T=0.2 (connected high diagonal), segment midpoints lie in opposite low corners (1,0) and (0,1)
    assert_ne!(connected[0].paths[0].points, separated[0].paths[0].points);

    // Case 10: v00 = -0.2, v10 = 1.0, v11 = -0.2, v01 = 1.0 -> saddle = +0.4
    let case10 = ScalarField2D::new(2, 2, vec![-0.2, 1.0, 1.0, -0.2], false, 2);
    let c10_conn = MarchingSquares::extract_at_thresholds(&case10, &[0.2]);
    let c10_sep = MarchingSquares::extract_at_thresholds(&case10, &[0.6]);
    assert_eq!(c10_conn[0].paths.len(), 2);
    assert_eq!(c10_sep[0].paths.len(), 2);
    assert_ne!(c10_conn[0].paths[0].points, c10_sep[0].paths[0].points);
}

#[test]
#[verified_engine::verified]
fn test_periodic_scalar_field_includes_seam_wrap_cells() {
    let (width, height) = (16_usize, 16_usize);
    let mut values = vec![0.0_f64; width * height];
    // Place Gaussian peak at (0, 0) wrapping across periodic edges
    for y in 0..height {
        for x in 0..width {
            let dx = (x as f64).min((width - x) as f64);
            let dy = (y as f64).min((height - y) as f64);
            values[flatten_2d_index(x, y, width)] = (-0.25 * (dx * dx + dy * dy)).exp();
        }
    }
    let periodic_field = ScalarField2D::new(width, height, values, true, 4);
    assert_eq!(periodic_field.domain_extents(), (16.0, 16.0));
    let levels = MarchingSquares::extract_levels(&periodic_field, 4);
    assert_eq!(levels.len(), 4);
    let total_paths: usize = levels.iter().map(|l| l.paths.len()).sum();
    assert!(total_paths >= 4);
}

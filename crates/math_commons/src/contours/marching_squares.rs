//! 2D marching-squares isoline extractor with linear edge interpolation,
//! bilinear saddle disambiguation, and continuous polyline chaining.

use super::path_order::chain_segments;
use super::{ContourPath, GridBounds2D, Point2D, Segment2D};
use crate::error::{MathError, MathResult};
use crate::math_kernel::types::Dimension;
use crate::registry::{MAX_GRID_SIZE, TOLERANCE_HIGH};

/// Extracts chained 2D contours at `iso_level` from a row-major `width x height` scalar field.
///
/// Grid dimensions are bounded by `2..=MAX_GRID_SIZE`. Ambiguous saddle cells (cases `5` and `10`)
/// are disambiguated using the four-corner bilinear center value `0.25 * (v00 + v10 + v11 + v01)`.
#[verified_engine::verified]
pub fn extract_contours(
    values: &[f64],
    width: usize,
    height: usize,
    bounds: &GridBounds2D,
    iso_level: f64,
) -> MathResult<Vec<ContourPath>> {
    let (dx, dy) = validate_grid_inputs(values, width, height, bounds, iso_level)?;
    let mut segments = Vec::new();

    for iy in 0..(height - 1) {
        let y0 = bounds.min_y + (iy as f64) * dy;
        let y1 = if iy + 2 == height {
            bounds.max_y
        } else {
            bounds.min_y + ((iy + 1) as f64) * dy
        };
        let row0 = iy * width;
        let row1 = (iy + 1) * width;

        for ix in 0..(width - 1) {
            let vals = [
                values[row0 + ix],
                values[row0 + ix + 1],
                values[row1 + ix + 1],
                values[row1 + ix],
            ];
            if !vals.iter().all(|v| v.is_finite()) {
                continue;
            }
            let x0 = bounds.min_x + (ix as f64) * dx;
            let x1 = if ix + 2 == width {
                bounds.max_x
            } else {
                bounds.min_x + ((ix + 1) as f64) * dx
            };
            let corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
            let (cell_segs, count) = cell_contour_segments(corners, vals, iso_level);
            for seg in cell_segs.iter().take(count) {
                segments.push(*seg);
            }
        }
    }

    let tol = (dx.min(dy) * 1e-6).max(1e-11);
    Ok(chain_segments(&segments, tol))
}

/// Extracts chained marching-squares contours across multiple `iso_levels`.
#[verified_engine::verified]
pub fn extract_multi_contours(
    values: &[f64],
    width: usize,
    height: usize,
    bounds: &GridBounds2D,
    iso_levels: &[f64],
) -> MathResult<Vec<(f64, Vec<ContourPath>)>> {
    if iso_levels.len() > MAX_GRID_SIZE {
        return Err(MathError::NumericalError {
            reason: format!("Too many contour iso_levels: {}", iso_levels.len()),
        });
    }
    let mut all = Vec::with_capacity(iso_levels.len());
    for &iso in iso_levels {
        let paths = extract_contours(values, width, height, bounds, iso)?;
        all.push((iso, paths));
    }
    Ok(all)
}

#[verified_engine::verified]
fn validate_grid_inputs(
    values: &[f64],
    width: usize,
    height: usize,
    bounds: &GridBounds2D,
    iso_level: f64,
) -> MathResult<(f64, f64)> {
    if !(2..=MAX_GRID_SIZE).contains(&width) || !(2..=MAX_GRID_SIZE).contains(&height) {
        return Err(MathError::NumericalError {
            reason: format!("Grid dimensions {width}x{height} must be within 2..={MAX_GRID_SIZE}"),
        });
    }
    let expected = width * height;
    if values.len() != expected {
        return Err(MathError::DimensionMismatch {
            expected: Dimension(expected),
            actual: Dimension(values.len()),
        });
    }
    if !iso_level.is_finite() {
        return Err(MathError::NumericalError {
            reason: "Contour iso_level must be finite".to_string(),
        });
    }
    let valid_bounds = GridBounds2D::new(bounds.min_x, bounds.max_x, bounds.min_y, bounds.max_y)?;
    let dx = valid_bounds.span_x() / ((width - 1) as f64);
    let dy = valid_bounds.span_y() / ((height - 1) as f64);
    Ok((dx, dy))
}

#[verified_engine::verified]
fn interpolate_edge(p0: Point2D, p1: Point2D, v0: f64, v1: f64, iso: f64) -> Point2D {
    let denom = v1 - v0;
    let t = if denom.abs() <= TOLERANCE_HIGH {
        0.5
    } else {
        ((iso - v0) / denom).clamp(0.0, 1.0)
    };
    [p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])]
}

#[verified_engine::verified]
fn cell_contour_segments(
    corners: [Point2D; 4],
    vals: [f64; 4],
    iso: f64,
) -> ([Segment2D; 2], usize) {
    let mut mask = 0_u8;
    if vals[0] >= iso {
        mask |= 1;
    }
    if vals[1] >= iso {
        mask |= 2;
    }
    if vals[2] >= iso {
        mask |= 4;
    }
    if vals[3] >= iso {
        mask |= 8;
    }

    let dummy = [[0.0, 0.0], [0.0, 0.0]];
    if mask == 0 || mask == 15 {
        return ([dummy, dummy], 0);
    }

    let e0 = interpolate_edge(corners[0], corners[1], vals[0], vals[1], iso);
    let e1 = interpolate_edge(corners[1], corners[2], vals[1], vals[2], iso);
    let e2 = interpolate_edge(corners[3], corners[2], vals[3], vals[2], iso);
    let e3 = interpolate_edge(corners[0], corners[3], vals[0], vals[3], iso);

    match mask {
        1 | 14 => ([[e3, e0], dummy], 1),
        2 | 13 => ([[e0, e1], dummy], 1),
        4 | 11 => ([[e1, e2], dummy], 1),
        8 | 7 => ([[e2, e3], dummy], 1),
        3 | 12 => ([[e3, e1], dummy], 1),
        6 | 9 => ([[e0, e2], dummy], 1),
        5 => saddle_case_5(e0, e1, e2, e3, vals, iso),
        10 => saddle_case_10(e0, e1, e2, e3, vals, iso),
        _ => ([dummy, dummy], 0),
    }
}

#[verified_engine::verified]
fn saddle_case_5(
    e0: Point2D,
    e1: Point2D,
    e2: Point2D,
    e3: Point2D,
    vals: [f64; 4],
    iso: f64,
) -> ([Segment2D; 2], usize) {
    let center = 0.25 * (vals[0] + vals[1] + vals[2] + vals[3]);
    if center >= iso {
        ([[e0, e1], [e2, e3]], 2)
    } else {
        ([[e3, e0], [e1, e2]], 2)
    }
}

#[verified_engine::verified]
fn saddle_case_10(
    e0: Point2D,
    e1: Point2D,
    e2: Point2D,
    e3: Point2D,
    vals: [f64; 4],
    iso: f64,
) -> ([Segment2D; 2], usize) {
    let center = 0.25 * (vals[0] + vals[1] + vals[2] + vals[3]);
    if center >= iso {
        ([[e3, e0], [e1, e2]], 2)
    } else {
        ([[e0, e1], [e2, e3]], 2)
    }
}

// [cite:math_commons]

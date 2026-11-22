//! Two-dimensional Marching Squares isoline extraction with asymptotic saddle disambiguation.

use crate::math_kernel::types::flatten_2d_index;
use std::collections::HashMap;

#[cfg(test)]
mod tests;

/// Maximum contour level count supported by interactive figure exports.
pub const MAX_CONTOUR_LEVELS: usize = 16;

/// Continuous or discrete 2D scalar field exported by spatial simulation tools.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarField2D {
    /// Grid width in samples.
    pub width: usize,
    /// Grid height in samples.
    pub height: usize,
    /// Row-major scalar samples of length `width * height`.
    pub values: Vec<f64>,
    /// Whether opposite grid edges wrap toroidally without seams.
    pub periodic: bool,
    /// Default number of contour thresholds (`1..=16`).
    pub default_levels: usize,
}

impl ScalarField2D {
    /// Constructs a new [`ScalarField2D`] with clamped default contour levels (`1..=16`).
    #[verified_engine::verified]
    pub fn new(
        width: usize,
        height: usize,
        values: Vec<f64>,
        periodic: bool,
        default_levels: usize,
    ) -> Self {
        Self {
            width,
            height,
            values,
            periodic,
            default_levels: default_levels.clamp(1, MAX_CONTOUR_LEVELS),
        }
    }

    /// Returns the coordinate domain extents `(domain_width, domain_height)` in cell units.
    #[verified_engine::verified]
    pub fn domain_extents(&self) -> (f64, f64) {
        if self.periodic {
            (self.width.max(1) as f64, self.height.max(1) as f64)
        } else {
            (
                self.width.saturating_sub(1).max(1) as f64,
                self.height.saturating_sub(1).max(1) as f64,
            )
        }
    }

    /// Samples `(x, y)` using periodic wrap when `self.periodic` is true, or edge clamping otherwise.
    #[inline]
    #[verified_engine::verified]
    pub fn sample_grid(&self, x: usize, y: usize) -> f64 {
        if self.width == 0 || self.height == 0 || self.values.is_empty() {
            return 0.0;
        }
        let sx = if self.periodic {
            x % self.width
        } else {
            x.min(self.width - 1)
        };
        let sy = if self.periodic {
            y % self.height
        } else {
            y.min(self.height - 1)
        };
        let idx = flatten_2d_index(sx, sy, self.width);
        self.values.get(idx).copied().unwrap_or(0.0)
    }

    /// Computes the `(min, max)` finite scalar range across `self.values`.
    #[verified_engine::verified]
    pub fn finite_min_max(&self) -> Option<(f64, f64)> {
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        let mut found = false;
        for &val in &self.values {
            if val.is_finite() {
                min_v = min_v.min(val);
                max_v = max_v.max(val);
                found = true;
            }
        }
        found.then_some((min_v, max_v))
    }
}

/// A chained 2D contour polyline or closed loop in grid coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct ContourPath {
    /// Ordered `[x, y]` vertices along the contour.
    pub points: Vec<[f64; 2]>,
    /// `true` when the last vertex connects back to the first vertex (`Z` in SVG).
    pub closed: bool,
}

/// Extracted contour paths at a single scalar isovalue threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct ContourLevel {
    /// Scalar threshold value of this isoline level.
    pub threshold: f64,
    /// Normalized position of this threshold within `[0.0, 1.0]`.
    pub normalized_level: f64,
    /// Chained polylines and closed loops for this threshold.
    pub paths: Vec<ContourPath>,
}

/// Stateless 2D Marching Squares contour extractor.
#[derive(Debug, Clone, Copy, Default)]
pub struct MarchingSquares;

impl MarchingSquares {
    /// Extracts `num_levels` (`1..=16`) evenly spaced interior contour thresholds from `field`.
    #[verified_engine::verified]
    pub fn extract_levels(field: &ScalarField2D, num_levels: usize) -> Vec<ContourLevel> {
        let clamped = num_levels.clamp(1, MAX_CONTOUR_LEVELS);
        let Some((min_v, max_v)) = field.finite_min_max() else {
            return Vec::new();
        };
        let span = max_v - min_v;
        if !span.is_finite() || span <= 1e-12 {
            return Vec::new();
        }
        let mut thresholds = Vec::with_capacity(clamped);
        for k in 0..clamped {
            let frac = (k + 1) as f64 / (clamped + 1) as f64;
            thresholds.push((min_v + frac * span, frac));
        }
        Self::extract_with_normalized(field, &thresholds)
    }

    /// Extracts contours at explicit scalar `thresholds`.
    #[verified_engine::verified]
    pub fn extract_at_thresholds(field: &ScalarField2D, thresholds: &[f64]) -> Vec<ContourLevel> {
        let range = field.finite_min_max().unwrap_or((0.0, 1.0));
        let span = (range.1 - range.0).max(1e-12);
        let pairs: Vec<(f64, f64)> = thresholds
            .iter()
            .copied()
            .filter(|t| t.is_finite())
            .map(|t| (t, ((t - range.0) / span).clamp(0.0, 1.0)))
            .collect();
        Self::extract_with_normalized(field, &pairs)
    }

    #[verified_engine::verified]
    fn extract_with_normalized(
        field: &ScalarField2D,
        thresholds: &[(f64, f64)],
    ) -> Vec<ContourLevel> {
        if field.width < 2 || field.height < 2 || field.values.len() < field.width * field.height {
            return Vec::new();
        }
        let cells_x = if field.periodic {
            field.width
        } else {
            field.width - 1
        };
        let cells_y = if field.periodic {
            field.height
        } else {
            field.height - 1
        };

        let mut levels = Vec::with_capacity(thresholds.len());
        for &(threshold, normalized_level) in thresholds {
            let segments = collect_cell_segments(field, cells_x, cells_y, threshold);
            let paths = chain_segments(&segments);
            levels.push(ContourLevel {
                threshold,
                normalized_level,
                paths,
            });
        }
        levels
    }
}

#[inline]
#[verified_engine::verified]
fn interp_fraction(va: f64, vb: f64, threshold: f64) -> f64 {
    let denom = vb - va;
    if denom.abs() < 1e-12 {
        0.5
    } else {
        ((threshold - va) / denom).clamp(0.0, 1.0)
    }
}

#[inline]
#[verified_engine::verified]
fn edge_point(edge: u8, x0: f64, y0: f64, corners: [f64; 4], threshold: f64) -> [f64; 2] {
    let [v00, v10, v11, v01] = corners;
    match edge {
        0 => [x0 + interp_fraction(v00, v10, threshold), y0],
        1 => [x0 + 1.0, y0 + interp_fraction(v10, v11, threshold)],
        2 => [x0 + interp_fraction(v01, v11, threshold), y0 + 1.0],
        _ => [x0, y0 + interp_fraction(v00, v01, threshold)],
    }
}

#[inline]
#[verified_engine::verified]
fn asymptotic_saddle_value(corners: [f64; 4]) -> f64 {
    let [v00, v10, v11, v01] = corners;
    let denom = v00 + v11 - v10 - v01;
    if denom.abs() < 1e-12 {
        0.25 * (v00 + v10 + v11 + v01)
    } else {
        (v00 * v11 - v10 * v01) / denom
    }
}

#[inline]
#[verified_engine::verified]
fn cell_edge_pairs(mask: u8, corners: [f64; 4], threshold: f64) -> ([(u8, u8); 2], usize) {
    match mask {
        0 | 15 => ([(0, 0); 2], 0),
        1 | 14 => ([(3, 0), (0, 0)], 1),
        2 | 13 => ([(0, 1), (0, 0)], 1),
        3 | 12 => ([(3, 1), (0, 0)], 1),
        4 | 11 => ([(1, 2), (0, 0)], 1),
        6 | 9 => ([(0, 2), (0, 0)], 1),
        7 | 8 => ([(3, 2), (0, 0)], 1),
        5 => {
            if asymptotic_saddle_value(corners) >= threshold {
                ([(0, 1), (2, 3)], 2)
            } else {
                ([(3, 0), (1, 2)], 2)
            }
        }
        10 => {
            if asymptotic_saddle_value(corners) >= threshold {
                ([(3, 0), (1, 2)], 2)
            } else {
                ([(0, 1), (2, 3)], 2)
            }
        }
        _ => ([(0, 0); 2], 0),
    }
}

#[verified_engine::verified]
fn collect_cell_segments(
    field: &ScalarField2D,
    cells_x: usize,
    cells_y: usize,
    threshold: f64,
) -> Vec<([f64; 2], [f64; 2])> {
    let mut segments = Vec::new();
    for cy in 0..cells_y {
        for cx in 0..cells_x {
            let v00 = field.sample_grid(cx, cy);
            let v10 = field.sample_grid(cx + 1, cy);
            let v11 = field.sample_grid(cx + 1, cy + 1);
            let v01 = field.sample_grid(cx, cy + 1);
            if !(v00.is_finite() && v10.is_finite() && v11.is_finite() && v01.is_finite()) {
                continue;
            }
            let corners = [v00, v10, v11, v01];
            let mask = u8::from(v00 >= threshold)
                | (u8::from(v10 >= threshold) << 1)
                | (u8::from(v11 >= threshold) << 2)
                | (u8::from(v01 >= threshold) << 3);
            let (pairs, count) = cell_edge_pairs(mask, corners, threshold);
            let (x0, y0) = (cx as f64, cy as f64);
            for &(ea, eb) in pairs.iter().take(count) {
                let pa = edge_point(ea, x0, y0, corners, threshold);
                let pb = edge_point(eb, x0, y0, corners, threshold);
                if (pa[0] - pb[0]).hypot(pa[1] - pb[1]) > 1e-9 {
                    segments.push((pa, pb));
                }
            }
        }
    }
    segments
}

type QuantKey = (i64, i64);

#[inline]
fn quantize_point(pt: [f64; 2]) -> QuantKey {
    (
        (pt[0] * 1_000_000.0).round() as i64,
        (pt[1] * 1_000_000.0).round() as i64,
    )
}

fn trace_chain(
    start_seg: usize,
    start_forward: bool,
    segments: &[([f64; 2], [f64; 2])],
    adj: &HashMap<QuantKey, Vec<usize>>,
    used: &mut [bool],
) -> ContourPath {
    used[start_seg] = true;
    let (first_pt, second_pt) = if start_forward {
        (segments[start_seg].0, segments[start_seg].1)
    } else {
        (segments[start_seg].1, segments[start_seg].0)
    };
    let start_key = quantize_point(first_pt);
    let mut points = vec![first_pt, second_pt];
    let mut curr_key = quantize_point(second_pt);

    while curr_key != start_key {
        let Some(candidates) = adj.get(&curr_key) else {
            break;
        };
        let Some(&next_idx) = candidates.iter().find(|&&idx| !used[idx]) else {
            break;
        };
        used[next_idx] = true;
        let (pa, pb) = segments[next_idx];
        let next_pt = if quantize_point(pa) == curr_key {
            pb
        } else {
            pa
        };
        curr_key = quantize_point(next_pt);
        points.push(next_pt);
    }

    let closed = points.len() >= 4 && curr_key == start_key;
    if closed {
        points.pop();
    }
    ContourPath { points, closed }
}

fn chain_segments(segments: &[([f64; 2], [f64; 2])]) -> Vec<ContourPath> {
    if segments.is_empty() {
        return Vec::new();
    }
    let mut adj: HashMap<QuantKey, Vec<usize>> = HashMap::with_capacity(segments.len() * 2);
    for (idx, &(pa, pb)) in segments.iter().enumerate() {
        adj.entry(quantize_point(pa)).or_default().push(idx);
        adj.entry(quantize_point(pb)).or_default().push(idx);
    }

    let mut used = vec![false; segments.len()];
    let mut paths = Vec::new();

    // Pass 1: Trace open polylines starting from endpoints with odd/boundary degree != 2.
    for (idx, &(pa, pb)) in segments.iter().enumerate() {
        if used[idx] {
            continue;
        }
        let deg_a = adj.get(&quantize_point(pa)).map_or(0, Vec::len);
        let deg_b = adj.get(&quantize_point(pb)).map_or(0, Vec::len);
        if deg_a != 2 {
            paths.push(trace_chain(idx, true, segments, &adj, &mut used));
        } else if deg_b != 2 {
            paths.push(trace_chain(idx, false, segments, &adj, &mut used));
        }
    }

    // Pass 2: Remaining unused segments form closed loops.
    for idx in 0..segments.len() {
        if !used[idx] {
            paths.push(trace_chain(idx, true, segments, &adj, &mut used));
        }
    }
    paths
}

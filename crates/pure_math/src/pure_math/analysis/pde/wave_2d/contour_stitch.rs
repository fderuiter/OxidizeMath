//! Cold-path multi-threshold Marching-Squares contour extraction with bilinear asymptotic
//! saddle disambiguation and endpoint welding into continuous open and closed chains.

// [cite:advanced_linear_algebra]

use std::collections::HashMap;

use math_commons::math_kernel::types::flatten_2d_index;

/// Upper bound on welded contour chains emitted per export call to bound WebAssembly memory.
pub const MAX_CONTOUR_CHAINS: usize = 4_096;

/// Upper bound on total vertices across all welded contour chains per export call.
pub const MAX_CONTOUR_VERTICES: usize = 65_536;

const ENDPOINT_QUANT_SCALE: f64 = 1_000_000.0;

/// Continuous 2D iso-contour polyline or closed loop in grid-cell coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct ContourChain {
    /// Scalar threshold value traced by this contour chain.
    pub iso_level: f64,
    /// Ordered 2D points `[x, y]` in `[0..=width - 1] x [0..=height - 1]` cell-center coordinates.
    pub points: Vec<[f64; 2]>,
    /// `true` when the last vertex connects back to the first vertex (`Z` in SVG `<path>`).
    pub closed: bool,
}

/// Evaluates the bilinear saddle-point sign relative to `0.0` for shifted corner values
/// `(d00, d10, d11, d01)` to disambiguate Marching-Squares cases `5` and `10`.
#[inline]
#[verified_engine::verified]
pub fn saddle_decider_positive(d00: f64, d10: f64, d11: f64, d01: f64) -> bool {
    let denom = d00 + d11 - d10 - d01;
    let saddle_val = if denom.abs() > 1.0e-12 {
        (d00 * d11 - d10 * d01) / denom
    } else {
        0.25 * (d00 + d10 + d11 + d01)
    };
    saddle_val >= 0.0
}

#[inline]
#[verified_engine::verified]
pub(crate) fn interp_edge_zero(a: f64, b: f64) -> f64 {
    let denom = a - b;
    if denom.abs() < 1.0e-15 {
        0.5
    } else {
        (a / denom).clamp(0.0, 1.0)
    }
}

#[inline]
#[verified_engine::verified]
pub(crate) fn cell_case_segments(
    mask: u8,
    corners: [f64; 4],
    edges: [[f64; 2]; 4],
    out: &mut Vec<[[f64; 2]; 2]>,
) {
    let [d00, d10, d11, d01] = corners;
    let [b, r, t, l] = edges;
    match mask {
        1 | 14 => out.push([l, b]),
        2 | 13 => out.push([b, r]),
        3 | 12 => out.push([l, r]),
        4 | 11 => out.push([r, t]),
        5 => {
            if saddle_decider_positive(d00, d10, d11, d01) {
                out.push([l, t]);
                out.push([b, r]);
            } else {
                out.push([l, b]);
                out.push([r, t]);
            }
        }
        6 | 9 => out.push([b, t]),
        7 | 8 => out.push([l, t]),
        10 => {
            if saddle_decider_positive(d00, d10, d11, d01) {
                out.push([l, b]);
                out.push([r, t]);
            } else {
                out.push([l, t]);
                out.push([b, r]);
            }
        }
        _ => {}
    }
}

/// Extracts raw Marching-Squares line segments at `iso_level` over active cells.
#[verified_engine::verified]
pub fn extract_iso_segments(
    field: &[f64],
    active_mask: Option<&[bool]>,
    width: usize,
    height: usize,
    iso_level: f64,
    max_segments: usize,
) -> Vec<[[f64; 2]; 2]> {
    if width < 2 || height < 2 || field.len() < width * height || !iso_level.is_finite() {
        return Vec::new();
    }
    if let Some(m) = active_mask
        && m.len() < width * height
    {
        return Vec::new();
    }
    let cap = max_segments.clamp(16, MAX_CONTOUR_VERTICES);
    let mut segments = Vec::new();
    for y in 0..(height - 1) {
        for x in 0..(width - 1) {
            if segments.len() + 2 > cap {
                return segments;
            }
            extract_single_cell(field, active_mask, width, x, y, iso_level, &mut segments);
        }
    }
    segments
}

#[verified_engine::verified]
fn extract_single_cell(
    field: &[f64],
    active_mask: Option<&[bool]>,
    width: usize,
    x: usize,
    y: usize,
    iso: f64,
    out: &mut Vec<[[f64; 2]; 2]>,
) {
    let i00 = flatten_2d_index(x, y, width);
    let i10 = flatten_2d_index(x + 1, y, width);
    let i11 = flatten_2d_index(x + 1, y + 1, width);
    let i01 = flatten_2d_index(x, y + 1, width);
    if let Some(m) = active_mask
        && (!m[i00] || !m[i10] || !m[i11] || !m[i01])
    {
        return;
    }
    let (v00, v10, v11, v01) = (field[i00], field[i10], field[i11], field[i01]);
    if !v00.is_finite() || !v10.is_finite() || !v11.is_finite() || !v01.is_finite() {
        return;
    }
    let (d00, d10, d11, d01) = (v00 - iso, v10 - iso, v11 - iso, v01 - iso);
    let mask = ((d00 >= 0.0) as u8)
        | (((d10 >= 0.0) as u8) << 1)
        | (((d11 >= 0.0) as u8) << 2)
        | (((d01 >= 0.0) as u8) << 3);
    if mask == 0 || mask == 15 {
        return;
    }
    let (xf, yf) = (x as f64, y as f64);
    let edges = [
        [xf + interp_edge_zero(d00, d10), yf],
        [xf + 1.0, yf + interp_edge_zero(d10, d11)],
        [xf + interp_edge_zero(d01, d11), yf + 1.0],
        [xf, yf + interp_edge_zero(d00, d01)],
    ];
    cell_case_segments(mask, [d00, d10, d11, d01], edges, out);
}

#[inline]
fn quantize_pt(p: [f64; 2]) -> (i64, i64) {
    (
        (p[0] * ENDPOINT_QUANT_SCALE).round() as i64,
        (p[1] * ENDPOINT_QUANT_SCALE).round() as i64,
    )
}

/// Welds disconnected 2-point cell segments into continuous open polylines and closed loops.
#[verified_engine::verified]
pub fn weld_segments_to_chains(
    raw_segments: &[[[f64; 2]; 2]],
    iso_level: f64,
    max_chains: usize,
    max_vertices: usize,
) -> Vec<ContourChain> {
    let segments: Vec<[[f64; 2]; 2]> = raw_segments
        .iter()
        .copied()
        .filter(|s| quantize_pt(s[0]) != quantize_pt(s[1]))
        .collect();
    if segments.is_empty() {
        return Vec::new();
    }
    let mut adj: HashMap<(i64, i64), Vec<(usize, usize)>> = HashMap::with_capacity(segments.len());
    for (idx, seg) in segments.iter().enumerate() {
        adj.entry(quantize_pt(seg[0])).or_default().push((idx, 0));
        adj.entry(quantize_pt(seg[1])).or_default().push((idx, 1));
    }

    let mut visited = vec![false; segments.len()];
    let mut chains = Vec::new();
    let mut vertex_budget = max_vertices.min(MAX_CONTOUR_VERTICES);
    let chain_limit = max_chains.min(MAX_CONTOUR_CHAINS);

    for pass_open_only in [true, false] {
        for start_seg in 0..segments.len() {
            if visited[start_seg] || chains.len() >= chain_limit || vertex_budget < 2 {
                continue;
            }
            let k0 = quantize_pt(segments[start_seg][0]);
            let k1 = quantize_pt(segments[start_seg][1]);
            let deg0 = adj.get(&k0).map_or(0, Vec::len);
            let deg1 = adj.get(&k1).map_or(0, Vec::len);
            let start_side = if pass_open_only {
                if deg0 != 2 {
                    0
                } else if deg1 != 2 {
                    1
                } else {
                    continue;
                }
            } else {
                0
            };
            let chain = trace_chain(
                &segments,
                &adj,
                &mut visited,
                start_seg,
                start_side,
                iso_level,
                vertex_budget,
            );
            vertex_budget = vertex_budget.saturating_sub(chain.points.len());
            if chain.points.len() >= 2 {
                chains.push(chain);
            }
        }
    }
    chains
}

fn trace_chain(
    segments: &[[[f64; 2]; 2]],
    adj: &HashMap<(i64, i64), Vec<(usize, usize)>>,
    visited: &mut [bool],
    start_seg: usize,
    start_side: usize,
    iso_level: f64,
    vertex_budget: usize,
) -> ContourChain {
    visited[start_seg] = true;
    let first_pt = segments[start_seg][start_side];
    let mut curr_pt = segments[start_seg][1 - start_side];
    let start_key = quantize_pt(first_pt);
    let mut points = vec![first_pt, curr_pt];
    let mut closed = false;

    while points.len() < vertex_budget {
        let curr_key = quantize_pt(curr_pt);
        if curr_key == start_key && points.len() >= 3 {
            points.pop();
            closed = true;
            break;
        }
        let Some(neighbors) = adj.get(&curr_key) else {
            break;
        };
        let Some(&(next_seg, hit_side)) = neighbors.iter().find(|&&(s, _)| !visited[s]) else {
            break;
        };
        visited[next_seg] = true;
        curr_pt = segments[next_seg][1 - hit_side];
        points.push(curr_pt);
    }
    ContourChain {
        iso_level,
        points,
        closed,
    }
}

/// Extracts stitched open and closed [`ContourChain`]s across multiple `iso_levels`.
#[verified_engine::verified]
pub fn extract_stitched_contours(
    field: &[f64],
    active_mask: Option<&[bool]>,
    width: usize,
    height: usize,
    iso_levels: &[f64],
) -> Vec<ContourChain> {
    let mut all_chains = Vec::new();
    let mut remaining_vertices = MAX_CONTOUR_VERTICES;
    for &iso in iso_levels {
        if all_chains.len() >= MAX_CONTOUR_CHAINS || remaining_vertices < 2 {
            break;
        }
        let raw = extract_iso_segments(field, active_mask, width, height, iso, remaining_vertices);
        let chains = weld_segments_to_chains(
            &raw,
            iso,
            MAX_CONTOUR_CHAINS - all_chains.len(),
            remaining_vertices,
        );
        for c in chains {
            remaining_vertices = remaining_vertices.saturating_sub(c.points.len());
            all_chains.push(c);
        }
    }
    all_chains
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_peak_forms_single_closed_loop() {
        let (w, h) = (17, 17);
        let mut field = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let dx = x as f64 - 8.0;
                let dy = y as f64 - 8.0;
                field[flatten_2d_index(x, y, w)] = (-(dx * dx + dy * dy) / 12.0).exp();
            }
        }
        let chains = extract_stitched_contours(&field, None, w, h, &[0.5]);
        assert_eq!(chains.len(), 1);
        assert!(chains[0].closed);
        assert!(chains[0].points.len() >= 12);
    }

    #[test]
    fn test_saddle_point_asymptotic_decider_avoids_self_intersection() {
        // 4x4 grid with hyperbolic saddle f(x, y) = (x - 1.5)*(y - 1.5) + 0.15 around center cell (1,1)
        let (w, h) = (4, 4);
        let mut field = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let u = x as f64 - 1.5;
                let v = y as f64 - 1.5;
                field[flatten_2d_index(x, y, w)] = u * v + 0.12;
            }
        }
        assert!(saddle_decider_positive(0.37, -0.13, 0.37, -0.13));
        assert!(!saddle_decider_positive(0.13, -0.37, 0.13, -0.37));

        let chains = extract_stitched_contours(&field, None, w, h, &[0.0]);
        assert_eq!(
            chains.len(),
            2,
            "Hyperbolic saddle should split into 2 disjoint open branches"
        );
        assert!(chains.iter().all(|c| !c.closed && c.points.len() >= 3));
    }
}

//! Shared-segment deduplication, connected polyline chaining, and greedy nearest-neighbor
//! path ordering with open-path endpoint reversal for pen plotters and laser cutters.

use super::{ContourPath, Point2D, Segment2D, euclidean_distance};
use std::collections::{HashMap, HashSet};

type GridKey = (i64, i64);

/// Deduplicates undirected 2D segments within spatial `tolerance`.
///
/// Segments that share identical quantized endpoints in either direction (`[a, b]` or `[b, a]`)
/// or collapse to zero length within `tolerance` are removed while preserving first-seen order.
#[must_use]
#[verified_engine::verified]
pub fn deduplicate_segments(segments: &[Segment2D], tolerance: f64) -> Vec<Segment2D> {
    let tol = sanitize_tolerance(tolerance);
    let inv_tol = 1.0 / tol;
    let mut seen: HashSet<(GridKey, GridKey)> = HashSet::with_capacity(segments.len());
    let mut unique = Vec::with_capacity(segments.len());

    for &seg in segments {
        if !is_finite_segment(seg) {
            continue;
        }
        let k0 = quantize_point(seg[0], inv_tol);
        let k1 = quantize_point(seg[1], inv_tol);
        if k0 == k1 || euclidean_distance(seg[0], seg[1]) <= tol * 0.25 {
            continue;
        }
        let edge_key = if k0 <= k1 { (k0, k1) } else { (k1, k0) };
        if seen.insert(edge_key) {
            unique.push(seg);
        }
    }
    unique
}

/// Chains undirected 2D segments into continuous open polylines and closed loops.
///
/// Duplicate segments are removed first via [`deduplicate_segments`]. Open chains are traced
/// from non-degree-2 vertices before closed degree-2 cycles so boundary contours never split
/// unnecessarily in the interior.
#[must_use]
#[verified_engine::verified]
pub fn chain_segments(segments: &[Segment2D], tolerance: f64) -> Vec<ContourPath> {
    let unique = deduplicate_segments(segments, tolerance);
    if unique.is_empty() {
        return Vec::new();
    }
    let inv_tol = 1.0 / sanitize_tolerance(tolerance);
    let mut adjacency: HashMap<GridKey, Vec<(usize, usize)>> = HashMap::new();
    let mut seg_keys: Vec<[GridKey; 2]> = Vec::with_capacity(unique.len());

    for (idx, &seg) in unique.iter().enumerate() {
        let k0 = quantize_point(seg[0], inv_tol);
        let k1 = quantize_point(seg[1], inv_tol);
        seg_keys.push([k0, k1]);
        adjacency.entry(k0).or_default().push((idx, 0));
        adjacency.entry(k1).or_default().push((idx, 1));
    }

    let mut visited = vec![false; unique.len()];
    let mut paths = Vec::new();

    for pass in 0..2 {
        for start_idx in 0..unique.len() {
            if visited[start_idx] {
                continue;
            }
            let [k0, k1] = seg_keys[start_idx];
            let deg0 = adjacency.get(&k0).map_or(0, Vec::len);
            let deg1 = adjacency.get(&k1).map_or(0, Vec::len);
            if pass == 0 && deg0 == 2 && deg1 == 2 {
                continue;
            }
            let start_end = if pass == 0 && deg0 == 2 && deg1 != 2 {
                1
            } else {
                0
            };
            paths.push(trace_single_chain(
                start_idx,
                start_end,
                &unique,
                &seg_keys,
                &adjacency,
                &mut visited,
            ));
        }
    }
    paths
}

#[verified_engine::verified]
fn trace_single_chain(
    start_idx: usize,
    start_end: usize,
    unique: &[Segment2D],
    seg_keys: &[[GridKey; 2]],
    adjacency: &HashMap<GridKey, Vec<(usize, usize)>>,
    visited: &mut [bool],
) -> ContourPath {
    visited[start_idx] = true;
    let other_end = 1 - start_end;
    let start_key = seg_keys[start_idx][start_end];
    let mut cur_key = seg_keys[start_idx][other_end];
    let mut pts = vec![unique[start_idx][start_end], unique[start_idx][other_end]];

    for _ in 0..unique.len() {
        if cur_key == start_key {
            break;
        }
        let Some(next) = next_unvisited_edge(cur_key, adjacency, visited) else {
            break;
        };
        let (seg_i, matched_end) = next;
        visited[seg_i] = true;
        let far_end = 1 - matched_end;
        cur_key = seg_keys[seg_i][far_end];
        pts.push(unique[seg_i][far_end]);
    }

    let closed = cur_key == start_key && pts.len() >= 4;
    if closed {
        pts.pop();
    }
    ContourPath::new(pts, closed)
}

#[verified_engine::verified]
fn next_unvisited_edge(
    key: GridKey,
    adjacency: &HashMap<GridKey, Vec<(usize, usize)>>,
    visited: &[bool],
) -> Option<(usize, usize)> {
    let neighbors = adjacency.get(&key)?;
    neighbors
        .iter()
        .copied()
        .find(|&(seg_idx, _)| !visited[seg_idx])
}

/// Orders paths with a greedy nearest-neighbor heuristic, reversing open paths and rotating
/// closed loops to minimize non-cutting pen-lift / rapid-travel distance from `[0.0, 0.0]`.
#[must_use]
#[verified_engine::verified]
pub fn order_paths_nearest_neighbor(paths: &[ContourPath]) -> Vec<ContourPath> {
    let valid: Vec<ContourPath> = paths.iter().filter(|p| !p.is_empty()).cloned().collect();
    if valid.len() <= 1 {
        return valid;
    }

    let mut used = vec![false; valid.len()];
    let mut ordered = Vec::with_capacity(valid.len());
    let mut pen = [0.0, 0.0];

    for _ in 0..valid.len() {
        let (best_idx, reverse_or_rot) = select_nearest_path(&valid, &used, pen);
        used[best_idx] = true;
        let mut chosen = valid[best_idx].clone();
        if chosen.closed {
            if reverse_or_rot > 0 && reverse_or_rot < chosen.points.len() {
                chosen.points.rotate_left(reverse_or_rot);
            }
        } else if reverse_or_rot == 1 {
            chosen.points.reverse();
        }
        if let Some(end_pt) = chosen.end_pen_position() {
            pen = end_pt;
        }
        ordered.push(chosen);
    }
    ordered
}

#[verified_engine::verified]
fn select_nearest_path(paths: &[ContourPath], used: &[bool], pen: Point2D) -> (usize, usize) {
    let mut best_idx = 0;
    let mut best_action = 0;
    let mut best_dist = f64::INFINITY;

    for (idx, path) in paths.iter().enumerate() {
        if used[idx] {
            continue;
        }
        if path.closed {
            for (v_idx, &pt) in path.points.iter().enumerate() {
                let d = euclidean_distance(pen, pt);
                if d < best_dist {
                    best_dist = d;
                    best_idx = idx;
                    best_action = v_idx;
                }
            }
        } else {
            let d_start = euclidean_distance(pen, path.points[0]);
            if d_start < best_dist {
                best_dist = d_start;
                best_idx = idx;
                best_action = 0;
            }
            let d_end = euclidean_distance(pen, path.points[path.points.len() - 1]);
            if d_end < best_dist {
                best_dist = d_end;
                best_idx = idx;
                best_action = 1;
            }
        }
    }
    (best_idx, best_action)
}

/// Computes total rapid / pen-lift Euclidean travel distance starting from `[0.0, 0.0]`.
#[must_use]
#[verified_engine::verified]
pub fn pen_lift_travel_distance(paths: &[ContourPath]) -> f64 {
    let mut pen = [0.0, 0.0];
    let mut travel = 0.0;
    for path in paths {
        if path.is_empty() {
            continue;
        }
        travel += euclidean_distance(pen, path.points[0]);
        if let Some(next_pen) = path.end_pen_position() {
            pen = next_pen;
        }
    }
    travel
}

/// Returns non-cutting rapid / pen-lift travel segments starting from `[0.0, 0.0]`.
#[must_use]
#[verified_engine::verified]
pub fn pen_lift_segments(paths: &[ContourPath]) -> Vec<Segment2D> {
    let mut pen = [0.0, 0.0];
    let mut jumps = Vec::with_capacity(paths.len());
    for path in paths {
        if path.is_empty() {
            continue;
        }
        let start = path.points[0];
        if euclidean_distance(pen, start) > 1e-9 {
            jumps.push([pen, start]);
        }
        if let Some(next_pen) = path.end_pen_position() {
            pen = next_pen;
        }
    }
    jumps
}

#[verified_engine::verified]
fn sanitize_tolerance(tolerance: f64) -> f64 {
    if tolerance.is_finite() && tolerance > 1e-12 {
        tolerance
    } else {
        1e-7
    }
}

#[verified_engine::verified]
fn is_finite_segment(seg: Segment2D) -> bool {
    seg[0][0].is_finite() && seg[0][1].is_finite() && seg[1][0].is_finite() && seg[1][1].is_finite()
}

#[verified_engine::verified]
fn quantize_point(pt: Point2D, inv_tol: f64) -> GridKey {
    let qx = (pt[0] * inv_tol)
        .round()
        .clamp(i64::MIN as f64, i64::MAX as f64) as i64;
    let qy = (pt[1] * inv_tol)
        .round()
        .clamp(i64::MIN as f64, i64::MAX as f64) as i64;
    (qx, qy)
}

// [cite:math_commons]

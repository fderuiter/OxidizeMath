//! Medial-axis `Zhang-Suen` skeletonization, interior eyelet hole filling, topological
//! dead-end whisker pruning, and iterative `Ramer-Douglas-Peucker` polyline simplification.

use super::engine::PhysarumEngine;
use super::params::PhysarumTerminal;
use math_commons::math_kernel::types::flatten_2d_index;

pub(crate) const NEIGHBORS_8: [(isize, isize); 8] = [
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
];

const NEIGHBORS_4: [(isize, isize); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];

#[derive(Clone, Copy)]
pub(crate) struct MaskView<'a> {
    pub(crate) mask: &'a [u8],
    pub(crate) w: usize,
    pub(crate) h: usize,
}

pub(crate) type RawChain = (usize, usize, Vec<[f64; 2]>);

#[verified_engine::verified]
pub(crate) fn build_foreground_mask(engine: &PhysarumEngine) -> Vec<u8> {
    let w = engine.width();
    let h = engine.height();
    let ref_peak = engine.telemetry().peak_trail.min(8.0);
    let thresh = (0.14 * ref_peak).max(0.26);
    let mut mask = vec![0u8; w * h];
    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let idx = flatten_2d_index(x, y, w);
            if engine.obstacles()[idx] == 0 && engine.trail()[idx] >= thresh {
                mask[idx] = 1;
            }
        }
    }
    debug_assert_eq!(mask.len(), w * h);
    assert!(thresh > 0.0);
    mask
}

#[verified_engine::verified]
pub(crate) fn fill_small_interior_holes(
    mask: &mut [u8],
    obstacles: &[u8],
    w: usize,
    h: usize,
    max_hole_area: usize,
) {
    let mut seen = vec![0u8; w * h];
    let mut comp = Vec::with_capacity(64);
    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let start = flatten_2d_index(x, y, w);
            if mask[start] != 0 || seen[start] != 0 {
                continue;
            }
            comp.clear();
            comp.push([x, y]);
            seen[start] = 1;
            let mut touches_exterior_or_wall = obstacles[start] != 0;
            let mut head = 0usize;
            while head < comp.len() {
                let [cx, cy] = comp[head];
                head += 1;
                if cx == 0 || cx + 1 == w || cy == 0 || cy + 1 == h {
                    touches_exterior_or_wall = true;
                    continue;
                }
                for &(dx, dy) in &NEIGHBORS_4 {
                    let nx = (cx as isize + dx) as usize;
                    let ny = (cy as isize + dy) as usize;
                    let nidx = flatten_2d_index(nx, ny, w);
                    if mask[nidx] == 0 && seen[nidx] == 0 {
                        seen[nidx] = 1;
                        if obstacles[nidx] != 0 {
                            touches_exterior_or_wall = true;
                        }
                        comp.push([nx, ny]);
                    }
                }
            }
            if !touches_exterior_or_wall && comp.len() <= max_hole_area {
                for &[hx, hy] in &comp {
                    mask[flatten_2d_index(hx, hy, w)] = 1;
                }
            }
        }
    }
    debug_assert_eq!(seen.len(), w * h);
    assert_eq!(mask.len(), w * h);
}

#[verified_engine::verified]
pub(crate) fn build_pinned_terminal_mask(
    terms: &[PhysarumTerminal],
    w: usize,
    h: usize,
    mask: &mut [u8],
) -> Vec<u8> {
    let mut pinned = vec![0u8; w * h];
    for t in terms {
        let tx = (t.x.round() as usize).clamp(1, w - 2);
        let ty = (t.y.round() as usize).clamp(1, h - 2);
        pinned[flatten_2d_index(tx, ty, w)] = 1;
        for dy in -2isize..=2 {
            for dx in -2isize..=2 {
                let nx = (tx as isize + dx).clamp(1, (w - 2) as isize) as usize;
                let ny = (ty as isize + dy).clamp(1, (h - 2) as isize) as usize;
                if dx * dx + dy * dy <= 4 {
                    mask[flatten_2d_index(nx, ny, w)] = 1;
                }
            }
        }
    }
    assert_eq!(pinned.len(), w * h);
    debug_assert!(!terms.is_empty());
    pinned
}

#[verified_engine::verified]
pub(crate) fn neighbor_ring(mask: &[u8], x: usize, y: usize, w: usize) -> [u8; 8] {
    let mut p = [0u8; 8];
    for (k, &(dx, dy)) in NEIGHBORS_8.iter().enumerate() {
        let nx = (x as isize + dx) as usize;
        let ny = (y as isize + dy) as usize;
        p[k] = mask[flatten_2d_index(nx, ny, w)];
    }
    debug_assert!(x >= 1 && y >= 1);
    assert!(w >= 4);
    p
}

#[verified_engine::verified]
fn zhang_suen_deletable(p: [u8; 8], pass: u8) -> bool {
    let b: u8 = p.iter().sum();
    if !(2..=6).contains(&b) {
        return false;
    }
    let mut transitions = 0u8;
    for k in 0..8 {
        if p[k] == 0 && p[(k + 1) % 8] == 1 {
            transitions += 1;
        }
    }
    if transitions != 1 {
        return false;
    }
    let cond = if pass == 0 {
        (p[0] * p[2] * p[4] == 0) && (p[2] * p[4] * p[6] == 0)
    } else {
        (p[0] * p[2] * p[6] == 0) && (p[0] * p[4] * p[6] == 0)
    };
    debug_assert!(pass <= 1);
    assert!(transitions == 1);
    cond
}

#[verified_engine::verified]
pub(crate) fn zhang_suen_thin(mask: &mut [u8], pinned: &[u8], w: usize, h: usize) {
    let mut to_clear = Vec::with_capacity(w * h / 8);
    for _iter in 0..96 {
        let mut changed = false;
        for pass in 0..2u8 {
            to_clear.clear();
            for y in 1..(h - 1) {
                for x in 1..(w - 1) {
                    let idx = flatten_2d_index(x, y, w);
                    if mask[idx] == 1 && pinned[idx] == 0 {
                        let ring = neighbor_ring(mask, x, y, w);
                        if zhang_suen_deletable(ring, pass) {
                            to_clear.push(idx);
                        }
                    }
                }
            }
            if !to_clear.is_empty() {
                changed = true;
                for &idx in &to_clear {
                    mask[idx] = 0;
                }
            }
        }
        if !changed {
            break;
        }
    }
    assert_eq!(mask.len(), w * h);
    debug_assert_eq!(pinned.len(), w * h);
}

#[verified_engine::verified]
pub(crate) fn prune_short_spurs(mask: &mut [u8], pinned: &[u8], w: usize, h: usize, rounds: usize) {
    let mut endpoints = Vec::with_capacity(64);
    for _ in 0..rounds {
        endpoints.clear();
        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let idx = flatten_2d_index(x, y, w);
                if mask[idx] == 1 && pinned[idx] == 0 {
                    let deg: u8 = neighbor_ring(mask, x, y, w).iter().sum();
                    if deg <= 1 {
                        endpoints.push(idx);
                    }
                }
            }
        }
        if endpoints.is_empty() {
            break;
        }
        for &idx in &endpoints {
            mask[idx] = 0;
        }
    }
    debug_assert!(rounds <= 32);
    assert_eq!(mask.len(), w * h);
}

#[verified_engine::verified]
pub(crate) fn cluster_keypoints(
    view: MaskView<'_>,
    terms: &[PhysarumTerminal],
) -> (Vec<[f64; 2]>, Vec<usize>) {
    let none_id = usize::MAX;
    let mut pixel_to_cluster = vec![none_id; view.w * view.h];
    let mut positions = Vec::with_capacity(terms.len() + 16);

    for (ti, t) in terms.iter().enumerate() {
        positions.push([t.x, t.y]);
        let cap_r = (t.radius + 2.6).ceil() as isize;
        let cap_r_sq = (t.radius + 2.6) * (t.radius + 2.6);
        let tx = (t.x.round() as isize).clamp(1, (view.w - 2) as isize);
        let ty = (t.y.round() as isize).clamp(1, (view.h - 2) as isize);
        for dy in -cap_r..=cap_r {
            for dx in -cap_r..=cap_r {
                let nx = (tx + dx).clamp(1, (view.w - 2) as isize) as usize;
                let ny = (ty + dy).clamp(1, (view.h - 2) as isize) as usize;
                let idx = flatten_2d_index(nx, ny, view.w);
                if view.mask[idx] == 1 && ((dx * dx + dy * dy) as f64) <= cap_r_sq {
                    pixel_to_cluster[idx] = ti;
                }
            }
        }
    }

    for y in 1..(view.h - 1) {
        for x in 1..(view.w - 1) {
            let idx = flatten_2d_index(x, y, view.w);
            if view.mask[idx] == 0 || pixel_to_cluster[idx] != none_id {
                continue;
            }
            let deg: u8 = neighbor_ring(view.mask, x, y, view.w).iter().sum();
            if deg != 2 {
                let cid =
                    assign_or_merge_cluster((x, y), view, &mut pixel_to_cluster, &mut positions);
                debug_assert!(cid < positions.len());
            }
        }
    }
    assert!(positions.len() >= terms.len());
    (positions, pixel_to_cluster)
}

#[verified_engine::verified]
fn assign_or_merge_cluster(
    seed: (usize, usize),
    view: MaskView<'_>,
    pixel_to_cluster: &mut [usize],
    positions: &mut Vec<[f64; 2]>,
) -> usize {
    let cid = positions.len();
    let mut queue = vec![[seed.0, seed.1]];
    pixel_to_cluster[flatten_2d_index(seed.0, seed.1, view.w)] = cid;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut count = 0usize;
    let mut head = 0usize;

    while head < queue.len() && head < 64 {
        let [cx, cy] = queue[head];
        head += 1;
        sum_x += cx as f64;
        sum_y += cy as f64;
        count += 1;
        for dy in -2isize..=2 {
            for dx in -2isize..=2 {
                let nx = (cx as isize + dx).clamp(1, (view.w - 2) as isize) as usize;
                let ny = (cy as isize + dy).clamp(1, (view.h - 2) as isize) as usize;
                let nidx = flatten_2d_index(nx, ny, view.w);
                if view.mask[nidx] == 1 && pixel_to_cluster[nidx] == usize::MAX {
                    let ndeg: u8 = neighbor_ring(view.mask, nx, ny, view.w).iter().sum();
                    if ndeg != 2 {
                        pixel_to_cluster[nidx] = cid;
                        queue.push([nx, ny]);
                    }
                }
            }
        }
    }
    let centroid = [sum_x / (count as f64), sum_y / (count as f64)];
    positions.push(centroid);
    assert!(count >= 1);
    debug_assert!(centroid[0].is_finite());
    cid
}

#[verified_engine::verified]
pub(crate) fn trace_skeleton_chains(
    view: MaskView<'_>,
    pixel_to_cluster: &[usize],
) -> Vec<RawChain> {
    let mut visited_step = vec![0u8; view.w * view.h];
    let mut chains = Vec::new();
    for y in 1..(view.h - 1) {
        for x in 1..(view.w - 1) {
            let start_idx = flatten_2d_index(x, y, view.w);
            let u_cid = pixel_to_cluster[start_idx];
            if view.mask[start_idx] == 0 || u_cid == usize::MAX {
                continue;
            }
            for &(dx, dy) in &NEIGHBORS_8 {
                let nx = (x as isize + dx) as usize;
                let ny = (y as isize + dy) as usize;
                let nidx = flatten_2d_index(nx, ny, view.w);
                if view.mask[nidx] == 0 {
                    continue;
                }
                let n_cid = pixel_to_cluster[nidx];
                if n_cid != usize::MAX && n_cid != u_cid {
                    chains.push((
                        u_cid,
                        n_cid,
                        vec![[x as f64, y as f64], [nx as f64, ny as f64]],
                    ));
                } else if n_cid == usize::MAX
                    && visited_step[nidx] == 0
                    && let Some((v_cid, poly)) = walk_degree2_chain(
                        (x, y),
                        (nx, ny),
                        view,
                        pixel_to_cluster,
                        &mut visited_step,
                    )
                {
                    chains.push((u_cid, v_cid, poly));
                }
            }
        }
    }
    debug_assert_eq!(visited_step.len(), view.w * view.h);
    assert!(view.w >= 4);
    chains
}

#[verified_engine::verified]
fn walk_degree2_chain(
    start: (usize, usize),
    first: (usize, usize),
    view: MaskView<'_>,
    pixel_to_cluster: &[usize],
    visited: &mut [u8],
) -> Option<(usize, Vec<[f64; 2]>)> {
    let start_cid = pixel_to_cluster[flatten_2d_index(start.0, start.1, view.w)];
    let mut poly = Vec::with_capacity(64);
    poly.push([start.0 as f64, start.1 as f64]);
    let mut prev = start;
    let mut curr = first;

    for _step in 0..(view.mask.len()) {
        let cidx = flatten_2d_index(curr.0, curr.1, view.w);
        poly.push([curr.0 as f64, curr.1 as f64]);
        let cid = pixel_to_cluster[cidx];
        if cid != usize::MAX {
            debug_assert!(poly.len() >= 2);
            return (cid != start_cid).then_some((cid, poly));
        }
        visited[cidx] = 1;
        let mut next_opt = None;
        for &(dx, dy) in &NEIGHBORS_8 {
            let nx = (curr.0 as isize + dx) as usize;
            let ny = (curr.1 as isize + dy) as usize;
            let nidx = flatten_2d_index(nx, ny, view.w);
            if (nx, ny) != prev && view.mask[nidx] == 1 {
                let ncid = pixel_to_cluster[nidx];
                if ncid != usize::MAX && ncid != start_cid {
                    next_opt = Some((nx, ny));
                    break;
                }
                if ncid == usize::MAX && visited[nidx] == 0 {
                    next_opt = Some((nx, ny));
                }
            }
        }
        let next = next_opt?;
        prev = curr;
        curr = next;
    }
    assert!(!poly.is_empty());
    None
}

#[verified_engine::verified]
pub(crate) fn simplify_polyline_rdp(pts: &[[f64; 2]], epsilon: f64) -> Vec<[f64; 2]> {
    let n = pts.len();
    if n <= 2 {
        return pts.to_vec();
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack = vec![(0usize, n - 1)];
    let mut guard = 0usize;

    while let Some((a, b)) = stack.pop() {
        guard += 1;
        if guard > n * 2 || b <= a + 1 {
            continue;
        }
        let (max_idx, max_dist) = farthest_point_from_chord(pts, a, b);
        if max_dist > epsilon {
            keep[max_idx] = true;
            stack.push((a, max_idx));
            stack.push((max_idx, b));
        }
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if keep[i] {
            out.push(pts[i]);
        }
    }
    assert!(out.len() >= 2);
    debug_assert!(epsilon > 0.0);
    out
}

#[verified_engine::verified]
fn farthest_point_from_chord(pts: &[[f64; 2]], a: usize, b: usize) -> (usize, f64) {
    let pa = pts[a];
    let pb = pts[b];
    let dx = pb[0] - pa[0];
    let dy = pb[1] - pa[1];
    let denom = dx.hypot(dy);
    let mut best_i = a + 1;
    let mut best_d = -1.0;
    for (idx, p) in pts.iter().enumerate().take(b).skip(a + 1) {
        let d = if denom <= 1e-9 {
            (p[0] - pa[0]).hypot(p[1] - pa[1])
        } else {
            (dy * p[0] - dx * p[1] + pb[0] * pa[1] - pb[1] * pa[0]).abs() / denom
        };
        if d > best_d {
            best_d = d;
            best_i = idx;
        }
    }
    debug_assert!(best_i > a && best_i < b);
    assert!(best_d >= 0.0);
    (best_i, best_d)
}

//! Zero-allocation 2D polyline intersection solvers for monotone and general curves.
// [cite:math_commons]

use crate::series::PointSeries;

/// Default maximum intersections collected by [`intersect_polylines`].
pub const MAX_POLYLINE_INTERSECTIONS: usize = 256;

/// Writes intersections `[x, y]` between polylines `a` and `b` within `x_bounds` into `out`.
///
/// Uses an $O(|a| + |b|)$ two-cursor knot sweep when both polylines are strictly increasing
/// in `x`, and falls back to bounded segment-pair AABB filtering for parametric curves.
/// Performs zero heap allocations.
#[verified_engine::verified]
pub fn intersect_polylines_into<A, B>(
    a: &A,
    b: &B,
    x_bounds: [f64; 2],
    out: &mut [[f64; 2]],
) -> usize
where
    A: PointSeries + ?Sized,
    B: PointSeries + ?Sized,
{
    if a.len() < 2 || b.len() < 2 || out.is_empty() {
        return 0;
    }
    if is_strictly_increasing_x(a) && is_strictly_increasing_x(b) {
        intersect_monotone_into(a, b, x_bounds, out)
    } else {
        intersect_general_into(a, b, x_bounds, out)
    }
}

/// Convenience wrapper returning up to [`MAX_POLYLINE_INTERSECTIONS`] polyline intersections.
pub fn intersect_polylines<A, B>(a: &A, b: &B, x_bounds: [f64; 2]) -> Vec<[f64; 2]>
where
    A: PointSeries + ?Sized,
    B: PointSeries + ?Sized,
{
    let mut stack_buf = [[0.0_f64; 2]; MAX_POLYLINE_INTERSECTIONS];
    let count = intersect_polylines_into(a, b, x_bounds, &mut stack_buf);
    stack_buf[..count].to_vec()
}

#[verified_engine::verified]
fn is_strictly_increasing_x<P: PointSeries + ?Sized>(pts: &P) -> bool {
    let n = pts.len();
    for i in 0..n {
        let p = pts.point_at(i);
        if !p[0].is_finite() || !p[1].is_finite() {
            return false;
        }
        if i > 0 && p[0] <= pts.x_at(i - 1) {
            return false;
        }
    }
    true
}

#[verified_engine::verified]
fn intersect_monotone_into<A, B>(a: &A, b: &B, x_bounds: [f64; 2], out: &mut [[f64; 2]]) -> usize
where
    A: PointSeries + ?Sized,
    B: PointSeries + ?Sized,
{
    let na = a.len();
    let nb = b.len();
    let x_lo = a.x_at(0).max(b.x_at(0)).max(x_bounds[0]);
    let x_hi = a.x_at(na - 1).min(b.x_at(nb - 1)).min(x_bounds[1]);
    if x_hi <= x_lo {
        return 0;
    }
    let mut ka = first_index_gt(a, x_lo);
    let mut kb = first_index_gt(b, x_lo);
    let mut ia = 0usize;
    let mut ib = 0usize;
    let mut x0 = x_lo;
    let mut written = 0usize;

    while x0 < x_hi - 1e-12 && written < out.len() {
        let x1 = next_merged_knot(a, b, &mut ka, &mut kb, x0, x_hi);
        if x1 <= x0 + 1e-13 {
            x0 = x1;
            continue;
        }
        while ia + 2 < na && a.x_at(ia + 1) <= x0 + 1e-12 {
            ia += 1;
        }
        while ib + 2 < nb && b.x_at(ib + 1) <= x0 + 1e-12 {
            ib += 1;
        }
        if let Some(hit) = monotone_slab_hit(a, b, ia, ib, x0, x1) {
            written = push_unique_point(out, written, hit);
        }
        x0 = x1;
    }
    written
}

#[verified_engine::verified]
fn first_index_gt<P: PointSeries + ?Sized>(pts: &P, x_lo: f64) -> usize {
    let mut idx = 0usize;
    let n = pts.len();
    while idx < n && pts.x_at(idx) <= x_lo + 1e-12 {
        idx += 1;
    }
    idx
}

#[verified_engine::verified]
fn next_merged_knot<A, B>(
    a: &A,
    b: &B,
    ka: &mut usize,
    kb: &mut usize,
    prev_x: f64,
    x_hi: f64,
) -> f64
where
    A: PointSeries + ?Sized,
    B: PointSeries + ?Sized,
{
    while *ka < a.len() && a.x_at(*ka) <= prev_x + 1e-12 {
        *ka += 1;
    }
    while *kb < b.len() && b.x_at(*kb) <= prev_x + 1e-12 {
        *kb += 1;
    }
    let ca = if *ka < a.len() { a.x_at(*ka) } else { x_hi };
    let cb = if *kb < b.len() { b.x_at(*kb) } else { x_hi };
    let next = ca.min(cb).min(x_hi);
    if *ka < a.len() && (a.x_at(*ka) - next).abs() <= 1e-12 {
        *ka += 1;
    }
    if *kb < b.len() && (b.x_at(*kb) - next).abs() <= 1e-12 {
        *kb += 1;
    }
    next
}

#[verified_engine::verified]
fn monotone_slab_hit<A, B>(a: &A, b: &B, ia: usize, ib: usize, x0: f64, x1: f64) -> Option<[f64; 2]>
where
    A: PointSeries + ?Sized,
    B: PointSeries + ?Sized,
{
    let ya0 = eval_segment_at_x(a.point_at(ia), a.point_at(ia + 1), x0);
    let ya1 = eval_segment_at_x(a.point_at(ia), a.point_at(ia + 1), x1);
    let yb0 = eval_segment_at_x(b.point_at(ib), b.point_at(ib + 1), x0);
    let yb1 = eval_segment_at_x(b.point_at(ib), b.point_at(ib + 1), x1);
    let d0 = ya0 - yb0;
    let d1 = ya1 - yb1;
    if d0.abs() <= 1e-12 && d1.abs() <= 1e-12 {
        None
    } else if d0.abs() <= 1e-12 {
        Some([x0, 0.5 * (ya0 + yb0)])
    } else if d1.abs() <= 1e-12 {
        Some([x1, 0.5 * (ya1 + yb1)])
    } else if (d0 < 0.0 && d1 > 0.0) || (d0 > 0.0 && d1 < 0.0) {
        let t = (d0 / (d0 - d1)).clamp(0.0, 1.0);
        let xh = x0 + t * (x1 - x0);
        let yh = 0.5 * ((ya0 + t * (ya1 - ya0)) + (yb0 + t * (yb1 - yb0)));
        Some([xh, yh])
    } else {
        None
    }
}

#[verified_engine::verified]
fn intersect_general_into<A, B>(a: &A, b: &B, x_bounds: [f64; 2], out: &mut [[f64; 2]]) -> usize
where
    A: PointSeries + ?Sized,
    B: PointSeries + ?Sized,
{
    let mut written = 0usize;
    for i in 0..(a.len() - 1) {
        let a0 = a.point_at(i);
        let a1 = a.point_at(i + 1);
        if !finite_seg(a0, a1) {
            continue;
        }
        let a_min_x = a0[0].min(a1[0]);
        let a_max_x = a0[0].max(a1[0]);
        if a_max_x < x_bounds[0] || a_min_x > x_bounds[1] {
            continue;
        }
        written = scan_general_against_b(a0, a1, a_min_x, a_max_x, b, x_bounds, out, written);
        if written == out.len() {
            break;
        }
    }
    written
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn scan_general_against_b<B: PointSeries + ?Sized>(
    a0: [f64; 2],
    a1: [f64; 2],
    a_min_x: f64,
    a_max_x: f64,
    b: &B,
    x_bounds: [f64; 2],
    out: &mut [[f64; 2]],
    mut written: usize,
) -> usize {
    let a_min_y = a0[1].min(a1[1]);
    let a_max_y = a0[1].max(a1[1]);
    for j in 0..(b.len() - 1) {
        let b0 = b.point_at(j);
        let b1 = b.point_at(j + 1);
        if !finite_seg(b0, b1) {
            continue;
        }
        if b0[0].max(b1[0]) < a_min_x - 1e-12 || b0[0].min(b1[0]) > a_max_x + 1e-12 {
            continue;
        }
        if b0[1].max(b1[1]) < a_min_y - 1e-12 || b0[1].min(b1[1]) > a_max_y + 1e-12 {
            continue;
        }
        if let Some(hit) = segment_intersection_2d(a0, a1, b0, b1) {
            if hit[0] >= x_bounds[0] - 1e-9 && hit[0] <= x_bounds[1] + 1e-9 {
                written = push_unique_point(out, written, hit);
                if written == out.len() {
                    break;
                }
            }
        }
    }
    written
}

/// Linearly interpolates segment `(p0, p1)` at horizontal coordinate `x`.
#[inline]
#[verified_engine::verified]
pub fn eval_segment_at_x(p0: [f64; 2], p1: [f64; 2], x: f64) -> f64 {
    let dx = p1[0] - p0[0];
    if dx.abs() <= 1e-15 {
        return 0.5 * (p0[1] + p1[1]);
    }
    let t = ((x - p0[0]) / dx).clamp(0.0, 1.0);
    p0[1] + t * (p1[1] - p0[1])
}

/// Computes the exact 2D intersection of closed line segments `(a0, a1)` and `(b0, b1)`.
#[verified_engine::verified]
pub fn segment_intersection_2d(
    a0: [f64; 2],
    a1: [f64; 2],
    b0: [f64; 2],
    b1: [f64; 2],
) -> Option<[f64; 2]> {
    let rx = a1[0] - a0[0];
    let ry = a1[1] - a0[1];
    let sx = b1[0] - b0[0];
    let sy = b1[1] - b0[1];
    let denom = rx * sy - ry * sx;
    if denom.abs() <= 1e-14 {
        return None;
    }
    let qpx = b0[0] - a0[0];
    let qpy = b0[1] - a0[1];
    let t = (qpx * sy - qpy * sx) / denom;
    let u = (qpx * ry - qpy * rx) / denom;
    if (-1e-9..=1.0 + 1e-9).contains(&t) && (-1e-9..=1.0 + 1e-9).contains(&u) {
        let tc = t.clamp(0.0, 1.0);
        Some([a0[0] + tc * rx, a0[1] + tc * ry])
    } else {
        None
    }
}

#[inline]
#[verified_engine::verified]
fn finite_seg(p0: [f64; 2], p1: [f64; 2]) -> bool {
    p0[0].is_finite() && p0[1].is_finite() && p1[0].is_finite() && p1[1].is_finite()
}

#[inline]
#[verified_engine::verified]
fn push_unique_point(out: &mut [[f64; 2]], written: usize, candidate: [f64; 2]) -> usize {
    for prev in &out[..written] {
        if (prev[0] - candidate[0]).abs() <= 1e-7 && (prev[1] - candidate[1]).abs() <= 1e-7 {
            return written;
        }
    }
    out[written] = candidate;
    written + 1
}

//! Zero-allocation 2D polyline intersection solvers for monotone and general curves.
// [cite:math_commons]

use crate::series::PointSeries;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use math_scalars::RealField;

/// Default maximum intersections collected by [`intersect_polylines`].
pub const MAX_POLYLINE_INTERSECTIONS: usize = 256;

/// Writes intersections `[x, y]` between polylines `a` and `b` within `x_bounds` into `out`.
///
/// Uses an $O(|a| + |b|)$ two-cursor knot sweep when both polylines are strictly increasing
/// in `x`, and falls back to bounded segment-pair AABB filtering for parametric curves.
/// Performs zero heap allocations.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn intersect_polylines_into<T, A, B>(
    a: &A,
    b: &B,
    x_bounds: [T; 2],
    out: &mut [[T; 2]],
) -> usize
where
    T: RealField,
    A: PointSeries<T> + ?Sized,
    B: PointSeries<T> + ?Sized,
{
    intersect_polylines_into_raw(a, b, x_bounds, out)
}

fn intersect_polylines_into_raw<T, A, B>(
    a: &A,
    b: &B,
    x_bounds: [T; 2],
    out: &mut [[T; 2]],
) -> usize
where
    T: RealField,
    A: PointSeries<T> + ?Sized,
    B: PointSeries<T> + ?Sized,
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
#[cfg(feature = "alloc")]
pub fn intersect_polylines<T, A, B>(a: &A, b: &B, x_bounds: [T; 2]) -> Vec<[T; 2]>
where
    T: RealField,
    A: PointSeries<T> + ?Sized,
    B: PointSeries<T> + ?Sized,
{
    let mut stack_buf = [[T::ZERO; 2]; MAX_POLYLINE_INTERSECTIONS];
    let count = intersect_polylines_into_raw(a, b, x_bounds, &mut stack_buf);
    stack_buf[..count].to_vec()
}

fn is_strictly_increasing_x<T: RealField, P: PointSeries<T> + ?Sized>(pts: &P) -> bool {
    let n = pts.len();
    let mut prev_x = T::NEG_INFINITY;
    for i in 0..n {
        let p = pts.point_at(i);
        if !p[0].is_finite() || !p[1].is_finite() {
            return false;
        }
        if i > 0 && p[0] <= prev_x {
            return false;
        }
        prev_x = p[0];
    }
    true
}

fn intersect_monotone_into<T, A, B>(a: &A, b: &B, x_bounds: [T; 2], out: &mut [[T; 2]]) -> usize
where
    T: RealField,
    A: PointSeries<T> + ?Sized,
    B: PointSeries<T> + ?Sized,
{
    let na = a.len();
    let nb = b.len();
    let x_lo = a.x_at(0).max(b.x_at(0)).max(x_bounds[0]);
    let x_hi = a.x_at(na - 1).min(b.x_at(nb - 1)).min(x_bounds[1]);
    if x_hi <= x_lo {
        return 0;
    }
    let tol = T::from_f64(1e-12);
    let tiny = T::from_f64(1e-13);
    let mut ka = first_index_gt(a, x_lo);
    let mut kb = first_index_gt(b, x_lo);
    let mut ia = 0usize;
    let mut ib = 0usize;
    let mut x0 = x_lo;
    let mut written = 0usize;

    while x0 < x_hi - tol && written < out.len() {
        let x1 = next_merged_knot(a, b, &mut ka, &mut kb, x0, x_hi);
        if x1 <= x0 + tiny {
            x0 = x1;
            continue;
        }
        while ia + 2 < na && a.x_at(ia + 1) <= x0 + tol {
            ia += 1;
        }
        while ib + 2 < nb && b.x_at(ib + 1) <= x0 + tol {
            ib += 1;
        }
        if let Some(hit) = monotone_slab_hit(a, b, ia, ib, x0, x1) {
            written = push_unique_point(out, written, hit);
        }
        x0 = x1;
    }
    written
}

fn first_index_gt<T: RealField, P: PointSeries<T> + ?Sized>(pts: &P, x_lo: T) -> usize {
    let mut idx = 0usize;
    let n = pts.len();
    let tol = T::from_f64(1e-12);
    while idx < n && pts.x_at(idx) <= x_lo + tol {
        idx += 1;
    }
    idx
}

fn next_merged_knot<T, A, B>(a: &A, b: &B, ka: &mut usize, kb: &mut usize, prev_x: T, x_hi: T) -> T
where
    T: RealField,
    A: PointSeries<T> + ?Sized,
    B: PointSeries<T> + ?Sized,
{
    let tol = T::from_f64(1e-12);
    while *ka < a.len() && a.x_at(*ka) <= prev_x + tol {
        *ka += 1;
    }
    while *kb < b.len() && b.x_at(*kb) <= prev_x + tol {
        *kb += 1;
    }
    let ca = if *ka < a.len() { a.x_at(*ka) } else { x_hi };
    let cb = if *kb < b.len() { b.x_at(*kb) } else { x_hi };
    let next = ca.min(cb).min(x_hi);
    if *ka < a.len() && (a.x_at(*ka) - next).abs() <= tol {
        *ka += 1;
    }
    if *kb < b.len() && (b.x_at(*kb) - next).abs() <= tol {
        *kb += 1;
    }
    next
}

fn monotone_slab_hit<T, A, B>(a: &A, b: &B, ia: usize, ib: usize, x0: T, x1: T) -> Option<[T; 2]>
where
    T: RealField,
    A: PointSeries<T> + ?Sized,
    B: PointSeries<T> + ?Sized,
{
    let tol = T::from_f64(1e-12);
    let a0 = a.point_at(ia);
    let a1 = a.point_at(ia + 1);
    let b0 = b.point_at(ib);
    let b1 = b.point_at(ib + 1);
    let ya0 = eval_segment_at_x_raw(a0, a1, x0);
    let ya1 = eval_segment_at_x_raw(a0, a1, x1);
    let yb0 = eval_segment_at_x_raw(b0, b1, x0);
    let yb1 = eval_segment_at_x_raw(b0, b1, x1);
    let d0 = ya0 - yb0;
    let d1 = ya1 - yb1;
    if d0.abs() <= tol && d1.abs() <= tol {
        None
    } else if d0.abs() <= tol {
        Some([x0, T::HALF * (ya0 + yb0)])
    } else if d1.abs() <= tol {
        Some([x1, T::HALF * (ya1 + yb1)])
    } else if (d0 < T::ZERO && d1 > T::ZERO) || (d0 > T::ZERO && d1 < T::ZERO) {
        let t = (d0 / (d0 - d1)).clamp(T::ZERO, T::ONE);
        let xh = x0 + t * (x1 - x0);
        let yh = T::HALF * ((ya0 + t * (ya1 - ya0)) + (yb0 + t * (yb1 - yb0)));
        Some([xh, yh])
    } else {
        None
    }
}

fn intersect_general_into<T, A, B>(a: &A, b: &B, x_bounds: [T; 2], out: &mut [[T; 2]]) -> usize
where
    T: RealField,
    A: PointSeries<T> + ?Sized,
    B: PointSeries<T> + ?Sized,
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
fn scan_general_against_b<T: RealField, B: PointSeries<T> + ?Sized>(
    a0: [T; 2],
    a1: [T; 2],
    a_min_x: T,
    a_max_x: T,
    b: &B,
    x_bounds: [T; 2],
    out: &mut [[T; 2]],
    mut written: usize,
) -> usize {
    let a_min_y = a0[1].min(a1[1]);
    let a_max_y = a0[1].max(a1[1]);
    let pad = T::from_f64(1e-12);
    let bpad = T::from_f64(1e-9);
    for j in 0..(b.len() - 1) {
        let b0 = b.point_at(j);
        let b1 = b.point_at(j + 1);
        if !finite_seg(b0, b1) {
            continue;
        }
        if b0[0].max(b1[0]) < a_min_x - pad || b0[0].min(b1[0]) > a_max_x + pad {
            continue;
        }
        if b0[1].max(b1[1]) < a_min_y - pad || b0[1].min(b1[1]) > a_max_y + pad {
            continue;
        }
        if let Some(hit) = segment_intersection_2d_raw(a0, a1, b0, b1)
            && hit[0] >= x_bounds[0] - bpad
            && hit[0] <= x_bounds[1] + bpad
        {
            written = push_unique_point(out, written, hit);
            if written == out.len() {
                break;
            }
        }
    }
    written
}

/// Linearly interpolates segment `(p0, p1)` at horizontal coordinate `x`.
#[inline]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn eval_segment_at_x<T: RealField>(p0: [T; 2], p1: [T; 2], x: T) -> T {
    eval_segment_at_x_raw(p0, p1, x)
}

#[inline(always)]
fn eval_segment_at_x_raw<T: RealField>(p0: [T; 2], p1: [T; 2], x: T) -> T {
    let dx = p1[0] - p0[0];
    if dx.abs() <= T::from_f64(1e-15) {
        return T::HALF * (p0[1] + p1[1]);
    }
    let t = ((x - p0[0]) / dx).clamp(T::ZERO, T::ONE);
    p0[1] + t * (p1[1] - p0[1])
}

/// Computes the exact 2D intersection of closed line segments `(a0, a1)` and `(b0, b1)`.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn segment_intersection_2d<T: RealField>(
    a0: [T; 2],
    a1: [T; 2],
    b0: [T; 2],
    b1: [T; 2],
) -> Option<[T; 2]> {
    segment_intersection_2d_raw(a0, a1, b0, b1)
}

#[inline]
fn segment_intersection_2d_raw<T: RealField>(
    a0: [T; 2],
    a1: [T; 2],
    b0: [T; 2],
    b1: [T; 2],
) -> Option<[T; 2]> {
    let rx = a1[0] - a0[0];
    let ry = a1[1] - a0[1];
    let sx = b1[0] - b0[0];
    let sy = b1[1] - b0[1];
    let denom = rx * sy - ry * sx;
    if denom.abs() <= T::from_f64(1e-14).max(T::EPSILON * T::from_f64(4.0)) {
        return None;
    }
    let qpx = b0[0] - a0[0];
    let qpy = b0[1] - a0[1];
    let t = (qpx * sy - qpy * sx) / denom;
    let u = (qpx * ry - qpy * rx) / denom;
    let eps = T::from_f64(1e-9).max(T::EPSILON * T::from_f64(16.0));
    if t >= -eps && t <= T::ONE + eps && u >= -eps && u <= T::ONE + eps {
        let tc = t.clamp(T::ZERO, T::ONE);
        Some([a0[0] + tc * rx, a0[1] + tc * ry])
    } else {
        None
    }
}

#[inline(always)]
fn finite_seg<T: RealField>(p0: [T; 2], p1: [T; 2]) -> bool {
    p0[0].is_finite() && p0[1].is_finite() && p1[0].is_finite() && p1[1].is_finite()
}

#[inline]
fn push_unique_point<T: RealField>(out: &mut [[T; 2]], written: usize, candidate: [T; 2]) -> usize {
    let dup = T::from_f64(1e-7).max(T::EPSILON * T::from_f64(64.0));
    for prev in &out[..written] {
        if (prev[0] - candidate[0]).abs() <= dup && (prev[1] - candidate[1]).abs() <= dup {
            return written;
        }
    }
    out[written] = candidate;
    written + 1
}

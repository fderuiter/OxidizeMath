//! Sub-segment Hermite threshold crossings and vertical line crossings.
// [cite:math_commons]

use crate::series::PointSeries;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use math_scalars::RealField;

/// Writes sub-segment `x` coordinates where `pts` crosses `y = y_target` into `out`.
///
/// Uses Fritsch-Carlson monotone cubic Hermite slopes with bracketed Newton-bisection
/// on strictly monotone 4-point stencils and falls back to linear interpolation otherwise.
/// Returns the number of crossings written (up to `out.len()`).
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn threshold_crossings_into<T: RealField, P: PointSeries<T> + ?Sized>(
    pts: &P,
    y_target: T,
    out: &mut [T],
) -> usize {
    threshold_crossings_into_raw(pts, y_target, out)
}

pub(super) fn threshold_crossings_into_raw<T: RealField, P: PointSeries<T> + ?Sized>(
    pts: &P,
    y_target: T,
    out: &mut [T],
) -> usize {
    let n = pts.len();
    if n < 2 || out.is_empty() || !y_target.is_finite() {
        return 0;
    }
    let mut written = 0;
    let mut p0 = pts.point_at(0);
    let mut p0_fin = is_finite_pair(p0);
    for i in 0..(n - 1) {
        let p1 = pts.point_at(i + 1);
        let p1_fin = is_finite_pair(p1);
        if p0_fin
            && p1_fin
            && let Some(x_hit) = segment_threshold_hit(pts, i, n, p0, p1, y_target)
        {
            written = push_unique_scalar(out, written, x_hit);
            if written == out.len() {
                break;
            }
        }
        p0 = p1;
        p0_fin = p1_fin;
    }
    written
}

/// Finds the sub-segment `y = y_target` crossing closest to `x_ref` with zero heap allocations.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn closest_threshold_crossing<T: RealField, P: PointSeries<T> + ?Sized>(
    pts: &P,
    y_target: T,
    x_ref: T,
) -> Option<T> {
    closest_threshold_crossing_raw(pts, y_target, x_ref)
}

fn closest_threshold_crossing_raw<T: RealField, P: PointSeries<T> + ?Sized>(
    pts: &P,
    y_target: T,
    x_ref: T,
) -> Option<T> {
    let n = pts.len();
    if n < 2 || !y_target.is_finite() || !x_ref.is_finite() {
        return None;
    }
    let mut best: Option<T> = None;
    let mut p0 = pts.point_at(0);
    let mut p0_fin = is_finite_pair(p0);
    for i in 0..(n - 1) {
        let p1 = pts.point_at(i + 1);
        let p1_fin = is_finite_pair(p1);
        if p0_fin
            && p1_fin
            && let Some(x_hit) = segment_threshold_hit(pts, i, n, p0, p1, y_target)
            && best.is_none_or(|prev| (x_hit - x_ref).abs() < (prev - x_ref).abs())
        {
            best = Some(x_hit);
        }
        p0 = p1;
        p0_fin = p1_fin;
    }
    best
}

/// Convenience wrapper collecting all sub-segment `y = y_target` crossings into a `Vec<T>`.
#[cfg(feature = "alloc")]
pub fn interpolate_threshold_crossings<T: RealField, P: PointSeries<T> + ?Sized>(
    pts: &P,
    y_target: T,
) -> Vec<T> {
    let n = pts.len();
    let mut out = Vec::new();
    if n < 2 || !y_target.is_finite() {
        return out;
    }
    let mut p0 = pts.point_at(0);
    let mut p0_fin = is_finite_pair(p0);
    for i in 0..(n - 1) {
        let p1 = pts.point_at(i + 1);
        let p1_fin = is_finite_pair(p1);
        if p0_fin
            && p1_fin
            && let Some(x_hit) = segment_threshold_hit(pts, i, n, p0, p1, y_target)
        {
            push_unique_vec(&mut out, x_hit);
        }
        p0 = p1;
        p0_fin = p1_fin;
    }
    out
}

#[inline]
pub(super) fn segment_threshold_hit<T: RealField, P: PointSeries<T> + ?Sized>(
    pts: &P,
    i: usize,
    n: usize,
    p0: [T; 2],
    p1: [T; 2],
    y_target: T,
) -> Option<T> {
    let tol = T::from_f64(1e-12);
    let d0 = p0[1] - y_target;
    let d1 = p1[1] - y_target;
    if d0.abs() <= tol && d1.abs() <= tol {
        None
    } else if d0.abs() <= tol {
        Some(p0[0])
    } else if d1.abs() <= tol {
        (i + 2 == n).then_some(p1[0])
    } else if (d0 < T::ZERO && d1 > T::ZERO) || (d0 > T::ZERO && d1 < T::ZERO) {
        Some(crossing_on_segment(pts, i, p0, p1, y_target))
    } else {
        None
    }
}

pub(super) fn crossing_on_segment<T: RealField, P: PointSeries<T> + ?Sized>(
    pts: &P,
    i: usize,
    p0: [T; 2],
    p1: [T; 2],
    y_target: T,
) -> T {
    let dx = p1[0] - p0[0];
    if dx > T::from_f64(1e-12)
        && let Some(xh) = hermite_root_on_segment(pts, i, p0, p1, y_target)
    {
        return xh;
    }
    let t = ((y_target - p0[1]) / (p1[1] - p0[1])).clamp(T::ZERO, T::ONE);
    p0[0] + t * dx
}

fn hermite_root_on_segment<T: RealField, P: PointSeries<T> + ?Sized>(
    pts: &P,
    i: usize,
    p0: [T; 2],
    p1: [T; 2],
    y_target: T,
) -> Option<T> {
    let h = p1[0] - p0[0];
    if h <= T::from_f64(1e-12) {
        return None;
    }
    let sec = (p1[1] - p0[1]) / h;
    let m0 = endpoint_monotone_slope_left(pts, i, p0, sec)?;
    let m1 = endpoint_monotone_slope_right(pts, i, p1, sec)?;
    solve_hermite_bracket(p0, p1, h, m0, m1, y_target)
}

fn endpoint_monotone_slope_left<T: RealField, P: PointSeries<T> + ?Sized>(
    pts: &P,
    i: usize,
    p0: [T; 2],
    sec: T,
) -> Option<T> {
    if i == 0 {
        return Some(sec);
    }
    let pm1 = pts.point_at(i - 1);
    if !is_finite_pair(pm1) || p0[0] - pm1[0] <= T::from_f64(1e-12) {
        return None;
    }
    let s_prev = (p0[1] - pm1[1]) / (p0[0] - pm1[0]);
    Some(if s_prev * sec > T::ZERO {
        T::HALF * (s_prev + sec)
    } else {
        T::ZERO
    })
}

fn endpoint_monotone_slope_right<T: RealField, P: PointSeries<T> + ?Sized>(
    pts: &P,
    i: usize,
    p1: [T; 2],
    sec: T,
) -> Option<T> {
    if i + 2 >= pts.len() {
        return Some(sec);
    }
    let p2 = pts.point_at(i + 2);
    if !is_finite_pair(p2) || p2[0] - p1[0] <= T::from_f64(1e-12) {
        return None;
    }
    let s_next = (p2[1] - p1[1]) / (p2[0] - p1[0]);
    Some(if sec * s_next > T::ZERO {
        T::HALF * (sec + s_next)
    } else {
        T::ZERO
    })
}

fn solve_hermite_bracket<T: RealField>(
    p0: [T; 2],
    p1: [T; 2],
    h: T,
    mut m0: T,
    mut m1: T,
    y_target: T,
) -> Option<T> {
    let sec = (p1[1] - p0[1]) / h;
    let three = T::from_f64(3.0);
    if sec.abs() <= T::from_f64(1e-15) {
        m0 = T::ZERO;
        m1 = T::ZERO;
    } else {
        let a = m0 / sec;
        let b = m1 / sec;
        let hypot = a.hypot(b);
        if hypot > three {
            let tau = three / hypot;
            m0 = tau * a * sec;
            m1 = tau * b * sec;
        }
    }
    let mut lo = T::ZERO;
    let mut hi = T::ONE;
    let f_lo = p0[1] - y_target;
    let mut t = (-f_lo / (p1[1] - p0[1])).clamp(T::ZERO, T::ONE);
    let root_tol = T::from_f64(1e-12).max(T::EPSILON * T::from_f64(4.0));
    let deriv_tol = T::from_f64(1e-14).max(T::EPSILON);
    for _ in 0..16 {
        let (ft, dft) = eval_hermite_val_deriv(p0[1], p1[1], h, m0, m1, t, y_target);
        if ft.abs() <= root_tol {
            break;
        }
        if (ft > T::ZERO) == (f_lo > T::ZERO) {
            lo = t;
        } else {
            hi = t;
        }
        let cand = if dft.abs() > deriv_tol {
            t - ft / dft
        } else {
            -T::ONE
        };
        t = if cand > lo && cand < hi {
            cand
        } else {
            T::HALF * (lo + hi)
        };
    }
    Some(p0[0] + t.clamp(T::ZERO, T::ONE) * h)
}

#[inline(always)]
fn eval_hermite_val_deriv<T: RealField>(
    y0: T,
    y1: T,
    h: T,
    m0: T,
    m1: T,
    t: T,
    y_target: T,
) -> (T, T) {
    let t2 = t * t;
    let t3 = t2 * t;
    let three = T::from_f64(3.0);
    let four = T::from_f64(4.0);
    let six = T::from_f64(6.0);
    let val = (T::TWO * t3 - three * t2 + T::ONE) * y0
        + (t3 - T::TWO * t2 + t) * h * m0
        + (-T::TWO * t3 + three * t2) * y1
        + (t3 - t2) * h * m1
        - y_target;
    let deriv = (six * t2 - six * t) * y0
        + (three * t2 - four * t + T::ONE) * h * m0
        + (-six * t2 + six * t) * y1
        + (three * t2 - T::TWO * t) * h * m1;
    (val, deriv)
}

/// Writes interpolated `y` values where `pts` crosses vertical line `x = x_target` into `out`.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn y_at_x_into<T: RealField, P: PointSeries<T> + ?Sized>(
    pts: &P,
    x_target: T,
    out: &mut [T],
) -> usize {
    y_at_x_into_raw(pts, x_target, out)
}

fn y_at_x_into_raw<T: RealField, P: PointSeries<T> + ?Sized>(
    pts: &P,
    x_target: T,
    out: &mut [T],
) -> usize {
    let n = pts.len();
    if n < 2 || out.is_empty() || !x_target.is_finite() {
        return 0;
    }
    let mut written = 0;
    let mut p0 = pts.point_at(0);
    let mut p0_fin = is_finite_pair(p0);
    for i in 0..(n - 1) {
        let p1 = pts.point_at(i + 1);
        let p1_fin = is_finite_pair(p1);
        if p0_fin
            && p1_fin
            && let Some(yh) = segment_vertical_hit(i, n, p0, p1, x_target)
        {
            written = push_unique_scalar(out, written, yh);
            if written == out.len() {
                break;
            }
        }
        p0 = p1;
        p0_fin = p1_fin;
    }
    written
}

/// Convenience wrapper collecting interpolated `y` values at `x = x_target` into a `Vec<T>`.
#[cfg(feature = "alloc")]
pub fn interpolate_y_at_x<T: RealField, P: PointSeries<T> + ?Sized>(
    pts: &P,
    x_target: T,
) -> Vec<T> {
    let n = pts.len();
    let mut out = Vec::new();
    if n < 2 || !x_target.is_finite() {
        return out;
    }
    let mut p0 = pts.point_at(0);
    let mut p0_fin = is_finite_pair(p0);
    for i in 0..(n - 1) {
        let p1 = pts.point_at(i + 1);
        let p1_fin = is_finite_pair(p1);
        if p0_fin
            && p1_fin
            && let Some(yh) = segment_vertical_hit(i, n, p0, p1, x_target)
        {
            push_unique_vec(&mut out, yh);
        }
        p0 = p1;
        p0_fin = p1_fin;
    }
    out
}

#[inline]
pub(super) fn segment_vertical_hit<T: RealField>(
    i: usize,
    n: usize,
    p0: [T; 2],
    p1: [T; 2],
    x_target: T,
) -> Option<T> {
    let tol = T::from_f64(1e-12);
    let d0 = p0[0] - x_target;
    let d1 = p1[0] - x_target;
    if d0.abs() <= tol && d1.abs() <= tol {
        None
    } else if d0.abs() <= tol {
        Some(p0[1])
    } else if d1.abs() <= tol {
        (i + 2 == n).then_some(p1[1])
    } else if (d0 < T::ZERO && d1 > T::ZERO) || (d0 > T::ZERO && d1 < T::ZERO) {
        let t = ((x_target - p0[0]) / (p1[0] - p0[0])).clamp(T::ZERO, T::ONE);
        Some(p0[1] + t * (p1[1] - p0[1]))
    } else {
        None
    }
}

#[inline(always)]
pub(super) fn is_finite_pair<T: RealField>(p: [T; 2]) -> bool {
    p[0].is_finite() && p[1].is_finite()
}

#[inline(always)]
fn push_unique_scalar<T: RealField>(out: &mut [T], written: usize, val: T) -> usize {
    if written > 0 && (out[written - 1] - val).abs() <= T::from_f64(1e-9) {
        written
    } else {
        out[written] = val;
        written + 1
    }
}

#[cfg(feature = "alloc")]
#[inline(always)]
pub(super) fn push_unique_vec<T: RealField>(out: &mut Vec<T>, val: T) {
    if out
        .last()
        .is_none_or(|&prev| (prev - val).abs() > T::from_f64(1e-9))
    {
        out.push(val);
    }
}

//! Sub-segment Hermite threshold crossings and vertical line crossings.
// [cite:math_commons]

use crate::series::PointSeries;

/// Writes sub-segment `x` coordinates where `pts` crosses `y = y_target` into `out`.
///
/// Uses Fritsch-Carlson monotone cubic Hermite slopes with bracketed Newton-bisection
/// on strictly monotone 4-point stencils and falls back to linear interpolation otherwise.
/// Returns the number of crossings written (up to `out.len()`).
#[verified_engine::verified]
pub fn threshold_crossings_into<P: PointSeries + ?Sized>(
    pts: &P,
    y_target: f64,
    out: &mut [f64],
) -> usize {
    threshold_crossings_into_raw(pts, y_target, out)
}

pub(super) fn threshold_crossings_into_raw<P: PointSeries + ?Sized>(
    pts: &P,
    y_target: f64,
    out: &mut [f64],
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
#[verified_engine::verified]
pub fn closest_threshold_crossing<P: PointSeries + ?Sized>(
    pts: &P,
    y_target: f64,
    x_ref: f64,
) -> Option<f64> {
    closest_threshold_crossing_raw(pts, y_target, x_ref)
}

fn closest_threshold_crossing_raw<P: PointSeries + ?Sized>(
    pts: &P,
    y_target: f64,
    x_ref: f64,
) -> Option<f64> {
    let n = pts.len();
    if n < 2 || !y_target.is_finite() || !x_ref.is_finite() {
        return None;
    }
    let mut best: Option<f64> = None;
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

/// Convenience wrapper collecting all sub-segment `y = y_target` crossings into a `Vec<f64>`.
pub fn interpolate_threshold_crossings<P: PointSeries + ?Sized>(
    pts: &P,
    y_target: f64,
) -> Vec<f64> {
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
pub(super) fn segment_threshold_hit<P: PointSeries + ?Sized>(
    pts: &P,
    i: usize,
    n: usize,
    p0: [f64; 2],
    p1: [f64; 2],
    y_target: f64,
) -> Option<f64> {
    let d0 = p0[1] - y_target;
    let d1 = p1[1] - y_target;
    if d0.abs() <= 1e-12 && d1.abs() <= 1e-12 {
        None
    } else if d0.abs() <= 1e-12 {
        Some(p0[0])
    } else if d1.abs() <= 1e-12 {
        (i + 2 == n).then_some(p1[0])
    } else if (d0 < 0.0 && d1 > 0.0) || (d0 > 0.0 && d1 < 0.0) {
        Some(crossing_on_segment(pts, i, p0, p1, y_target))
    } else {
        None
    }
}

pub(super) fn crossing_on_segment<P: PointSeries + ?Sized>(
    pts: &P,
    i: usize,
    p0: [f64; 2],
    p1: [f64; 2],
    y_target: f64,
) -> f64 {
    let dx = p1[0] - p0[0];
    if dx > 1e-12
        && let Some(xh) = hermite_root_on_segment(pts, i, p0, p1, y_target)
    {
        return xh;
    }
    let t = ((y_target - p0[1]) / (p1[1] - p0[1])).clamp(0.0, 1.0);
    p0[0] + t * dx
}

fn hermite_root_on_segment<P: PointSeries + ?Sized>(
    pts: &P,
    i: usize,
    p0: [f64; 2],
    p1: [f64; 2],
    y_target: f64,
) -> Option<f64> {
    let h = p1[0] - p0[0];
    if h <= 1e-12 {
        return None;
    }
    let sec = (p1[1] - p0[1]) / h;
    let m0 = endpoint_monotone_slope_left(pts, i, p0, sec)?;
    let m1 = endpoint_monotone_slope_right(pts, i, p1, sec)?;
    solve_hermite_bracket(p0, p1, h, m0, m1, y_target)
}

fn endpoint_monotone_slope_left<P: PointSeries + ?Sized>(
    pts: &P,
    i: usize,
    p0: [f64; 2],
    sec: f64,
) -> Option<f64> {
    if i == 0 {
        return Some(sec);
    }
    let pm1 = pts.point_at(i - 1);
    if !is_finite_pair(pm1) || p0[0] - pm1[0] <= 1e-12 {
        return None;
    }
    let s_prev = (p0[1] - pm1[1]) / (p0[0] - pm1[0]);
    Some(if s_prev * sec > 0.0 {
        0.5 * (s_prev + sec)
    } else {
        0.0
    })
}

fn endpoint_monotone_slope_right<P: PointSeries + ?Sized>(
    pts: &P,
    i: usize,
    p1: [f64; 2],
    sec: f64,
) -> Option<f64> {
    if i + 2 >= pts.len() {
        return Some(sec);
    }
    let p2 = pts.point_at(i + 2);
    if !is_finite_pair(p2) || p2[0] - p1[0] <= 1e-12 {
        return None;
    }
    let s_next = (p2[1] - p1[1]) / (p2[0] - p1[0]);
    Some(if sec * s_next > 0.0 {
        0.5 * (sec + s_next)
    } else {
        0.0
    })
}

fn solve_hermite_bracket(
    p0: [f64; 2],
    p1: [f64; 2],
    h: f64,
    mut m0: f64,
    mut m1: f64,
    y_target: f64,
) -> Option<f64> {
    let sec = (p1[1] - p0[1]) / h;
    if sec.abs() <= 1e-15 {
        m0 = 0.0;
        m1 = 0.0;
    } else {
        let a = m0 / sec;
        let b = m1 / sec;
        let hypot = (a * a + b * b).sqrt();
        if hypot > 3.0 {
            let tau = 3.0 / hypot;
            m0 = tau * a * sec;
            m1 = tau * b * sec;
        }
    }
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    let f_lo = p0[1] - y_target;
    let mut t = (-f_lo / (p1[1] - p0[1])).clamp(0.0, 1.0);
    for _ in 0..16 {
        let (ft, dft) = eval_hermite_val_deriv(p0[1], p1[1], h, m0, m1, t, y_target);
        if ft.abs() <= 1e-12 {
            break;
        }
        if (ft > 0.0) == (f_lo > 0.0) {
            lo = t;
        } else {
            hi = t;
        }
        let cand = if dft.abs() > 1e-14 {
            t - ft / dft
        } else {
            -1.0
        };
        t = if cand > lo && cand < hi {
            cand
        } else {
            0.5 * (lo + hi)
        };
    }
    Some(p0[0] + t.clamp(0.0, 1.0) * h)
}

#[inline(always)]
fn eval_hermite_val_deriv(
    y0: f64,
    y1: f64,
    h: f64,
    m0: f64,
    m1: f64,
    t: f64,
    y_target: f64,
) -> (f64, f64) {
    let t2 = t * t;
    let t3 = t2 * t;
    let val = (2.0 * t3 - 3.0 * t2 + 1.0) * y0
        + (t3 - 2.0 * t2 + t) * h * m0
        + (-2.0 * t3 + 3.0 * t2) * y1
        + (t3 - t2) * h * m1
        - y_target;
    let deriv = (6.0 * t2 - 6.0 * t) * y0
        + (3.0 * t2 - 4.0 * t + 1.0) * h * m0
        + (-6.0 * t2 + 6.0 * t) * y1
        + (3.0 * t2 - 2.0 * t) * h * m1;
    (val, deriv)
}

/// Writes interpolated `y` values where `pts` crosses vertical line `x = x_target` into `out`.
#[verified_engine::verified]
pub fn y_at_x_into<P: PointSeries + ?Sized>(pts: &P, x_target: f64, out: &mut [f64]) -> usize {
    y_at_x_into_raw(pts, x_target, out)
}

fn y_at_x_into_raw<P: PointSeries + ?Sized>(pts: &P, x_target: f64, out: &mut [f64]) -> usize {
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

/// Convenience wrapper collecting interpolated `y` values at `x = x_target` into a `Vec<f64>`.
pub fn interpolate_y_at_x<P: PointSeries + ?Sized>(pts: &P, x_target: f64) -> Vec<f64> {
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
pub(super) fn segment_vertical_hit(
    i: usize,
    n: usize,
    p0: [f64; 2],
    p1: [f64; 2],
    x_target: f64,
) -> Option<f64> {
    let d0 = p0[0] - x_target;
    let d1 = p1[0] - x_target;
    if d0.abs() <= 1e-12 && d1.abs() <= 1e-12 {
        None
    } else if d0.abs() <= 1e-12 {
        Some(p0[1])
    } else if d1.abs() <= 1e-12 {
        (i + 2 == n).then_some(p1[1])
    } else if (d0 < 0.0 && d1 > 0.0) || (d0 > 0.0 && d1 < 0.0) {
        let t = ((x_target - p0[0]) / (p1[0] - p0[0])).clamp(0.0, 1.0);
        Some(p0[1] + t * (p1[1] - p0[1]))
    } else {
        None
    }
}

#[inline(always)]
pub(super) fn is_finite_pair(p: [f64; 2]) -> bool {
    (p[0] * 0.0 + p[1] * 0.0) == 0.0
}

#[inline(always)]
fn push_unique_scalar(out: &mut [f64], written: usize, val: f64) -> usize {
    if written > 0 && (out[written - 1] - val).abs() <= 1e-9 {
        written
    } else {
        out[written] = val;
        written + 1
    }
}

#[inline(always)]
pub(super) fn push_unique_vec(out: &mut Vec<f64>, val: f64) {
    if out.last().is_none_or(|&prev| (prev - val).abs() > 1e-9) {
        out.push(val);
    }
}

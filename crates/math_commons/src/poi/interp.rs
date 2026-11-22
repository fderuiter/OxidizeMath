//! Sub-segment Hermite threshold crossings, vertical crossings, and parabolic extrema.
// [cite:math_commons]

use super::LocalExtremum;
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
    let n = pts.len();
    if n < 2 || out.is_empty() || !y_target.is_finite() {
        return 0;
    }
    let mut written = 0;
    for i in 0..(n - 1) {
        let p0 = pts.point_at(i);
        let p1 = pts.point_at(i + 1);
        if !is_finite_pair(p0) || !is_finite_pair(p1) {
            continue;
        }
        let d0 = p0[1] - y_target;
        let d1 = p1[1] - y_target;
        if d0.abs() <= 1e-12 && d1.abs() <= 1e-12 {
            continue;
        }
        let next_x = if d0.abs() <= 1e-12 {
            Some(p0[0])
        } else if d1.abs() <= 1e-12 {
            (i + 2 == n).then_some(p1[0])
        } else if (d0 < 0.0 && d1 > 0.0) || (d0 > 0.0 && d1 < 0.0) {
            Some(crossing_on_segment(pts, i, p0, p1, y_target))
        } else {
            None
        };
        if let Some(x_hit) = next_x {
            written = push_unique_scalar(out, written, x_hit);
            if written == out.len() {
                break;
            }
        }
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
    let n = pts.len();
    if n < 2 || !y_target.is_finite() || !x_ref.is_finite() {
        return None;
    }
    let mut best: Option<f64> = None;
    for i in 0..(n - 1) {
        let p0 = pts.point_at(i);
        let p1 = pts.point_at(i + 1);
        if !is_finite_pair(p0) || !is_finite_pair(p1) {
            continue;
        }
        let d0 = p0[1] - y_target;
        let d1 = p1[1] - y_target;
        if d0.abs() <= 1e-12 && d1.abs() <= 1e-12 {
            continue;
        }
        let candidate = if d0.abs() <= 1e-12 {
            Some(p0[0])
        } else if d1.abs() <= 1e-12 {
            (i + 2 == n).then_some(p1[0])
        } else if (d0 < 0.0 && d1 > 0.0) || (d0 > 0.0 && d1 < 0.0) {
            Some(crossing_on_segment(pts, i, p0, p1, y_target))
        } else {
            None
        };
        if let Some(x_hit) = candidate {
            let replace = best.is_none_or(|prev| (x_hit - x_ref).abs() < (prev - x_ref).abs());
            if replace {
                best = Some(x_hit);
            }
        }
    }
    best
}

/// Convenience wrapper collecting all sub-segment `y = y_target` crossings into a `Vec<f64>`.
pub fn interpolate_threshold_crossings<P: PointSeries + ?Sized>(
    pts: &P,
    y_target: f64,
) -> Vec<f64> {
    let cap = pts.len();
    if cap < 2 {
        return Vec::new();
    }
    let mut buf = vec![0.0; cap];
    let count = threshold_crossings_into(pts, y_target, &mut buf);
    buf.truncate(count);
    buf
}

#[verified_engine::verified]
fn crossing_on_segment<P: PointSeries + ?Sized>(
    pts: &P,
    i: usize,
    p0: [f64; 2],
    p1: [f64; 2],
    y_target: f64,
) -> f64 {
    let dx = p1[0] - p0[0];
    if dx > 1e-12
        && let Some(xh) = hermite_root_on_segment(pts, i, y_target)
    {
        return xh;
    }
    let t = ((y_target - p0[1]) / (p1[1] - p0[1])).clamp(0.0, 1.0);
    p0[0] + t * dx
}

#[verified_engine::verified]
fn hermite_root_on_segment<P: PointSeries + ?Sized>(
    pts: &P,
    i: usize,
    y_target: f64,
) -> Option<f64> {
    let p0 = pts.point_at(i);
    let p1 = pts.point_at(i + 1);
    let h = p1[0] - p0[0];
    if h <= 1e-12 {
        return None;
    }
    let sec = (p1[1] - p0[1]) / h;
    let m0 = endpoint_monotone_slope_left(pts, i, p0, sec)?;
    let m1 = endpoint_monotone_slope_right(pts, i, p1, sec)?;
    solve_hermite_bracket(p0, p1, h, m0, m1, y_target)
}

#[verified_engine::verified]
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

#[verified_engine::verified]
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

#[verified_engine::verified]
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

#[inline]
#[verified_engine::verified]
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
    let n = pts.len();
    if n < 2 || out.is_empty() || !x_target.is_finite() {
        return 0;
    }
    let mut written = 0;
    for i in 0..(n - 1) {
        let p0 = pts.point_at(i);
        let p1 = pts.point_at(i + 1);
        if !is_finite_pair(p0) || !is_finite_pair(p1) {
            continue;
        }
        let d0 = p0[0] - x_target;
        let d1 = p1[0] - x_target;
        if d0.abs() <= 1e-12 && d1.abs() <= 1e-12 {
            continue;
        }
        let next_y = if d0.abs() <= 1e-12 {
            Some(p0[1])
        } else if d1.abs() <= 1e-12 {
            (i + 2 == n).then_some(p1[1])
        } else if (d0 < 0.0 && d1 > 0.0) || (d0 > 0.0 && d1 < 0.0) {
            let t = ((x_target - p0[0]) / (p1[0] - p0[0])).clamp(0.0, 1.0);
            Some(p0[1] + t * (p1[1] - p0[1]))
        } else {
            None
        };
        if let Some(yh) = next_y {
            written = push_unique_scalar(out, written, yh);
            if written == out.len() {
                break;
            }
        }
    }
    written
}

/// Convenience wrapper collecting interpolated `y` values at `x = x_target` into a `Vec<f64>`.
pub fn interpolate_y_at_x<P: PointSeries + ?Sized>(pts: &P, x_target: f64) -> Vec<f64> {
    let cap = pts.len();
    if cap < 2 {
        return Vec::new();
    }
    let mut buf = vec![0.0; cap];
    let count = y_at_x_into(pts, x_target, &mut buf);
    buf.truncate(count);
    buf
}

/// Writes interior local extrema refined via 3-point parabolic interpolation into `out`.
#[verified_engine::verified]
pub fn detect_local_extrema_into<P: PointSeries + ?Sized>(
    pts: &P,
    out: &mut [LocalExtremum],
) -> usize {
    let n = pts.len();
    if n < 3 || out.is_empty() {
        return 0;
    }
    let mut written = 0;
    for i in 1..(n - 1) {
        let p0 = pts.point_at(i - 1);
        let p1 = pts.point_at(i);
        let p2 = pts.point_at(i + 1);
        if !is_finite_pair(p0) || !is_finite_pair(p1) || !is_finite_pair(p2) {
            continue;
        }
        let is_max = p1[1] >= p0[1] && p1[1] >= p2[1] && (p1[1] > p0[1] || p1[1] > p2[1]);
        let is_min = p1[1] <= p0[1] && p1[1] <= p2[1] && (p1[1] < p0[1] || p1[1] < p2[1]);
        if is_max || is_min {
            out[written] = LocalExtremum {
                point: parabolic_vertex(p0, p1, p2),
                is_max,
            };
            written += 1;
            if written == out.len() {
                break;
            }
        }
    }
    written
}

/// Convenience wrapper collecting interior local extrema into a `Vec<LocalExtremum>`.
pub fn detect_local_extrema<P: PointSeries + ?Sized>(pts: &P) -> Vec<LocalExtremum> {
    let cap = pts.len().saturating_sub(2);
    if cap == 0 {
        return Vec::new();
    }
    let mut buf = vec![
        LocalExtremum {
            point: [0.0, 0.0],
            is_max: true,
        };
        cap
    ];
    let count = detect_local_extrema_into(pts, &mut buf);
    buf.truncate(count);
    buf
}

/// Finds the global maximum in `pts`, refining interior peaks via [`parabolic_vertex`].
///
/// Executes with zero heap allocations on any [`PointSeries`].
#[verified_engine::verified]
pub fn peak_extremum<P: PointSeries + ?Sized>(pts: &P) -> Option<[f64; 2]> {
    let n = pts.len();
    if n == 0 {
        return None;
    }
    let mut best_idx: Option<usize> = None;
    let mut best_pt = [0.0_f64; 2];
    for i in 0..n {
        let p = pts.point_at(i);
        if is_finite_pair(p) && best_idx.is_none_or(|_| p[1] > best_pt[1]) {
            best_idx = Some(i);
            best_pt = p;
        }
    }
    let idx = best_idx?;
    if idx > 0 && idx + 1 < n {
        let p0 = pts.point_at(idx - 1);
        let p2 = pts.point_at(idx + 1);
        if is_finite_pair(p0)
            && is_finite_pair(p2)
            && p0[0] < best_pt[0]
            && best_pt[0] < p2[0]
            && (best_pt[1] > p0[1] || best_pt[1] > p2[1])
        {
            let [vx, vy] = parabolic_vertex(p0, best_pt, p2);
            return Some([vx, vy.max(best_pt[1])]);
        }
    }
    Some(best_pt)
}

/// Fits a 3-point Lagrange parabola through `(p0, p1, p2)` and returns its clamped vertex `[x, y]`.
#[verified_engine::verified]
pub fn parabolic_vertex(p0: [f64; 2], p1: [f64; 2], p2: [f64; 2]) -> [f64; 2] {
    let (x0, y0) = (p0[0], p0[1]);
    let (x1, y1) = (p1[0], p1[1]);
    let (x2, y2) = (p2[0], p2[1]);
    let denom = (x0 - x1) * (x0 - x2) * (x1 - x2);
    if denom.abs() <= 1e-15 || x2 <= x0 {
        return p1;
    }
    let a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom;
    let b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom;
    if a.abs() <= 1e-15 {
        return p1;
    }
    let vx = (-b / (2.0 * a)).clamp(x0, x2);
    let d01 = x0 - x1;
    let d02 = x0 - x2;
    let d12 = x1 - x2;
    let l0 = ((vx - x1) * (vx - x2)) / (d01 * d02);
    let l1 = ((vx - x0) * (vx - x2)) / (-d01 * d12);
    let l2 = ((vx - x0) * (vx - x1)) / (d02 * d12);
    let vy = y0 * l0 + y1 * l1 + y2 * l2;
    if vx.is_finite() && vy.is_finite() {
        [vx, vy]
    } else {
        p1
    }
}

#[inline]
#[verified_engine::verified]
fn is_finite_pair(p: [f64; 2]) -> bool {
    p[0].is_finite() && p[1].is_finite()
}

#[inline]
#[verified_engine::verified]
fn push_unique_scalar(out: &mut [f64], written: usize, val: f64) -> usize {
    if written > 0 && (out[written - 1] - val).abs() <= 1e-9 {
        written
    } else {
        out[written] = val;
        written + 1
    }
}

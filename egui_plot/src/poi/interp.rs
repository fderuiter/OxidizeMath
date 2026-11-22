//! Sub-segment cubic Hermite, secant, and parabolic interpolation for critical points.

use crate::items::PlotPoint;

/// Locate all $x$ positions where a polyline crosses `y_target` (including step drops).
pub fn interpolate_threshold_crossings(pts: &[PlotPoint], y_target: f64) -> Vec<f64> {
    if pts.len() < 2 || !y_target.is_finite() {
        return Vec::new();
    }
    let mut roots = Vec::new();
    let n = pts.len();
    for i in 0..n - 1 {
        let p0 = pts[i];
        let p1 = pts[i + 1];
        if !is_finite_pt(p0) || !is_finite_pt(p1) {
            continue;
        }
        let f0 = p0.y - y_target;
        let f1 = p1.y - y_target;
        if f0 == 0.0 && f1 == 0.0 {
            continue;
        }
        if f0 == 0.0 {
            roots.push(p0.x);
            continue;
        }
        if (f0 > 0.0 && f1 < 0.0) || (f0 < 0.0 && f1 > 0.0) {
            if let Some(x) = solve_segment_crossing(pts, i, y_target) {
                roots.push(x);
            }
        } else if i + 2 == n && f1 == 0.0 {
            roots.push(p1.x);
        }
    }
    roots
}

/// Evaluate or interpolate all $y$ values where a polyline crosses `x_target`.
pub fn interpolate_y_at_x(pts: &[PlotPoint], x_target: f64) -> Vec<f64> {
    if pts.len() < 2 || !x_target.is_finite() {
        return Vec::new();
    }
    let mut values = Vec::new();
    let n = pts.len();
    for i in 0..n - 1 {
        let p0 = pts[i];
        let p1 = pts[i + 1];
        if !is_finite_pt(p0) || !is_finite_pt(p1) {
            continue;
        }
        let dx0 = p0.x - x_target;
        let dx1 = p1.x - x_target;
        if dx0 == 0.0 && dx1 == 0.0 {
            continue;
        }
        if dx0 == 0.0 {
            values.push(p0.y);
            continue;
        }
        if (dx0 > 0.0 && dx1 < 0.0) || (dx0 < 0.0 && dx1 > 0.0) {
            if let Some(y) = eval_segment_at_x(pts, i, x_target) {
                values.push(y);
            }
        } else if i + 2 == n && dx1 == 0.0 {
            values.push(p1.y);
        }
    }
    values
}

/// Detect local maxima and minima via three-point parabolic interpolation.
pub(crate) fn detect_local_extrema(pts: &[PlotPoint]) -> Vec<([f64; 2], bool)> {
    if pts.len() < 3 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for i in 1..pts.len() - 1 {
        let (p0, p1, p2) = (pts[i - 1], pts[i], pts[i + 1]);
        if !is_finite_pt(p0) || !is_finite_pt(p1) || !is_finite_pt(p2) {
            continue;
        }
        let x_inc = p0.x < p1.x && p1.x < p2.x;
        let x_dec = p0.x > p1.x && p1.x > p2.x;
        if !x_inc && !x_dec {
            continue;
        }
        let Some(next_y) = next_distinct_y(pts, i) else {
            continue;
        };
        let is_max = p1.y > p0.y && p1.y > next_y && p1.y >= p2.y;
        let is_min = p1.y < p0.y && p1.y < next_y && p1.y <= p2.y;
        if is_max || is_min {
            out.push((parabolic_vertex(p0, p1, p2), is_max));
        }
    }
    out
}

fn next_distinct_y(pts: &[PlotPoint], idx: usize) -> Option<f64> {
    let base_y = pts[idx].y;
    for pt in &pts[idx + 1..] {
        if !is_finite_pt(*pt) {
            return None;
        }
        if (pt.y - base_y).abs() > 0.0 {
            return Some(pt.y);
        }
    }
    None
}

fn parabolic_vertex(p0: PlotPoint, p1: PlotPoint, p2: PlotPoint) -> [f64; 2] {
    let (q0, q2) = if p0.x < p2.x { (p0, p2) } else { (p2, p0) };
    let u0 = q0.x - p1.x;
    let u2 = q2.x - p1.x;
    let d0 = q0.y - p1.y;
    let d2 = q2.y - p1.y;
    let denom = u0 * u2 * (u2 - u0);
    if !denom.is_finite() || denom.abs() <= 1e-18 {
        return [p1.x, p1.y];
    }
    let a = (u0 * d2 - u2 * d0) / denom;
    let b = (u2 * u2 * d0 - u0 * u0 * d2) / denom;
    if !a.is_finite() || !b.is_finite() || a.abs() <= 1e-15 {
        return [p1.x, p1.y];
    }
    let u_vertex = (-b / (2.0 * a)).clamp(u0, u2);
    let x_v = p1.x + u_vertex;
    let y_v = p1.y + b * u_vertex + a * u_vertex * u_vertex;
    if x_v.is_finite() && y_v.is_finite() {
        [x_v, y_v]
    } else {
        [p1.x, p1.y]
    }
}

fn solve_segment_crossing(pts: &[PlotPoint], idx: usize, y_target: f64) -> Option<f64> {
    let p0 = pts[idx];
    let p1 = pts[idx + 1];
    let dx = p1.x - p0.x;
    if dx.abs() <= 1e-12 {
        return Some(p0.x);
    }
    let f0 = p0.y - y_target;
    let f1 = p1.y - y_target;
    let dy = f1 - f0;
    if dy.abs() <= 1e-18 {
        return None;
    }
    let s_secant = (-f0 / dy).clamp(0.0, 1.0);
    let Some((m0, m1)) = monotone_hermite_slopes(pts, idx) else {
        let x = p0.x + s_secant * dx;
        return x.is_finite().then_some(x);
    };
    let s = solve_monotone_hermite_root(f0, f1, m0 * dx, m1 * dx, s_secant);
    let x = p0.x + s * dx;
    x.is_finite().then_some(x)
}

pub(crate) fn eval_segment_at_x(pts: &[PlotPoint], idx: usize, x_target: f64) -> Option<f64> {
    let p0 = pts[idx];
    let p1 = pts[idx + 1];
    let dx = p1.x - p0.x;
    if dx.abs() <= 1e-12 {
        return Some(0.5 * (p0.y + p1.y));
    }
    let s = ((x_target - p0.x) / dx).clamp(0.0, 1.0);
    if let Some((m0, m1)) = monotone_hermite_slopes(pts, idx) {
        let y = eval_hermite(p0.y, p1.y, m0 * dx, m1 * dx, s);
        if y.is_finite() {
            return Some(y);
        }
    }
    let y = p0.y + s * (p1.y - p0.y);
    y.is_finite().then_some(y)
}

fn monotone_hermite_slopes(pts: &[PlotPoint], idx: usize) -> Option<(f64, f64)> {
    let p0 = pts[idx];
    let p1 = pts[idx + 1];
    let dx = p1.x - p0.x;
    if dx <= 1e-12 {
        return None;
    }
    let secant = (p1.y - p0.y) / dx;
    if !secant.is_finite() || secant.abs() <= 1e-18 {
        return Some((0.0, 0.0));
    }
    let prev = idx
        .checked_sub(1)
        .map(|j| pts[j])
        .filter(|p| is_finite_pt(*p) && p.x < p0.x);
    let next = pts
        .get(idx + 2)
        .copied()
        .filter(|p| is_finite_pt(*p) && p.x > p1.x);
    if prev.is_none() && next.is_none() {
        return None;
    }
    let raw_m0 = estimate_endpoint_slope(prev, p0, p1, next, true);
    let raw_m1 = estimate_endpoint_slope(prev, p0, p1, next, false);
    Some(clamp_fritsch_carlson(raw_m0, raw_m1, secant))
}

fn estimate_endpoint_slope(
    prev: Option<PlotPoint>,
    p0: PlotPoint,
    p1: PlotPoint,
    next: Option<PlotPoint>,
    at_left: bool,
) -> f64 {
    let h = p1.x - p0.x;
    let sec = (p1.y - p0.y) / h;
    if at_left {
        if let Some(pm) = prev {
            let h_prev = p0.x - pm.x;
            let sec_prev = (p0.y - pm.y) / h_prev;
            return (h * sec_prev + h_prev * sec) / (h_prev + h);
        }
        if let Some(pn) = next {
            let h_next = pn.x - p1.x;
            let sec_next = (pn.y - p1.y) / h_next;
            return ((2.0 * h + h_next) * sec - h * sec_next) / (h + h_next);
        }
    } else {
        if let Some(pn) = next {
            let h_next = pn.x - p1.x;
            let sec_next = (pn.y - p1.y) / h_next;
            return (h_next * sec + h * sec_next) / (h + h_next);
        }
        if let Some(pm) = prev {
            let h_prev = p0.x - pm.x;
            let sec_prev = (p0.y - pm.y) / h_prev;
            return ((2.0 * h + h_prev) * sec - h * sec_prev) / (h_prev + h);
        }
    }
    sec
}

fn clamp_fritsch_carlson(mut m0: f64, mut m1: f64, secant: f64) -> (f64, f64) {
    if !m0.is_finite() || m0 * secant <= 0.0 {
        m0 = 0.0;
    }
    if !m1.is_finite() || m1 * secant <= 0.0 {
        m1 = 0.0;
    }
    let alpha = m0 / secant;
    let beta = m1 / secant;
    let norm_sq = alpha * alpha + beta * beta;
    if norm_sq > 9.0 {
        let tau = 3.0 / norm_sq.sqrt();
        m0 = tau * alpha * secant;
        m1 = tau * beta * secant;
    }
    (m0, m1)
}

fn eval_hermite(y0: f64, y1: f64, t0: f64, t1: f64, s: f64) -> f64 {
    let s2 = s * s;
    let s3 = s2 * s;
    (2.0 * s3 - 3.0 * s2 + 1.0) * y0
        + (s3 - 2.0 * s2 + s) * t0
        + (-2.0 * s3 + 3.0 * s2) * y1
        + (s3 - s2) * t1
}

fn eval_hermite_deriv(y0: f64, y1: f64, t0: f64, t1: f64, s: f64) -> f64 {
    let s2 = s * s;
    (6.0 * s2 - 6.0 * s) * (y0 - y1) + (3.0 * s2 - 4.0 * s + 1.0) * t0 + (3.0 * s2 - 2.0 * s) * t1
}

fn solve_monotone_hermite_root(f0: f64, f1: f64, t0: f64, t1: f64, s_init: f64) -> f64 {
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    let mut s = s_init.clamp(0.0, 1.0);
    for _ in 0..12 {
        let val = eval_hermite(f0, f1, t0, t1, s);
        if !val.is_finite() || val.abs() <= 1e-14 {
            return s;
        }
        if (val > 0.0) == (f0 > 0.0) {
            lo = s;
        } else {
            hi = s;
        }
        let deriv = eval_hermite_deriv(f0, f1, t0, t1, s);
        let next = if deriv.is_finite() && deriv.abs() > 1e-14 {
            s - val / deriv
        } else {
            0.5 * (lo + hi)
        };
        s = if next > lo && next < hi {
            next
        } else {
            0.5 * (lo + hi)
        };
    }
    s
}

#[inline]
pub(crate) fn is_finite_pt(pt: PlotPoint) -> bool {
    pt.x.is_finite() && pt.y.is_finite()
}

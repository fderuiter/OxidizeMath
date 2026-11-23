//! Safeguarded zero-allocation cubic Hermite interpolation and Newton-bisection root-bracketing.
// [cite:engine_diffeq]

use crate::error::DiffeqError;
use crate::math::hypot_f64;

/// Evaluates the scalar cubic Hermite polynomial on normalized coordinate $s \in [0, 1]$ over step `dt`,
/// given endpoint values `(v0, v1)` and time/abscissa derivatives `(d0, d1)`.
#[must_use]
#[inline]
#[verified_engine::verified]
pub fn cubic_hermite_scalar(s: f64, dt: f64, v0: f64, v1: f64, d0: f64, d1: f64) -> f64 {
    let s2 = s * s;
    let s3 = s2 * s;
    let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    let h10 = s3 - 2.0 * s2 + s;
    let h01 = -2.0 * s3 + 3.0 * s2;
    let h11 = s3 - s2;
    h00 * v0 + h10 * (dt * d0) + h01 * v1 + h11 * (dt * d1)
}

/// Evaluates the derivative $\frac{dP}{ds}$ of the cubic Hermite interpolant with respect to normalized $s \in [0, 1]$.
#[must_use]
#[inline]
#[verified_engine::verified]
pub fn cubic_hermite_slope(s: f64, dt: f64, v0: f64, v1: f64, d0: f64, d1: f64) -> f64 {
    let s2 = s * s;
    let dh00 = 6.0 * s2 - 6.0 * s;
    let dh10 = 3.0 * s2 - 4.0 * s + 1.0;
    let dh01 = -6.0 * s2 + 6.0 * s;
    let dh11 = 3.0 * s2 - 2.0 * s;
    dh00 * v0 + dh10 * (dt * d0) + dh01 * v1 + dh11 * (dt * d1)
}

/// Evaluates both value $P(s)$ and normalized slope $\frac{dP}{ds}(s)$ of the cubic Hermite polynomial in one pass.
#[must_use]
#[inline]
#[verified_engine::verified]
pub fn cubic_hermite_val_deriv(s: f64, dt: f64, v0: f64, v1: f64, d0: f64, d1: f64) -> (f64, f64) {
    (
        cubic_hermite_scalar(s, dt, v0, v1, d0, d1),
        cubic_hermite_slope(s, dt, v0, v1, d0, d1),
    )
}

/// Clamps endpoint slopes `(m0, m1)` to preserve monotonicity via Fritsch & Carlson's $\alpha^2 + \beta^2 \le 9$ circle.
#[must_use]
#[verified_engine::verified]
pub fn clamp_fritsch_carlson_slopes(
    h: f64,
    y0: f64,
    y1: f64,
    mut m0: f64,
    mut m1: f64,
) -> (f64, f64) {
    if !h.is_finite() || h.abs() <= 1e-30 {
        return (0.0, 0.0);
    }
    let sec = (y1 - y0) / h;
    if sec.abs() <= 1e-15 {
        return (0.0, 0.0);
    }
    let a = m0 / sec;
    let b = m1 / sec;
    let rad = hypot_f64(a, b);
    if rad > 3.0 {
        let tau = 3.0 / rad;
        m0 = tau * a * sec;
        m1 = tau * b * sec;
    }
    (m0, m1)
}

#[verified_engine::verified]
fn newton_bisect_hermite_loop(
    dt: f64,
    g0: f64,
    g1: f64,
    dg0: f64,
    dg1: f64,
    tol: f64,
    iters: usize,
) -> (f64, f64, f64) {
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    let mut g_lo = g0;
    let mut s = (-g0 / (g1 - g0)).clamp(0.0, 1.0);
    for _ in 0..iters {
        let (f, df) = cubic_hermite_val_deriv(s, dt, g0, g1, dg0, dg1);
        if f.abs() <= tol || (hi - lo).abs() <= tol {
            return (s.clamp(0.0, 1.0), f.abs(), (hi - lo).abs());
        }
        if (f > 0.0) == (g_lo > 0.0) {
            lo = s;
            g_lo = f;
        } else {
            hi = s;
        }
        let candidate = if df.abs() > 1e-14 { s - f / df } else { -1.0 };
        s = if candidate > lo && candidate < hi {
            candidate
        } else {
            0.5 * (lo + hi)
        };
    }
    let f_end = cubic_hermite_scalar(s, dt, g0, g1, dg0, dg1).abs();
    (s.clamp(0.0, 1.0), f_end, (hi - lo).abs())
}

/// Validates and solves for the normalized root $s \in [0, 1]$ of a bracketed cubic Hermite polynomial
/// using safeguarded Newton-bisection with zero heap allocations.
#[verified_engine::verified]
pub fn solve_hermite_unit_bracket(
    dt: f64,
    g0: f64,
    g1: f64,
    dg0: f64,
    dg1: f64,
    tol: f64,
    max_iter: usize,
) -> Result<f64, DiffeqError> {
    if !dt.is_finite() || dt == 0.0 {
        return Err(DiffeqError::InvalidStepSize { dt });
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(DiffeqError::InvalidTolerance { tol });
    }
    if !g0.is_finite() || !g1.is_finite() || !dg0.is_finite() || !dg1.is_finite() {
        return Err(DiffeqError::NonFiniteEvaluation);
    }
    if g0 == 0.0 {
        return Ok(0.0);
    }
    if g1 == 0.0 {
        return Ok(1.0);
    }
    if (g0 > 0.0) == (g1 > 0.0) {
        return Err(DiffeqError::NotBracketed { fa: g0, fb: g1 });
    }
    let iters = if max_iter == 0 { 16 } else { max_iter };
    let (s, _, _) = newton_bisect_hermite_loop(dt, g0, g1, dg0, dg1, tol, iters);
    Ok(s)
}

/// Solves for the normalized crossing fraction $s \in [0, 1]$ of a cubic Hermite trajectory segment,
/// returning `0.5` when endpoints are nearly flat (`|g1 - g0| <= 1e-15`).
#[must_use]
#[verified_engine::verified]
pub fn solve_hermite_root(dt: f64, g0: f64, g1: f64, dg0: f64, dg1: f64) -> f64 {
    if (g1 - g0).abs() <= 1e-15 {
        return 0.5;
    }
    if let Ok(s) = solve_hermite_unit_bracket(dt, g0, g1, dg0, dg1, 1e-14, 16) {
        return s;
    }
    newton_bisect_hermite_loop(dt, g0, g1, dg0, dg1, 1e-14, 16).0
}

/// Solves for the abscissa $x \in [x_0, x_1]$ (where `p0 = [x0, y0]` and `p1 = [x1, y1]`) where a
/// Fritsch-Carlson monotone cubic Hermite segment crosses `y_target`.
#[must_use]
#[verified_engine::verified]
pub fn solve_monotone_hermite_crossing(
    p0: [f64; 2],
    p1: [f64; 2],
    h: f64,
    m0: f64,
    m1: f64,
    y_target: f64,
) -> Option<f64> {
    if !h.is_finite() || h <= 0.0 {
        return None;
    }
    let (cm0, cm1) = clamp_fritsch_carlson_slopes(h, p0[1], p1[1], m0, m1);
    let g0 = p0[1] - y_target;
    let g1 = p1[1] - y_target;
    let s = match solve_hermite_unit_bracket(h, g0, g1, cm0, cm1, 1e-12, 16) {
        Ok(val) => val,
        Err(_) => solve_hermite_root(h, g0, g1, cm0, cm1),
    };
    Some(p0[0] + s * h)
}

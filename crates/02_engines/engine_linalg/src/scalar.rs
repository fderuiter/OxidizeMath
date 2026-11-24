#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:engine_linalg]
// [cite:advanced_linear_algebra]
//! `#![no_std]` unordered [`FieldScalar`] (`f32`, `f64`, [`Complex<f32>`], [`Complex<f64>`])
//! and ordered [`RealScalar`] / [`Scalar`] (`f32`, `f64`) abstractions re-exported from [`math_scalars`].

pub use math_scalars::{
    Complex, Complex32, Complex64, FieldScalar, RealScalar, RealScalar as Scalar,
};

/// Pure `core` IEEE-754 square root for `f64` using exponent halving and five Heron steps.
/// `[cite:engine_linalg]`
#[inline]
#[must_use]
pub fn sqrt_f64(x: f64) -> f64 {
    if x == 0.0 {
        return x;
    }
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if !x.is_finite() {
        return x;
    }
    if x < f64::MIN_POSITIVE {
        const SCALE_UP: f64 = 4_503_599_627_370_496.0;
        const SCALE_DOWN: f64 = 1.0 / 67_108_864.0;
        return sqrt_f64_normal(x * SCALE_UP) * SCALE_DOWN;
    }
    sqrt_f64_normal(x)
}

#[inline]
fn sqrt_f64_normal(x: f64) -> f64 {
    let mut g = f64::from_bits((x.to_bits() >> 1) + 0x1ff8_0000_0000_0000);
    g = 0.5 * (g + x / g);
    g = 0.5 * (g + x / g);
    g = 0.5 * (g + x / g);
    g = 0.5 * (g + x / g);
    0.5 * (g + x / g)
}

/// Pure `core` overflow-safe Euclidean hypotenuse $\sqrt{x^2 + y^2}$ for `f64`.
/// `[cite:engine_linalg]`
#[inline]
#[must_use]
pub fn hypot_f64(x: f64, y: f64) -> f64 {
    if x.is_infinite() || y.is_infinite() {
        return f64::INFINITY;
    }
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    let ay = y.abs();
    let (hi, lo) = if ax >= ay { (ax, ay) } else { (ay, ax) };
    if hi == 0.0 {
        return 0.0;
    }
    let r = lo / hi;
    hi * sqrt_f64(1.0 + r * r)
}

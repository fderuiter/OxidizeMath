// [cite:math_scalars]
//! Portable IEEE-754 bit-level directed rounding and error-free bound helpers.
//!
//! Operates purely on IEEE-754 binary32/binary64 bit representations and deterministic
//! `libm` fused multiply-add residuals without mutating hardware FPU rounding modes,
//! ensuring identical outward-rounded bounds on `x86_64` and `wasm32-unknown-unknown`.

use crate::traits::RealScalar;

/// Smallest IEEE-754 binary64 value strictly greater than `x`.
#[inline]
#[must_use]
pub fn next_up_f64(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    let bits = x.to_bits();
    if bits == 0x8000_0000_0000_0001 {
        return 0.0;
    }
    let next = if x > 0.0 { bits + 1 } else { bits - 1 };
    let out = f64::from_bits(next);
    debug_assert!(out > x);
    out
}

/// Largest IEEE-754 binary64 value strictly less than `x`.
#[inline]
#[must_use]
pub fn next_down_f64(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return x;
    }
    let out = -next_up_f64(-x);
    debug_assert!(out < x);
    out
}

/// Smallest IEEE-754 binary32 value strictly greater than `x`.
#[inline]
#[must_use]
pub fn next_up_f32(x: f32) -> f32 {
    if x.is_nan() || x == f32::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f32::from_bits(1);
    }
    let bits = x.to_bits();
    if bits == 0x8000_0001 {
        return 0.0;
    }
    let next = if x > 0.0 { bits + 1 } else { bits - 1 };
    let out = f32::from_bits(next);
    debug_assert!(out > x);
    out
}

/// Largest IEEE-754 binary32 value strictly less than `x`.
#[inline]
#[must_use]
pub fn next_down_f32(x: f32) -> f32 {
    if x.is_nan() || x == f32::NEG_INFINITY {
        return x;
    }
    let out = -next_up_f32(-x);
    debug_assert!(out < x);
    out
}

/// Steps `x` upward by `ulps` IEEE-754 binary64 representable increments.
#[inline]
#[must_use]
pub fn step_up_f64(mut x: f64, ulps: u32) -> f64 {
    for _ in 0..ulps {
        x = next_up_f64(x);
    }
    x
}

/// Steps `x` downward by `ulps` IEEE-754 binary64 representable increments.
#[inline]
#[must_use]
pub fn step_down_f64(mut x: f64, ulps: u32) -> f64 {
    for _ in 0..ulps {
        x = next_down_f64(x);
    }
    x
}

/// Steps `x` upward by `ulps` IEEE-754 binary32 representable increments.
#[inline]
#[must_use]
pub fn step_up_f32(mut x: f32, ulps: u32) -> f32 {
    for _ in 0..ulps {
        x = next_up_f32(x);
    }
    x
}

/// Steps `x` downward by `ulps` IEEE-754 binary32 representable increments.
#[inline]
#[must_use]
pub fn step_down_f32(mut x: f32, ulps: u32) -> f32 {
    for _ in 0..ulps {
        x = next_down_f32(x);
    }
    x
}

/// Ordered IEEE-754 binary64 ULP distance between `a` and `b` (`-0.0` and `+0.0` have distance `0`).
///
/// Returns `u64::MAX` if either operand is `NaN`.
#[inline]
#[must_use]
pub fn ulp_distance_f64(a: f64, b: f64) -> u64 {
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    let rank = |v: f64| -> u64 {
        if v == 0.0 {
            0x8000_0000_0000_0000
        } else if v > 0.0 {
            0x8000_0000_0000_0000 + v.to_bits()
        } else {
            0x8000_0000_0000_0000 - (v.to_bits() & 0x7fff_ffff_ffff_ffff)
        }
    };
    rank(a).abs_diff(rank(b))
}

/// Ordered IEEE-754 binary32 ULP distance between `a` and `b` (`-0.0` and `+0.0` have distance `0`).
///
/// Returns `u64::MAX` if either operand is `NaN`.
#[inline]
#[must_use]
pub fn ulp_distance_f32(a: f32, b: f32) -> u64 {
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    let rank = |v: f32| -> u64 {
        if v == 0.0 {
            0x8000_0000
        } else if v > 0.0 {
            0x8000_0000 + u64::from(v.to_bits())
        } else {
            0x8000_0000 - u64::from(v.to_bits() & 0x7fff_ffff)
        }
    };
    rank(a).abs_diff(rank(b))
}

/// Tight directed-rounding `(lo, hi)` bounds for `a + b` via Knuth TwoSum residual inspection.
#[inline]
pub(crate) fn add_bounds<T: RealScalar>(a: T, b: T) -> (T, T) {
    let s = a + b;
    if s.is_nan() {
        return (T::NAN, T::NAN);
    }
    if !s.is_finite() {
        if a.is_infinite() || b.is_infinite() {
            return (s, s);
        }
        return if s > T::ZERO {
            (T::MAX_VAL, T::INFINITY)
        } else {
            (T::NEG_INFINITY, -T::MAX_VAL)
        };
    }
    let a_prime = s - b;
    let b_prime = s - a_prime;
    let err = (a - a_prime) + (b - b_prime);
    let lo = if err < T::ZERO { s.next_down() } else { s };
    let hi = if err > T::ZERO { s.next_up() } else { s };
    (lo, hi)
}

/// Tight directed-rounding `(lo, hi)` bounds for `a - b`.
#[inline]
pub(crate) fn sub_bounds<T: RealScalar>(a: T, b: T) -> (T, T) {
    add_bounds(a, -b)
}

/// Tight directed-rounding `(lo, hi)` bounds for `a * b` via deterministic `libm` FMA residual.
#[inline]
pub(crate) fn mul_bounds<T: RealScalar>(a: T, b: T) -> (T, T) {
    let p = a * b;
    if p.is_nan() {
        return (T::NAN, T::NAN);
    }
    if a == T::ZERO || b == T::ZERO {
        return (T::ZERO, T::ZERO);
    }
    if p.is_finite() && p.abs() >= T::MIN_POSITIVE {
        let err = a.fma(b, -p);
        if err.is_finite() {
            let lo = if err < T::ZERO { p.next_down() } else { p };
            let hi = if err > T::ZERO { p.next_up() } else { p };
            return (lo, hi);
        }
    }
    if !p.is_finite() {
        if a.is_infinite() || b.is_infinite() {
            return (p, p);
        }
        return if p > T::ZERO {
            (T::MAX_VAL, T::INFINITY)
        } else {
            (T::NEG_INFINITY, -T::MAX_VAL)
        };
    }
    (p.next_down(), p.next_up())
}

/// Tight directed-rounding `(lo, hi)` bounds for `a / b` (`b != 0`).
#[inline]
pub(crate) fn div_bounds<T: RealScalar>(a: T, b: T) -> (T, T) {
    let q = a / b;
    if q.is_nan() {
        return (T::NAN, T::NAN);
    }
    if a == T::ZERO && b.is_finite() && b != T::ZERO {
        return (T::ZERO, T::ZERO);
    }
    if q.is_finite()
        && a.is_finite()
        && b.is_finite()
        && q.abs() >= T::MIN_POSITIVE
        && q.abs() <= T::MAX_VAL * T::HALF
    {
        let rem = (-q).fma(b, a);
        if rem.is_finite() {
            let signed_rem = if b > T::ZERO { rem } else { -rem };
            let lo = if signed_rem < T::ZERO {
                q.next_down()
            } else {
                q
            };
            let hi = if signed_rem > T::ZERO { q.next_up() } else { q };
            return (lo, hi);
        }
    }
    if !q.is_finite() {
        if a.is_infinite() || b == T::ZERO {
            return (q, q);
        }
        return if q > T::ZERO {
            (T::MAX_VAL, T::INFINITY)
        } else {
            (T::NEG_INFINITY, -T::MAX_VAL)
        };
    }
    (q.next_down(), q.next_up())
}

/// Tight directed-rounding `(lo, hi)` bounds for `sqrt(a)` (`a >= 0`).
#[inline]
pub(crate) fn sqrt_bounds<T: RealScalar>(a: T) -> (T, T) {
    if a.is_nan() || a < T::ZERO {
        return (T::NAN, T::NAN);
    }
    if a == T::ZERO || !a.is_finite() {
        return (a, a);
    }
    let s = a.sqrt();
    if s >= T::MIN_POSITIVE && a <= T::MAX_VAL * T::HALF {
        let rem = (-s).fma(s, a);
        if rem.is_finite() {
            let lo = if rem < T::ZERO { s.next_down() } else { s };
            let hi = if rem > T::ZERO { s.next_up() } else { s };
            return (lo, hi);
        }
    }
    (s.next_down().max_val(T::ZERO), s.next_up())
}

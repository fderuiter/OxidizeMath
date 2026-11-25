//! Portable IEEE-754 bit-level directed ULP stepping and distance utilities.
//!
//! Avoids FPU rounding-mode mutations (`fesetround`) so native (`x86_64`, `aarch64`)
//! and `wasm32-unknown-unknown` targets produce identical outward-rounded bounds.
// [cite:math_commons]

const F64_TINY_BITS: u64 = 0x1;
const F64_NEG_TINY_BITS: u64 = 0x8000_0000_0000_0001;
const F64_SIGN_MASK: u64 = 0x8000_0000_0000_0000;

const F32_TINY_BITS: u32 = 0x1;
const F32_NEG_TINY_BITS: u32 = 0x8000_0001;
const F32_SIGN_MASK: u32 = 0x8000_0000;

/// Returns the least `f64` value strictly greater than `x` in IEEE-754 order.
#[inline]
#[must_use]
pub fn next_up_f64(x: f64) -> f64 {
    let bits = x.to_bits();
    if x.is_nan() || bits == f64::INFINITY.to_bits() {
        return x;
    }
    let abs = bits & !F64_SIGN_MASK;
    let next_bits = if abs == 0 {
        F64_TINY_BITS
    } else if bits == abs {
        bits + 1
    } else {
        bits - 1
    };
    f64::from_bits(next_bits)
}

/// Returns the greatest `f64` value strictly less than `x` in IEEE-754 order.
#[inline]
#[must_use]
pub fn next_down_f64(x: f64) -> f64 {
    let bits = x.to_bits();
    if x.is_nan() || bits == f64::NEG_INFINITY.to_bits() {
        return x;
    }
    let abs = bits & !F64_SIGN_MASK;
    let next_bits = if abs == 0 {
        F64_NEG_TINY_BITS
    } else if bits == abs {
        bits - 1
    } else {
        bits + 1
    };
    f64::from_bits(next_bits)
}

/// Returns the least `f32` value strictly greater than `x` in IEEE-754 order.
#[inline]
#[must_use]
pub fn next_up_f32(x: f32) -> f32 {
    let bits = x.to_bits();
    if x.is_nan() || bits == f32::INFINITY.to_bits() {
        return x;
    }
    let abs = bits & !F32_SIGN_MASK;
    let next_bits = if abs == 0 {
        F32_TINY_BITS
    } else if bits == abs {
        bits + 1
    } else {
        bits - 1
    };
    f32::from_bits(next_bits)
}

/// Returns the greatest `f32` value strictly less than `x` in IEEE-754 order.
#[inline]
#[must_use]
pub fn next_down_f32(x: f32) -> f32 {
    let bits = x.to_bits();
    if x.is_nan() || bits == f32::NEG_INFINITY.to_bits() {
        return x;
    }
    let abs = bits & !F32_SIGN_MASK;
    let next_bits = if abs == 0 {
        F32_NEG_TINY_BITS
    } else if bits == abs {
        bits - 1
    } else {
        bits + 1
    };
    f32::from_bits(next_bits)
}

/// Steps `x` upward by `steps` IEEE-754 ULPs (`f64`).
#[inline]
#[must_use]
pub fn pad_ulp_up_f64(mut x: f64, steps: u32) -> f64 {
    if x.is_nan() {
        return f64::INFINITY;
    }
    let mut i = 0;
    while i < steps {
        x = next_up_f64(x);
        i += 1;
    }
    x
}

/// Steps `x` downward by `steps` IEEE-754 ULPs (`f64`).
#[inline]
#[must_use]
pub fn pad_ulp_down_f64(mut x: f64, steps: u32) -> f64 {
    if x.is_nan() {
        return f64::NEG_INFINITY;
    }
    let mut i = 0;
    while i < steps {
        x = next_down_f64(x);
        i += 1;
    }
    x
}

/// Steps `x` upward by `steps` IEEE-754 ULPs (`f32`).
#[inline]
#[must_use]
pub fn pad_ulp_up_f32(mut x: f32, steps: u32) -> f32 {
    if x.is_nan() {
        return f32::INFINITY;
    }
    let mut i = 0;
    while i < steps {
        x = next_up_f32(x);
        i += 1;
    }
    x
}

/// Steps `x` downward by `steps` IEEE-754 ULPs (`f32`).
#[inline]
#[must_use]
pub fn pad_ulp_down_f32(mut x: f32, steps: u32) -> f32 {
    if x.is_nan() {
        return f32::NEG_INFINITY;
    }
    let mut i = 0;
    while i < steps {
        x = next_down_f32(x);
        i += 1;
    }
    x
}

#[inline]
fn ordered_key_f64(x: f64) -> u64 {
    let bits = x.to_bits();
    if (bits & F64_SIGN_MASK) != 0 {
        F64_SIGN_MASK.wrapping_sub(bits & !F64_SIGN_MASK)
    } else {
        bits | F64_SIGN_MASK
    }
}

#[inline]
fn ordered_key_f32(x: f32) -> u32 {
    let bits = x.to_bits();
    if (bits & F32_SIGN_MASK) != 0 {
        F32_SIGN_MASK.wrapping_sub(bits & !F32_SIGN_MASK)
    } else {
        bits | F32_SIGN_MASK
    }
}

/// Computes the exact IEEE-754 ULP distance between two `f64` values (`u64::MAX` if non-finite).
#[inline]
#[must_use]
pub fn ulp_distance_f64(a: f64, b: f64) -> u64 {
    if !a.is_finite() || !b.is_finite() {
        return u64::MAX;
    }
    let ka = ordered_key_f64(a);
    let kb = ordered_key_f64(b);
    ka.abs_diff(kb)
}

/// Computes the exact IEEE-754 ULP distance between two `f32` values (`u64::MAX` if non-finite).
#[inline]
#[must_use]
pub fn ulp_distance_f32(a: f32, b: f32) -> u64 {
    if !a.is_finite() || !b.is_finite() {
        return u64::MAX;
    }
    let ka = ordered_key_f32(a);
    let kb = ordered_key_f32(b);
    u64::from(ka.abs_diff(kb))
}

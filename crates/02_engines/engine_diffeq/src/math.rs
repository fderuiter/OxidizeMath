//! Pure-`core` IEEE-754 elementary functions for `#![no_std]` step control and norms.
// [cite:engine_diffeq]

const EXP_MASK: u64 = 0x7ff0_0000_0000_0000;
const MANT_MASK: u64 = 0x000f_ffff_ffff_ffff;
const EXP_BIAS: i32 = 1023;
const LN_2: f64 = core::f64::consts::LN_2;
const SQRT_HALF: f64 = core::f64::consts::FRAC_1_SQRT_2;

/// Computes $\sqrt{x}$ in pure `#![no_std]` IEEE-754 arithmetic via bit-halving and Heron iterations.
#[must_use]
#[inline]
#[verified_engine::verified]
pub fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return if x == 0.0 { 0.0 } else { f64::NAN };
    }
    if !x.is_finite() {
        return x;
    }
    let (norm_x, post_scale) = if x < f64::MIN_POSITIVE {
        // 2^54 and 2^-27
        (x * 18_014_398_509_481_984.0, 1.0 / 134_217_728.0)
    } else {
        (x, 1.0)
    };
    let seed_bits = (norm_x.to_bits() >> 1) + (0x3ff0_0000_0000_0000_u64 >> 1);
    let mut g = f64::from_bits(seed_bits);
    g = 0.5 * (g + norm_x / g);
    g = 0.5 * (g + norm_x / g);
    g = 0.5 * (g + norm_x / g);
    g = 0.5 * (g + norm_x / g);
    g * post_scale
}

/// Computes $\sqrt{a^2 + b^2}$ without overflow or underflow in `#![no_std]`.
#[must_use]
#[inline]
#[verified_engine::verified]
pub fn hypot_f64(a: f64, b: f64) -> f64 {
    let aa = a.abs();
    let bb = b.abs();
    let hi = aa.max(bb);
    let lo = aa.min(bb);
    if hi == 0.0 {
        return 0.0;
    }
    if !hi.is_finite() {
        return hi;
    }
    let r = lo / hi;
    hi * sqrt_f64(1.0 + r * r)
}

/// Computes $\ln(x)$ for $x > 0$ in `#![no_std]` via IEEE-754 mantissa reduction and an odd Padé/Taylor series.
#[must_use]
#[verified_engine::verified]
pub fn ln_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return if x == 0.0 {
            f64::NEG_INFINITY
        } else {
            f64::NAN
        };
    }
    if !x.is_finite() {
        return x;
    }
    let (norm_x, sub_exp) = if x < f64::MIN_POSITIVE {
        (x * 18_014_398_509_481_984.0, -54_i32)
    } else {
        (x, 0_i32)
    };
    let bits = norm_x.to_bits();
    let mut exp = ((bits & EXP_MASK) >> 52) as i32 - EXP_BIAS + sub_exp;
    let mut m = f64::from_bits((bits & MANT_MASK) | 0x3ff0_0000_0000_0000_u64);
    if m < SQRT_HALF {
        m *= 2.0;
        exp -= 1;
    }
    let z = (m - 1.0) / (m + 1.0);
    let z2 = z * z;
    let mut poly = 1.0 / 15.0;
    poly = poly * z2 + 1.0 / 13.0;
    poly = poly * z2 + 1.0 / 11.0;
    poly = poly * z2 + 1.0 / 9.0;
    poly = poly * z2 + 1.0 / 7.0;
    poly = poly * z2 + 1.0 / 5.0;
    poly = poly * z2 + 1.0 / 3.0;
    poly = poly * z2 + 1.0;
    (exp as f64) * LN_2 + 2.0 * z * poly
}

/// Computes $e^x$ in `#![no_std]` via Cody-Waite range reduction $x = k \ln 2 + r$ ($|r| \le \tfrac{1}{2}\ln 2$).
#[must_use]
#[verified_engine::verified]
pub fn exp_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x >= 709.782_712_893_384 {
        return f64::INFINITY;
    }
    if x <= -708.396_418_532_264_1 {
        return 0.0;
    }
    let kf = x / LN_2;
    let k = (if kf >= 0.0 { kf + 0.5 } else { kf - 0.5 }) as i32;
    let r = x - (k as f64) * LN_2;
    let mut p = 1.0 / 39_916_800.0; // 1/11!
    p = p * r + 1.0 / 3_628_800.0;
    p = p * r + 1.0 / 362_880.0;
    p = p * r + 1.0 / 40_320.0;
    p = p * r + 1.0 / 5_040.0;
    p = p * r + 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    let exp_r = p * r + 1.0;
    let scale_bits = ((k + EXP_BIAS) as u64) << 52;
    exp_r * f64::from_bits(scale_bits)
}

/// Computes $x^p = \exp(p \ln x)$ for $x > 0$ in pure `#![no_std]`.
#[must_use]
#[inline]
#[verified_engine::verified]
pub fn pow_pos_f64(base: f64, exponent: f64) -> f64 {
    if base <= 0.0 {
        return 0.0;
    }
    exp_f64(exponent * ln_f64(base))
}

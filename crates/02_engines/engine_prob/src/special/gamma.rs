//! `#![no_std]` Lanczos `ln_gamma`, `gamma`, `beta`, `ln_beta`, and `ln_factorial`.

use core::f64::consts::PI;

/// 9-term Lanczos approximation ($g = 7$) coefficients (Godfrey / Boost double precision).
const LANCZOS_G: f64 = 7.0;
const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_7;

const LANCZOS_COEFFS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Exact integer factorials $0! \dots 20!$ representable in `f64` without rounding error.
const EXACT_FACTORIALS: [f64; 21] = [
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5_040.0,
    40_320.0,
    362_880.0,
    3_628_800.0,
    39_916_800.0,
    479_001_600.0,
    6_227_020_800.0,
    87_178_291_200.0,
    1_307_674_368_000.0,
    20_922_789_888_000.0,
    355_687_428_096_000.0,
    6_402_373_705_728_000.0,
    121_645_100_408_832_000.0,
    2_432_902_008_176_640_000.0,
];

#[inline]
#[cfg_attr(feature = "verified", verified_engine::verified)]
fn lanczos_series(z_minus_one: f64) -> f64 {
    debug_assert!(z_minus_one.is_finite());
    let mut acc = LANCZOS_COEFFS[0];
    let mut idx = 1usize;
    while idx < LANCZOS_COEFFS.len() {
        acc += LANCZOS_COEFFS[idx] / (z_minus_one + idx as f64);
        idx += 1;
    }
    acc
}

#[inline]
#[cfg_attr(feature = "verified", verified_engine::verified)]
fn lanczos_ln_gamma_ge_half(x: f64) -> f64 {
    debug_assert!(x >= 0.5);
    let z = x - 1.0;
    let t = z + LANCZOS_G + 0.5;
    let sum = lanczos_series(z);
    LN_SQRT_2PI + (z + 0.5) * libm::log(t) - t + libm::log(sum)
}

#[inline]
#[cfg_attr(feature = "verified", verified_engine::verified)]
fn lanczos_gamma_ge_half(x: f64) -> f64 {
    debug_assert!(x >= 0.5);
    let rounded = libm::rint(x);
    if (x - rounded).abs() == 0.0 && (1.0..=21.0).contains(&rounded) {
        return EXACT_FACTORIALS[(rounded as usize) - 1];
    }
    let z = x - 1.0;
    let t = z + LANCZOS_G + 0.5;
    let sum = lanczos_series(z);
    libm::sqrt(2.0 * PI) * libm::pow(t, z + 0.5) * libm::exp(-t) * sum
}

/// Evaluates $\ln \Gamma(x)$ for $x > 0$ via the 9-term Lanczos approximation with Euler reflection.
///
/// Returns `f64::INFINITY` at $x = 0$ or $x = +\infty$, and `f64::NAN` for $x < 0$ or `NaN`.
///
/// # Example
///
/// ```rust
/// # use engine_prob::__doctest_facade as math_explorer;
/// use math_explorer::engine_prob::ln_gamma;
///
/// assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-14);
/// assert!((ln_gamma(0.5) - 0.5 * core::f64::consts::PI.ln()).abs() < 1e-14);
/// ```
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn ln_gamma(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 || x.is_infinite() {
        return f64::INFINITY;
    }
    if x == 1.0 || x == 2.0 {
        return 0.0;
    }
    if x < 0.5 {
        let sin_pi_x = libm::sin(PI * x);
        if sin_pi_x <= 0.0 {
            return f64::INFINITY;
        }
        let res = libm::log(PI / sin_pi_x) - lanczos_ln_gamma_ge_half(1.0 - x);
        debug_assert!(!res.is_nan());
        return res;
    }
    let out = lanczos_ln_gamma_ge_half(x);
    debug_assert!(!out.is_nan());
    out
}

/// Evaluates Euler's Gamma function $\Gamma(x)$ across $\mathbb{R} \setminus \{0, -1, -2, \dots\}$.
///
/// Uses exact lookup for small positive integers $1 \le n \le 21$ and Lanczos + reflection elsewhere.
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn gamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x.is_sign_positive() {
            f64::INFINITY
        } else {
            f64::NAN
        };
    }
    if x <= 0.0 {
        let rounded = libm::rint(x);
        if (x - rounded).abs() == 0.0 {
            return f64::NAN;
        }
        let sin_pi_x = libm::sin(PI * x);
        if sin_pi_x == 0.0 {
            return f64::NAN;
        }
        return PI / (sin_pi_x * lanczos_gamma_ge_half(1.0 - x));
    }
    if x < 0.5 {
        return PI / (libm::sin(PI * x) * lanczos_gamma_ge_half(1.0 - x));
    }
    lanczos_gamma_ge_half(x)
}

/// Evaluates $\ln(k!) = \ln \Gamma(k + 1)$ without intermediate overflow.
#[inline]
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn ln_factorial(k: u64) -> f64 {
    if k <= 1 {
        0.0
    } else if (k as usize) < EXACT_FACTORIALS.len() {
        libm::log(EXACT_FACTORIALS[k as usize])
    } else {
        let res = ln_gamma((k as f64) + 1.0);
        debug_assert!(res > 0.0);
        res
    }
}

/// Evaluates $\ln B(a, b) = \ln \Gamma(a) + \ln \Gamma(b) - \ln \Gamma(a + b)$ for $a > 0, b > 0$.
#[inline]
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn ln_beta(a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 || a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    let res = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    debug_assert!(!res.is_nan());
    res
}

/// Evaluates Euler's Beta function $B(a, b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a + b)}$ for $a > 0, b > 0$.
#[inline]
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn beta(a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 || a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    let res = libm::exp(ln_beta(a, b));
    debug_assert!(res >= 0.0);
    res
}

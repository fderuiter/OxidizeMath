//! `#![no_std]` error functions (`erf`, `erfc`, `erf_inv`, `erfc_inv`) and standard normal
//! PDF, log-PDF, CDF, survival, and Halley-refined quantile routines.

use crate::error::ProbError;
use core::f64::consts::{FRAC_1_SQRT_2, PI, SQRT_2};

const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_7;
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

const ACKLAM_A: [f64; 6] = [
    -3.969_683_028_665_376e1,
    2.209_460_984_245_205e2,
    -2.759_285_104_469_687e2,
    1.383_577_518_672_69e2,
    -3.066_479_806_614_716e1,
    2.506_628_277_459_239,
];

const ACKLAM_B: [f64; 5] = [
    -5.447_609_879_822_406e1,
    1.615_858_368_580_409e2,
    -1.556_989_798_598_866e2,
    6.680_131_188_771_972e1,
    -1.328_068_155_288_572e1,
];

const ACKLAM_C: [f64; 6] = [
    -7.784_894_002_430_293e-3,
    -3.223_964_580_411_365e-1,
    -2.400_758_277_161_838,
    -2.549_732_539_343_734,
    4.374_664_141_464_968,
    2.938_163_982_698_783,
];

const ACKLAM_D: [f64; 4] = [
    7.784_695_709_041_462e-3,
    3.224_671_290_700_398e-1,
    2.445_134_137_142_996,
    3.754_408_661_907_416,
];

const ACKLAM_P_LOW: f64 = 0.024_25;

/// Evaluates the Gauss error function $\operatorname{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2}\,dt$.
#[inline]
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn erf(x: f64) -> f64 {
    let y = libm::erf(x);
    debug_assert!(x.is_nan() || (-1.0..=1.0).contains(&y));
    y
}

/// Evaluates the complementary error function $\operatorname{erfc}(x) = 1 - \operatorname{erf}(x)$.
#[inline]
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn erfc(x: f64) -> f64 {
    let y = libm::erfc(x);
    debug_assert!(x.is_nan() || (0.0..=2.0).contains(&y));
    y
}

/// Evaluates the standard normal density $\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2 / 2}$.
#[inline]
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn standard_normal_pdf(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    let val = INV_SQRT_2PI * libm::exp(-0.5 * z * z);
    debug_assert!(val >= 0.0);
    val
}

/// Evaluates the standard normal log-density $\ln \phi(z) = -\frac{1}{2}\ln(2\pi) - \frac{z^2}{2}$.
#[inline]
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn standard_normal_ln_pdf(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    -LN_SQRT_2PI - 0.5 * z * z
}

/// Evaluates the standard normal CDF $\Phi(z) = \frac{1}{2}\operatorname{erfc}(-z / \sqrt{2})$.
#[inline]
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn standard_normal_cdf(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    let val = 0.5 * libm::erfc(-z * FRAC_1_SQRT_2);
    debug_assert!((0.0..=1.0).contains(&val));
    val
}

/// Evaluates the standard normal upper-tail survival probability $1 - \Phi(z) = \frac{1}{2}\operatorname{erfc}(z / \sqrt{2})$.
#[inline]
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn standard_normal_survival(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    let val = 0.5 * libm::erfc(z * FRAC_1_SQRT_2);
    debug_assert!((0.0..=1.0).contains(&val));
    val
}

#[inline]
fn acklam_tail_seed(q: f64) -> f64 {
    let t = libm::sqrt(-2.0 * libm::log(q));
    let num = (((((ACKLAM_C[0] * t + ACKLAM_C[1]) * t + ACKLAM_C[2]) * t + ACKLAM_C[3]) * t
        + ACKLAM_C[4])
        * t)
        + ACKLAM_C[5];
    let den = (((ACKLAM_D[0] * t + ACKLAM_D[1]) * t + ACKLAM_D[2]) * t + ACKLAM_D[3]) * t + 1.0;
    num / den
}

#[inline]
fn acklam_central_seed(p: f64) -> f64 {
    let q = p - 0.5;
    let r = q * q;
    let num = (((((ACKLAM_A[0] * r + ACKLAM_A[1]) * r + ACKLAM_A[2]) * r + ACKLAM_A[3]) * r
        + ACKLAM_A[4])
        * r
        + ACKLAM_A[5])
        * q;
    let den = ((((ACKLAM_B[0] * r + ACKLAM_B[1]) * r + ACKLAM_B[2]) * r + ACKLAM_B[3]) * r
        + ACKLAM_B[4])
        * r
        + 1.0;
    num / den
}

#[inline]
fn halley_refine_normal(mut z: f64, p: f64) -> f64 {
    for _ in 0..2 {
        let delta = if p <= 0.5 {
            0.5 * libm::erfc(-z * FRAC_1_SQRT_2) - p
        } else {
            (1.0 - p) - 0.5 * libm::erfc(z * FRAC_1_SQRT_2)
        };
        let pdf = INV_SQRT_2PI * libm::exp(-0.5 * z * z);
        if pdf <= f64::MIN_POSITIVE {
            break;
        }
        let ratio = delta / pdf;
        z -= ratio / (1.0 + 0.5 * z * ratio);
    }
    z
}

/// Evaluates the standard normal quantile $\Phi^{-1}(p)$ to full IEEE-754 double precision ($< 10^{-15}$).
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn standard_normal_quantile(p: f64) -> Result<f64, ProbError> {
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err(ProbError::InvalidProbability { value: p });
    }
    if p == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if p == 1.0 {
        return Ok(f64::INFINITY);
    }
    if p == 0.5 {
        return Ok(0.0);
    }
    let z0 = if p < ACKLAM_P_LOW {
        acklam_tail_seed(p)
    } else if p > 1.0 - ACKLAM_P_LOW {
        -acklam_tail_seed(1.0 - p)
    } else {
        acklam_central_seed(p)
    };
    let z = halley_refine_normal(z0, p);
    debug_assert!(z.is_finite());
    Ok(z)
}

/// Evaluates $\operatorname{erfc}^{-1}(y)$ for $y \in [0, 2]$ to full double precision.
///
/// Uses $\operatorname{erfc}^{-1}(y) = -\Phi^{-1}(y / 2) / \sqrt{2}$ with direct Halley refinement on $\operatorname{erfc}$.
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn erfc_inv(y: f64) -> f64 {
    if y.is_nan() || !(0.0..=2.0).contains(&y) {
        return f64::NAN;
    }
    if y == 0.0 {
        return f64::INFINITY;
    }
    if y == 2.0 {
        return f64::NEG_INFINITY;
    }
    if y == 1.0 {
        return 0.0;
    }
    let p = 0.5 * y;
    let Ok(z) = standard_normal_quantile(p) else {
        return f64::NAN;
    };
    let mut x = -z * FRAC_1_SQRT_2;
    let two_over_sqrt_pi = 2.0 / libm::sqrt(PI);
    let err = libm::erfc(x) - y;
    let deriv = -two_over_sqrt_pi * libm::exp(-x * x);
    if deriv.abs() > f64::MIN_POSITIVE {
        let ratio = err / deriv;
        x -= ratio / (1.0 + x * ratio);
    }
    debug_assert!(x.is_finite());
    x
}

/// Evaluates $\operatorname{erf}^{-1}(x)$ for $x \in [-1, 1]$ with Halley refinement on $\operatorname{erf}$.
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn erf_inv(x: f64) -> f64 {
    if x.is_nan() || !(-1.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return f64::INFINITY;
    }
    if x == -1.0 {
        return f64::NEG_INFINITY;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let mut y = sign * erfc_inv(1.0 - x.abs());
    let two_over_sqrt_pi = 2.0 / libm::sqrt(PI);
    if x.abs() <= 0.75 {
        let err = libm::erf(y) - x;
        let deriv = two_over_sqrt_pi * libm::exp(-y * y);
        if deriv > f64::MIN_POSITIVE {
            let ratio = err / deriv;
            y -= ratio / (1.0 - y * ratio);
        }
    }
    let _ = SQRT_2;
    y
}

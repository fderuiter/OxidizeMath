//! Generic `#![no_std]` IEEE-754 `f32`/`f64` uniform, Gaussian, and bounded-index samplers.
// [cite:clinical_trials_statistics]
#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]

use rand_core::RngCore;

const INV_TWO_POW_24_F32: f32 = 1.0 / ((1_u32 << 24) as f32);
const INV_TWO_POW_53_F64: f64 = 1.0 / ((1_u64 << 53) as f64);
const F32_MAX_BELOW_ONE: f32 = f32::from_bits(0x3f7f_ffff);
const F32_MIN_ABOVE_ZERO: f32 = 1.0e-7_f32;
const F64_MAX_BELOW_ONE: f64 = 1.0 - 1.0e-15_f64;
const F64_MIN_ABOVE_ZERO: f64 = 1.0e-15_f64;

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Sealed IEEE-754 floating-point scalar contract (`f32` and `f64`) compatible with
/// `engine_linalg::Scalar` for deterministic PRNG and QMC point generation.
pub trait RngScalar: Copy + PartialOrd + private::Sealed {
    /// Additive identity `0.0`.
    const ZERO: Self;
    /// Multiplicative identity `1.0`.
    const ONE: Self;

    /// Converts an `f64` value in `[0, 1]` into `Self`.
    fn from_unit_f64(value: f64) -> Self;

    /// Converts an `f64` value in `[0, 1]` into a strictly open `(0, 1)` scalar.
    fn clamp_open_unit_f64(value: f64) -> Self;

    /// Converts a 53-bit dyadic integer `raw_53 < 2^53` into a strictly open `(0, 1)` midpoint `(raw_53 + 0.5) * 2^-53`.
    fn from_dyadic_53_open(raw_53: u64) -> Self;

    /// Draws a uniform scalar on `[0, 1)` using full IEEE-754 significand precision.
    fn sample_closed_open<R: RngCore + ?Sized>(rng: &mut R) -> Self;

    /// Draws a strictly open uniform scalar on `(0, 1)` safe for `ln()` and `probit()` transforms.
    fn sample_open<R: RngCore + ?Sized>(rng: &mut R) -> Self;

    /// Draws a symmetric open uniform scalar on `(-1, 1)`.
    fn sample_symmetric_open<R: RngCore + ?Sized>(rng: &mut R) -> Self;
}

impl RngScalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline]
    fn from_unit_f64(value: f64) -> Self {
        debug_assert!(value.is_finite());
        value as f32
    }

    #[inline]
    fn clamp_open_unit_f64(value: f64) -> Self {
        let cast = (value as f32).clamp(F32_MIN_ABOVE_ZERO, F32_MAX_BELOW_ONE);
        debug_assert!(cast > 0.0 && cast < 1.0);
        cast
    }

    #[inline]
    fn from_dyadic_53_open(raw_53: u64) -> Self {
        debug_assert!(raw_53 < (1_u64 << 53));
        let unit = ((raw_53 as f64) + 0.5) * INV_TWO_POW_53_F64;
        Self::clamp_open_unit_f64(unit)
    }

    #[inline]
    fn sample_closed_open<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        let bits = (rng.next_u64() >> 40) as u32;
        let u = (bits as f32) * INV_TWO_POW_24_F32;
        debug_assert!((0.0..1.0).contains(&u));
        u
    }

    #[inline]
    fn sample_open<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        let bits = (rng.next_u64() >> 40) as u32;
        let u = (((bits as f32) + 0.5) * INV_TWO_POW_24_F32)
            .clamp(F32_MIN_ABOVE_ZERO, F32_MAX_BELOW_ONE);
        debug_assert!(u > 0.0 && u < 1.0);
        u
    }

    #[inline]
    fn sample_symmetric_open<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        let u = Self::sample_open(rng);
        let sym = 2.0 * u - 1.0;
        debug_assert!(sym > -1.0 && sym < 1.0);
        sym
    }
}

impl RngScalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline]
    fn from_unit_f64(value: f64) -> Self {
        debug_assert!(value.is_finite());
        value
    }

    #[inline]
    fn clamp_open_unit_f64(value: f64) -> Self {
        let clamped = value.clamp(F64_MIN_ABOVE_ZERO, F64_MAX_BELOW_ONE);
        debug_assert!(clamped > 0.0 && clamped < 1.0);
        clamped
    }

    #[inline]
    fn from_dyadic_53_open(raw_53: u64) -> Self {
        debug_assert!(raw_53 < (1_u64 << 53));
        let unit = (((raw_53 as f64) + 0.5) * INV_TWO_POW_53_F64)
            .clamp(F64_MIN_ABOVE_ZERO, F64_MAX_BELOW_ONE);
        debug_assert!(unit > 0.0 && unit < 1.0);
        unit
    }

    #[inline]
    fn sample_closed_open<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        let bits = rng.next_u64() >> 11;
        let u = (bits as f64) * INV_TWO_POW_53_F64;
        debug_assert!((0.0..1.0).contains(&u));
        u
    }

    #[inline]
    fn sample_open<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        let bits = rng.next_u64() >> 11;
        let u = (((bits as f64) + 0.5) * INV_TWO_POW_53_F64)
            .clamp(F64_MIN_ABOVE_ZERO, F64_MAX_BELOW_ONE);
        debug_assert!(u > 0.0 && u < 1.0);
        u
    }

    #[inline]
    fn sample_symmetric_open<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        let u = Self::sample_open(rng);
        let sym = 2.0 * u - 1.0;
        debug_assert!(sym > -1.0 && sym < 1.0);
        sym
    }
}

/// Samples a uniform floating-point scalar in `[0, 1)`.
#[inline]
pub fn sample_closed_open<T: RngScalar, R: RngCore + ?Sized>(rng: &mut R) -> T {
    T::sample_closed_open(rng)
}

/// Samples a strictly open uniform floating-point scalar in `(0, 1)`.
#[inline]
pub fn sample_open<T: RngScalar, R: RngCore + ?Sized>(rng: &mut R) -> T {
    T::sample_open(rng)
}

/// Samples a symmetric open uniform floating-point scalar in `(-1, 1)`.
#[inline]
pub fn sample_symmetric_open<T: RngScalar, R: RngCore + ?Sized>(rng: &mut R) -> T {
    T::sample_symmetric_open(rng)
}

/// Typed stack-resident error returned by fallible bounded-index samplers in [`crate::scalar`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RngError {
    /// Requested exclusive `upper` bound was `0`, leaving the sampling interval `0..0` empty.
    ZeroUpperBound,
}

impl RngError {
    /// Returns the static type identifier `"RngError"`.
    #[inline]
    pub const fn error_type(&self) -> &'static str {
        "RngError"
    }

    /// Returns the stable diagnostic error code (`RNG-001`).
    #[inline]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::ZeroUpperBound => "RNG-001",
        }
    }
}

impl core::fmt::Display for RngError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Self::ZeroUpperBound => f.write_str("Bounded sampler upper bound must be > 0"),
        }
    }
}

impl core::error::Error for RngError {}

impl math_diagnostics::Diagnostic for RngError {
    fn severity(&self) -> math_diagnostics::Severity {
        math_diagnostics::Severity::Error
    }

    fn visit_fields(&self, visitor: &mut dyn math_diagnostics::DiagnosticVisitor) {
        visitor.record_str("error_type", self.error_type());
        visitor.record_str("code", self.code());
        visitor.record_display("description", self);
    }
}

/// Fallible companion to [`sample_bounded_u64`] returning [`RngError::ZeroUpperBound`] if `upper == 0`.
#[inline]
pub fn try_sample_bounded_u64<R: RngCore + ?Sized>(
    rng: &mut R,
    upper: u64,
) -> Result<u64, RngError> {
    if upper == 0 {
        return Err(RngError::ZeroUpperBound);
    }
    if upper == 1 {
        return Ok(0);
    }
    let threshold = upper.wrapping_neg() % upper;
    loop {
        let r = rng.next_u64();
        let prod = (r as u128) * (upper as u128);
        if (prod as u64) >= threshold {
            let idx = (prod >> 64) as u64;
            debug_assert!(idx < upper);
            return Ok(idx);
        }
    }
}

/// Samples an unbiased `u64` in `0..upper` via rejection sampling (`upper >= 1`).
///
/// # Panics
///
/// Panics if `upper == 0`. Prefer [`try_sample_bounded_u64`] when `upper` is caller-supplied.
#[inline]
pub fn sample_bounded_u64<R: RngCore + ?Sized>(rng: &mut R, upper: u64) -> u64 {
    try_sample_bounded_u64(rng, upper).expect("sample_bounded_u64 upper bound must be > 0")
}

/// Fallible companion to [`sample_bounded_usize`] returning [`RngError::ZeroUpperBound`] if `upper == 0`.
#[inline]
pub fn try_sample_bounded_usize<R: RngCore + ?Sized>(
    rng: &mut R,
    upper: usize,
) -> Result<usize, RngError> {
    if upper == 0 {
        return Err(RngError::ZeroUpperBound);
    }
    let idx = try_sample_bounded_u64(rng, upper as u64)? as usize;
    debug_assert!(idx < upper);
    Ok(idx)
}

/// Samples an unbiased `usize` in `0..upper` using 64-bit Lemire multiplication so 32-bit `wasm32`
/// and 64-bit native targets consume identical `u64` draws and return identical indices.
///
/// # Panics
///
/// Panics if `upper == 0`. Prefer [`try_sample_bounded_usize`] when `upper` is caller-supplied.
#[inline]
pub fn sample_bounded_usize<R: RngCore + ?Sized>(rng: &mut R, upper: usize) -> usize {
    try_sample_bounded_usize(rng, upper).expect("sample_bounded_usize upper bound must be > 0")
}

/// Fills `shifts` with independent `[0, 1)` Cranley-Patterson shift scalars without heap allocation.
#[inline]
pub fn fill_cranley_patterson_shifts<T: RngScalar, R: RngCore + ?Sized>(
    shifts: &mut [T],
    rng: &mut R,
) {
    for slot in shifts.iter_mut() {
        *slot = T::sample_closed_open(rng);
    }
}

/// Draws a standard normal $\mathcal{N}(0, 1)$ `f64` variate via a deterministic Rational/Acklam
/// inverse-normal transform on a single open-unit uniform draw (100% `libm`-free `#![no_std]`).
#[inline]
pub fn sample_standard_normal_f64<R: RngCore + ?Sized>(rng: &mut R) -> f64 {
    let u: f64 = sample_open(rng);
    acklam_probit_f64(u)
}

/// Draws a standard normal $\mathcal{N}(0, 1)$ variate converted to `T`.
#[inline]
pub fn sample_standard_normal<T: RngScalar, R: RngCore + ?Sized>(rng: &mut R) -> T {
    T::from_unit_f64(sample_standard_normal_f64(rng))
}

#[inline]
fn ln_open_unit_f64(u: f64) -> f64 {
    debug_assert!(u > 0.0 && u < 1.0);
    let bits = u.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    let y = (m - 1.0) / (m + 1.0);
    let y2 = y * y;
    let mut term = y;
    let mut sum = y;
    let mut k = 3_i32;
    while k <= 21 {
        term *= y2;
        sum += term / (k as f64);
        k += 2;
    }
    2.0 * sum + (exp as f64) * core::f64::consts::LN_2
}

#[inline]
fn sqrt_pos_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut g = f64::from_bits((x.to_bits() >> 1) + (0x3ff0_0000_0000_0000_u64 >> 1));
    let mut iter = 0_u32;
    while iter < 6 {
        g = 0.5 * (g + x / g);
        iter += 1;
    }
    g
}

#[inline]
fn acklam_probit_f64(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    if !(0.02425..=0.97575).contains(&p) {
        let sign = if p < 0.5 { -1.0 } else { 1.0 };
        let tail = if p < 0.5 { p } else { 1.0 - p };
        let q = sqrt_pos_f64(-2.0 * ln_open_unit_f64(tail));
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        return sign * (num / den);
    }
    let q = p - 0.5;
    let r = q * q;
    let num = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q;
    let den = ((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0;
    num / den
}

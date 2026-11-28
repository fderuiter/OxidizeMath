//! Generic `#![no_std]` IEEE-754 `f32`/`f64` uniform, Gaussian, and bounded-index samplers.
// [cite:clinical_trials_statistics]
#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]

use math_scalars::RealField;
use rand_core::RngCore;

const INV_TWO_POW_53_F64: f64 = 1.0 / ((1_u64 << 53) as f64);

/// Real floating-point scalar contract unified with [`math_scalars::RealField`] for deterministic
/// PRNG and QMC point generation across all `RealField` types (`f32`, `f64`, etc.).
pub trait RngScalar: RealField {
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

impl<T: RealField> RngScalar for T {
    #[inline]
    fn from_unit_f64(value: f64) -> Self {
        debug_assert!(value.is_finite());
        T::from_f64(value)
    }

    #[inline]
    fn clamp_open_unit_f64(value: f64) -> Self {
        let clamped = T::open_unit_from_f64(value);
        debug_assert!(clamped > T::ZERO && clamped < T::ONE);
        clamped
    }

    #[inline]
    fn from_dyadic_53_open(raw_53: u64) -> Self {
        debug_assert!(raw_53 < (1_u64 << 53));
        let unit = ((raw_53 as f64) + 0.5) * INV_TWO_POW_53_F64;
        let clamped = T::open_unit_from_f64(unit);
        debug_assert!(clamped > T::ZERO && clamped < T::ONE);
        clamped
    }

    #[inline]
    fn sample_closed_open<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        let u = T::uniform_closed_open_from_u64(rng.next_u64());
        debug_assert!(u >= T::ZERO && u < T::ONE);
        u
    }

    #[inline]
    fn sample_open<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        let u = T::uniform_open_from_u64(rng.next_u64());
        debug_assert!(u > T::ZERO && u < T::ONE);
        u
    }

    #[inline]
    fn sample_symmetric_open<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        let u = Self::sample_open(rng);
        let sym = T::TWO * u - T::ONE;
        debug_assert!(sym > -T::ONE && sym < T::ONE);
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

/// Draws a standard normal $\mathcal{N}(0, 1)$ `f64` variate via Wichura's AS 241 minimax
/// rational inverse-CDF (`probit`) transform on a single open-unit uniform draw (`0`-ULP `libm::sqrt` `#![no_std]`).
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
pub(crate) fn ln_open_unit_f64(u: f64) -> f64 {
    const LN2_HI: f64 = 6.931_471_803_691_238e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
    const MIN_NORMAL_F64: f64 = 2.225_073_858_507_201_4e-308;
    const TWO_POW_54: f64 = (1_u64 << 54) as f64;

    debug_assert!(u > 0.0 && u < 1.0);
    let (bits, mut exp) = if u < MIN_NORMAL_F64 {
        let b = (u * TWO_POW_54).to_bits();
        (b, ((b >> 52) & 0x7ff) as i64 - 1023 - 54)
    } else {
        let b = u.to_bits();
        (b, ((b >> 52) & 0x7ff) as i64 - 1023)
    };
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if m >= core::f64::consts::SQRT_2 {
        m *= 0.5;
        exp += 1;
    }
    let f = m - 1.0;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w
        * (3.999_999_999_940_942e-1
            + w * (2.222_219_843_214_978_4e-1 + w * 1.531_383_769_920_937_3e-1));
    let t2 = z
        * (6.666_666_666_666_735e-1
            + w * (2.857_142_874_366_239e-1
                + w * (1.818_357_216_161_805e-1 + w * 1.479_819_860_511_658_6e-1)));
    let hfsq = 0.5 * f * f;
    let dk = exp as f64;
    dk * LN2_HI - ((hfsq - (s * (hfsq + t2 + t1) + dk * LN2_LO)) - f)
}

#[inline]
fn sqrt_pos_f64(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        math_scalars::sqrt_f64(x)
    }
}

const WICHURA_A: [f64; 8] = [
    3.387_132_872_796_366_5,
    1.331_416_678_917_843_8e2,
    1.971_590_950_306_551_4e3,
    1.373_169_376_550_946e4,
    4.592_195_393_154_987e4,
    6.726_577_092_700_87e4,
    3.343_057_558_358_813e4,
    2.509_080_928_730_122_7e3,
];
const WICHURA_B: [f64; 8] = [
    1.0,
    4.231_333_070_160_091e1,
    6.871_870_074_920_579e2,
    5.394_196_021_424_751e3,
    2.121_379_430_158_659_7e4,
    3.930_789_580_009_271e4,
    2.872_908_573_572_194_3e4,
    5.226_495_278_852_854e3,
];
const WICHURA_C: [f64; 8] = [
    1.423_437_110_749_683_5,
    4.630_337_846_156_546,
    5.769_497_221_460_691,
    3.647_848_324_763_204_5,
    1.270_458_252_452_368_4,
    2.417_807_251_774_506e-1,
    2.272_384_498_926_918_4e-2,
    7.745_450_142_783_414e-4,
];
const WICHURA_D: [f64; 8] = [
    1.0,
    2.053_191_626_637_759,
    1.676_384_830_183_803_8,
    6.897_673_349_851e-1,
    1.481_039_764_274_800_8e-1,
    1.519_866_656_361_645_8e-2,
    5.475_938_084_995_345e-4,
    1.050_750_071_644_416_9e-9,
];
const WICHURA_E: [f64; 8] = [
    6.657_904_643_501_104,
    5.463_784_911_164_114,
    1.784_826_539_917_291_3,
    2.965_605_718_285_049e-1,
    2.653_218_952_657_612_4e-2,
    1.242_660_947_388_078_4e-3,
    2.711_555_568_743_487_6e-5,
    2.010_334_399_292_288e-7,
];
const WICHURA_F: [f64; 8] = [
    1.0,
    5.998_322_065_558_88e-1,
    1.369_298_809_227_358e-1,
    1.487_536_129_085_061_5e-2,
    7.868_691_311_456_133e-4,
    1.846_318_317_510_054_8e-5,
    1.421_511_758_316_446e-7,
    2.044_263_103_389_939_7e-15,
];

#[inline]
fn eval_poly8(c: &[f64; 8], x: f64) -> f64 {
    ((((((c[7] * x + c[6]) * x + c[5]) * x + c[4]) * x + c[3]) * x + c[2]) * x + c[1]) * x + c[0]
}

#[inline]
pub(crate) fn acklam_probit_f64(p: f64) -> f64 {
    debug_assert!(p > 0.0 && p < 1.0);
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180_625 - q * q;
        return q * eval_poly8(&WICHURA_A, r) / eval_poly8(&WICHURA_B, r);
    }
    let r = if q < 0.0 { p } else { 1.0 - p };
    let s = sqrt_pos_f64(-ln_open_unit_f64(r));
    let mag = if s <= 5.0 {
        let t = s - 1.6;
        eval_poly8(&WICHURA_C, t) / eval_poly8(&WICHURA_D, t)
    } else {
        let t = s - 5.0;
        eval_poly8(&WICHURA_E, t) / eval_poly8(&WICHURA_F, t)
    };
    if q < 0.0 { -mag } else { mag }
}

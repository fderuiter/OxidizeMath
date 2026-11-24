//! Pure-Rust IEEE-754 software transcendental functions backed by `libm`.
//!
//! Stochastic samplers, Monte Carlo kernels, and cross-platform trajectory generators call these
//! wrappers instead of platform C `libm` intrinsics so native Linux, Windows, macOS, and
//! `wasm32-unknown-unknown` targets produce bit-identical floating-point results.

/// Pure-Rust natural logarithm $\ln(x)$ (`f64`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn ln(x: f64) -> f64 {
    libm::log(x)
}

/// Pure-Rust $\ln(1 + x)$ accurate near zero (`f64`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn ln_1p(x: f64) -> f64 {
    libm::log1p(x)
}

/// Pure-Rust base-2 logarithm $\log_2(x)$ (`f64`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn log2(x: f64) -> f64 {
    libm::log2(x)
}

/// Pure-Rust base-10 logarithm $\log_{10}(x)$ (`f64`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn log10(x: f64) -> f64 {
    libm::log10(x)
}

/// Pure-Rust exponential $e^x$ (`f64`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}

/// Pure-Rust $e^x - 1$ accurate near zero (`f64`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn exp_m1(x: f64) -> f64 {
    libm::expm1(x)
}

/// Pure-Rust real power $x^y$ (`f64`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn powf(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}

/// Pure-Rust IEEE-754 square root $\sqrt{x}$ (`f64`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

/// Pure-Rust cube root $x^{1/3}$ (`f64`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn cbrt(x: f64) -> f64 {
    libm::cbrt(x)
}

/// Pure-Rust Euclidean norm $\sqrt{x^2 + y^2}$ without intermediate overflow (`f64`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn hypot(x: f64, y: f64) -> f64 {
    libm::hypot(x, y)
}

/// Pure-Rust sine $\sin(x)$ (`f64`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn sin(x: f64) -> f64 {
    libm::sin(x)
}

/// Pure-Rust cosine $\cos(x)$ (`f64`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn cos(x: f64) -> f64 {
    libm::cos(x)
}

/// Pure-Rust simultaneous $(\sin(x), \cos(x))$ (`f64`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn sincos(x: f64) -> (f64, f64) {
    libm::sincos(x)
}

/// Pure-Rust tangent $\tan(x)$ (`f64`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn tan(x: f64) -> f64 {
    libm::tan(x)
}

/// Pure-Rust four-quadrant inverse tangent $\operatorname{atan2}(y, x)$ (`f64`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn atan2(y: f64, x: f64) -> f64 {
    libm::atan2(y, x)
}

/// Pure-Rust hyperbolic tangent $\tanh(x)$ (`f64`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn tanh(x: f64) -> f64 {
    libm::tanh(x)
}

/// Pure-Rust natural logarithm $\ln(x)$ (`f32`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn lnf(x: f32) -> f32 {
    libm::logf(x)
}

/// Pure-Rust exponential $e^x$ (`f32`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn expf(x: f32) -> f32 {
    libm::expf(x)
}

/// Pure-Rust real power $x^y$ (`f32`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn powff(x: f32, y: f32) -> f32 {
    libm::powf(x, y)
}

/// Pure-Rust IEEE-754 square root $\sqrt{x}$ (`f32`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn sqrtf(x: f32) -> f32 {
    libm::sqrtf(x)
}

/// Pure-Rust Euclidean norm $\sqrt{x^2 + y^2}$ (`f32`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn hypotf(x: f32, y: f32) -> f32 {
    libm::hypotf(x, y)
}

/// Pure-Rust sine $\sin(x)$ (`f32`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn sinf(x: f32) -> f32 {
    libm::sinf(x)
}

/// Pure-Rust cosine $\cos(x)$ (`f32`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn cosf(x: f32) -> f32 {
    libm::cosf(x)
}

/// Pure-Rust simultaneous $(\sin(x), \cos(x))$ (`f32`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn sincosf(x: f32) -> (f32, f32) {
    libm::sincosf(x)
}

/// Pure-Rust tangent $\tan(x)$ (`f32`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn tanf(x: f32) -> f32 {
    libm::tanf(x)
}

/// Pure-Rust four-quadrant inverse tangent $\operatorname{atan2}(y, x)$ (`f32`).
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn atan2f(y: f32, x: f32) -> f32 {
    libm::atan2f(y, x)
}

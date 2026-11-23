//! Bracketing scalar root-finding methods: [`Bisection`] and [`Brent`].

use super::{AnalysisError, RootFinder};
use crate::pure_math::analysis::dense_linear::{canonical_zero, has_strict_sign_change};

#[verified_engine::verified]
fn validate_bisection_bounds(min: f64, max: f64, tolerance: f64) -> Result<(), AnalysisError> {
    if !min.is_finite() || !max.is_finite() || !tolerance.is_finite() || tolerance < 0.0 {
        return Err(AnalysisError::InvalidBracket { min, max });
    }
    Ok(())
}

#[verified_engine::verified]
fn validate_brent_bounds(min: f64, max: f64, tolerance: f64) -> Result<(), AnalysisError> {
    if !min.is_finite() || !max.is_finite() || !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(AnalysisError::InvalidBracket { min, max });
    }
    Ok(())
}

/// Bisection method implementation.
///
/// Iteratively halves a sign-changing bracket `[min, max]`, rejecting `NaN` or infinite
/// function values, normalizing exact `-0.0` roots to `0.0`, and checking strict sign
/// changes without calling `f64::signum()`.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::roots::{Bisection, RootFinder};
///
/// let solver = Bisection::default();
/// let root = solver.find_root(|x| x * x - 2.0, 1.0, 2.0).unwrap();
/// assert!((root - std::f64::consts::SQRT_2).abs() < math_explorer::registry::TOLERANCE_FAST);
/// assert!(solver.find_root(|x| x.ln() - 1.0, -1.0, 5.0).is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bisection {
    /// Maximum number of iterations before giving up.
    pub max_iterations: usize,
    /// Absolute tolerance for the root interval and residual value.
    pub tolerance: f64,
}

impl Bisection {
    /// Creates a new [`Bisection`] solver.
    #[verified_engine::verified]
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }
}

impl Default for Bisection {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            max_iterations: math_commons::registry::MAX_ITERATIONS,
            tolerance: math_commons::registry::TOLERANCE_FAST,
        }
    }
}

impl RootFinder for Bisection {
    #[verified_engine::verified]
    fn find_root<F>(&self, f: F, min: f64, max: f64) -> Result<f64, AnalysisError>
    where
        F: Fn(f64) -> f64,
    {
        validate_bisection_bounds(min, max, self.tolerance)?;
        let mut low = min;
        let mut high = max;
        let mut f_low = f(low);
        let f_high = f(high);
        if !f_low.is_finite() || !f_high.is_finite() {
            return Err(AnalysisError::RootNotBracketed {
                min: low,
                f_min: f_low,
                max: high,
                f_max: f_high,
            });
        }
        if f_low == 0.0 || f_low.abs() < self.tolerance {
            return Ok(canonical_zero(low));
        }
        if f_high == 0.0 || f_high.abs() < self.tolerance {
            return Ok(canonical_zero(high));
        }
        if !has_strict_sign_change(f_low, f_high) {
            return Err(AnalysisError::RootNotBracketed {
                min: low,
                f_min: f_low,
                max: high,
                f_max: f_high,
            });
        }

        let mut mid = low;
        for _ in 0..self.max_iterations {
            mid = low + 0.5 * (high - low);
            if (high - low).abs() < self.tolerance {
                return Ok(canonical_zero(mid));
            }
            let f_mid = f(mid);
            if !f_mid.is_finite() {
                return Err(AnalysisError::InvalidParameters(
                    "Non-finite midpoint evaluation during Bisection",
                ));
            }
            if f_mid == 0.0 || f_mid.abs() < self.tolerance {
                return Ok(canonical_zero(mid));
            }
            if has_strict_sign_change(f_low, f_mid) {
                high = mid;
            } else {
                low = mid;
                f_low = f_mid;
            }
        }

        Err(AnalysisError::ConvergenceError(canonical_zero(mid)))
    }
}

/// Bracketed Brent-Dekker root-finding solver.
///
/// Combines inverse quadratic interpolation, secant steps, and guaranteed bisection fallback
/// for superlinear convergence without derivative evaluations or dynamic allocation.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::roots::{Brent, RootFinder};
///
/// fn main() -> Result<(), math_explorer::pure_math::analysis::roots::AnalysisError> {
///     let solver = Brent::default();
///     let root = solver.find_root(|x| x.cos() - x, 0.0, 1.0)?;
///     assert!((root - 0.739_085_133_215).abs() < math_explorer::registry::TOLERANCE_STANDARD);
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Brent {
    /// Maximum number of iterations before returning [`AnalysisError::ConvergenceError`].
    pub max_iterations: usize,
    /// Absolute bracket convergence tolerance.
    pub tolerance: f64,
}

impl Brent {
    /// Creates a new [`Brent`] solver with explicit iteration and tolerance bounds.
    #[verified_engine::verified]
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }
}

impl Default for Brent {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            max_iterations: math_commons::registry::MAX_ITERATIONS,
            tolerance: math_commons::registry::TOLERANCE_HIGH,
        }
    }
}

#[derive(Clone, Copy)]
struct BrentIterate {
    a: f64,
    b: f64,
    c: f64,
    fa: f64,
    fb: f64,
    fc: f64,
    d: f64,
    e: f64,
}

#[verified_engine::verified]
fn brent_propose_step(st: &BrentIterate, xm: f64, tol1: f64) -> (f64, f64) {
    if st.e.abs() < tol1 || st.fa.abs() <= st.fb.abs() {
        return (xm, xm);
    }
    let s = st.fb / st.fa;
    let (mut p, mut q) = if (st.a - st.c).abs() <= f64::EPSILON {
        (2.0 * xm * s, 1.0 - s)
    } else {
        let qi = st.fa / st.fc;
        let r = st.fb / st.fc;
        (
            s * (2.0 * xm * qi * (qi - r) - (st.b - st.a) * (r - 1.0)),
            (qi - 1.0) * (r - 1.0) * (s - 1.0),
        )
    };
    if p > 0.0 {
        q = -q;
    }
    p = p.abs();
    let bound_bracket = 3.0 * xm * q - (tol1 * q).abs();
    let bound_prev = (st.e * q).abs();
    if q != 0.0 && 2.0 * p < bound_bracket.min(bound_prev) {
        (p / q, st.d)
    } else {
        (xm, xm)
    }
}

impl RootFinder for Brent {
    #[verified_engine::verified]
    fn find_root<F>(&self, f: F, min: f64, max: f64) -> Result<f64, AnalysisError>
    where
        F: Fn(f64) -> f64,
    {
        validate_brent_bounds(min, max, self.tolerance)?;
        let fa0 = f(min);
        let fb0 = f(max);
        if !fa0.is_finite() || !fb0.is_finite() {
            return Err(AnalysisError::RootNotBracketed {
                min,
                f_min: fa0,
                max,
                f_max: fb0,
            });
        }
        if fa0 == 0.0 {
            return Ok(canonical_zero(min));
        }
        if fb0 == 0.0 {
            return Ok(canonical_zero(max));
        }
        if !has_strict_sign_change(fa0, fb0) {
            return Err(AnalysisError::RootNotBracketed {
                min,
                f_min: fa0,
                max,
                f_max: fb0,
            });
        }

        let mut st = BrentIterate {
            a: min,
            b: max,
            c: max,
            fa: fa0,
            fb: fb0,
            fc: fb0,
            d: max - min,
            e: max - min,
        };

        for _ in 0..self.max_iterations {
            if !has_strict_sign_change(st.fb, st.fc) {
                st.c = st.a;
                st.fc = st.fa;
                st.d = st.b - st.a;
                st.e = st.d;
            }
            if st.fc.abs() < st.fb.abs() {
                st.a = st.b;
                st.b = st.c;
                st.c = st.a;
                st.fa = st.fb;
                st.fb = st.fc;
                st.fc = st.fa;
            }
            let tol1 = 2.0 * f64::EPSILON * st.b.abs() + 0.5 * self.tolerance;
            let xm = 0.5 * (st.c - st.b);
            if xm.abs() <= tol1 || st.fb == 0.0 {
                return Ok(canonical_zero(st.b));
            }
            let (next_d, next_e) = brent_propose_step(&st, xm, tol1);
            st.d = next_d;
            st.e = next_e;
            st.a = st.b;
            st.fa = st.fb;
            if st.d.abs() > tol1 {
                st.b += st.d;
            } else {
                st.b += tol1.copysign(xm);
            }
            st.fb = f(st.b);
            if !st.fb.is_finite() {
                return Err(AnalysisError::ConvergenceError(canonical_zero(st.b)));
            }
        }

        Err(AnalysisError::ConvergenceError(canonical_zero(st.b)))
    }
}

// [cite:graph_parameters_rust]
// [cite:advanced_linear_algebra]

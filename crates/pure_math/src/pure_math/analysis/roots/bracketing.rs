//! Bracketing scalar root-finding methods: [`Bisection`] and [`Brent`].

use super::{AnalysisError, RootFinder};
use crate::pure_math::analysis::dense_linear::{canonical_zero, has_strict_sign_change};

#[verified_engine::verified]
fn validate_bracket_inputs(
    min: f64,
    max: f64,
    tolerance: f64,
    f_min: f64,
    f_max: f64,
) -> Result<(), AnalysisError> {
    if !min.is_finite() || !max.is_finite() {
        return Err(AnalysisError::InvalidParameters(
            "Interval bounds must be finite numbers.".into(),
        ));
    }
    if min > max {
        return Err(AnalysisError::InvalidParameters(format!(
            "Lower bound ({min}) must not exceed upper bound ({max})."
        )));
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(AnalysisError::InvalidParameters(
            "Tolerance must be a finite non-negative number.".into(),
        ));
    }
    if !f_min.is_finite() || !f_max.is_finite() {
        return Err(AnalysisError::InvalidParameters(format!(
            "Non-finite endpoint evaluation: f({min})={f_min}, f({max})={f_max}."
        )));
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
#[derive(Debug, Clone, Copy)]
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
        let mut low = min;
        let mut high = max;
        let mut f_low = f(low);
        let f_high = f(high);
        validate_bracket_inputs(low, high, self.tolerance, f_low, f_high)?;

        if f_low == 0.0 || f_low.abs() < self.tolerance {
            return Ok(canonical_zero(low));
        }
        if f_high == 0.0 || f_high.abs() < self.tolerance {
            return Ok(canonical_zero(high));
        }
        if !has_strict_sign_change(f_low, f_high) {
            return Err(AnalysisError::InvalidParameters(format!(
                "Root not bracketed: f({low})={f_low}, f({high})={f_high}. Signs must differ."
            )));
        }

        let mut mid = low;
        for _ in 0..self.max_iterations {
            mid = low + 0.5 * (high - low);
            let f_mid = f(mid);
            if !f_mid.is_finite() {
                return Err(AnalysisError::InvalidParameters(format!(
                    "Non-finite midpoint evaluation: f({mid})={f_mid}."
                )));
            }
            if f_mid == 0.0 || f_mid.abs() < self.tolerance || (high - low).abs() < self.tolerance {
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

/// Brent-Dekker hybrid bracketing root-finding method.
///
/// Combines inverse quadratic interpolation, secant steps, and bisection fallback to achieve
/// superlinear convergence while preserving guaranteed bracket contraction.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::roots::{Brent, RootFinder};
///
/// let solver = Brent::default();
/// let root = solver.find_root(|x| x.cos() - x, 0.0, 1.0).unwrap();
/// assert!((root - 0.739_085_133).abs() < math_explorer::registry::TOLERANCE_FAST);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Brent {
    /// Maximum number of iterations before returning [`AnalysisError::ConvergenceError`].
    pub max_iterations: usize,
    /// Convergence tolerance for bracket half-width and residual magnitude.
    pub tolerance: f64,
}

impl Brent {
    /// Creates a new [`Brent`] solver.
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
            tolerance: math_commons::registry::TOLERANCE_FAST,
        }
    }
}

#[verified_engine::verified]
fn brent_interpolated_candidate(a: f64, b: f64, c: f64, fa: f64, fb: f64, fc: f64) -> f64 {
    if (fa - fc).abs() > f64::EPSILON && (fb - fc).abs() > f64::EPSILON {
        let term_a = a * fb * fc / ((fa - fb) * (fa - fc));
        let term_b = b * fa * fc / ((fb - fa) * (fb - fc));
        let term_c = c * fa * fb / ((fc - fa) * (fc - fb));
        term_a + term_b + term_c
    } else {
        b - fb * (b - a) / (fb - fa)
    }
}

#[verified_engine::verified]
fn reject_brent_step(s: f64, a: f64, b: f64, c: f64, d: f64, mflag: bool, tol1: f64) -> bool {
    if !s.is_finite() {
        return true;
    }
    let quarter = 0.25 * (3.0 * a + b);
    let lo = quarter.min(b);
    let hi = quarter.max(b);
    let outside = s <= lo || s >= hi;
    let step_len = (s - b).abs();
    let prev_span = if mflag { (b - c).abs() } else { (c - d).abs() };
    outside || step_len >= 0.5 * prev_span || prev_span < tol1
}

impl RootFinder for Brent {
    #[verified_engine::verified]
    fn find_root<F>(&self, f: F, min: f64, max: f64) -> Result<f64, AnalysisError>
    where
        F: Fn(f64) -> f64,
    {
        let mut a = min;
        let mut b = max;
        let mut fa = f(a);
        let mut fb = f(b);
        validate_bracket_inputs(a, b, self.tolerance, fa, fb)?;

        if fa == 0.0 || fa.abs() <= self.tolerance {
            return Ok(canonical_zero(a));
        }
        if fb == 0.0 || fb.abs() <= self.tolerance {
            return Ok(canonical_zero(b));
        }
        if !has_strict_sign_change(fa, fb) {
            return Err(AnalysisError::InvalidParameters(format!(
                "Root not bracketed for Brent: f({a})={fa}, f({b})={fb}."
            )));
        }
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }

        let mut c = a;
        let mut fc = fa;
        let mut d = b - a;
        let mut mflag = true;

        for _ in 0..self.max_iterations {
            let tol1 = 2.0 * f64::EPSILON * b.abs() + 0.5 * self.tolerance;
            let xm = 0.5 * (a - b);
            if fb == 0.0 || fb.abs() <= self.tolerance || xm.abs() <= tol1 {
                return Ok(canonical_zero(b));
            }

            let mut s = brent_interpolated_candidate(a, b, c, fa, fb, fc);
            if reject_brent_step(s, a, b, c, d, mflag, tol1) {
                s = b + xm;
                mflag = true;
            } else {
                mflag = false;
            }

            d = c;
            c = b;
            fc = fb;
            let fs = f(s);
            if !fs.is_finite() {
                return Err(AnalysisError::InvalidParameters(format!(
                    "Non-finite evaluation during Brent step: f({s})={fs}."
                )));
            }
            if has_strict_sign_change(fa, fs) {
                b = s;
                fb = fs;
            } else {
                a = s;
                fa = fs;
            }
            if fa.abs() < fb.abs() {
                std::mem::swap(&mut a, &mut b);
                std::mem::swap(&mut fa, &mut fb);
            }
        }

        Err(AnalysisError::ConvergenceError(canonical_zero(b)))
    }
}

// [cite:graph_parameters_rust]

//! Forward-mode AD Newton-Raphson and certified outward-rounded interval root solvers.

use math_scalars::{Ball, Dual, Interval, Scalar};

use super::{AnalysisError, NewtonRaphson, validate_solver_bounds};

impl NewtonRaphson {
    /// Finds a root using exact forward-mode [`Dual<f64>`] automatic differentiation.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] if bounds are invalid, the derivative vanishes, or iteration fails to converge.
    #[verified_engine::verified]
    pub fn find_root_ad<F>(&self, f_ad: F, guess: f64) -> Result<f64, AnalysisError>
    where
        F: Fn(Dual<f64>) -> Dual<f64>,
    {
        validate_solver_bounds(guess, guess, self.tolerance)?;
        let mut x = guess;

        for _ in 0..self.max_iterations {
            let jet = f_ad(Dual::variable(x));
            if !jet.val.is_finite() {
                return Err(AnalysisError::ConvergenceError(x));
            }
            if jet.val.abs() <= self.tolerance {
                return Ok(x);
            }
            if !jet.eps.is_finite() || jet.eps.abs() < 1e-14 {
                return Err(AnalysisError::DerivativeTooSmall {
                    x,
                    derivative: jet.eps,
                });
            }
            let next_x = x - jet.val / jet.eps;
            if !next_x.is_finite() {
                return Err(AnalysisError::ConvergenceError(x));
            }
            let step_tol = self.tolerance.max(4.0 * f64::EPSILON * next_x.abs());
            if (next_x - x).abs() <= step_tol {
                return Ok(next_x);
            }
            x = next_x;
        }

        Err(AnalysisError::ConvergenceError(x))
    }
}

#[verified_engine::verified]
fn validate_interval_bracket<F>(f: &F, bracket: Interval<f64>) -> Result<(), AnalysisError>
where
    F: Fn(Dual<Interval<f64>>) -> Dual<Interval<f64>>,
{
    let (lo, hi) = (bracket.lo, bracket.hi);
    if !lo.is_finite() || !hi.is_finite() || lo > hi {
        return Err(AnalysisError::InvalidBracket { min: lo, max: hi });
    }
    let f_lo = f(Dual::constant(Interval::point(lo))).val;
    let f_hi = f(Dual::constant(Interval::point(hi))).val;
    if !f_lo.lo.is_finite() || !f_lo.hi.is_finite() || !f_hi.lo.is_finite() || !f_hi.hi.is_finite()
    {
        return Err(AnalysisError::RootNotBracketed {
            min: lo,
            f_min: f_lo.midpoint(),
            max: hi,
            f_max: f_hi.midpoint(),
        });
    }
    let strictly_pos = f_lo.lo > 0.0 && f_hi.lo > 0.0;
    let strictly_neg = f_lo.hi < 0.0 && f_hi.hi < 0.0;
    if strictly_pos || strictly_neg {
        return Err(AnalysisError::RootNotBracketed {
            min: lo,
            f_min: f_lo.midpoint(),
            max: hi,
            f_max: f_hi.midpoint(),
        });
    }
    Ok(())
}

#[verified_engine::verified]
fn bisect_certified_subinterval<F>(
    f: &F,
    x: Interval<f64>,
) -> Result<Option<Interval<f64>>, AnalysisError>
where
    F: Fn(Dual<Interval<f64>>) -> Dual<Interval<f64>>,
{
    let m = x.midpoint();
    if m <= x.lo || m >= x.hi {
        return Ok(None);
    }
    let left = Interval::new(x.lo, m);
    if f(Dual::constant(left)).val.contains(0.0) {
        let f_lo = f(Dual::constant(Interval::point(x.lo))).val;
        let f_m = f(Dual::constant(Interval::point(m))).val;
        if f_lo.contains(0.0) || f_m.contains(0.0) || (f_lo.lo > 0.0) != (f_m.lo > 0.0) {
            return Ok(Some(left));
        }
    }
    let right = Interval::new(m, x.hi);
    if f(Dual::constant(right)).val.contains(0.0) {
        return Ok(Some(right));
    }
    Err(AnalysisError::ConvergenceError(m))
}

/// Certified outward-rounded Interval Newton root-enclosure solver:
///
/// $$N(X) = m(X) - \frac{f(m(X))}{f'(X)}, \qquad X_{k+1} = X_k \cap N(X_k)$$
///
/// Reference: R. E. Moore, R. B. Kearfott & M. J. Cloud (2009), *Introduction to Interval Analysis*,
/// SIAM (DOI: `10.1137/1.9780898717716`); see the
/// [pure_math specification](../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
///
/// Outward-rounded [`Interval<f64>`] and [`Dual<Interval<f64>>`] arithmetic guarantees every
/// root $x^* \in X_k$ satisfies $x^* \in N(X_k)$, with quadratic width contraction near simple roots
/// and certified subinterval bisection if floating-point rounding stalls contraction before `max_ulp_width`.
///
/// # Domain of Validity
///
/// Requires a finite ordered bracket $X_0 = [\underline{x}, \overline{x}]$ whose endpoint enclosures do not
/// share a strict sign, and non-vanishing interval slope $0 \notin f'(X_k)$ (`AnalysisError::DerivativeTooSmall`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntervalNewton {
    /// Maximum contractor iterations before returning [`AnalysisError::ConvergenceError`].
    pub max_iterations: usize,
    /// Target certified enclosure width in IEEE-754 ULPs.
    pub max_ulp_width: u64,
}

impl IntervalNewton {
    /// Constructs a new [`IntervalNewton`] solver.
    #[must_use]
    #[verified_engine::verified]
    pub const fn new(max_iterations: usize, max_ulp_width: u64) -> Self {
        Self {
            max_iterations,
            max_ulp_width,
        }
    }

    /// Computes a certified outward-rounded [`Interval<f64>`] enclosing a unique root in `bracket`.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] if `bracket` is invalid/unbracketed, $0 \in f'(X)$, or convergence stalls.
    #[verified_engine::verified]
    pub fn enclose_root<F>(
        &self,
        f: F,
        bracket: Interval<f64>,
    ) -> Result<Interval<f64>, AnalysisError>
    where
        F: Fn(Dual<Interval<f64>>) -> Dual<Interval<f64>>,
    {
        validate_interval_bracket(&f, bracket)?;
        let mut x = bracket;

        for _ in 0..self.max_iterations {
            if x.width_ulps() <= self.max_ulp_width && f(Dual::constant(x)).val.contains(0.0) {
                return Ok(x);
            }
            let m = x.midpoint();
            let m_iv = Interval::point(m);
            let fm = f(Dual::constant(m_iv)).val;
            let df_x = f(Dual::variable(x)).eps;
            if df_x.contains(0.0) {
                return Err(AnalysisError::DerivativeTooSmall {
                    x: m,
                    derivative: 0.0,
                });
            }
            let newton_step = m_iv - (fm / df_x);
            let Some(next_x) = x.intersect(newton_step) else {
                return Err(AnalysisError::RootNotBracketed {
                    min: x.lo,
                    f_min: fm.lo,
                    max: x.hi,
                    f_max: fm.hi,
                });
            };
            if next_x.width_ulps() <= self.max_ulp_width {
                return Ok(next_x);
            }
            if next_x == x {
                if let Some(sub) = bisect_certified_subinterval(&f, x)? {
                    x = sub;
                    continue;
                }
                return Ok(next_x);
            }
            x = next_x;
        }

        if x.width_ulps() <= self.max_ulp_width {
            Ok(x)
        } else {
            Err(AnalysisError::ConvergenceError(x.midpoint()))
        }
    }

    /// Computes a certified midpoint-radius [`Ball<f64>`] enclosing the root in `bracket`.
    ///
    /// # Errors
    ///
    /// Propagates any [`AnalysisError`] from [`Self::enclose_root`].
    #[verified_engine::verified]
    pub fn enclose_root_ball<F>(
        &self,
        f: F,
        bracket: Interval<f64>,
    ) -> Result<Ball<f64>, AnalysisError>
    where
        F: Fn(Dual<Interval<f64>>) -> Dual<Interval<f64>>,
    {
        let iv = self.enclose_root(f, bracket)?;
        Ok(Ball::from_interval(iv))
    }
}

impl Default for IntervalNewton {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            max_iterations: 64,
            max_ulp_width: 2,
        }
    }
}

/// Certified outward-rounded Krawczyk contractor for nonlinear root verification:
///
/// $$K(X) = m(X) - Y f(m(X)) + \bigl(1 - Y f'(X)\bigr)\bigl(X - m(X)\bigr), \qquad X_{k+1} = X_k \cap K(X_k)$$
///
/// Reference: R. Krawczyk (1969), "Newton-Algorithmen zur Bestimmung von Nullstellen mit Fehlerschranken",
/// *Computing* 4(3), 187–201 (DOI: `10.1007/BF02242387`); see the
/// [pure_math specification](../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
///
/// Avoids dividing by the wide derivative enclosure $f'(X)$ directly by preconditioning with the scalar
/// midpoint inverse $Y \approx f'(m(X))^{-1}$, preventing division blow-up while preserving rigorous
/// outward-rounded root containment and certified bisection fallback.
///
/// # Domain of Validity
///
/// Requires a finite ordered bracket $X_0 = [\underline{x}, \overline{x}]$ without identical strict endpoint
/// signs and non-zero midpoint derivative $|f'(m(X_k))| \ge 10^{-14}$ (`AnalysisError::DerivativeTooSmall`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Krawczyk {
    /// Maximum contractor iterations before returning [`AnalysisError::ConvergenceError`].
    pub max_iterations: usize,
    /// Target certified enclosure width in IEEE-754 ULPs.
    pub max_ulp_width: u64,
}

impl Krawczyk {
    /// Constructs a new [`Krawczyk`] root-enclosure contractor.
    #[must_use]
    #[verified_engine::verified]
    pub const fn new(max_iterations: usize, max_ulp_width: u64) -> Self {
        Self {
            max_iterations,
            max_ulp_width,
        }
    }

    /// Computes a certified outward-rounded [`Interval<f64>`] enclosing a root via $K(X) \cap X$.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] if `bracket` is invalid, the midpoint Jacobian vanishes, or contraction fails.
    #[verified_engine::verified]
    pub fn enclose_root<F>(
        &self,
        f: F,
        bracket: Interval<f64>,
    ) -> Result<Interval<f64>, AnalysisError>
    where
        F: Fn(Dual<Interval<f64>>) -> Dual<Interval<f64>>,
    {
        validate_interval_bracket(&f, bracket)?;
        let mut x = bracket;

        for _ in 0..self.max_iterations {
            if x.width_ulps() <= self.max_ulp_width && f(Dual::constant(x)).val.contains(0.0) {
                return Ok(x);
            }
            let m = x.midpoint();
            let m_iv = Interval::point(m);
            let jet_m = f(Dual::variable(m_iv));
            let c = jet_m.eps.midpoint();
            if !c.is_finite() || c.abs() < 1e-14 {
                return Err(AnalysisError::DerivativeTooSmall {
                    x: m,
                    derivative: c,
                });
            }
            let y_iv = Interval::point(1.0 / c);
            let df_x = f(Dual::variable(x)).eps;
            let r_iv = Interval::ONE - (y_iv * df_x);
            let krawczyk_iv = m_iv - (y_iv * jet_m.val) + r_iv * (x - m_iv);

            let Some(next_x) = x.intersect(krawczyk_iv) else {
                return Err(AnalysisError::RootNotBracketed {
                    min: x.lo,
                    f_min: jet_m.val.lo,
                    max: x.hi,
                    f_max: jet_m.val.hi,
                });
            };
            if next_x.width_ulps() <= self.max_ulp_width {
                return Ok(next_x);
            }
            if next_x == x {
                if let Some(sub) = bisect_certified_subinterval(&f, x)? {
                    x = sub;
                    continue;
                }
                return Ok(next_x);
            }
            x = next_x;
        }

        if x.width_ulps() <= self.max_ulp_width {
            Ok(x)
        } else {
            Err(AnalysisError::ConvergenceError(x.midpoint()))
        }
    }

    /// Computes a certified midpoint-radius [`Ball<f64>`] enclosing the root in `bracket`.
    ///
    /// # Errors
    ///
    /// Propagates any [`AnalysisError`] from [`Self::enclose_root`].
    #[verified_engine::verified]
    pub fn enclose_root_ball<F>(
        &self,
        f: F,
        bracket: Interval<f64>,
    ) -> Result<Ball<f64>, AnalysisError>
    where
        F: Fn(Dual<Interval<f64>>) -> Dual<Interval<f64>>,
    {
        let iv = self.enclose_root(f, bracket)?;
        Ok(Ball::from_interval(iv))
    }
}

impl Default for Krawczyk {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            max_iterations: 64,
            max_ulp_width: 2,
        }
    }
}

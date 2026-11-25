// [cite:advanced_linear_algebra]
use super::AnalysisError;
use math_scalars::{Ball, Dual, Interval, interval_newton_enclose, krawczyk_enclose};

/// Rigorous Interval-Newton root-enclosure solver over IEEE-754 outward-rounded [`Interval<f64>`].
///
/// Iterates $N(X) = m - \frac{f([m, m])}{F'(X)}$ and $X_{k+1} = X_k \cap N(X_k)$ until the
/// enclosure contracts within `ulp_tolerance` ULPs or proves root non-existence ($N(X_k) \cap X_k = \emptyset$).
///
/// Reference: Moore (1966), *Interval Analysis*, Prentice-Hall; Neumaier (1990), *Interval Methods
/// for Systems of Equations*, Cambridge University Press.
///
/// # Stability
/// Outward-rounded directed ULP arithmetic guarantees every simple root inside `[min, max]` remains
/// strictly enclosed across native `x86_64` and `wasm32-unknown-unknown` without FPU mode mutations.
///
/// # Domain of Validity
/// Requires finite `min <= max` and $0 \notin F'([\text{min}, \text{max}])$ at convergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntervalNewton {
    /// Maximum contractor iterations before returning [`AnalysisError::ConvergenceError`].
    pub max_iterations: usize,
    /// Target maximum ULP width of the certified root bracket (clamped to `>= 1`).
    pub ulp_tolerance: u64,
}

impl IntervalNewton {
    /// Creates an [`IntervalNewton`] solver with explicit iteration and ULP width bounds.
    #[verified_engine::verified]
    pub const fn new(max_iterations: usize, ulp_tolerance: u64) -> Self {
        Self {
            max_iterations,
            ulp_tolerance: if ulp_tolerance == 0 { 1 } else { ulp_tolerance },
        }
    }

    /// Certifies a root enclosure inside `[min, max]` from a single forward-AD interval closure
    /// over [`Dual<Interval<f64>>`], automatically deriving the rigorous slope enclosure $F'(X)$.
    #[verified_engine::verified]
    pub fn enclose_root_ad<G>(
        &self,
        g: G,
        min: f64,
        max: f64,
    ) -> Result<Interval<f64>, AnalysisError>
    where
        G: Fn(Dual<Interval<f64>>) -> Dual<Interval<f64>>,
    {
        if !min.is_finite() || !max.is_finite() || min > max {
            return Err(AnalysisError::InvalidBracket { min, max });
        }
        let bracket = Interval::new(min, max);
        let f_box = g(Dual::constant(bracket)).val;
        if !f_box.contains_zero() {
            let f_min = g(Dual::constant(Interval::point(min))).val.midpoint();
            let f_max = g(Dual::constant(Interval::point(max))).val.midpoint();
            return Err(AnalysisError::RootNotBracketed {
                min,
                f_min,
                max,
                f_max,
            });
        }
        let cert = interval_newton_enclose(&g, bracket, self.max_iterations)
            .ok_or(AnalysisError::ConvergenceError(0.5 * (min + max)))?;
        if cert.enclosure.ulp_width() <= self.ulp_tolerance.max(4) {
            Ok(cert.enclosure)
        } else {
            Err(AnalysisError::ConvergenceError(cert.enclosure.midpoint()))
        }
    }

    /// Certifies a root enclosure inside `[min, max]` given separate interval extensions `f` and `df`.
    #[verified_engine::verified]
    pub fn enclose_root<F, DF>(
        &self,
        f: F,
        df: DF,
        min: f64,
        max: f64,
    ) -> Result<Interval<f64>, AnalysisError>
    where
        F: Fn(Interval<f64>) -> Interval<f64>,
        DF: Fn(Interval<f64>) -> Interval<f64>,
    {
        self.enclose_root_ad(
            |x: Dual<Interval<f64>>| Dual::new(f(x.val), df(x.val) * x.eps),
            min,
            max,
        )
    }
}

impl Default for IntervalNewton {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new(64, 4)
    }
}

/// Rigorous Krawczyk contractor root-enclosure solver returning certified midpoint-radius [`Ball<f64>`] bounds.
///
/// Iterates $K(X) = m - c f([m, m]) + (1 - c F'(X))(X - m)$ with $c = 1 / \operatorname{mid}(F'(X))$,
/// certifying existence and uniqueness whenever $K(X) \subseteq \operatorname{int}(X)$.
///
/// Reference: Krawczyk (1969), *Computing* 4(3), 187–201, DOI: 10.1007/BF02234767;
/// Rump (2010), *Acta Numerica* 19, 287–449, DOI: 10.1017/S096249291000005X.
///
/// # Stability
/// Avoids interval division by $F'(X)$, remaining well-defined near small derivative enclosures while
/// preserving outward-rounded containment guarantees down to `<= 4` ULPs.
///
/// # Domain of Validity
/// Requires finite `min <= max` and non-singular midpoint derivative $\operatorname{mid}(F'(X)) \neq 0$.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Krawczyk {
    /// Maximum contractor iterations before returning [`AnalysisError::ConvergenceError`].
    pub max_iterations: usize,
    /// Target maximum ULP width of the certified root ball (clamped to `>= 1`).
    pub ulp_tolerance: u64,
}

impl Krawczyk {
    /// Creates a [`Krawczyk`] solver with explicit iteration and ULP width bounds.
    #[verified_engine::verified]
    pub const fn new(max_iterations: usize, ulp_tolerance: u64) -> Self {
        Self {
            max_iterations,
            ulp_tolerance: if ulp_tolerance == 0 { 1 } else { ulp_tolerance },
        }
    }

    /// Certifies a [`Ball<f64>`] root enclosure inside `[min, max]` from a single [`Dual<Interval<f64>>`] closure.
    #[verified_engine::verified]
    pub fn enclose_root_ad<G>(&self, g: G, min: f64, max: f64) -> Result<Ball<f64>, AnalysisError>
    where
        G: Fn(Dual<Interval<f64>>) -> Dual<Interval<f64>>,
    {
        if !min.is_finite() || !max.is_finite() || min > max {
            return Err(AnalysisError::InvalidBracket { min, max });
        }
        let bracket = Interval::new(min, max);
        if !g(Dual::constant(bracket)).val.contains_zero() {
            let f_min = g(Dual::constant(Interval::point(min))).val.midpoint();
            let f_max = g(Dual::constant(Interval::point(max))).val.midpoint();
            return Err(AnalysisError::RootNotBracketed {
                min,
                f_min,
                max,
                f_max,
            });
        }
        let initial_ball = Ball::from_interval(bracket);
        let cert = krawczyk_enclose(&g, initial_ball, self.max_iterations)
            .ok_or(AnalysisError::ConvergenceError(0.5 * (min + max)))?;
        if cert.enclosure.ulp_width() <= self.ulp_tolerance.max(4) {
            Ok(cert.enclosure)
        } else {
            Err(AnalysisError::ConvergenceError(cert.enclosure.mid))
        }
    }

    /// Certifies a [`Ball<f64>`] root enclosure inside `[min, max]` given separate interval extensions `f` and `df`.
    #[verified_engine::verified]
    pub fn enclose_root<F, DF>(
        &self,
        f: F,
        df: DF,
        min: f64,
        max: f64,
    ) -> Result<Ball<f64>, AnalysisError>
    where
        F: Fn(Interval<f64>) -> Interval<f64>,
        DF: Fn(Interval<f64>) -> Interval<f64>,
    {
        self.enclose_root_ad(
            |x: Dual<Interval<f64>>| Dual::new(f(x.val), df(x.val) * x.eps),
            min,
            max,
        )
    }
}

impl Default for Krawczyk {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new(64, 4)
    }
}

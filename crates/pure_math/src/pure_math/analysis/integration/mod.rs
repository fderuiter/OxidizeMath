//! Numerical Integration Strategies (1D quadrature and hyperrectangle QMC/MC integration).
//!
//! Provides a common interface for scalar quadrature (`ClenshawCurtis`, `Trapezoidal` with
//! two-grid Richardson error estimation, composite `Simpson`, and stack-bounded adaptive
//! `GaussKronrod15`) alongside a preallocated [`MultivariateIntegrator`] over hyperrectangles
//! $[a_1, b_1] \times \dots \times [a_d, b_d]$ backed by Sobol, Halton, Latin Hypercube, or IID MC.

pub mod gauss_kronrod;
pub mod multivariate;

#[cfg(test)]
mod tests;

pub use gauss_kronrod::{GK15_MAX_PANELS, GaussKronrod15, GkSubinterval};
pub use multivariate::{MultivariateIntegrator, MultivariateRule};
use quadrature::clenshaw_curtis;

/// Result of an integration operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntegrationResult {
    /// The estimated value of the integral.
    pub value: f64,
    /// The estimated absolute error.
    pub error: f64,
}

/// A trait for numerical integration strategies.
pub trait Integrator: Send + Sync {
    /// Integrates the function `f` over the interval `[min, max]` with a target error `eps`.
    #[verified_engine::verified]
    fn integrate<F>(&self, f: F, min: f64, max: f64, eps: f64) -> IntegrationResult
    where
        F: Fn(f64) -> f64;
}

/// Clenshaw-Curtis Quadrature Integration.
///
/// Uses the `quadrature` crate.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClenshawCurtis;

impl Integrator for ClenshawCurtis {
    #[verified_engine::verified]
    fn integrate<F>(&self, f: F, min: f64, max: f64, eps: f64) -> IntegrationResult
    where
        F: Fn(f64) -> f64,
    {
        let result = clenshaw_curtis::integrate(f, min, max, eps);
        IntegrationResult {
            value: result.integral,
            error: result.error_estimate,
        }
    }
}

/// Composite Trapezoidal Rule Integration with two-grid Richardson error estimation.
///
/// Evaluates `steps` composite trapezoids alongside midpoint refinements (`T_{2n}`) to return
/// both the composite integral and a non-cancelling Richardson / local-variation error bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Trapezoidal {
    /// Number of steps (segments) to use.
    pub steps: usize,
}

impl Trapezoidal {
    /// Creates a new Trapezoidal integrator with the specified number of steps.
    #[verified_engine::verified]
    pub fn new(steps: usize) -> Self {
        Self { steps }
    }
}

impl Default for Trapezoidal {
    #[verified_engine::verified]
    fn default() -> Self {
        Self { steps: 100 }
    }
}

impl Integrator for Trapezoidal {
    #[verified_engine::verified]
    fn integrate<F>(&self, f: F, min: f64, max: f64, _eps: f64) -> IntegrationResult
    where
        F: Fn(f64) -> f64,
    {
        if self.steps == 0 || !min.is_finite() || !max.is_finite() || min == max {
            return IntegrationResult {
                value: 0.0,
                error: 0.0,
            };
        }

        let n = self.steps;
        let h = (max - min) / (n as f64);
        let h_abs = h.abs();
        let mut f_left = f(min);
        let f_end = f(max);
        let mut interior_sum = 0.0;
        let mut mid_sum = 0.0;
        let mut local_err = 0.0;
        let mut total_tv = 0.0;

        for i in 0..n {
            let f_mid = f(min + ((i as f64) + 0.5) * h);
            let f_right = if i + 1 == n {
                f_end
            } else {
                let y = f(min + ((i + 1) as f64) * h);
                interior_sum += y;
                y
            };
            mid_sum += f_mid;
            let d_i = (f_mid - 0.5 * (f_left + f_right)).abs();
            let tv_i = (f_right - f_mid).abs() + (f_mid - f_left).abs();
            total_tv += tv_i;
            let kappa = if tv_i > 0.0 {
                (2.0 * d_i / tv_i).clamp(0.0, 1.0)
            } else {
                0.0
            };
            local_err += h_abs * d_i * (2.0 / 3.0 + (4.0 / 3.0) * kappa);
            f_left = f_right;
        }

        let val_n = (0.5 * (f(min) + f_end) + interior_sum) * h;
        let val_2n = 0.5 * (val_n + mid_sum * h);
        let richardson = (val_2n - val_n).abs() * (4.0 / 3.0);
        let roundoff = if total_tv > 0.0 {
            16.0 * f64::EPSILON * val_n.abs().max(h_abs * total_tv)
        } else {
            0.0
        };

        IntegrationResult {
            value: val_n,
            error: richardson.max(local_err).max(roundoff),
        }
    }
}

/// Composite Simpson's 1/3 Rule Integration with two-grid Richardson error estimation.
///
/// Rounds odd `steps` up to the next even subinterval count and pairs coarse (`S_n`) and fine
/// (`S_{2n}`) parabolic panels with a fourth-difference kink/jump guard.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::integration::{Integrator, Simpson};
///
/// let res = Simpson::new(100).integrate(|x| x.sin(), 0.0, std::f64::consts::PI, 1e-10);
/// assert!((res.value - 2.0).abs() <= res.error);
/// assert!(res.error < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Simpson {
    /// Number of composite subintervals (rounded up to the next even integer `>= 2`).
    pub steps: usize,
}

impl Simpson {
    /// Creates a new composite `Simpson` integrator with `steps` subintervals.
    #[verified_engine::verified]
    pub fn new(steps: usize) -> Self {
        Self { steps }
    }
}

impl Default for Simpson {
    #[verified_engine::verified]
    fn default() -> Self {
        Self { steps: 100 }
    }
}

impl Integrator for Simpson {
    #[verified_engine::verified]
    fn integrate<F>(&self, f: F, min: f64, max: f64, _eps: f64) -> IntegrationResult
    where
        F: Fn(f64) -> f64,
    {
        if self.steps == 0 || !min.is_finite() || !max.is_finite() || min == max {
            return IntegrationResult {
                value: 0.0,
                error: 0.0,
            };
        }

        let base = self.steps.max(2);
        let n = if base % 2 == 1 { base + 1 } else { base };
        let panels = n / 2;
        let h = (max - min) / (n as f64);
        let mut y0 = f(min);
        let mut coarse_sum = 0.0;
        let mut fine_sum = 0.0;
        let mut local_err = 0.0;
        let mut total_tv = 0.0;

        for k in 0..panels {
            let x_base = min + ((2 * k) as f64) * h;
            let y1 = f(x_base + 0.5 * h);
            let y2 = f(x_base + h);
            let y3 = f(x_base + 1.5 * h);
            let y4 = if k + 1 == panels {
                f(max)
            } else {
                f(min + ((2 * (k + 1)) as f64) * h)
            };

            let s_coarse = (h / 3.0) * (y0 + 4.0 * y2 + y4);
            let s_fine = (h / 6.0) * (y0 + 4.0 * y1 + 2.0 * y2 + 4.0 * y3 + y4);
            coarse_sum += s_coarse;
            fine_sum += s_fine;

            let delta_s = (s_fine - s_coarse).abs();
            let d4 = (-y0 + 4.0 * y1 - 6.0 * y2 + 4.0 * y3 - y4).abs();
            let tv = (y1 - y0).abs() + (y2 - y1).abs() + (y3 - y2).abs() + (y4 - y3).abs();
            total_tv += tv;
            let eta = if tv > 0.0 { d4 / (4.0 * tv) } else { 0.0 };
            let factor = if eta >= 0.08 { 3.0 } else { 16.0 / 15.0 };
            local_err += factor * delta_s;
            y0 = y4;
        }

        let richardson = (fine_sum - coarse_sum).abs() * (16.0 / 15.0);
        let roundoff = if total_tv > 0.0 {
            16.0 * f64::EPSILON * coarse_sum.abs().max(h.abs() * total_tv)
        } else {
            0.0
        };

        IntegrationResult {
            value: coarse_sum,
            error: richardson.max(local_err).max(roundoff),
        }
    }
}

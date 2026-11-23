//! Lumped two-dimensional ordinary differential equation model for normal and cannibalistic cohorts.

// dN/dt = beta_N(N, C) * N + beta_C(N, C) * C - K(N) * N - phi(N, C) - mu_N(N, C) * N
// dC/dt = K(N) * N - mu_C(N, C) * C

use super::death_rate::sanitize_density;
use nalgebra::Vector2;
use pure_math::pure_math::analysis::ode::{OdeSystem, RungeKutta4, Solver};

/// Parameters for the two-dimensional normal-cannibal ODE system.
///
/// Models density-dependent recruitment, cannibalistic morph transition $k_N N$, and guarded
/// Holling Type I/II predation $\phi(N, C) = \frac{\phi_{NC} N_+ C_+}{1 + h N_+}$ so trajectories
/// starting in $\mathbb{R}_{\ge 0}^2$ remain non-negative.
///
/// # Example
///
/// ```rust
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::cannibalism::CannibalismModel;
/// use math_explorer::nalgebra::Vector2;
/// use math_explorer::pure_math::analysis::ode::OdeSystem;
///
/// let model = CannibalismModel::with_handling_time(0.8, 0.3, 0.15, 0.25, 0.1, 0.2, 0.05);
/// let deriv_at_zero = model.derivative(0.0, &Vector2::new(0.0, 5.0));
/// assert!(deriv_at_zero[0] >= 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CannibalismModel {
    /// Birth rate of normal individuals ($\beta_N$)
    pub beta_n: f64,
    /// Birth rate contribution from cannibalistic individuals ($\beta_C$)
    pub beta_c: f64,
    /// Rate at which normal individuals transition to cannibals ($k_N$)
    pub k_n: f64,
    /// Attack rate coefficient for loss of normal individuals due to cannibalism ($\phi_{NC}$)
    pub phi_n_c: f64,
    /// Mortality rate of normal individuals ($\mu_N$)
    pub mu_n: f64,
    /// Mortality rate of cannibalistic individuals ($\mu_C$)
    pub mu_c: f64,
    /// Holling Type II handling time ($h \ge 0$) saturating predation at high prey density
    pub handling_time: f64,
}

impl CannibalismModel {
    /// Creates a `CannibalismModel` with Holling Type I mass-action predation (`handling_time = 0.0`).
    #[verified_engine::verified]
    pub fn new(beta_n: f64, beta_c: f64, k_n: f64, phi_n_c: f64, mu_n: f64, mu_c: f64) -> Self {
        Self::with_handling_time(beta_n, beta_c, k_n, phi_n_c, mu_n, mu_c, 0.0)
    }

    /// Creates a `CannibalismModel` with explicit Holling Type II `handling_time` ($h \ge 0$).
    #[verified_engine::verified]
    pub fn with_handling_time(
        beta_n: f64,
        beta_c: f64,
        k_n: f64,
        phi_n_c: f64,
        mu_n: f64,
        mu_c: f64,
        handling_time: f64,
    ) -> Self {
        let clamped_h = if handling_time.is_finite() {
            handling_time.max(0.0)
        } else {
            0.0
        };
        debug_assert!(clamped_h >= 0.0);
        Self {
            beta_n,
            beta_c,
            k_n,
            phi_n_c,
            mu_n,
            mu_c,
            handling_time: clamped_h,
        }
    }

    /// Advances `(N(t), C(t))` by one 4th-order [`RungeKutta4`] step with non-negative population clamping.
    #[verified_engine::verified]
    pub fn step_rk4(&self, t: f64, state: &Vector2<f64>, dt: f64) -> Vector2<f64> {
        let mut clamped = Vector2::new(sanitize_density(state[0]), sanitize_density(state[1]));
        if !dt.is_finite() || dt <= 0.0 || (clamped[0] == 0.0 && clamped[1] == 0.0) {
            return clamped;
        }
        let mut rk4 = RungeKutta4::new(&clamped);
        rk4.step(self, t, &mut clamped, dt);
        Vector2::new(sanitize_density(clamped[0]), sanitize_density(clamped[1]))
    }
}

/// Implementation of the ODE System trait.
///
/// State vector components:
/// - index 0: Normal population ($N$)
/// - index 1: Cannibal population ($C$)
impl OdeSystem<Vector2<f64>> for CannibalismModel {
    #[verified_engine::verified]
    fn derivative(&self, _t: f64, state: &Vector2<f64>) -> Vector2<f64> {
        let n = if state[0].is_finite() {
            state[0].max(0.0)
        } else {
            0.0
        };
        let c = if state[1].is_finite() {
            state[1].max(0.0)
        } else {
            0.0
        };
        let h = self.handling_time.max(0.0);
        let denom = (1.0 + h * n).max(f64::EPSILON);
        let predation = self.phi_n_c.max(0.0) * n * c / denom;
        debug_assert!(predation >= 0.0);

        let dndt =
            self.beta_n * n + self.beta_c.max(0.0) * c - self.k_n * n - predation - self.mu_n * n;
        let dcdt = self.k_n.max(0.0) * n - self.mu_c * c;

        Vector2::new(dndt, dcdt)
    }
}

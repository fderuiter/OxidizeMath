//! Euler-Maruyama numerical solver for the Ornstein-Uhlenbeck process.

use super::core::OuParams;
use crate::error::OuError;
use engine_diffeq::{EulerMaruyamaSolver, SdeSystem, StochasticCalculus};
use math_commons::math_kernel::types::SolverWorkspace;
use rand::Rng;

struct ScalarOuSde {
    mu: f64,
    theta: f64,
    sigma: f64,
}

impl SdeSystem<f64> for ScalarOuSde {
    fn state_dim(&self) -> usize {
        1
    }
    fn wiener_dim(&self) -> usize {
        1
    }
    fn drift(&self, _t: f64, x: &[f64], out_drift: &mut [f64]) {
        out_drift[0] = self.theta * (self.mu - x[0]);
    }
    fn diffusion(&self, _t: f64, _x: &[f64], out_diffusion: &mut [f64]) {
        out_diffusion[0] = self.sigma;
    }
}

/// Time step for numerical integration (Δt).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeStep(f64);

impl TimeStep {
    /// Creates a new time step.
    ///
    /// # Arguments
    ///
    /// * `value` - The time step value (must be positive)
    ///
    /// # Returns
    ///
    /// * `Result<TimeStep, OuError>` - The validated time step or an error
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, OuError> {
        if value <= 0.0 || !value.is_finite() {
            return Err(OuError::InvalidTimeStep { value });
        }
        Ok(Self(value))
    }

    /// Returns the raw time step value.
    #[verified_engine::verified]
    pub fn value(&self) -> f64 {
        self.0
    }
}

/// Euler-Maruyama solver for the scalar Ornstein-Uhlenbeck SDE
/// $dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t$, advancing
/// $X_{t+\Delta t} = X_t + \theta(\mu - X_t)\Delta t + \sigma\sqrt{\Delta t}\,Z$ with $Z \sim \mathcal{N}(0, 1)$
/// via [`engine_diffeq::EulerMaruyamaSolver`].
///
/// Reference: Kloeden & Platen (1992), *Numerical Solution of Stochastic Differential Equations*,
/// DOI: 10.1007/978-3-662-12616-5; [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Mean-reverting drift contraction requires $0 < \theta \Delta t < 2$ (asymptotic stationary variance
/// approaches $\frac{\sigma^2}{2\theta(1 - \theta\Delta t / 2)}$ as $\theta\Delta t \to 0$).
///
/// # Domain of Validity
/// Valid for finite initial state `x0`, validated [`OuParams`] ($\theta > 0$, $\sigma \ge 0$, finite $\mu$),
/// and strictly positive finite [`TimeStep`] ($\Delta t > 0$).
pub struct EulerMaruyama {
    params: OuParams,
    dt: TimeStep,
    engine: EulerMaruyamaSolver,
}

impl EulerMaruyama {
    /// Creates a new Euler-Maruyama solver.
    ///
    /// # Arguments
    ///
    /// * `params` - The OU process parameters
    /// * `dt` - The time step for integration
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::ou_process::{
    ///     EulerMaruyama, OuParams, TimeStep
    /// };
    ///
    /// let params = OuParams::from_values(0.5, 1.0, 0.3).unwrap();
    /// let dt = TimeStep::new(0.01).unwrap();
    /// let solver = EulerMaruyama::new(params, dt);
    /// ```
    #[verified_engine::verified]
    pub fn new(params: OuParams, dt: TimeStep) -> Self {
        debug_assert!(dt.value() > 0.0);
        Self {
            params,
            dt,
            engine: EulerMaruyamaSolver::new(StochasticCalculus::Ito),
        }
    }

    /// Performs a single Euler-Maruyama step.
    ///
    /// # Arguments
    ///
    /// * `x_current` - The current value of the process
    /// * `rng` - Mutable reference to a random number generator
    ///
    /// # Returns
    ///
    /// The next value X_{t+Δt}
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::ou_process::{
    ///     EulerMaruyama, OuParams, TimeStep
    /// };
    /// use math_explorer::rand::SeedableRng;
    /// use math_explorer::rand::rngs::StdRng;
    ///
    /// let params = OuParams::from_values(0.5, 1.0, 0.3).unwrap();
    /// let dt = TimeStep::new(0.01).unwrap();
    /// let solver = EulerMaruyama::new(params, dt);
    ///
    /// let mut rng = math_explorer::rng::OxidizeRng::default();
    /// let x0 = 0.6;
    /// let x1 = solver.step(x0, &mut rng);
    /// ```
    #[verified_engine::verified]
    pub fn step<R: Rng>(&self, x_current: f64, rng: &mut R) -> f64 {
        let sde = ScalarOuSde {
            mu: self.params.mu.value(),
            theta: self.params.theta.value(),
            sigma: self.params.sigma.value(),
        };
        let mut state = [x_current];
        let mut scratch = [0.0_f64; 4];
        let mut ws = SolverWorkspace::new(&mut scratch);
        self.engine
            .step(&sde, 0.0, self.dt.value(), &mut state, &mut ws, rng);
        debug_assert!(state[0].is_finite() || !x_current.is_finite());
        state[0]
    }

    /// Simulates a trajectory of the OU process.
    ///
    /// # Arguments
    ///
    /// * `x0` - Initial value
    /// * `n_steps` - Number of time steps to simulate
    /// * `rng` - Mutable reference to a random number generator
    ///
    /// # Returns
    ///
    /// A vector containing the trajectory [X_0, X_1, ..., X_n]
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::ou_process::{
    ///     EulerMaruyama, OuParams, TimeStep
    /// };
    /// use math_explorer::rand::SeedableRng;
    /// use math_explorer::rand::rngs::StdRng;
    ///
    /// let params = OuParams::from_values(0.5, 1.0, 0.3).unwrap();
    /// let dt = TimeStep::new(0.01).unwrap();
    /// let solver = EulerMaruyama::new(params, dt);
    ///
    /// let mut rng = math_explorer::rng::OxidizeRng::default();
    /// let trajectory = solver.simulate(0.6, 100, &mut rng);
    /// assert_eq!(trajectory.len(), 101);  // Initial + 100 steps
    /// ```
    #[verified_engine::verified]
    pub fn simulate<R: Rng>(&self, x0: f64, n_steps: usize, rng: &mut R) -> Vec<f64> {
        let mut trajectory = Vec::with_capacity(n_steps + 1);
        trajectory.push(x0);

        let mut x = x0;
        for _ in 0..n_steps {
            x = self.step(x, rng);
            trajectory.push(x);
        }

        trajectory
    }

    /// Simulates multiple trajectories (Monte Carlo paths).
    ///
    /// # Arguments
    ///
    /// * `x0` - Initial value
    /// * `n_steps` - Number of time steps per trajectory
    /// * `n_paths` - Number of trajectories to simulate
    /// * `rng` - Mutable reference to a random number generator
    ///
    /// # Returns
    ///
    /// A vector of trajectories, each containing [X_0, X_1, ..., X_n]
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::ou_process::{
    ///     EulerMaruyama, OuParams, TimeStep
    /// };
    /// use math_explorer::rand::SeedableRng;
    /// use math_explorer::rand::rngs::StdRng;
    ///
    /// let params = OuParams::from_values(0.5, 1.0, 0.3).unwrap();
    /// let dt = TimeStep::new(0.01).unwrap();
    /// let solver = EulerMaruyama::new(params, dt);
    ///
    /// let mut rng = math_explorer::rng::OxidizeRng::default();
    /// let paths = solver.simulate_paths(0.6, 100, 1000, &mut rng);
    /// assert_eq!(paths.len(), 1000);  // 1000 paths
    /// assert_eq!(paths[0].len(), 101);  // Each path has 101 points
    /// ```
    #[verified_engine::verified]
    pub fn simulate_paths<R: Rng>(
        &self,
        x0: f64,
        n_steps: usize,
        n_paths: usize,
        rng: &mut R,
    ) -> Vec<Vec<f64>> {
        (0..n_paths)
            .map(|_| self.simulate(x0, n_steps, rng))
            .collect()
    }

    /// Returns the parameters of the OU process.
    #[verified_engine::verified]
    pub fn params(&self) -> OuParams {
        self.params
    }

    /// Returns the time step.
    #[verified_engine::verified]
    pub fn time_step(&self) -> TimeStep {
        self.dt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_time_step_valid() {
        let dt = TimeStep::new(0.01).unwrap();
        assert_eq!(dt.value(), 0.01);
    }

    #[test]
    #[verified_engine::verified]
    fn test_time_step_invalid() {
        assert!(TimeStep::new(-0.01).is_err());
        assert!(TimeStep::new(0.0).is_err());
        assert!(TimeStep::new(f64::NAN).is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_euler_maruyama_deterministic() {
        let params = OuParams::from_values(0.5, 1.0, 0.3).unwrap();
        let dt = TimeStep::new(0.01).unwrap();
        let solver = EulerMaruyama::new(params, dt);

        let mut rng1 = oxidize_core::rng::OxidizeRng::default();
        let mut rng2 = oxidize_core::rng::OxidizeRng::default();

        let x1 = solver.step(0.6, &mut rng1);
        let x2 = solver.step(0.6, &mut rng2);

        // Should be identical with same seed
        assert_eq!(x1, x2);
    }

    #[test]
    #[verified_engine::verified]
    fn test_euler_maruyama_simulate() {
        let params = OuParams::from_values(0.5, 1.0, 0.3).unwrap();
        let dt = TimeStep::new(0.01).unwrap();
        let solver = EulerMaruyama::new(params, dt);

        let mut rng = oxidize_core::rng::OxidizeRng::default();
        let trajectory = solver.simulate(0.6, 100, &mut rng);

        assert_eq!(trajectory.len(), 101);
        assert_eq!(trajectory[0], 0.6);
    }

    #[test]
    #[verified_engine::verified]
    fn test_euler_maruyama_mean_reversion() {
        // With high theta, should revert to mean quickly
        let params = OuParams::from_values(0.5, 5.0, 0.1).unwrap();
        let dt = TimeStep::new(0.01).unwrap();
        let solver = EulerMaruyama::new(params, dt);

        let mut rng = oxidize_core::rng::OxidizeRng::default();
        let trajectory = solver.simulate(1.0, 1000, &mut rng);

        // Final value should be close to mean (0.5)
        let final_value = trajectory.last().unwrap();
        assert!((final_value - 0.5).abs() < 0.3);
    }

    #[test]
    #[verified_engine::verified]
    fn test_euler_maruyama_multiple_paths() {
        let params = OuParams::from_values(0.5, 1.0, 0.3).unwrap();
        let dt = TimeStep::new(0.01).unwrap();
        let solver = EulerMaruyama::new(params, dt);

        let mut rng = oxidize_core::rng::OxidizeRng::default();
        let paths = solver.simulate_paths(0.6, 100, 10, &mut rng);

        assert_eq!(paths.len(), 10);
        for path in paths {
            assert_eq!(path.len(), 101);
            assert_eq!(path[0], 0.6);
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_euler_maruyama_no_volatility() {
        // With sigma = 0, should be purely deterministic drift
        let params = OuParams::from_values(0.5, 1.0, 0.0).unwrap();
        let dt = TimeStep::new(0.01).unwrap();
        let solver = EulerMaruyama::new(params, dt);

        let mut rng = oxidize_core::rng::OxidizeRng::default();
        let x0 = 0.6;
        let x1 = solver.step(x0, &mut rng);

        // Should be x0 + theta*(mu - x0)*dt = 0.6 + 1.0*(0.5 - 0.6)*0.01 = 0.599
        let expected = x0 + 1.0 * (0.5 - x0) * 0.01;
        assert_relative_eq!(x1, expected, epsilon = 1e-12);
    }

    #[test]
    #[verified_engine::verified]
    fn test_monte_carlo_mean() {
        // Simulate many paths and check the mean converges to mu
        let mu = 0.5;
        let params = OuParams::from_values(mu, 2.0, 0.2).unwrap();
        let dt = TimeStep::new(0.01).unwrap();
        let solver = EulerMaruyama::new(params, dt);

        let mut rng = oxidize_core::rng::OxidizeRng::new(12345);
        let n_steps = 500; // Long enough to reach equilibrium
        let n_paths = 5000;

        let paths = solver.simulate_paths(mu, n_steps, n_paths, &mut rng);

        // Average final value across all paths
        let final_mean: f64 = paths.iter().map(|p| p.last().unwrap()).sum::<f64>() / n_paths as f64;

        // Should be close to mu
        assert!((final_mean - mu).abs() < 0.05);
    }
}

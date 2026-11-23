//! Euler-Maruyama numerical solver for the Ornstein-Uhlenbeck process.

use super::core::OuParams;
pub use super::exact_solver::ExactOuSolver;
use crate::error::OuError;
use oxidize_core::rng::OxidizeRng;
use rand::Rng;
use rand_distr::{Distribution, Normal};

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

/// Euler-Maruyama solver for the OU process SDE.
///
/// Implements the numerical approximation:
///
/// ```text
/// X_{t+Δt} = X_t + θ(μ - X_t)Δt + σ√(Δt)Z
/// ```
///
/// where Z ~ N(0,1) is a standard normal random variable.
///
/// # References
///
/// Kloeden, P. E., & Platen, E. (1992). *Numerical Solution of Stochastic Differential Equations*.
/// Springer-Verlag.
pub struct EulerMaruyama {
    params: OuParams,
    dt: TimeStep,
    normal: Normal<f64>,
    diffusion_factor: f64,
}

impl EulerMaruyama {
    /// Creates a new Euler-Maruyama solver.
    ///
    /// # Arguments
    ///
    /// * `params` - The OU process parameters
    /// * `dt` - The time step for integration
    ///
    /// # Panics
    ///
    /// Panics if the standard normal distribution (`Normal::new(0.0, 1.0)`) fails to initialize.
    /// This should theoretically never happen as 0.0 and 1.0 are valid parameters for the mean
    /// and standard deviation.
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
        let normal = Normal::new(0.0, 1.0).unwrap();
        let diffusion_factor = params.sigma.value() * dt.value().sqrt();
        Self {
            params,
            dt,
            normal,
            diffusion_factor,
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
        let mu = self.params.mu.value();
        let theta = self.params.theta.value();
        let dt = self.dt.value();

        // Drift term: θ(μ - X_t)Δt
        let drift = theta * (mu - x_current) * dt;

        // Diffusion term: σ√(Δt)Z where Z ~ N(0,1)
        let diffusion = self.normal.sample(rng) * self.diffusion_factor;

        // X_{t+Δt} = X_t + drift + diffusion
        x_current + drift + diffusion
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

    /// Fills `out` in-place using `OxidizeRng` with **zero heap allocations**.
    #[verified_engine::verified]
    pub fn simulate_into(&self, x0: f64, out: &mut [f64], rng: &mut OxidizeRng) {
        if out.is_empty() {
            return;
        }
        out[0] = x0;
        let mut x = x0;
        for slot in out.iter_mut().skip(1) {
            x = self.step(x, rng);
            *slot = x;
        }
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

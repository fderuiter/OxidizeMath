//! Exact discrete-time Gaussian transition solver for the Ornstein-Uhlenbeck process.
//!
//! Uses the exact conditional transition law:
//! $$X_{t+\Delta t} \mid X_t = x \sim \mathcal{N}\!\left(\mu + (x - \mu)e^{-\theta \Delta t},\,
//! \frac{\sigma^2}{2\theta}(1 - e^{-2\theta \Delta t})\right).$$

use super::core::OuParams;
use super::solver::TimeStep;
use oxidize_core::rng::OxidizeRng;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Exact Gaussian transition solver for the Ornstein-Uhlenbeck SDE.
#[derive(Debug, Clone)]
pub struct ExactOuSolver {
    params: OuParams,
    dt: TimeStep,
    normal: Normal<f64>,
    decay: f64,
    std_step: f64,
}

impl ExactOuSolver {
    /// Creates an [`ExactOuSolver`] precomputing exact exponential decay and innovation scale.
    ///
    /// # Panics
    ///
    /// Panics only if `Normal::new(0.0, 1.0)` fails (infallible for standard normal constants).
    #[must_use]
    #[verified_engine::verified]
    pub fn new(params: OuParams, dt: TimeStep) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let theta = params.theta.value();
        let dt_val = dt.value();
        let decay = (-theta * dt_val).exp();
        let std_step = exact_conditional_variance(theta, params.sigma.value(), dt_val).sqrt();
        Self {
            params,
            dt,
            normal,
            decay,
            std_step,
        }
    }

    /// Exact conditional expectation $\mathbb{E}[X_{t+\tau} \mid X_t = x_0] = \mu + (x_0 - \mu)e^{-\theta \tau}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn conditional_mean(&self, x0: f64, horizon: f64) -> f64 {
        let mu = self.params.mu.value();
        let theta = self.params.theta.value();
        mu + (x0 - mu) * (-theta * horizon.max(0.0)).exp()
    }

    /// Exact conditional variance $\operatorname{Var}(X_{t+\tau} \mid X_t = x_0) = \frac{\sigma^2}{2\theta}(1 - e^{-2\theta \tau})$.
    #[must_use]
    #[verified_engine::verified]
    pub fn conditional_variance(&self, horizon: f64) -> f64 {
        exact_conditional_variance(
            self.params.theta.value(),
            self.params.sigma.value(),
            horizon.max(0.0),
        )
    }

    /// Advances a single exact OU transition step using any `rand::Rng`.
    #[verified_engine::verified]
    pub fn step<R: Rng>(&self, x_current: f64, rng: &mut R) -> f64 {
        let mu = self.params.mu.value();
        let mean_next = mu + (x_current - mu) * self.decay;
        mean_next + self.std_step * self.normal.sample(rng)
    }

    /// Fills `out` (`[X_0, X_1, ..., X_{n-1}]`) in-place using `OxidizeRng` with **zero heap allocations**.
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

    /// Simulates a trajectory `[X_0, X_1, ..., X_{n_steps}]` of length `n_steps + 1`.
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

    /// Simulates `n_paths` independent exact OU trajectories.
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

    /// Returns the OU parameters.
    #[must_use]
    #[verified_engine::verified]
    pub fn params(&self) -> OuParams {
        self.params
    }

    /// Returns the integration time step.
    #[must_use]
    #[verified_engine::verified]
    pub fn time_step(&self) -> TimeStep {
        self.dt
    }
}

#[verified_engine::verified]
fn exact_conditional_variance(theta: f64, sigma: f64, tau: f64) -> f64 {
    if tau <= 0.0 || sigma <= 0.0 {
        return 0.0;
    }
    let two_theta_tau = 2.0 * theta * tau;
    if two_theta_tau < 1e-8 {
        // Stable Taylor expansion for small 2*theta*tau
        let half = theta * tau;
        sigma * sigma * tau * (1.0 - half + (2.0 / 3.0) * half * half)
    } else {
        (sigma * sigma / (2.0 * theta)) * (-two_theta_tau).exp_m1().neg_val()
    }
}

trait NegExt {
    fn neg_val(self) -> Self;
}

impl NegExt for f64 {
    #[verified_engine::verified]
    fn neg_val(self) -> Self {
        -self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_exact_ou_solver_conditional_moments_and_zero_alloc() {
        let params = OuParams::from_values(1.2, 3.5, 0.8).unwrap();
        // Use a large step dt = 0.5 where Euler-Maruyama would be severely biased (theta*dt = 1.75)
        let dt = TimeStep::new(0.5).unwrap();
        let solver = ExactOuSolver::new(params, dt);

        let x0 = 4.0;
        let expected_mean = solver.conditional_mean(x0, 0.5);
        let expected_var = solver.conditional_variance(0.5);

        let mut rng = OxidizeRng::new(2026);
        let n_paths = 20_000_usize;
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let mut traj_buf = [0.0_f64; 8];

        let (_, metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            for _ in 0..n_paths {
                let x1 = solver.step(x0, &mut rng);
                sum += x1;
                sum_sq += x1 * x1;
            }
            solver.simulate_into(x0, &mut traj_buf, &mut rng);
        });
        assert!(metrics.function_calls > 0);
        assert_eq!(traj_buf[0], x0);

        let sample_mean = sum / n_paths as f64;
        let sample_var = (sum_sq / n_paths as f64) - sample_mean * sample_mean;
        assert_relative_eq!(sample_mean, expected_mean, epsilon = 0.015);
        assert_relative_eq!(sample_var, expected_var, epsilon = 0.015);
    }
}

// [cite:clinical_trials_statistics]

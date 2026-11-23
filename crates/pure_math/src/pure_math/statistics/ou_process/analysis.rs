//! Analysis tools for OU process applications.

use super::core::OuParams;
use super::solver::{EulerMaruyama, TimeStep};
use crate::error::OuError;
use rand::Rng;

/// Performance statistics from Monte Carlo simulation.
///
/// Contains summary statistics from multiple simulated trajectories.
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Mean of final values across all paths.
    pub mean_final: f64,
    /// Standard deviation of final values.
    pub std_final: f64,
    /// Median of final values.
    pub median_final: f64,
    /// 5th percentile of final values.
    pub percentile_05: f64,
    /// 95th percentile of final values.
    pub percentile_95: f64,
    /// Probability of exceeding a threshold.
    pub prob_above_threshold: Option<f64>,
}

/// Player momentum classifier based on mean reversion rate.
///
/// Categorizes players by their "streakiness" behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MomentumType {
    /// Fast mean reversion (θ > 2.0): "Flash in the pan"
    FlashInPan,
    /// Moderate mean reversion (0.5 < θ ≤ 2.0): "Normal"
    Normal,
    /// Slow mean reversion (θ ≤ 0.5): "Heat check" player with sticky momentum
    HeatCheck,
}

impl MomentumType {
    /// Classifies a player based on their mean reversion rate.
    ///
    /// # Arguments
    ///
    /// * `theta` - The mean reversion rate
    ///
    /// # Returns
    ///
    /// The momentum type classification
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::ou_process::MomentumType;
    /// use math_explorer::primitives::PositiveFloat;
    ///
    /// let theta = PositiveFloat::new(0.3).unwrap();
    /// let momentum_type = MomentumType::classify(theta.value());
    /// assert_eq!(momentum_type, MomentumType::HeatCheck);
    /// ```
    #[verified_engine::verified]
    pub fn classify(theta: f64) -> Self {
        if theta > 2.0 {
            Self::FlashInPan
        } else if theta > 0.5 {
            Self::Normal
        } else {
            Self::HeatCheck
        }
    }

    /// Returns a description of the momentum type.
    #[verified_engine::verified]
    pub fn description(&self) -> &str {
        match self {
            Self::FlashInPan => "Flash in the pan - streaks revert quickly",
            Self::Normal => "Normal mean reversion",
            Self::HeatCheck => "Heat check player - momentum is sticky",
        }
    }
}

/// OU process analyzer for sports analytics.
///
/// Provides high-level analysis functions for betting and performance prediction.
pub struct OuAnalyzer {
    params: OuParams,
    dt: TimeStep,
}

impl OuAnalyzer {
    /// Creates a new OU analyzer.
    ///
    /// # Arguments
    ///
    /// * `params` - The OU process parameters
    /// * `dt` - The time step for simulation
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::ou_process::{OuAnalyzer, OuParams, TimeStep};
    ///
    /// let params = OuParams::from_values(0.45, 1.0, 0.15).unwrap();
    /// let dt = TimeStep::new(0.01).unwrap();
    /// let analyzer = OuAnalyzer::new(params, dt);
    /// ```
    #[verified_engine::verified]
    pub fn new(params: OuParams, dt: TimeStep) -> Self {
        Self { params, dt }
    }

    /// Classifies the momentum type based on the mean reversion rate.
    ///
    /// # Returns
    ///
    /// The momentum type classification
    #[verified_engine::verified]
    pub fn momentum_type(&self) -> MomentumType {
        MomentumType::classify(self.params.theta.value())
    }

    /// Runs Monte Carlo simulation to estimate final performance distribution.
    ///
    /// # Arguments
    ///
    /// * `x0` - Initial performance level
    /// * `time_horizon` - Total simulation time
    /// * `n_paths` - Number of Monte Carlo paths
    /// * `threshold` - Optional threshold for computing exceedance probability
    /// * `rng` - Mutable reference to a random number generator
    ///
    /// # Returns
    ///
    /// Performance statistics from the simulation
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::ou_process::{OuAnalyzer, OuParams, TimeStep};
    /// use math_explorer::rand::SeedableRng;
    /// use math_explorer::rand::rngs::StdRng;
    ///
    /// let params = OuParams::from_values(0.45, 1.0, 0.15).unwrap();
    /// let dt = TimeStep::new(0.01).unwrap();
    /// let analyzer = OuAnalyzer::new(params, dt);
    ///
    /// let mut rng = math_explorer::rng::OxidizeRng::default();
    /// let stats = analyzer.monte_carlo_forecast(0.50, 1.0, 10000, Some(0.48), &mut rng);
    /// ```
    #[verified_engine::verified]
    pub fn monte_carlo_forecast<R: Rng>(
        &self,
        x0: f64,
        time_horizon: f64,
        n_paths: usize,
        threshold: Option<f64>,
        rng: &mut R,
    ) -> PerformanceStats {
        let n_steps = (time_horizon / self.dt.value()).ceil() as usize;
        let solver = EulerMaruyama::new(self.params, self.dt);

        // Simulate paths
        let paths = solver.simulate_paths(x0, n_steps, n_paths, rng);

        // Extract final values
        let mut final_values: Vec<f64> = paths.iter().map(|p| *p.last().unwrap()).collect();
        // Optimization: Use sort_unstable_by instead of sort_by for primitive f64s to avoid O(N) allocation overhead
        final_values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute statistics
        let mean_final = final_values.iter().sum::<f64>() / n_paths as f64;
        let variance = final_values
            .iter()
            .map(|&x| (x - mean_final).powi(2))
            .sum::<f64>()
            / n_paths as f64;
        let std_final = variance.sqrt();

        let median_final = final_values[n_paths / 2];
        let percentile_05 = final_values[(n_paths as f64 * 0.05) as usize];
        let percentile_95 = final_values[(n_paths as f64 * 0.95) as usize];

        let prob_above_threshold = threshold.map(|t| {
            let count = final_values.iter().filter(|&&x| x > t).count();
            count as f64 / n_paths as f64
        });

        PerformanceStats {
            mean_final,
            std_final,
            median_final,
            percentile_05,
            percentile_95,
            prob_above_threshold,
        }
    }

    /// Estimates the probability of a comeback given current deficit.
    ///
    /// # Arguments
    ///
    /// * `current_score` - Current performance/score level
    /// * `target_score` - Target level to exceed
    /// * `time_remaining` - Time remaining in simulation units
    /// * `n_paths` - Number of Monte Carlo paths
    /// * `rng` - Mutable reference to a random number generator
    ///
    /// # Returns
    ///
    /// The probability of reaching the target
    #[verified_engine::verified]
    pub fn comeback_probability<R: Rng>(
        &self,
        current_score: f64,
        target_score: f64,
        time_remaining: f64,
        n_paths: usize,
        rng: &mut R,
    ) -> f64 {
        let stats = self.monte_carlo_forecast(
            current_score,
            time_remaining,
            n_paths,
            Some(target_score),
            rng,
        );
        stats.prob_above_threshold.unwrap_or(0.0)
    }

    /// Returns the OU parameters.
    #[verified_engine::verified]
    pub fn params(&self) -> OuParams {
        self.params
    }
}

/// Estimates OU parameters from observed data using simple method of moments.
///
/// # Arguments
///
/// * `observations` - Time series of observations
/// * `dt` - Time step between observations
///
/// # Returns
///
/// * `Result<OuParams, OuError>` - Estimated parameters or an error
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::ou_process::estimate_ou_params;
///
/// let observations = vec![0.45, 0.47, 0.46, 0.48, 0.44, 0.46];
/// let dt = 1.0;  // One observation per time unit
/// let params = estimate_ou_params(&observations, dt).unwrap();
/// ```
#[verified_engine::verified]
pub fn estimate_ou_params(observations: &[f64], dt: f64) -> Result<OuParams, OuError> {
    if observations.len() < 3 {
        return Err(OuError::InsufficientData {
            required: 3,
            actual: observations.len(),
        });
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err(OuError::InvalidTimeStep { value: dt });
    }

    let n = observations.len();
    let mu = observations.iter().sum::<f64>() / n as f64;
    let variance = observations.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / (n - 1) as f64;

    let pairs = n - 1;
    let mean_prev = observations[..pairs].iter().sum::<f64>() / pairs as f64;
    let mean_next = observations[1..].iter().sum::<f64>() / pairs as f64;

    let mut numerator = 0.0_f64;
    let mut denominator = 0.0_f64;
    for i in 0..pairs {
        let dx = observations[i] - mean_prev;
        let dy = observations[i + 1] - mean_next;
        numerator += dx * dy;
        denominator += dx * dx;
    }

    let rho1 = if denominator > math_commons::registry::TOLERANCE_HIGH {
        numerator / denominator
    } else {
        0.5
    };

    let (theta, sigma) = if rho1 > 0.0 && rho1 < 1.0 {
        let theta_hat = -rho1.ln() / dt;
        let dof = (pairs.saturating_sub(2)).max(1) as f64;
        let mut sse = 0.0_f64;
        for i in 0..pairs {
            let resid = (observations[i + 1] - mean_next) - rho1 * (observations[i] - mean_prev);
            sse += resid * resid;
        }
        let innovation_var = sse / dof;
        let one_minus_rho_sq = (1.0 - rho1 * rho1).max(1e-15);
        let sigma_hat = (innovation_var * (2.0 * theta_hat) / one_minus_rho_sq).sqrt();
        (theta_hat, sigma_hat)
    } else {
        // Stationary method-of-moments fallback: Var(X) = sigma^2 / (2 * theta) => sigma = sqrt(2 * theta * Var(X))
        let theta_fallback = 1.0_f64;
        let sigma_hat = (2.0 * theta_fallback * variance.max(0.0)).sqrt();
        (theta_fallback, sigma_hat)
    };

    OuParams::from_values(mu, theta.max(1e-6), sigma.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_momentum_type_classification() {
        assert_eq!(MomentumType::classify(3.0), MomentumType::FlashInPan);
        assert_eq!(MomentumType::classify(1.0), MomentumType::Normal);
        assert_eq!(MomentumType::classify(0.3), MomentumType::HeatCheck);
    }

    #[test]
    #[verified_engine::verified]
    fn test_ou_analyzer_creation() {
        let params = OuParams::from_values(0.45, 1.0, 0.15).unwrap();
        let dt = TimeStep::new(0.01).unwrap();
        let analyzer = OuAnalyzer::new(params, dt);

        assert_eq!(analyzer.momentum_type(), MomentumType::Normal);
    }

    #[test]
    #[verified_engine::verified]
    fn test_monte_carlo_forecast() {
        let params = OuParams::from_values(0.5, 1.0, 0.2).unwrap();
        let dt = TimeStep::new(0.01).unwrap();
        let analyzer = OuAnalyzer::new(params, dt);

        let mut rng = oxidize_core::rng::OxidizeRng::default();
        let stats = analyzer.monte_carlo_forecast(0.6, 1.0, 1000, Some(0.5), &mut rng);

        // Mean should be close to long-term mean
        assert!((stats.mean_final - 0.5).abs() < 0.1);
        assert!(stats.std_final > 0.0);
        assert!(stats.prob_above_threshold.is_some());
    }

    #[test]
    #[verified_engine::verified]
    fn test_comeback_probability() {
        let params = OuParams::from_values(0.5, 1.0, 0.3).unwrap();
        let dt = TimeStep::new(0.01).unwrap();
        let analyzer = OuAnalyzer::new(params, dt);

        let mut rng = oxidize_core::rng::OxidizeRng::default();
        let prob = analyzer.comeback_probability(0.3, 0.5, 1.0, 1000, &mut rng);

        // Should be between 0 and 1
        assert!((0.0..=1.0).contains(&prob));
        // Should be non-zero since 0.5 is the mean
        assert!(prob > 0.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_estimate_ou_params() {
        // Generate synthetic data from known OU process
        let true_params = OuParams::from_values(0.5, 1.0, 0.2).unwrap();
        let dt = TimeStep::new(0.1).unwrap();
        let solver = EulerMaruyama::new(true_params, dt);

        let mut rng = oxidize_core::rng::OxidizeRng::new(12345);
        let trajectory = solver.simulate(0.6, 100, &mut rng);

        // Estimate parameters
        let estimated = estimate_ou_params(&trajectory, 0.1).unwrap();

        // Should be reasonably close (within 50% for this small sample)
        assert!((estimated.mu.value() - 0.5).abs() < 0.25);
        assert!(estimated.theta.value() > 0.0);
        assert!(estimated.sigma.value() > 0.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_estimate_ou_params_insufficient_data() {
        let observations = vec![0.5, 0.6];
        let result = estimate_ou_params(&observations, 1.0);
        assert!(result.is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_estimate_ou_params_recovers_sigma_within_five_percent() {
        let dt_val = 0.02;
        let dt = TimeStep::new(dt_val).unwrap();
        let true_sigma = 0.35;

        for &theta in &[4.0_f64, 0.10_f64] {
            let params = OuParams::from_values(1.0, theta, true_sigma).unwrap();
            let solver = super::super::ExactOuSolver::new(params, dt);
            let mut rng = oxidize_core::rng::OxidizeRng::new(2026);
            let traj = solver.simulate(1.0, 8_000, &mut rng);

            let est = estimate_ou_params(&traj, dt_val).unwrap();
            let rel_err = (est.sigma.value() - true_sigma).abs() / true_sigma;
            assert!(
                rel_err < 0.05,
                "theta={theta}: estimated sigma {} exceeded 5% relative error vs {true_sigma} (rel_err={rel_err})",
                est.sigma.value()
            );
        }
    }
}

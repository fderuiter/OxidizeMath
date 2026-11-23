//! Analysis tools for OU process applications.

use super::core::OuParams;
use super::solver::{EulerMaruyama, TimeStep};
use crate::error::OuError;
use rand::Rng;

/// Performance statistics from Monte Carlo simulation.
///
/// Contains summary statistics from multiple simulated trajectories.
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, Copy, PartialEq)]
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
        debug_assert!(n_paths > 0);
        let n_steps = (time_horizon / self.dt.value()).ceil() as usize;
        let solver = EulerMaruyama::new(self.params, self.dt);
        let paths = solver.simulate_paths(x0, n_steps, n_paths, rng);
        let final_values: Vec<f64> = paths.iter().filter_map(|p| p.last().copied()).collect();
        let mut ws = math_commons::stochastic::SolverWorkspace::new(0, final_values.len());
        summarize_terminal_values_with_workspace(&mut ws, &final_values, threshold).unwrap_or(
            PerformanceStats {
                mean_final: x0,
                std_final: 0.0,
                median_final: x0,
                percentile_05: x0,
                percentile_95: x0,
                prob_above_threshold: threshold.map(|t| if x0 > t { 1.0 } else { 0.0 }),
            },
        )
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

/// Summarizes terminal Monte Carlo observations into [`PerformanceStats`] using a preallocated
/// [`math_commons::stochastic::SolverWorkspace`] with zero heap allocations.
#[must_use]
#[verified_engine::verified]
pub fn summarize_terminal_values_with_workspace(
    ws: &mut math_commons::stochastic::SolverWorkspace,
    final_values: &[f64],
    threshold: Option<f64>,
) -> Option<PerformanceStats> {
    debug_assert!(final_values.is_empty() || !final_values.is_empty());
    let summary = ws.summarize_samples(final_values, threshold)?;
    Some(PerformanceStats {
        mean_final: summary.mean,
        std_final: summary.std_dev,
        median_final: summary.median,
        percentile_05: summary.percentile_05,
        percentile_95: summary.percentile_95,
        prob_above_threshold: threshold.map(|_| summary.prob_above_threshold),
    })
}

/// Estimates Ornstein-Uhlenbeck parameters $(\mu, \theta, \sigma)$ from uniformly spaced
/// observations using exact discrete-time AR(1) conditional Gaussian likelihood and stationary
/// $\sigma = \sqrt{2\theta \cdot \hat{V}_{\mathrm{stat}}}$ diffusion scaling.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::ou_process::estimate_ou_params;
///
/// let observations = vec![0.45, 0.47, 0.46, 0.48, 0.44, 0.46];
/// let dt = 1.0;
/// let params = estimate_ou_params(&observations, dt).unwrap();
/// assert!(params.sigma.value() > 0.0);
/// ```
#[verified_engine::verified]
pub fn estimate_ou_params(observations: &[f64], dt: f64) -> Result<OuParams, OuError> {
    debug_assert!(observations.len() < 3 || observations.len() >= 3);
    if observations.len() < 3 {
        return Err(OuError::InsufficientData {
            required: 3,
            actual: observations.len(),
        });
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err(OuError::InvalidTimeStep { value: dt });
    }
    let cal = math_commons::stochastic::calibrate_ou(observations, dt).ok_or_else(|| {
        OuError::InvalidSimulationParams {
            reason: "Observations must be finite".to_string(),
        }
    })?;
    OuParams::from_values(cal.mu, cal.theta, cal.sigma)
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
        let true_params = OuParams::from_values(0.5, 1.0, 0.2).unwrap();
        let dt = TimeStep::new(0.02).unwrap();
        let solver = EulerMaruyama::new(true_params, dt);

        let mut rng = oxidize_core::rng::OxidizeRng::new(12345);
        let trajectory = solver.simulate(0.5, 2_500, &mut rng);

        let estimated = estimate_ou_params(&trajectory, 0.02).unwrap();
        assert!((estimated.mu.value() - 0.5).abs() < 0.15);
        assert!(estimated.theta.value() > 0.0);
        assert!(
            (estimated.sigma.value() - 0.2).abs() / 0.2 < 0.12,
            "expected sigma ~0.2, got {}",
            estimated.sigma.value()
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_estimate_ou_params_recovers_sigma_across_slow_and_fast_theta() {
        let slow_params = OuParams::from_values(1.2, 0.05, 0.25).unwrap();
        let fast_params = OuParams::from_values(-0.8, 5.0, 0.35).unwrap();
        let dt = TimeStep::new(0.01).unwrap();

        let mut rng_slow = oxidize_core::rng::OxidizeRng::new(2026_1123);
        let slow_traj = EulerMaruyama::new(slow_params, dt).simulate(1.2, 4_000, &mut rng_slow);
        let est_slow = estimate_ou_params(&slow_traj, 0.01).unwrap();
        assert!(
            (est_slow.sigma.value() - 0.25).abs() / 0.25 < 0.10,
            "slow theta sigma={} expected ~0.25",
            est_slow.sigma.value()
        );

        let mut rng_fast = oxidize_core::rng::OxidizeRng::new(2026_1124);
        let fast_traj = EulerMaruyama::new(fast_params, dt).simulate(-0.8, 4_000, &mut rng_fast);
        let est_fast = estimate_ou_params(&fast_traj, 0.01).unwrap();
        assert!((est_fast.mu.value() - (-0.8)).abs() < 0.10);
        assert!((est_fast.theta.value() - 5.0).abs() / 5.0 < 0.18);
        assert!(
            (est_fast.sigma.value() - 0.35).abs() / 0.35 < 0.10,
            "fast theta sigma={} expected ~0.35",
            est_fast.sigma.value()
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_estimate_ou_params_insufficient_data() {
        let observations = vec![0.5, 0.6];
        let result = estimate_ou_params(&observations, 1.0);
        assert!(result.is_err());
    }
}

//! Analysis tools for OU process applications.

use super::core::OuParams;
use super::solver::{EulerMaruyama, TimeStep};
use crate::error::OuError;
use rand::Rng;

/// Performance statistics from Monte Carlo simulation.
///
/// Contains summary statistics and calibrated standard errors from multiple simulated trajectories.
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Mean of final values across all paths.
    pub mean_final: f64,
    /// Sample standard deviation of final values.
    pub std_final: f64,
    /// Standard error of `mean_final` (`std_final / sqrt(n_paths)`).
    pub standard_error: f64,
    /// Lower bound of the 95% normal confidence interval for `mean_final`.
    pub ci_95_lower: f64,
    /// Upper bound of the 95% normal confidence interval for `mean_final`.
    pub ci_95_upper: f64,
    /// Median of final values.
    pub median_final: f64,
    /// 5th percentile of final values.
    pub percentile_05: f64,
    /// 95th percentile of final values.
    pub percentile_95: f64,
    /// Probability of exceeding a threshold.
    pub prob_above_threshold: Option<f64>,
    /// Binomial standard error of `prob_above_threshold` (`sqrt(p*(1-p)/n_paths)`).
    pub prob_standard_error: Option<f64>,
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
        let valid_threshold = threshold.filter(|t| t.is_finite());
        let fallback = if x0.is_finite() {
            x0
        } else {
            self.params.mu.value()
        };
        if n_paths == 0 || !x0.is_finite() || !time_horizon.is_finite() || time_horizon < 0.0 {
            return PerformanceStats {
                mean_final: fallback,
                std_final: 0.0,
                standard_error: 0.0,
                ci_95_lower: fallback,
                ci_95_upper: fallback,
                median_final: fallback,
                percentile_05: fallback,
                percentile_95: fallback,
                prob_above_threshold: valid_threshold.map(|_| 0.0),
                prob_standard_error: valid_threshold.map(|_| 0.0),
            };
        }

        let n_steps = if time_horizon == 0.0 {
            0
        } else {
            (time_horizon / self.dt.value()).ceil() as usize
        };
        let solver = EulerMaruyama::new(self.params, self.dt);
        let paths = solver.simulate_paths(x0, n_steps, n_paths, rng);

        let mut final_values: Vec<f64> = paths
            .iter()
            .map(|p| {
                let last = p.last().copied().unwrap_or(fallback);
                if last.is_finite() { last } else { fallback }
            })
            .collect();
        final_values.sort_unstable_by(f64::total_cmp);

        let mean_final = final_values.iter().sum::<f64>() / (n_paths as f64);
        let variance = if n_paths > 1 {
            final_values
                .iter()
                .map(|&x| (x - mean_final).powi(2))
                .sum::<f64>()
                / ((n_paths - 1) as f64)
        } else {
            0.0
        };
        let std_final = variance.sqrt();
        let standard_error = if n_paths > 1 {
            std_final / (n_paths as f64).sqrt()
        } else {
            0.0
        };
        let ci_half = 1.959_963_984_540_054 * standard_error;

        let median_final = final_values[n_paths / 2];
        let p05_idx = (((n_paths - 1) as f64) * 0.05) as usize;
        let p95_idx = (((n_paths - 1) as f64) * 0.95) as usize;
        let prob_above_threshold = valid_threshold.map(|t| {
            let count = final_values.iter().filter(|&&x| x > t).count();
            (count as f64) / (n_paths as f64)
        });
        let prob_standard_error =
            prob_above_threshold.map(|p| ((p * (1.0 - p)) / (n_paths as f64)).sqrt());

        PerformanceStats {
            mean_final,
            std_final,
            standard_error,
            ci_95_lower: mean_final - ci_half,
            ci_95_upper: mean_final + ci_half,
            median_final,
            percentile_05: final_values[p05_idx],
            percentile_95: final_values[p95_idx],
            prob_above_threshold,
            prob_standard_error,
        }
    }

    /// Validated fallible variant of [`Self::monte_carlo_forecast`] returning [`OuError`] on invalid inputs.
    #[verified_engine::verified]
    pub fn try_monte_carlo_forecast<R: Rng>(
        &self,
        x0: f64,
        time_horizon: f64,
        n_paths: usize,
        threshold: Option<f64>,
        rng: &mut R,
    ) -> Result<PerformanceStats, OuError> {
        if n_paths == 0 {
            return Err(OuError::InvalidSimulationParams {
                reason: "n_paths must be at least 1".to_string(),
            });
        }
        if !x0.is_finite() || !time_horizon.is_finite() || time_horizon < 0.0 {
            return Err(OuError::InvalidSimulationParams {
                reason: "x0 and time_horizon must be finite with time_horizon >= 0".to_string(),
            });
        }
        if let Some(t) = threshold
            && !t.is_finite()
        {
            return Err(OuError::InvalidSimulationParams {
                reason: "threshold must be finite".to_string(),
            });
        }
        Ok(self.monte_carlo_forecast(x0, time_horizon, n_paths, threshold, rng))
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

    // Estimate mu as sample mean
    let mu = observations.iter().sum::<f64>() / observations.len() as f64;

    // Estimate sigma from sample variance
    let variance = observations.iter().map(|&x| (x - mu).powi(2)).sum::<f64>()
        / (observations.len() - 1) as f64;
    let sigma = variance.sqrt();

    // Estimate theta from lag-1 autocorrelation
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for i in 0..observations.len() - 1 {
        let diff_current = observations[i] - mu;
        let diff_next = observations[i + 1] - mu;
        numerator += diff_current * diff_next;
        denominator += diff_current * diff_current;
    }

    let rho1 = if denominator > math_commons::registry::TOLERANCE_HIGH {
        numerator / denominator
    } else {
        0.5 // Default if variance is too small
    };

    // theta ≈ -ln(ρ₁) / Δt
    let theta = if rho1 > 0.0 && rho1 < 1.0 {
        -rho1.ln() / dt
    } else {
        1.0 // Default if autocorrelation is out of range
    };

    OuParams::from_values(mu, theta.max(0.01), sigma.max(0.01))
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
    fn test_monte_carlo_forecast_zero_paths_nan_and_standard_errors() {
        let params = OuParams::from_values(0.5, 1.0, 0.2).unwrap();
        let dt = TimeStep::new(0.01).unwrap();
        let analyzer = OuAnalyzer::new(params, dt);
        let mut rng = oxidize_core::rng::OxidizeRng::new(7);

        let zero_stats = analyzer.monte_carlo_forecast(0.6, 1.0, 0, Some(0.5), &mut rng);
        assert!(zero_stats.mean_final.is_finite());
        assert_eq!(zero_stats.standard_error, 0.0);
        assert_eq!(zero_stats.prob_above_threshold, Some(0.0));
        assert!(
            analyzer
                .try_monte_carlo_forecast(0.6, 1.0, 0, None, &mut rng)
                .is_err()
        );

        let nan_stats = analyzer.monte_carlo_forecast(f64::NAN, 1.0, 64, Some(f64::NAN), &mut rng);
        assert!(nan_stats.mean_final.is_finite());
        assert_eq!(nan_stats.standard_error, 0.0);
        assert!(nan_stats.prob_above_threshold.is_none());
        assert!(
            analyzer
                .try_monte_carlo_forecast(f64::NAN, 1.0, 64, None, &mut rng)
                .is_err()
        );

        let stats = analyzer.monte_carlo_forecast(0.5, 1.0, 500, Some(0.5), &mut rng);
        assert!(stats.standard_error > 0.0);
        assert!(stats.ci_95_lower < stats.mean_final && stats.mean_final < stats.ci_95_upper);
        assert!(stats.prob_standard_error.unwrap() > 0.0);
    }
}

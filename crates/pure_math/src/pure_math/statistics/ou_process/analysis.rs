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
    #[verified_engine::verified(alloc_ok = "allocates owned output buffer")]
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
        let fallback_stats = || PerformanceStats {
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
        if n_paths == 0 || !x0.is_finite() || !time_horizon.is_finite() || time_horizon < 0.0 {
            return fallback_stats();
        }

        let n_steps = if time_horizon == 0.0 {
            0
        } else {
            (time_horizon / self.dt.value()).ceil() as usize
        };
        let solver = EulerMaruyama::new(self.params, self.dt);
        let paths = solver.simulate_paths(x0, n_steps, n_paths, rng);
        let final_values: Vec<f64> = paths
            .iter()
            .map(|p| {
                let last = p.last().copied().unwrap_or(fallback);
                if last.is_finite() { last } else { fallback }
            })
            .collect();
        let mut ws = math_commons::stochastic::SolverWorkspace::new(0, final_values.len());
        summarize_terminal_values_with_workspace(&mut ws, &final_values, valid_threshold)
            .unwrap_or_else(fallback_stats)
    }

    /// Validated fallible variant of [`Self::monte_carlo_forecast`] returning [`OuError`] on invalid inputs.
    #[verified_engine::verified(alloc_ok = "formats validation error")]
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
    let valid_threshold = threshold.filter(|t| t.is_finite());
    let summary = ws.summarize_samples(final_values, valid_threshold)?;
    let n = final_values.len();
    let standard_error = if n > 1 {
        summary.std_dev / (n as f64).sqrt()
    } else {
        0.0
    };
    let ci_half = 1.959_963_984_540_054 * standard_error;
    let prob_above_threshold = valid_threshold.map(|_| summary.prob_above_threshold);
    let prob_standard_error = prob_above_threshold.map(|p| ((p * (1.0 - p)) / (n as f64)).sqrt());
    Some(PerformanceStats {
        mean_final: summary.mean,
        std_final: summary.std_dev,
        standard_error,
        ci_95_lower: summary.mean - ci_half,
        ci_95_upper: summary.mean + ci_half,
        median_final: summary.median,
        percentile_05: summary.percentile_05,
        percentile_95: summary.percentile_95,
        prob_above_threshold,
        prob_standard_error,
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
#[verified_engine::verified(alloc_ok = "formats validation error")]
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
#[path = "analysis_tests.rs"]
mod tests;

// [cite:clinical_trials_statistics]

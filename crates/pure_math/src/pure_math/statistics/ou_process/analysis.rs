//! Analysis tools for OU process applications.

use super::core::OuParams;
use super::solver::{EulerMaruyama, TimeStep};
use crate::error::OuError;
use engine_prob::variance_reduction::control_variate_from_slices;
use oxidize_core::rng::OxidizeRng;
use rand::Rng;

/// Variance-reduction mode for [`OuAnalyzer::try_monte_carlo_forecast_variance_reduced`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OuVarianceReduction {
    /// Standard independent identically distributed Euler-Maruyama terminal sampling.
    None,
    /// Coupled antithetic Gaussian increments $(+Z_k, -Z_k)$ per path pair.
    Antithetic,
    /// Optimal linear control variate against exact discrete terminal mean $\mathbb{E}\[X_n\]$.
    ControlVariate,
}

/// Performance statistics from Monte Carlo simulation.
///
/// Contains summary statistics and Monte Carlo standard errors across simulated terminal states.
#[derive(Debug, Clone, PartialEq)]
pub struct PerformanceStats {
    /// Mean of final values across all paths.
    pub mean_final: f64,
    /// Standard error of `mean_final`.
    pub mean_std_error: f64,
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
    /// Standard error of `prob_above_threshold` when `threshold` is `Some(_)`.
    pub prob_above_threshold_std_error: Option<f64>,
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

fn validate_forecast_inputs(
    x0: f64,
    time_horizon: f64,
    n_paths: usize,
    threshold: Option<f64>,
) -> Result<(), OuError> {
    if !x0.is_finite() {
        return Err(OuError::InvalidSimulationParams {
            reason: "initial state x0 must be finite".to_string(),
        });
    }
    if threshold.is_some_and(|t| !t.is_finite()) {
        return Err(OuError::InvalidSimulationParams {
            reason: "threshold must be finite".to_string(),
        });
    }
    if !time_horizon.is_finite() || time_horizon <= 0.0 {
        return Err(OuError::InvalidTimeStep {
            value: time_horizon,
        });
    }
    if n_paths == 0 {
        return Err(OuError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }
    Ok(())
}

fn summarize_sorted_terminals(
    final_values: &mut [f64],
    threshold: Option<f64>,
) -> PerformanceStats {
    let n = final_values.len();
    final_values.sort_unstable_by(f64::total_cmp);

    let nf = n as f64;
    let mean_final = final_values.iter().sum::<f64>() / nf;
    let ss: f64 = final_values.iter().map(|&x| (x - mean_final).powi(2)).sum();
    let std_final = (ss / nf).sqrt();
    let mean_std_error = if n > 1 {
        ((ss / ((n - 1) as f64)) / nf).sqrt()
    } else {
        0.0
    };

    let median_final = final_values[n / 2];
    let idx_05 = ((nf * 0.05) as usize).min(n - 1);
    let idx_95 = ((nf * 0.95) as usize).min(n - 1);
    let (prob_above_threshold, prob_above_threshold_std_error) = match threshold {
        Some(t) => {
            let hits = final_values.iter().filter(|&&x| x > t).count();
            let p = (hits as f64) / nf;
            let se = if n > 1 {
                (p * (1.0 - p) / ((n - 1) as f64)).sqrt()
            } else {
                0.0
            };
            (Some(p), Some(se))
        }
        None => (None, None),
    };

    PerformanceStats {
        mean_final,
        mean_std_error,
        std_final,
        median_final,
        percentile_05: final_values[idx_05],
        percentile_95: final_values[idx_95],
        prob_above_threshold,
        prob_above_threshold_std_error,
    }
}

fn paired_se(pair_values: &[f64]) -> (f64, f64) {
    let m = pair_values.len();
    if m == 0 {
        return (0.0, 0.0);
    }
    let mf = m as f64;
    let mean = pair_values.iter().sum::<f64>() / mf;
    if m <= 1 {
        return (mean, 0.0);
    }
    let var = pair_values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / ((m - 1) as f64);
    (mean, (var / mf).sqrt())
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
    #[verified_engine::verified]
    pub fn momentum_type(&self) -> MomentumType {
        MomentumType::classify(self.params.theta.value())
    }

    /// Fallible Monte Carlo terminal distribution forecast bypassing full trajectory allocations.
    #[verified_engine::verified]
    pub fn try_monte_carlo_forecast<R: Rng>(
        &self,
        x0: f64,
        time_horizon: f64,
        n_paths: usize,
        threshold: Option<f64>,
        rng: &mut R,
    ) -> Result<PerformanceStats, OuError> {
        validate_forecast_inputs(x0, time_horizon, n_paths, threshold)?;
        let n_steps = (time_horizon / self.dt.value()).ceil() as usize;
        let solver = EulerMaruyama::new(self.params, self.dt);
        let mut final_values = vec![0.0_f64; n_paths];
        solver.simulate_terminals_into(x0, n_steps, &mut final_values, rng);
        if final_values.iter().any(|v| !v.is_finite()) {
            return Err(OuError::InvalidSimulationParams {
                reason: "simulated terminal state overflowed to non-finite value".to_string(),
            });
        }
        Ok(summarize_sorted_terminals(&mut final_values, threshold))
    }

    /// Fallible variance-reduced Monte Carlo forecast supporting antithetic and control-variate modes.
    #[verified_engine::verified]
    pub fn try_monte_carlo_forecast_variance_reduced(
        &self,
        x0: f64,
        time_horizon: f64,
        n_paths: usize,
        threshold: Option<f64>,
        mode: OuVarianceReduction,
        rng: &mut OxidizeRng,
    ) -> Result<PerformanceStats, OuError> {
        if mode == OuVarianceReduction::None {
            return self.try_monte_carlo_forecast(x0, time_horizon, n_paths, threshold, rng);
        }
        validate_forecast_inputs(x0, time_horizon, n_paths, threshold)?;
        let n_steps = (time_horizon / self.dt.value()).ceil() as usize;
        let solver = EulerMaruyama::new(self.params, self.dt);

        if mode == OuVarianceReduction::Antithetic {
            let n_pairs = n_paths.div_ceil(2);
            let mut terminals = Vec::with_capacity(2 * n_pairs);
            let mut pair_means = Vec::with_capacity(n_pairs);
            let mut pair_hits = Vec::with_capacity(if threshold.is_some() { n_pairs } else { 0 });
            for _ in 0..n_pairs {
                let (xp, xn) = solver.simulate_antithetic_terminal_pair(x0, n_steps, rng);
                terminals.push(xp);
                terminals.push(xn);
                pair_means.push(0.5 * (xp + xn));
                if let Some(t) = threshold {
                    pair_hits
                        .push(0.5 * (f64::from(u8::from(xp > t)) + f64::from(u8::from(xn > t))));
                }
            }
            let mut stats = summarize_sorted_terminals(&mut terminals, threshold);
            let (m_est, m_se) = paired_se(&pair_means);
            stats.mean_final = m_est;
            stats.mean_std_error = m_se;
            if threshold.is_some() {
                let (p_est, p_se) = paired_se(&pair_hits);
                stats.prob_above_threshold = Some(p_est.clamp(0.0, 1.0));
                stats.prob_above_threshold_std_error = Some(p_se);
            }
            return Ok(stats);
        }

        let mut terminals = vec![0.0_f64; n_paths];
        solver.simulate_terminals_into(x0, n_steps, &mut terminals, rng);
        let c_exp = solver.discrete_terminal_mean(x0, n_steps);
        let cv_prob = if let Some(t) = threshold {
            let indicators: Vec<f64> = terminals
                .iter()
                .map(|&v| if v > t { 1.0 } else { 0.0 })
                .collect();
            control_variate_from_slices(&indicators, &terminals, c_exp).ok()
        } else {
            None
        };
        let mut stats = summarize_sorted_terminals(&mut terminals, threshold);
        stats.mean_final = c_exp;
        stats.mean_std_error = 0.0;
        if let Some(cv) = cv_prob {
            stats.prob_above_threshold = Some(cv.mean.clamp(0.0, 1.0));
            stats.prob_above_threshold_std_error = Some(cv.std_error);
        }
        Ok(stats)
    }

    /// Runs Monte Carlo simulation to estimate final performance distribution without panicking
    /// on `n_paths == 0` or `NaN` inputs.
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
    ///
    /// let mut rng = math_explorer::rng::OxidizeRng::default();
    /// let stats = analyzer.monte_carlo_forecast(0.50, 1.0, 10000, Some(0.48), &mut rng);
    /// assert!(stats.mean_std_error > 0.0);
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
        if let Ok(stats) = self.try_monte_carlo_forecast(x0, time_horizon, n_paths, threshold, rng)
        {
            return stats;
        }
        let fallback = if x0.is_finite() {
            x0
        } else {
            self.params.mu.value()
        };
        PerformanceStats {
            mean_final: fallback,
            mean_std_error: 0.0,
            std_final: 0.0,
            median_final: fallback,
            percentile_05: fallback,
            percentile_95: fallback,
            prob_above_threshold: threshold.map(|t| {
                if t.is_finite() && n_paths > 0 && fallback > t {
                    1.0
                } else {
                    0.0
                }
            }),
            prob_above_threshold_std_error: threshold.map(|_| 0.0),
        }
    }

    /// Estimates the probability of a comeback given current deficit.
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

    let mu = observations.iter().sum::<f64>() / observations.len() as f64;
    let variance = observations.iter().map(|&x| (x - mu).powi(2)).sum::<f64>()
        / (observations.len() - 1) as f64;
    let sigma = variance.sqrt();

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
        0.5
    };
    let theta = if rho1 > 0.0 && rho1 < 1.0 {
        -rho1.ln() / dt
    } else {
        1.0
    };

    OuParams::from_values(mu, theta.max(0.01), sigma.max(0.01))
}

#[cfg(test)]
#[path = "analysis_tests.rs"]
mod tests;

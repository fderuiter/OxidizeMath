//! Homogeneous and inhomogeneous Poisson arrival processes.
//!
//! Provides exact log-space count PMFs, inter-arrival sampling, and zero-allocation
//! Lewis-Shedler thinning trajectories driven by `OxidizeRng`.

use oxidize_core::rng::OxidizeRng;
use rand::Rng;
use thiserror::Error;

/// Errors returned by Poisson process constructors and simulators.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum PoissonError {
    /// Arrival rate must be strictly positive and finite.
    #[error("Invalid Poisson arrival rate: {rate} (must be > 0 and finite)")]
    InvalidRate {
        /// Offending rate value.
        rate: f64,
    },
    /// Time horizon must be non-negative and finite.
    #[error("Invalid Poisson time horizon: {horizon} (must be >= 0 and finite)")]
    InvalidHorizon {
        /// Offending time horizon.
        horizon: f64,
    },
    /// Parametric inhomogeneous intensity function violates non-negativity or finiteness.
    #[error("Invalid inhomogeneous intensity profile: {reason}")]
    InvalidIntensityProfile {
        /// Explanation of the invalid intensity configuration.
        reason: String,
    },
}

/// Homogeneous Poisson point process with constant rate $\lambda > 0$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HomogeneousPoisson {
    rate: f64,
}

impl HomogeneousPoisson {
    /// Creates a validated homogeneous Poisson process with rate $\lambda > 0$.
    ///
    /// # Errors
    ///
    /// Returns [`PoissonError::InvalidRate`] if `rate <= 0.0` or non-finite.
    #[verified_engine::verified]
    pub fn new(rate: f64) -> Result<Self, PoissonError> {
        if !rate.is_finite() || rate <= 0.0 {
            return Err(PoissonError::InvalidRate { rate });
        }
        Ok(Self { rate })
    }

    /// Returns the arrival rate $\lambda$.
    #[must_use]
    #[verified_engine::verified]
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Expected arrival count $\mathbb{E}[N(t)] = \lambda t$.
    ///
    /// # Errors
    ///
    /// Returns [`PoissonError::InvalidHorizon`] if `horizon < 0.0` or non-finite.
    #[verified_engine::verified]
    pub fn mean(&self, horizon: f64) -> Result<f64, PoissonError> {
        validate_horizon(horizon)?;
        Ok(self.rate * horizon)
    }

    /// Variance $\operatorname{Var}(N(t)) = \lambda t$.
    ///
    /// # Errors
    ///
    /// Returns [`PoissonError::InvalidHorizon`] if `horizon < 0.0` or non-finite.
    #[verified_engine::verified]
    pub fn variance(&self, horizon: f64) -> Result<f64, PoissonError> {
        self.mean(horizon)
    }

    /// Exact count probability $P(N(t) = k) = \frac{(\lambda t)^k e^{-\lambda t}}{k!}$ computed in log-space.
    ///
    /// # Errors
    ///
    /// Returns [`PoissonError::InvalidHorizon`] if `horizon < 0.0` or non-finite.
    #[verified_engine::verified]
    pub fn pmf(&self, k: usize, horizon: f64) -> Result<f64, PoissonError> {
        let mu = self.mean(horizon)?;
        if mu == 0.0 {
            return Ok(if k == 0 { 1.0 } else { 0.0 });
        }
        let mut log_fact = 0.0_f64;
        for m in 2..=k {
            log_fact += (m as f64).ln();
        }
        let log_p = (k as f64) * mu.ln() - mu - log_fact;
        Ok(log_p.exp())
    }

    /// Samples a single exponential inter-arrival time $\Delta t \sim \operatorname{Exp}(\lambda)$
    /// with zero heap allocations.
    #[must_use]
    #[verified_engine::verified]
    pub fn sample_interarrival(&self, rng: &mut OxidizeRng) -> f64 {
        let u = rng.r#gen::<f64>().max(f64::MIN_POSITIVE);
        -u.ln() / self.rate
    }

    /// Simulates arrivals in `(0, horizon]` into `out_times` with **zero heap allocations**.
    ///
    /// Returns the total arrival count observed in `(0, horizon]`; up to `out_times.len()`
    /// timestamps are stored in `out_times`.
    ///
    /// # Errors
    ///
    /// Returns [`PoissonError::InvalidHorizon`] if `horizon < 0.0` or non-finite.
    #[verified_engine::verified]
    pub fn simulate_arrivals_into(
        &self,
        horizon: f64,
        out_times: &mut [f64],
        rng: &mut OxidizeRng,
    ) -> Result<usize, PoissonError> {
        validate_horizon(horizon)?;
        let mut t = 0.0_f64;
        let mut total = 0_usize;
        while t < horizon {
            t += self.sample_interarrival(rng);
            if t <= horizon {
                if total < out_times.len() {
                    out_times[total] = t;
                }
                total += 1;
            }
        }
        Ok(total)
    }

    /// Convenience wrapper returning up to `max_events` arrival timestamps in `(0, horizon]`.
    ///
    /// # Errors
    ///
    /// Returns [`PoissonError::InvalidHorizon`] if `horizon < 0.0` or non-finite.
    #[verified_engine::verified]
    pub fn simulate_arrivals(
        &self,
        horizon: f64,
        max_events: usize,
        rng: &mut OxidizeRng,
    ) -> Result<Vec<f64>, PoissonError> {
        let mut buf = vec![0.0_f64; max_events];
        let count = self.simulate_arrivals_into(horizon, &mut buf, rng)?;
        buf.truncate(count.min(max_events));
        Ok(buf)
    }
}

/// Fixed parametric rate profile $\lambda(t)$ for inhomogeneous Poisson thinning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntensityProfile {
    /// Periodic diurnal rate $\lambda(t) = \lambda_0 + A \sin(\omega t + \phi)$ with $|A| \le \lambda_0$.
    Sinusoidal {
        /// Baseline rate $\lambda_0 > 0$.
        base_rate: f64,
        /// Oscillation amplitude $|A| \le \lambda_0$.
        amplitude: f64,
        /// Angular frequency $\omega \ge 0$.
        angular_freq: f64,
        /// Initial phase $\phi$ in radians.
        phase: f64,
    },
    /// Linearly trending intensity $\lambda(t) = \lambda_0 + \beta t$ ($\lambda_0 > 0, \beta \ge 0$).
    LinearRamp {
        /// Rate at $t = 0$.
        start_rate: f64,
        /// Non-negative slope $\beta \ge 0$.
        slope: f64,
    },
    /// Piecewise-constant surge $\lambda(t) = \lambda_{\text{burst}}$ on $[t_0, t_1]$, else $\lambda_0$.
    StepBurst {
        /// Baseline rate outside $[t_0, t_1]$.
        base_rate: f64,
        /// Peak rate inside $[t_0, t_1]$.
        burst_rate: f64,
        /// Burst start time $t_0 \ge 0$.
        burst_start: f64,
        /// Burst end time $t_1 \ge t_0$.
        burst_end: f64,
    },
}

impl IntensityProfile {
    /// Validates parameters of the intensity profile.
    ///
    /// # Errors
    ///
    /// Returns [`PoissonError::InvalidIntensityProfile`] if rates can become negative, zero-bounded,
    /// or non-finite.
    #[verified_engine::verified]
    pub fn validate(self) -> Result<Self, PoissonError> {
        match self {
            Self::Sinusoidal {
                base_rate,
                amplitude,
                angular_freq,
                phase,
            } => {
                if !base_rate.is_finite()
                    || !amplitude.is_finite()
                    || !angular_freq.is_finite()
                    || !phase.is_finite()
                    || base_rate <= 0.0
                    || angular_freq < 0.0
                    || amplitude.abs() > base_rate
                {
                    return Err(PoissonError::InvalidIntensityProfile {
                        reason:
                            "Sinusoidal requires base_rate > 0, freq >= 0, |amplitude| <= base_rate"
                                .to_string(),
                    });
                }
            }
            Self::LinearRamp { start_rate, slope } => {
                if !start_rate.is_finite() || !slope.is_finite() || start_rate <= 0.0 || slope < 0.0
                {
                    return Err(PoissonError::InvalidIntensityProfile {
                        reason: "LinearRamp requires start_rate > 0 and slope >= 0".to_string(),
                    });
                }
            }
            Self::StepBurst {
                base_rate,
                burst_rate,
                burst_start,
                burst_end,
            } => {
                if !base_rate.is_finite()
                    || !burst_rate.is_finite()
                    || !burst_start.is_finite()
                    || !burst_end.is_finite()
                    || base_rate <= 0.0
                    || burst_rate <= 0.0
                    || burst_start < 0.0
                    || burst_end < burst_start
                {
                    return Err(PoissonError::InvalidIntensityProfile {
                        reason:
                            "StepBurst requires positive rates and 0 <= burst_start <= burst_end"
                                .to_string(),
                    });
                }
            }
        }
        Ok(self)
    }

    /// Evaluates instantaneous intensity $\lambda(t)$ at $t \ge 0$.
    #[must_use]
    #[verified_engine::verified]
    pub fn rate_at(self, t: f64) -> f64 {
        match self {
            Self::Sinusoidal {
                base_rate,
                amplitude,
                angular_freq,
                phase,
            } => (base_rate + amplitude * (angular_freq * t + phase).sin()).max(0.0),
            Self::LinearRamp { start_rate, slope } => (start_rate + slope * t).max(0.0),
            Self::StepBurst {
                base_rate,
                burst_rate,
                burst_start,
                burst_end,
            } => {
                if t >= burst_start && t <= burst_end {
                    burst_rate
                } else {
                    base_rate
                }
            }
        }
    }

    /// Returns a tight majorizing bound $\bar{\lambda} \ge \sup_{0 \le t \le T} \lambda(t)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn upper_bound(self, horizon: f64) -> f64 {
        match self {
            Self::Sinusoidal {
                base_rate,
                amplitude,
                ..
            } => base_rate + amplitude.abs(),
            Self::LinearRamp { start_rate, slope } => start_rate + slope * horizon.max(0.0),
            Self::StepBurst {
                base_rate,
                burst_rate,
                ..
            } => base_rate.max(burst_rate),
        }
    }

    /// Computes exact cumulative intensity $\Lambda(T) = \int_0^T \lambda(t)\,dt$.
    #[must_use]
    #[verified_engine::verified]
    pub fn integrated_mean(self, horizon: f64) -> f64 {
        let t = horizon.max(0.0);
        match self {
            Self::Sinusoidal {
                base_rate,
                amplitude,
                angular_freq,
                phase,
            } => {
                if angular_freq <= 1e-12 {
                    (base_rate + amplitude * phase.sin()) * t
                } else {
                    base_rate * t
                        + amplitude * (phase.cos() - (angular_freq * t + phase).cos())
                            / angular_freq
                }
            }
            Self::LinearRamp { start_rate, slope } => start_rate * t + 0.5 * slope * t * t,
            Self::StepBurst {
                base_rate,
                burst_rate,
                burst_start,
                burst_end,
            } => {
                let overlap = (t.min(burst_end) - burst_start).max(0.0);
                base_rate * t + (burst_rate - base_rate) * overlap
            }
        }
    }
}

/// Inhomogeneous Poisson process simulated via Lewis-Shedler thinning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InhomogeneousPoisson {
    profile: IntensityProfile,
}

impl InhomogeneousPoisson {
    /// Creates a validated inhomogeneous Poisson process.
    ///
    /// # Errors
    ///
    /// Returns [`PoissonError::InvalidIntensityProfile`] if `profile` is invalid.
    #[verified_engine::verified]
    pub fn new(profile: IntensityProfile) -> Result<Self, PoissonError> {
        Ok(Self {
            profile: profile.validate()?,
        })
    }

    /// Returns the underlying parametric intensity profile.
    #[must_use]
    #[verified_engine::verified]
    pub fn profile(&self) -> IntensityProfile {
        self.profile
    }

    /// Expected arrival count $\Lambda(T) = \int_0^T \lambda(t)\,dt$.
    ///
    /// # Errors
    ///
    /// Returns [`PoissonError::InvalidHorizon`] if `horizon < 0.0` or non-finite.
    #[verified_engine::verified]
    pub fn expected_count(&self, horizon: f64) -> Result<f64, PoissonError> {
        validate_horizon(horizon)?;
        Ok(self.profile.integrated_mean(horizon))
    }

    /// Simulates arrivals on `(0, horizon]` via Lewis-Shedler thinning with **zero heap allocations**.
    ///
    /// # Errors
    ///
    /// Returns [`PoissonError::InvalidHorizon`] if `horizon < 0.0` or non-finite.
    #[verified_engine::verified]
    pub fn simulate_arrivals_into(
        &self,
        horizon: f64,
        out_times: &mut [f64],
        rng: &mut OxidizeRng,
    ) -> Result<usize, PoissonError> {
        validate_horizon(horizon)?;
        let lambda_bar = self.profile.upper_bound(horizon);
        if lambda_bar <= 0.0 || horizon == 0.0 {
            return Ok(0);
        }

        let mut t = 0.0_f64;
        let mut accepted = 0_usize;
        while t < horizon {
            let u1 = rng.r#gen::<f64>().max(f64::MIN_POSITIVE);
            t += -u1.ln() / lambda_bar;
            if t <= horizon {
                let u2: f64 = rng.r#gen();
                if u2 * lambda_bar <= self.profile.rate_at(t) {
                    if accepted < out_times.len() {
                        out_times[accepted] = t;
                    }
                    accepted += 1;
                }
            }
        }
        Ok(accepted)
    }

    /// Convenience wrapper returning up to `max_events` thinned arrival timestamps in `(0, horizon]`.
    ///
    /// # Errors
    ///
    /// Returns [`PoissonError::InvalidHorizon`] if `horizon < 0.0` or non-finite.
    #[verified_engine::verified]
    pub fn simulate_arrivals(
        &self,
        horizon: f64,
        max_events: usize,
        rng: &mut OxidizeRng,
    ) -> Result<Vec<f64>, PoissonError> {
        let mut buf = vec![0.0_f64; max_events];
        let count = self.simulate_arrivals_into(horizon, &mut buf, rng)?;
        buf.truncate(count.min(max_events));
        Ok(buf)
    }
}

#[verified_engine::verified]
fn validate_horizon(horizon: f64) -> Result<(), PoissonError> {
    if !horizon.is_finite() || horizon < 0.0 {
        return Err(PoissonError::InvalidHorizon { horizon });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_homogeneous_poisson_pmf_and_zero_alloc_sim() {
        let proc = HomogeneousPoisson::new(2.5).unwrap();
        let sum_pmf: f64 = (0..30).map(|k| proc.pmf(k, 2.0).unwrap()).sum();
        assert_relative_eq!(sum_pmf, 1.0, epsilon = 1e-10);
        assert_relative_eq!(proc.pmf(0, 2.0).unwrap(), (-5.0_f64).exp(), epsilon = 1e-12);

        let mut times = [0.0_f64; 128];
        let mut rng = OxidizeRng::new(99);
        let (count, metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            proc.simulate_arrivals_into(10.0, &mut times, &mut rng)
                .unwrap()
        });
        assert!(metrics.function_calls > 0);
        assert!(count > 0);
    }

    #[test]
    fn test_inhomogeneous_poisson_thinning_matches_integrated_mean() {
        let profile = IntensityProfile::Sinusoidal {
            base_rate: 4.0,
            amplitude: 2.0,
            angular_freq: std::f64::consts::PI,
            phase: 0.0,
        };
        let proc = InhomogeneousPoisson::new(profile).unwrap();
        let expected = proc.expected_count(4.0).unwrap();
        assert_relative_eq!(expected, 16.0, epsilon = 1e-12);

        let mut buf = [0.0_f64; 64];
        let mut rng = OxidizeRng::new(12345);
        let mut total_events = 0_usize;
        let trials = 1_500_usize;
        let (_, metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            for _ in 0..trials {
                total_events += proc
                    .simulate_arrivals_into(4.0, &mut buf, &mut rng)
                    .unwrap();
            }
        });
        assert!(metrics.function_calls > 0);
        let sample_mean = total_events as f64 / trials as f64;
        assert!((sample_mean - expected).abs() < 0.35);
    }
}

// [cite:clinical_trials_statistics]

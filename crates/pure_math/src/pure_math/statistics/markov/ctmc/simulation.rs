//! Gillespie trajectory simulation (allocating and zero-allocation variants) for CTMCs.

use super::{ContinuousMarkovChain, Result};
use crate::error::MarkovError;
use nalgebra::RealField;
use num_traits::ToPrimitive;
use oxidize_core::rng::OxidizeRng;
use rand::Rng;
use rand_distr::{Distribution, Exp};

impl<T: RealField + Copy + ToPrimitive> ContinuousMarkovChain<T> {
    /// Generates a sample trajectory `(time, state)` from the CTMC via Gillespie's algorithm.
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::InvalidState`] if `initial_state` is out of bounds or `max_time < 0`.
    #[verified_engine::verified]
    pub fn simulate_trajectory<R: rand::Rng>(
        &self,
        initial_state: usize,
        max_time: T,
        rng: &mut R,
    ) -> Result<Vec<(T, usize)>> {
        if initial_state >= self.num_states {
            return Err(MarkovError::InvalidState {
                reason: format!("Initial state {initial_state} out of bounds"),
            });
        }
        if !max_time.is_finite() || max_time < T::zero() {
            return Err(MarkovError::InvalidState {
                reason: "Simulation horizon max_time must be finite and non-negative".to_string(),
            });
        }

        let mut trajectory = vec![(T::zero(), initial_state)];
        let mut current_state = initial_state;
        let mut current_time = T::zero();

        while current_time < max_time {
            let rate = -self.generator[(current_state, current_state)];
            let rate_f64 = rate.to_f64().unwrap_or(0.0);
            if rate <= T::zero() || rate_f64 <= 0.0 {
                break;
            }

            let exp_dist = Exp::new(rate_f64).map_err(|_| {
                MarkovError::Math(math_commons::error::MathError::NumericalError {
                    reason: format!("Invalid rate for exponential: {rate_f64}"),
                })
            })?;
            let holding_time: f64 = exp_dist.sample(rng);
            current_time += T::from_f64(holding_time).unwrap();
            if current_time >= max_time {
                break;
            }

            let Some(next_state) = self.sample_jump_target(current_state, rate_f64, rng) else {
                break;
            };
            current_state = next_state;
            trajectory.push((current_time, current_state));
        }

        Ok(trajectory)
    }

    #[verified_engine::verified]
    fn sample_jump_target<R: Rng>(
        &self,
        current_state: usize,
        total_rate: f64,
        rng: &mut R,
    ) -> Option<usize> {
        let threshold: f64 = rng.r#gen::<f64>() * total_rate;
        let mut cumulative = 0.0_f64;
        let mut fallback = None;

        for j in 0..self.num_states {
            if j != current_state {
                let r = self.generator[(current_state, j)].to_f64().unwrap_or(0.0);
                if r > 0.0 {
                    fallback = Some(j);
                    cumulative += r;
                    if threshold < cumulative {
                        return Some(j);
                    }
                }
            }
        }
        fallback
    }
}

impl ContinuousMarkovChain<f64> {
    /// Simulates a Gillespie trajectory into pre-allocated buffers and accumulates exact state
    /// sojourn durations over `[0, max_time]` with **zero heap allocations**.
    ///
    /// Returns the number of recorded `(time, state)` events written to `out_times` / `out_states`.
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::InvalidState`] if `initial_state >= self.num_states()`, `max_time < 0`,
    /// or `sojourn_times.len() < self.num_states()`.
    #[verified_engine::verified]
    pub fn simulate_gillespie_into(
        &self,
        initial_state: usize,
        max_time: f64,
        out_times: &mut [f64],
        out_states: &mut [usize],
        sojourn_times: &mut [f64],
        rng: &mut OxidizeRng,
    ) -> Result<usize> {
        self.validate_gillespie_buffers(initial_state, max_time, sojourn_times)?;
        for s in sojourn_times.iter_mut().take(self.num_states) {
            *s = 0.0;
        }

        let cap = out_times.len().min(out_states.len());
        let mut count = 0_usize;
        if cap > 0 {
            out_times[0] = 0.0;
            out_states[0] = initial_state;
            count = 1;
        }

        let mut state = initial_state;
        let mut time = 0.0_f64;
        while time < max_time {
            let rate = -self.generator[(state, state)];
            if rate <= 0.0 {
                sojourn_times[state] += max_time - time;
                break;
            }
            let u = rng.r#gen::<f64>().max(f64::MIN_POSITIVE);
            let dt = -u.ln() / rate;
            if time + dt >= max_time {
                sojourn_times[state] += max_time - time;
                break;
            }
            sojourn_times[state] += dt;
            time += dt;
            state = self.step_jump_oxidize(state, rate, rng);
            if count < cap {
                out_times[count] = time;
                out_states[count] = state;
                count += 1;
            }
        }
        Ok(count)
    }

    #[verified_engine::verified]
    fn validate_gillespie_buffers(
        &self,
        initial_state: usize,
        max_time: f64,
        sojourn_times: &[f64],
    ) -> Result<()> {
        if initial_state >= self.num_states {
            return Err(MarkovError::InvalidState {
                reason: format!("Initial state {initial_state} out of bounds"),
            });
        }
        if !max_time.is_finite() || max_time < 0.0 {
            return Err(MarkovError::InvalidState {
                reason: "max_time must be finite and non-negative".to_string(),
            });
        }
        if sojourn_times.len() < self.num_states {
            return Err(MarkovError::InvalidState {
                reason: "sojourn_times buffer smaller than num_states".to_string(),
            });
        }
        Ok(())
    }

    #[verified_engine::verified]
    fn step_jump_oxidize(&self, state: usize, total_rate: f64, rng: &mut OxidizeRng) -> usize {
        let target = rng.r#gen::<f64>() * total_rate;
        let mut cumulative = 0.0_f64;
        let mut fallback = state;
        for j in 0..self.num_states {
            if j != state {
                let r = self.generator[(state, j)];
                if r > 0.0 {
                    fallback = j;
                    cumulative += r;
                    if target < cumulative {
                        return j;
                    }
                }
            }
        }
        fallback
    }
}

// [cite:clinical_trials_statistics]

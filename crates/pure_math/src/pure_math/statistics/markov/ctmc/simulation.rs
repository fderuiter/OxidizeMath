//! Gillespie trajectory simulation (allocating and zero-allocation variants) for CTMCs.

use super::{ContinuousMarkovChain, Result};
use crate::error::MarkovError;
use crate::pure_math::analysis::stochastic::{GillespieSolver, StochasticError, StochasticSystem};
use nalgebra::RealField;
use num_traits::ToPrimitive;
use oxidize_core::rng::OxidizeRng;
use rand::Rng;

impl<T: RealField + Copy + ToPrimitive> ContinuousMarkovChain<T> {
    /// Generates a sample trajectory `(time, state)` from the CTMC via Gillespie's algorithm.
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::InvalidState`] if `initial_state` is out of bounds or `max_time < 0`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::markov::ContinuousMarkovChain;
    /// use math_explorer::nalgebra::DMatrix;
    ///
    /// let generator = DMatrix::from_row_slice(2, 2, &[
    ///     -1.0,  1.0,
    ///      2.0, -2.0,
    /// ]);
    /// let chain = ContinuousMarkovChain::new(generator).unwrap();
    /// let mut rng = math_explorer::rng::OxidizeRng::default();
    ///
    /// let trajectory = chain.simulate_trajectory(0, 10.0, &mut rng).unwrap();
    /// assert!(!trajectory.is_empty());
    /// assert_eq!(trajectory[0], (0.0, 0));
    /// ```
    #[verified_engine::verified]
    pub fn simulate_trajectory<R: rand::Rng>(
        &self,
        initial_state: usize,
        max_time: T,
        rng: &mut R,
    ) -> Result<Vec<(T, usize)>> {
        debug_assert!(self.num_states > 0);
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
        let mut solver = GillespieSolver::with_capacity(rng, self.num_states);

        while current_time < max_time {
            let holding_time = solver.step(self, &mut current_state).map_err(|err| {
                MarkovError::Math(math_commons::error::MathError::NumericalError {
                    reason: err.to_string(),
                })
            })?;
            if !holding_time.is_finite() {
                break;
            }
            let Some(dt_val) = T::from_f64(holding_time) else {
                return Err(MarkovError::Math(
                    math_commons::error::MathError::ConversionError {
                        reason: format!("Failed to convert holding time {holding_time}"),
                    },
                ));
            };
            current_time += dt_val;
            if current_time >= max_time {
                break;
            }
            trajectory.push((current_time, current_state));
        }

        Ok(trajectory)
    }
}

impl<T: RealField + Copy + ToPrimitive> StochasticSystem<usize> for ContinuousMarkovChain<T> {
    #[verified_engine::verified]
    fn propensities(&self, state: &usize, out: &mut Vec<f64>) {
        debug_assert!(*state < self.num_states);
        let i = *state;
        let tol = math_commons::registry::TOLERANCE_HIGH;
        let total_exit = (-self.generator[(i, i)]).to_f64().unwrap_or(0.0);
        if total_exit <= 0.0 {
            return;
        }
        let min_edge = (total_exit * 1e-15).min(tol);
        for j in 0..self.num_states {
            if j == i {
                out.push(0.0);
            } else {
                let rate = self.generator[(i, j)].to_f64().unwrap_or(0.0);
                out.push(if rate > min_edge { rate } else { 0.0 });
            }
        }
    }

    #[verified_engine::verified]
    fn react(
        &self,
        state: &mut usize,
        reaction_index: usize,
    ) -> std::result::Result<(), StochasticError> {
        debug_assert!(*state < self.num_states);
        if reaction_index >= self.num_states {
            return Err(StochasticError::InvalidReactionIndex(reaction_index));
        }
        *state = reaction_index;
        Ok(())
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

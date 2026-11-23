//! Zero-allocation discrete-time Markov chain stepping with `OxidizeRng`.

use super::{MarkovChain, Result};
use crate::error::MarkovError;
use oxidize_core::rng::OxidizeRng;
use rand::Rng;

impl MarkovChain<f64> {
    /// Advances a single state transition using `OxidizeRng` without heap allocations.
    #[must_use]
    #[verified_engine::verified]
    pub fn step_state(&self, current_state: usize, rng: &mut OxidizeRng) -> usize {
        let n = self.num_states();
        if n == 0 || current_state >= n {
            return current_state;
        }

        let u: f64 = rng.r#gen();
        let mut cumulative = 0.0_f64;
        let mut fallback = current_state;
        for j in 0..n {
            let p = self.transition_matrix[(current_state, j)];
            if p > 0.0 {
                fallback = j;
                cumulative += p;
                if u < cumulative {
                    return j;
                }
            }
        }
        fallback
    }

    /// Simulates `steps` transitions in-place, accumulating empirical state visit counts into
    /// `visit_counts` (including the initial state at step 0 when `steps == 0` is false; specifically
    /// records `steps` post-step visits, or initial + transitions).
    ///
    /// Performs **zero heap allocations** in the hot loop under `VerifiedAllocator`.
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::InvalidState`] if `initial_state` is out of bounds or
    /// `visit_counts.len() < self.num_states()`.
    #[verified_engine::verified]
    pub fn simulate_visits_into(
        &self,
        initial_state: usize,
        steps: usize,
        visit_counts: &mut [usize],
        rng: &mut OxidizeRng,
    ) -> Result<usize> {
        let n = self.num_states();
        if initial_state >= n {
            return Err(MarkovError::InvalidState {
                reason: format!("Initial state {initial_state} out of bounds for {n} states"),
            });
        }
        if visit_counts.len() < n {
            return Err(MarkovError::InvalidState {
                reason: format!(
                    "visit_counts slice length {} smaller than state count {n}",
                    visit_counts.len()
                ),
            });
        }

        for count in visit_counts.iter_mut().take(n) {
            *count = 0;
        }

        let mut state = initial_state;
        for _ in 0..steps {
            state = self.step_state(state, rng);
            visit_counts[state] += 1;
        }
        Ok(state)
    }
}

// [cite:clinical_trials_statistics]

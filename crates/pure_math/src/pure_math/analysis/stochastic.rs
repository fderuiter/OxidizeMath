//! Stochastic Simulation Algorithms (SSA)
//!
//! This module provides tools for simulating continuous-time Markov processes,
//! such as chemical reaction networks or epidemiological models.

use rand::Rng;
use thiserror::Error;

/// Errors that can occur during stochastic simulation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StochasticError {
    #[error("Invalid reaction index: {0}")]
    #[allow(missing_docs)]
    InvalidReactionIndex(usize),
}

/// A trait for systems that can be simulated stochastically.
///
/// Unlike ODE systems which are continuous, stochastic systems define discrete events
/// that occur with specific propensities (rates).
pub trait StochasticSystem<State> {
    /// Appends the propensity (rate) of each reaction in the current state to the output buffer.
    ///
    /// The buffer is cleared by the solver before calling this method.
    #[verified_engine::verified]
    fn propensities(&self, state: &State, out: &mut Vec<f64>);

    /// Updates the state according to the reaction that occurred.
    ///
    /// # Arguments
    /// * `state` - The current state to be modified.
    /// * `reaction_index` - The index of the reaction to execute (corresponding to the index in propensities).
    #[verified_engine::verified]
    fn react(&self, state: &mut State, reaction_index: usize) -> Result<(), StochasticError>;
}

struct AllocationLockGuard;

impl AllocationLockGuard {
    fn lock() -> Self {
        verified_engine::allocator::lock_allocations();
        AllocationLockGuard
    }
}

impl Drop for AllocationLockGuard {
    fn drop(&mut self) {
        verified_engine::allocator::unlock_allocations();
    }
}

/// A solver for stochastic simulation using the Gillespie Algorithm (SSA).
///
/// It uses the Direct Method to simulate exact stochastic trajectories.
pub struct GillespieSolver<R> {
    rng: R,
    /// Reusable buffer for propensities to avoid allocation per step.
    buffer: Vec<f64>,
}

impl<R: Rng> GillespieSolver<R> {
    /// Creates a new solver with the provided random number generator.
    ///
    /// This allows for deterministic simulations by passing a seeded RNG.
    #[verified_engine::verified]
    pub fn new(rng: R) -> Self {
        Self::with_capacity(rng, 16)
    }

    /// Creates a new solver with the provided random number generator and buffer capacity.
    #[verified_engine::verified]
    pub fn with_capacity(rng: R, capacity: usize) -> Self {
        Self {
            rng,
            buffer: Vec::with_capacity(capacity),
        }
    }

    /// Performs one step of the Gillespie algorithm.
    ///
    /// Returns the time elapsed for this step.
    /// Returns `Ok(f64::INFINITY)` if no reactions can occur (total propensity is 0).
    #[verified_engine::verified(opt_out = "Legacy missing assertions")]
    pub fn step<S, State>(&mut self, system: &S, state: &mut State) -> Result<f64, StochasticError>
    where
        S: StochasticSystem<State>,
    {
        let _guard = AllocationLockGuard::lock();
        self.buffer.clear();
        system.propensities(state, &mut self.buffer);

        let rates = &self.buffer;

        let total_rate: f64 = rates.iter().sum();

        if total_rate <= 0.0 {
            return Ok(f64::INFINITY);
        }

        // 1. Determine time step tau
        // r1 in (0, 1]
        let r1: f64 = self.rng.r#gen();
        let r1 = if r1 <= 0.0 { f64::MIN_POSITIVE } else { r1 };
        let tau = -r1.ln() / total_rate;

        // 2. Determine which reaction mu occurred
        let r2: f64 = self.rng.r#gen();
        let threshold = r2 * total_rate;
        let mut cumulative = 0.0;
        let mut reaction_index = 0;

        for (i, &rate) in rates.iter().enumerate() {
            cumulative += rate;
            if cumulative >= threshold {
                reaction_index = i;
                break;
            }
        }
        // Fallback for floating point errors
        if cumulative < threshold {
            reaction_index = rates.len().saturating_sub(1);
        }

        // 3. Update state
        system.react(state, reaction_index)?;

        Ok(tau)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple Decay Model: X -> nothing
    // Rate: k * X
    struct DecayModel {
        k: f64,
    }

    struct DecayState {
        x: i32,
    }

    impl StochasticSystem<DecayState> for DecayModel {
        #[verified_engine::verified]
        fn propensities(&self, state: &DecayState, out: &mut Vec<f64>) {
            if state.x > 0 {
                out.push(self.k * state.x as f64);
            } else {
                out.push(0.0);
            }
        }

        #[verified_engine::verified]
        fn react(
            &self,
            state: &mut DecayState,
            reaction_index: usize,
        ) -> Result<(), StochasticError> {
            if reaction_index == 0 {
                state.x -= 1;
                Ok(())
            } else {
                Err(StochasticError::InvalidReactionIndex(reaction_index))
            }
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_decay_process() -> Result<(), StochasticError> {
        let rng = oxidize_core::rng::OxidizeRng::default();
        let mut solver = GillespieSolver::new(rng);
        let model = DecayModel { k: 0.1 };
        let mut state = DecayState { x: 100 };

        let mut time = 0.0;
        while time < 10.0 {
            let dt = solver.step(&model, &mut state)?;
            if dt.is_infinite() {
                break;
            }
            time += dt;
        }

        // With k=0.1, mean lifetime is 10.0. In 10.0 time units, approx 1/e remain.
        assert!(state.x < 100);
        assert!(state.x > 0);
        Ok(())
    }

    #[test]
    #[verified_engine::verified]
    fn test_gillespie_solver_verified_run() -> Result<(), StochasticError> {
        let rng = oxidize_core::rng::OxidizeRng::default();
        let mut solver = GillespieSolver::new(rng);
        let model = DecayModel { k: 0.1 };
        let mut state = DecayState { x: 100 };

        let (result, _metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            solver.step(&model, &mut state)
        });

        let dt = result?;
        assert!(dt > 0.0);
        assert_eq!(state.x, 99);
        Ok(())
    }

    struct MultiReactionModel {
        num_reactions: usize,
    }

    struct MultiState {
        counts: [i32; 32],
    }

    impl StochasticSystem<MultiState> for MultiReactionModel {
        #[verified_engine::verified]
        fn propensities(&self, _state: &MultiState, out: &mut Vec<f64>) {
            for _ in 0..self.num_reactions {
                out.push(1.0);
            }
        }

        #[verified_engine::verified]
        fn react(
            &self,
            state: &mut MultiState,
            reaction_index: usize,
        ) -> Result<(), StochasticError> {
            if reaction_index < state.counts.len() {
                state.counts[reaction_index] += 1;
                Ok(())
            } else {
                Err(StochasticError::InvalidReactionIndex(reaction_index))
            }
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_gillespie_solver_with_capacity() -> Result<(), StochasticError> {
        let n_reactions = 32;
        let rng = oxidize_core::rng::OxidizeRng::default();
        let mut solver = GillespieSolver::with_capacity(rng, n_reactions);
        let model = MultiReactionModel {
            num_reactions: n_reactions,
        };
        let mut state = MultiState { counts: [0; 32] };

        let (result, _metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            solver.step(&model, &mut state)
        });

        let dt = result?;
        assert!(dt > 0.0);
        assert_eq!(state.counts.iter().sum::<i32>(), 1);
        Ok(())
    }

    #[test]
    #[verified_engine::verified]
    fn test_gillespie_solver_zero_propensity() -> Result<(), StochasticError> {
        let rng = oxidize_core::rng::OxidizeRng::default();
        let mut solver = GillespieSolver::new(rng);
        let model = DecayModel { k: 0.1 };
        let mut state = DecayState { x: 0 };

        let initial_x = state.x;
        let dt = solver.step(&model, &mut state)?;
        assert_eq!(dt, f64::INFINITY);
        assert_eq!(state.x, initial_x);
        Ok(())
    }
}

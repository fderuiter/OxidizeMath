//! Period detection for discrete-time Markov chains via BFS chord GCD.

use super::{MarkovChain, Result};
use crate::error::MarkovError;
use nalgebra::RealField;
use num_traits::ToPrimitive;
use std::collections::VecDeque;

#[verified_engine::verified]
fn gcd_usize(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

impl<T: RealField + Copy + ToPrimitive> MarkovChain<T> {
    /// Computes the period of `state` as the GCD of all return-cycle lengths within its SCC.
    ///
    /// Returns `0` for transient singleton states with zero return probability.
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::InvalidState`] if `state >= self.num_states()`.
    #[verified_engine::verified]
    pub fn state_period(&self, state: usize) -> Result<usize> {
        let n = self.num_states();
        if state >= n {
            return Err(MarkovError::InvalidState {
                reason: format!("State index {state} out of bounds for {n} states"),
            });
        }

        let eps = T::from_f64(1e-15).unwrap();
        if self.transition_matrix[(state, state)] > eps {
            return Ok(1);
        }

        let classes = self.communicating_classes();
        let class = classes
            .iter()
            .find(|c| c.states.binary_search(&state).is_ok())
            .ok_or_else(|| MarkovError::InvalidState {
                reason: format!("State {state} missing from SCC partition"),
            })?;

        Ok(self.class_bfs_period(state, &class.states, eps))
    }

    /// Returns the period of the unique closed recurrent class, or `None` if multiple closed
    /// classes exist.
    #[must_use]
    #[verified_engine::verified]
    pub fn period(&self) -> Option<usize> {
        let closed = self.closed_classes();
        if closed.len() != 1 {
            return None;
        }
        let rep = *closed[0].states.first()?;
        self.state_period(rep).ok()
    }

    /// Returns `true` if every closed recurrent class in the chain has period `1`.
    #[must_use]
    #[verified_engine::verified]
    pub fn is_aperiodic(&self) -> bool {
        let closed = self.closed_classes();
        !closed.is_empty()
            && closed.iter().all(|c| {
                c.states
                    .first()
                    .and_then(|&s| self.state_period(s).ok())
                    .is_some_and(|p| p == 1)
            })
    }

    #[verified_engine::verified]
    fn class_bfs_period(&self, root: usize, class_states: &[usize], eps: T) -> usize {
        let n = self.num_states();
        let mut in_class = vec![false; n];
        for &s in class_states {
            in_class[s] = true;
        }

        let mut depth: Vec<Option<usize>> = vec![None; n];
        let mut queue = VecDeque::with_capacity(class_states.len());
        depth[root] = Some(0);
        queue.push_back(root);

        while let Some(u) = queue.pop_front() {
            let du = depth[u].unwrap_or(0);
            for &v in class_states {
                if in_class[v] && depth[v].is_none() && self.transition_matrix[(u, v)] > eps {
                    depth[v] = Some(du + 1);
                    queue.push_back(v);
                }
            }
        }

        let mut period_gcd = 0_usize;
        for &u in class_states {
            let Some(du) = depth[u] else { continue };
            for &v in class_states {
                if self.transition_matrix[(u, v)] > eps
                    && let Some(dv) = depth[v]
                {
                    let chord_diff = (du + 1).abs_diff(dv);
                    if chord_diff > 0 {
                        period_gcd = gcd_usize(period_gcd, chord_diff);
                    }
                }
            }
        }
        period_gcd
    }
}

// [cite:clinical_trials_statistics]

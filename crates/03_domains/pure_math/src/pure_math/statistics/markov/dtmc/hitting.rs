//! Mean first-passage / hitting times for discrete-time Markov chains.

use super::{MarkovChain, Result};
use crate::error::MarkovError;
use nalgebra::{DMatrix, DVector, RealField};
use num_traits::ToPrimitive;
use std::collections::VecDeque;

impl<T: RealField + Copy + ToPrimitive> MarkovChain<T> {
    /// Computes the mean first hitting time $h_i = \mathbb{E}_i[\inf\{n \ge 0 : X_n = \text{target}\}]$
    /// from every state $i$ to `target`.
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::InvalidState`] if `target` is out of bounds or unreachable from
    /// one or more states, or [`MarkovError::SingularMatrix`] if the reduced system is singular.
    #[verified_engine::verified]
    pub fn mean_hitting_times(&self, target: usize) -> Result<DVector<T>> {
        self.mean_hitting_times_to_set(&[target])
    }

    /// Computes the mean first hitting time to any state in `targets`.
    ///
    /// Solves $(I - P_{UU}) h_U = \mathbf{1}$ over non-target states $U = S \setminus \text{targets}$,
    /// setting $h_s = 0$ for $s \in \text{targets}$.
    ///
    /// # Errors
    ///
    /// Returns an error if `targets` is empty, contains an out-of-bounds index, or is not
    /// reachable from all non-target states.
    #[verified_engine::verified]
    pub fn mean_hitting_times_to_set(&self, targets: &[usize]) -> Result<DVector<T>> {
        let n = self.num_states();
        let is_target = self.validate_target_mask(targets)?;
        if !self.targets_reachable_from_all(&is_target) {
            return Err(MarkovError::InvalidState {
                reason: "Target set is unreachable from one or more states".to_string(),
            });
        }

        let non_targets: Vec<usize> = (0..n).filter(|&s| !is_target[s]).collect();
        let m = non_targets.len();
        if m == 0 {
            return Ok(DVector::zeros(n));
        }

        let mut lhs = DMatrix::identity(m, m);
        for (row, &i) in non_targets.iter().enumerate() {
            for (col, &j) in non_targets.iter().enumerate() {
                lhs[(row, col)] -= self.transition_matrix[(i, j)];
            }
        }

        let rhs = DVector::from_element(m, T::one());
        let solved = lhs
            .qr()
            .solve(&rhs)
            .ok_or_else(|| MarkovError::SingularMatrix {
                context: "Cannot solve (I - P_UU) for mean hitting times".to_string(),
            })?;

        let mut hitting = DVector::zeros(n);
        for (idx, &state) in non_targets.iter().enumerate() {
            hitting[state] = solved[idx];
        }
        Ok(hitting)
    }

    #[verified_engine::verified]
    fn validate_target_mask(&self, targets: &[usize]) -> Result<Vec<bool>> {
        let n = self.num_states();
        if targets.is_empty() {
            return Err(MarkovError::InvalidState {
                reason: "Target state set cannot be empty".to_string(),
            });
        }
        let mut mask = vec![false; n];
        for &t in targets {
            if t >= n {
                return Err(MarkovError::InvalidState {
                    reason: format!("Target state {t} out of bounds for {n} states"),
                });
            }
            mask[t] = true;
        }
        Ok(mask)
    }

    #[verified_engine::verified]
    fn targets_reachable_from_all(&self, is_target: &[bool]) -> bool {
        let n = self.num_states();
        let eps = T::from_f64(1e-15).unwrap();
        let mut can_reach = is_target.to_vec();
        let mut queue: VecDeque<usize> = (0..n).filter(|&s| is_target[s]).collect();

        while let Some(v) = queue.pop_front() {
            for (u, reached) in can_reach.iter_mut().enumerate().take(n) {
                if !*reached && self.transition_matrix[(u, v)] > eps {
                    *reached = true;
                    queue.push_back(u);
                }
            }
        }

        can_reach.into_iter().all(|r| r)
    }
}

// [cite:clinical_trials_statistics]

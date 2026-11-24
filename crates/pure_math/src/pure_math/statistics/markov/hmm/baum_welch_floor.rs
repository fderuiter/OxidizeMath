//! Simplex-floored Baum-Welch Expectation-Maximization built on [`BaumWelchWorkspace`].
// [cite:clinical_trials_statistics]

use super::baum_welch::{
    accumulate_backward_stats, apply_m_step, restore_parameters, run_forward_into,
    snapshot_parameters,
};
use super::baum_welch_types::{BaumWelchResult, BaumWelchWorkspace};
use super::model::HiddenMarkovModel;
use crate::error::MarkovError;
use nalgebra::RealField;
use num_traits::ToPrimitive;

#[verified_engine::verified]
fn clamp_and_normalize_positive<T: RealField + Copy>(probs: &mut [T]) -> bool {
    let k = probs.len();
    if k == 0 {
        return false;
    }
    let zero = T::zero();
    let one = T::one();
    let sum = probs
        .iter()
        .fold(zero, |acc, &x| acc + if x > zero { x } else { zero });
    if sum <= zero {
        let unif = one / T::from_usize(k).unwrap();
        probs.fill(unif);
        return false;
    }
    for p in probs.iter_mut() {
        *p = if *p > zero { *p / sum } else { zero };
    }
    true
}

#[verified_engine::verified(alloc_ok = "allocates active-set mask for simplex projection")]
fn project_floored_simplex<T: RealField + Copy>(probs: &mut [T], floor: T) {
    let zero = T::zero();
    if !clamp_and_normalize_positive(probs) || floor <= zero {
        return;
    }
    let k = probs.len();
    let one = T::one();
    let mut fixed = vec![false; k];
    for _ in 0..k {
        let mut changed = false;
        let mut fixed_count = 0_usize;
        let mut free_sum = zero;
        for (i, &p) in probs.iter().enumerate() {
            if fixed[i] || p < floor {
                if !fixed[i] {
                    fixed[i] = true;
                    changed = true;
                }
                fixed_count += 1;
            } else {
                free_sum += p;
            }
        }
        if !changed {
            break;
        }
        if fixed_count == k || free_sum <= zero {
            probs.fill(one / T::from_usize(k).unwrap());
            return;
        }
        let remaining = one - T::from_usize(fixed_count).unwrap() * floor;
        for (i, p) in probs.iter_mut().enumerate() {
            *p = if fixed[i] {
                floor
            } else {
                (*p / free_sum) * remaining
            };
        }
    }
}

impl<T: RealField + Copy + ToPrimitive> HiddenMarkovModel<T> {
    /// Estimates $(\pi, A, B)$ via Baum-Welch Expectation-Maximization with probability floor `prob_floor`.
    ///
    /// Enforces simplex floor constraints $p_k \ge \text{prob\_floor}$ across initial, transition,
    /// and emission rows to prevent zero-lock absorption during training while guaranteeing
    /// monotonically non-decreasing log-likelihoods.
    ///
    /// # Errors
    ///
    /// Returns an error if `observations` is invalid, `max_iters == 0`, `tolerance < 0`,
    /// `prob_floor` exceeds $1 / \max(N, M)$, or the sequence has zero initial likelihood.
    #[verified_engine::verified(
        alloc_ok = "allocates Baum-Welch workspace and log-likelihood trace"
    )]
    pub fn baum_welch(
        &mut self,
        observations: &[usize],
        max_iters: usize,
        tolerance: T,
        prob_floor: T,
    ) -> Result<BaumWelchResult<T>, MarkovError> {
        self.validate_baum_welch_args(observations, max_iters, tolerance, prob_floor)?;
        self.apply_parameter_floors(prob_floor);

        let mut ws =
            BaumWelchWorkspace::new(self.num_states, self.num_observations, observations.len());
        let mut current_ll = run_forward_into(self, observations, &mut ws);
        if !current_ll.is_finite() {
            return Err(MarkovError::InvalidObservation {
                reason: "Observation sequence has zero probability under current parameters"
                    .to_string(),
            });
        }

        let has_transitions = observations.len() > 1;
        let mut log_likelihoods = Vec::with_capacity(max_iters + 1);
        log_likelihoods.push(current_ll);
        let mut iterations = 0_usize;
        let mut converged = false;

        for _ in 0..max_iters {
            accumulate_backward_stats(self, observations, &mut ws);
            snapshot_parameters(self, &mut ws);
            apply_m_step(self, &ws, T::zero(), has_transitions);
            self.apply_parameter_floors(prob_floor);

            let next_ll = run_forward_into(self, observations, &mut ws);
            if !next_ll.is_finite() || next_ll < current_ll {
                restore_parameters(self, &ws);
                converged = true;
                break;
            }

            iterations += 1;
            log_likelihoods.push(next_ll);
            let improvement = next_ll - current_ll;
            current_ll = next_ll;
            if improvement <= tolerance {
                converged = true;
                break;
            }
        }

        Ok(BaumWelchResult {
            log_likelihoods,
            iterations,
            converged,
        })
    }

    #[verified_engine::verified(alloc_ok = "formats validation error on invalid input")]
    fn validate_baum_welch_args(
        &self,
        observations: &[usize],
        max_iters: usize,
        tolerance: T,
        prob_floor: T,
    ) -> Result<(), MarkovError> {
        self.validate_observations(observations)?;
        if max_iters == 0 {
            return Err(MarkovError::InvalidState {
                reason: "Baum-Welch max_iters must be at least 1".to_string(),
            });
        }
        let zero = T::zero();
        if !tolerance.is_finite() || tolerance < zero {
            return Err(MarkovError::InvalidState {
                reason: "Baum-Welch tolerance must be finite and non-negative".to_string(),
            });
        }
        let max_dim = self.num_states.max(self.num_observations);
        let max_floor = T::one() / T::from_usize(max_dim).unwrap();
        if !prob_floor.is_finite() || prob_floor < zero || prob_floor >= max_floor {
            return Err(MarkovError::InvalidProbability {
                value: prob_floor.to_f64().unwrap_or(f64::NAN),
            });
        }
        Ok(())
    }

    #[verified_engine::verified(alloc_ok = "allocates row scratch buffers for floor projection")]
    fn apply_parameter_floors(&mut self, floor: T) {
        if floor <= T::zero() {
            return;
        }
        project_floored_simplex(self.initial.as_mut_slice(), floor);
        let mut row_buf = vec![T::zero(); self.num_states];
        for i in 0..self.num_states {
            for (j, slot) in row_buf.iter_mut().enumerate() {
                *slot = self.transitions[(i, j)];
            }
            project_floored_simplex(&mut row_buf, floor);
            for (j, &val) in row_buf.iter().enumerate() {
                self.transitions[(i, j)] = val;
            }
        }
        let mut emit_buf = vec![T::zero(); self.num_observations];
        for i in 0..self.num_states {
            for (k, slot) in emit_buf.iter_mut().enumerate() {
                *slot = self.emissions[(i, k)];
            }
            project_floored_simplex(&mut emit_buf, floor);
            for (k, &val) in emit_buf.iter().enumerate() {
                self.emissions[(i, k)] = val;
            }
        }
    }
}

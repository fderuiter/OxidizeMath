//! Baum-Welch Expectation-Maximization (EM) parameter learning with probability floors.

use super::model::HiddenMarkovModel;
use crate::error::MarkovError;
/// Result alias for Baum-Welch Expectation-Maximization operations.
pub type Result<T> = std::result::Result<T, MarkovError>;
use nalgebra::{DMatrix, DVector, RealField};
use num_traits::ToPrimitive;

/// Summary returned by [`HiddenMarkovModel::baum_welch`].
#[derive(Debug, Clone)]
pub struct BaumWelchResult<T: RealField + Copy> {
    /// Log-likelihood trace $\ln P(Y \mid \theta^{(k)})$ at each accepted EM state.
    pub log_likelihoods: Vec<T>,
    /// Number of parameter update iterations completed.
    pub iterations: usize,
    /// True if consecutive log-likelihood improvement fell within `tolerance`.
    pub converged: bool,
}

#[verified_engine::verified]
fn project_floored_simplex<T: RealField + Copy>(probs: &mut [T], floor: T) {
    let k = probs.len();
    if k == 0 {
        return;
    }
    let zero = T::zero();
    let one = T::one();
    let sum = probs
        .iter()
        .fold(zero, |acc, &x| acc + if x > zero { x } else { zero });

    if sum <= zero {
        let unif = one / T::from_usize(k).unwrap();
        for p in probs.iter_mut() {
            *p = unif;
        }
        return;
    }
    for p in probs.iter_mut() {
        *p = if *p > zero { *p / sum } else { zero };
    }
    if floor <= zero {
        return;
    }

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
            let unif = one / T::from_usize(k).unwrap();
            for p in probs.iter_mut() {
                *p = unif;
            }
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
    #[verified_engine::verified]
    pub fn baum_welch(
        &mut self,
        observations: &[usize],
        max_iters: usize,
        tolerance: T,
        prob_floor: T,
    ) -> Result<BaumWelchResult<T>> {
        self.validate_baum_welch_args(observations, max_iters, tolerance, prob_floor)?;
        self.apply_parameter_floors(prob_floor);

        let (mut alpha, mut scales) = self.forward_probabilities(observations)?;
        let mut current_ll = Self::checked_log_likelihood(&scales)?;
        let mut log_likelihoods = Vec::with_capacity(max_iters + 1);
        log_likelihoods.push(current_ll);

        let mut iterations = 0_usize;
        let mut converged = false;

        for _ in 0..max_iters {
            let beta = self.backward_probabilities(observations, &scales)?;
            let mut candidate = self.clone();
            candidate.em_maximization_step(observations, &alpha, &beta, prob_floor);

            let (next_alpha, next_scales) = candidate.forward_probabilities(observations)?;
            let Ok(next_ll) = Self::checked_log_likelihood(&next_scales) else {
                converged = true;
                break;
            };

            if next_ll < current_ll {
                converged = true;
                break;
            }

            *self = candidate;
            iterations += 1;
            log_likelihoods.push(next_ll);
            let improvement = next_ll - current_ll;
            alpha = next_alpha;
            scales = next_scales;
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

    #[verified_engine::verified]
    fn validate_baum_welch_args(
        &self,
        observations: &[usize],
        max_iters: usize,
        tolerance: T,
        prob_floor: T,
    ) -> Result<()> {
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

    #[verified_engine::verified]
    fn checked_log_likelihood(scales: &[T]) -> Result<T> {
        let zero = T::zero();
        let mut ll = zero;
        for &c in scales {
            if c <= zero {
                return Err(MarkovError::InvalidObservation {
                    reason: "Observation sequence has zero probability under current parameters"
                        .to_string(),
                });
            }
            ll += c.ln();
        }
        Ok(ll)
    }

    #[verified_engine::verified]
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

    #[verified_engine::verified]
    fn em_maximization_step(
        &mut self,
        observations: &[usize],
        alpha: &DMatrix<T>,
        beta: &DMatrix<T>,
        floor: T,
    ) {
        let gamma = Self::compute_gamma_matrix(alpha, beta);
        self.update_initial_and_emissions(observations, &gamma, floor);
        if observations.len() >= 2 {
            self.update_transitions_from_xi(observations, alpha, beta, &gamma, floor);
        }
    }

    #[verified_engine::verified]
    fn compute_gamma_matrix(alpha: &DMatrix<T>, beta: &DMatrix<T>) -> DMatrix<T> {
        let n_states = alpha.nrows();
        let t_max = alpha.ncols();
        let zero = T::zero();
        let mut gamma = DMatrix::zeros(n_states, t_max);
        for t in 0..t_max {
            let mut sum = zero;
            for i in 0..n_states {
                let g = alpha[(i, t)] * beta[(i, t)];
                gamma[(i, t)] = g;
                sum += g;
            }
            if sum > zero {
                for i in 0..n_states {
                    gamma[(i, t)] /= sum;
                }
            }
        }
        gamma
    }

    #[verified_engine::verified]
    fn update_initial_and_emissions(
        &mut self,
        observations: &[usize],
        gamma: &DMatrix<T>,
        floor: T,
    ) {
        let zero = T::zero();
        let mut init_vec: DVector<T> = gamma.column(0).into_owned();
        project_floored_simplex(init_vec.as_mut_slice(), floor);
        self.initial = init_vec;

        let mut emit_row = vec![zero; self.num_observations];
        for i in 0..self.num_states {
            emit_row.fill(zero);
            let mut denom = zero;
            for (t, &obs) in observations.iter().enumerate() {
                let g = gamma[(i, t)];
                emit_row[obs] += g;
                denom += g;
            }
            if denom > zero {
                for val in &mut emit_row {
                    *val /= denom;
                }
                project_floored_simplex(&mut emit_row, floor);
                for (k, &val) in emit_row.iter().enumerate() {
                    self.emissions[(i, k)] = val;
                }
            }
        }
    }

    #[verified_engine::verified]
    fn update_transitions_from_xi(
        &mut self,
        observations: &[usize],
        alpha: &DMatrix<T>,
        beta: &DMatrix<T>,
        gamma: &DMatrix<T>,
        floor: T,
    ) {
        let zero = T::zero();
        let t_max = observations.len();
        let mut xi_sum = DMatrix::zeros(self.num_states, self.num_states);
        let mut xi_step = DMatrix::zeros(self.num_states, self.num_states);

        for t in 0..(t_max - 1) {
            let y_next = observations[t + 1];
            let mut step_norm = zero;
            for i in 0..self.num_states {
                for j in 0..self.num_states {
                    let w = alpha[(i, t)]
                        * self.transitions[(i, j)]
                        * self.emissions[(j, y_next)]
                        * beta[(j, t + 1)];
                    xi_step[(i, j)] = w;
                    step_norm += w;
                }
            }
            if step_norm > zero {
                xi_sum += &xi_step / step_norm;
            }
        }

        let mut trans_row = vec![zero; self.num_states];
        for i in 0..self.num_states {
            let denom = (0..(t_max - 1)).fold(zero, |acc, t| acc + gamma[(i, t)]);
            if denom > zero {
                for (j, slot) in trans_row.iter_mut().enumerate() {
                    *slot = xi_sum[(i, j)] / denom;
                }
                project_floored_simplex(&mut trans_row, floor);
                for (j, &val) in trans_row.iter().enumerate() {
                    self.transitions[(i, j)] = val;
                }
            }
        }
    }
}

// [cite:clinical_trials_statistics]

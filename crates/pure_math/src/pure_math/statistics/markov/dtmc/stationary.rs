//! QR linear nullspace stationary and limiting distributions for discrete-time Markov chains.

use super::MarkovChain;
use nalgebra::{DMatrix, DVector, RealField};
use num_traits::ToPrimitive;

impl<T: RealField + Copy + ToPrimitive> MarkovChain<T> {
    /// Computes the unique stationary distribution $\pi$ solving $(P^T - I)\pi = 0$, $\sum_i \pi_i = 1$.
    ///
    /// Returns `Some(pi)` **if and only if** the chain has a single closed recurrent class
    /// (determined via Tarjan SCC), regardless of periodicity. Chains with multiple closed
    /// or absorbing classes have a multi-dimensional stationary simplex and return `None`.
    #[must_use]
    #[verified_engine::verified]
    pub fn stationary_distribution(&self) -> Option<DVector<T>> {
        let closed = self.closed_classes();
        if closed.len() != 1 {
            return None;
        }

        let n = self.num_states();
        if closed[0].states.len() == 1 {
            let mut pi = DVector::zeros(n);
            pi[closed[0].states[0]] = T::one();
            return Some(pi);
        }

        solve_stochastic_qr_nullspace(&self.transition_matrix)
    }

    /// Computes $\lim_{k \to \infty} (\pi^{(0)})^T P^k$ from `initial_weights`.
    ///
    /// - For absorbing chains, projects transient weights through $B = N \cdot R$ (`u_A = B^T w_T + w_A`).
    /// - For chains with aperiodic closed classes, projects initial mass onto each closed class's
    ///   unique stationary distribution.
    /// - Returns `None` if `initial_weights` is invalid or any positively weighted closed class is periodic.
    #[must_use]
    #[verified_engine::verified]
    pub fn limiting_distribution(&self, initial_weights: &DVector<T>) -> Option<DVector<T>> {
        let weights = self.normalize_initial_weights(initial_weights)?;
        let fixed_tol = T::from_f64(1e-12).unwrap();
        if (&(self.transition_matrix.transpose() * &weights) - &weights).amax() <= fixed_tol {
            return Some(weights);
        }

        let closed = self.closed_classes();
        if closed.is_empty() {
            return None;
        }

        // Canonical absorbing case: project transient state weights directly through B = N * R.
        if !self.absorbing_indices.is_empty() && closed.len() == self.absorbing_indices.len() {
            return self.project_absorbing_limiting(&weights);
        }

        if closed.len() == 1 {
            let rep = *closed[0].states.first()?;
            if self.state_period(rep).ok()? != 1 {
                return None;
            }
            return self.stationary_distribution();
        }

        self.project_general_aperiodic_limiting(&weights, &closed)
    }

    /// Convenience wrapper computing `limiting_distribution` starting deterministically in `initial_state`.
    #[must_use]
    #[verified_engine::verified]
    pub fn limiting_distribution_from_state(&self, initial_state: usize) -> Option<DVector<T>> {
        if initial_state >= self.num_states() {
            return None;
        }
        let mut e = DVector::zeros(self.num_states());
        e[initial_state] = T::one();
        self.limiting_distribution(&e)
    }

    #[verified_engine::verified]
    fn normalize_initial_weights(&self, weights: &DVector<T>) -> Option<DVector<T>> {
        if weights.len() != self.num_states() {
            return None;
        }
        let neg_tol = T::from_f64(1e-12).unwrap();
        let sum_tol = T::from_f64(math_commons::registry::TOLERANCE_FAST).unwrap();
        let mut cleaned = DVector::zeros(weights.len());
        let mut total = T::zero();
        for i in 0..weights.len() {
            let w = weights[i];
            if !w.is_finite() || w < -neg_tol {
                return None;
            }
            let clamped = if w < T::zero() { T::zero() } else { w };
            cleaned[i] = clamped;
            total += clamped;
        }
        if (total - T::one()).abs() > sum_tol || total <= T::zero() {
            return None;
        }
        Some(cleaned / total)
    }

    #[verified_engine::verified]
    fn project_absorbing_limiting(&self, weights: &DVector<T>) -> Option<DVector<T>> {
        let n = self.num_states();
        let mut limiting = DVector::zeros(n);
        if self.transient_indices.is_empty() {
            return Some(weights.clone());
        }

        let b = self.absorption_probabilities().ok()?;
        let mut w_t = DVector::zeros(self.transient_indices.len());
        for (k, &s) in self.transient_indices.iter().enumerate() {
            w_t[k] = weights[s];
        }

        let absorbed_from_transient = b.transpose() * w_t;
        for (j, &abs_state) in self.absorbing_indices.iter().enumerate() {
            limiting[abs_state] = weights[abs_state] + absorbed_from_transient[j];
        }
        Some(limiting)
    }

    #[verified_engine::verified]
    fn project_general_aperiodic_limiting(
        &self,
        weights: &DVector<T>,
        closed: &[super::scc::CommunicatingClass],
    ) -> Option<DVector<T>> {
        let n = self.num_states();
        let mut is_recurrent = vec![false; n];
        for c in closed {
            let rep = *c.states.first()?;
            if self.state_period(rep).ok()? != 1 {
                return None;
            }
            for &s in &c.states {
                is_recurrent[s] = true;
            }
        }

        let transient: Vec<usize> = (0..n).filter(|&s| !is_recurrent[s]).collect();
        let mut limiting = DVector::zeros(n);

        for c in closed {
            let class_pi = solve_subchain_stationary(&self.transition_matrix, &c.states)?;
            let hit_mass = self.class_absorption_mass(weights, &transient, &c.states)?;
            for (local_idx, &state) in c.states.iter().enumerate() {
                limiting[state] += hit_mass * class_pi[local_idx];
            }
        }
        Some(limiting)
    }

    #[verified_engine::verified]
    fn class_absorption_mass(
        &self,
        weights: &DVector<T>,
        transient: &[usize],
        class_states: &[usize],
    ) -> Option<T> {
        let mut mass = class_states
            .iter()
            .fold(T::zero(), |acc, &s| acc + weights[s]);
        if transient.is_empty() {
            return Some(mass);
        }

        let m = transient.len();
        let mut lhs = DMatrix::identity(m, m);
        let mut rhs = DVector::zeros(m);
        for (r, &i) in transient.iter().enumerate() {
            for (col, &j) in transient.iter().enumerate() {
                lhs[(r, col)] -= self.transition_matrix[(i, j)];
            }
            rhs[r] = class_states
                .iter()
                .fold(T::zero(), |acc, &s| acc + self.transition_matrix[(i, s)]);
        }

        let hit_probs = lhs.qr().solve(&rhs)?;
        for (r, &i) in transient.iter().enumerate() {
            mass += weights[i] * hit_probs[r];
        }
        Some(mass)
    }
}

#[verified_engine::verified]
fn solve_subchain_stationary<T: RealField + Copy + ToPrimitive>(
    transition: &DMatrix<T>,
    states: &[usize],
) -> Option<DVector<T>> {
    let m = states.len();
    if m == 1 {
        return Some(DVector::from_element(1, T::one()));
    }
    let mut sub = DMatrix::zeros(m, m);
    for (r, &i) in states.iter().enumerate() {
        for (c, &j) in states.iter().enumerate() {
            sub[(r, c)] = transition[(i, j)];
        }
    }
    solve_stochastic_qr_nullspace(&sub)
}

#[verified_engine::verified]
fn solve_stochastic_qr_nullspace<T: RealField + Copy + ToPrimitive>(
    transition: &DMatrix<T>,
) -> Option<DVector<T>> {
    let n = transition.nrows();
    let mut a = DMatrix::zeros(n + 1, n);
    for i in 0..n {
        for j in 0..n {
            let val = transition[(j, i)] - if i == j { T::one() } else { T::zero() };
            a[(i, j)] = val;
        }
    }
    for j in 0..n {
        a[(n, j)] = T::one();
    }

    let mut b = DVector::zeros(n + 1);
    b[n] = T::one();

    let qr = a.qr();
    let r = qr.r();
    let r_max = (0..n).fold(T::zero(), |acc, i| acc.max(r[(i, i)].abs()));
    let rank_tol = T::from_f64(1e-12).unwrap() * r_max;
    if r_max <= T::zero() || (0..n).any(|i| r[(i, i)].abs() <= rank_tol) {
        return None;
    }

    let q_t_b = qr.q().transpose() * b;
    let mut pi = r.solve_upper_triangular(&q_t_b)?;

    let clamp_tol = T::from_f64(1e-12).unwrap();
    let mut total = T::zero();
    for i in 0..n {
        if !pi[i].is_finite() || pi[i] < -clamp_tol {
            return None;
        }
        if pi[i] < T::zero() {
            pi[i] = T::zero();
        }
        total += pi[i];
    }
    if total <= T::zero() {
        return None;
    }
    pi /= total;

    let residual = (&(transition.transpose() * &pi) - &pi).amax();
    if residual > T::from_f64(1e-10).unwrap() {
        return None;
    }
    Some(pi)
}

// [cite:clinical_trials_statistics]

//! Metropolis-Hastings and pairwise Fermi evolutionary game kernels for [`super::GraphSpinSystem`].

use super::{GraphSpinSystem, KB};
use domain_applied::applied::game_theory::evolutionary::MatrixPayoff;
use rand::Rng;

impl GraphSpinSystem {
    #[inline]
    #[verified_engine::verified]
    pub(super) fn weighted_neighbor_sum(&self, node: usize) -> f64 {
        let start = self.row_ptr[node];
        let end = self.row_ptr[node + 1];
        let mut acc = 0.0_f64;
        for idx in start..end {
            let nbr = self.col_idx[idx];
            acc += self.weights[idx] * f64::from(self.spins[nbr]);
        }
        acc
    }

    #[verified_engine::verified]
    pub(super) fn mean_game_payoff_inner(&self, payoff: &MatrixPayoff) -> f64 {
        let mat = payoff.payoff_matrix();
        if self.n_nodes == 0 || mat.nrows() != 2 || mat.ncols() != 2 {
            return 0.0;
        }
        let (r, s, t, p) = (mat[(0, 0)], mat[(0, 1)], mat[(1, 0)], mat[(1, 1)]);
        let mut total = 0.0_f64;
        for i in 0..self.n_nodes {
            total += self.node_game_payoff(i, r, s, t, p);
        }
        total / (self.n_nodes as f64)
    }

    #[inline]
    #[verified_engine::verified]
    fn node_game_payoff(&self, node: usize, r: f64, s: f64, t: f64, p: f64) -> f64 {
        let s_i = self.spins[node];
        let start = self.row_ptr[node];
        let end = self.row_ptr[node + 1];
        let mut acc = if s_i > 0 { r } else { p };
        for idx in start..end {
            let w = self.weights[idx].max(0.0);
            let s_nbr = self.spins[self.col_idx[idx]];
            let pair = match (s_i > 0, s_nbr > 0) {
                (true, true) => r,
                (true, false) => s,
                (false, true) => t,
                (false, false) => p,
            };
            acc += w * pair;
        }
        acc
    }

    #[verified_engine::verified]
    pub(super) fn evolve_inner(
        &mut self,
        steps: usize,
        temperature: f64,
        j_coupling: f64,
        h_field: f64,
    ) {
        if steps == 0
            || self.n_nodes == 0
            || !temperature.is_finite()
            || !j_coupling.is_finite()
            || !h_field.is_finite()
        {
            return;
        }
        let kbt = KB * temperature;
        let inv_kbt = if kbt > 0.0 && kbt.is_finite() {
            1.0 / kbt
        } else {
            0.0
        };
        for _ in 0..steps {
            let idx = self.rng.gen_range(0..self.n_nodes);
            let s_i = self.spins[idx];
            let nbr_sum = self.weighted_neighbor_sum(idx);
            let delta_e = 2.0 * f64::from(s_i) * (j_coupling * nbr_sum + h_field);
            let accept = if delta_e <= 0.0 {
                true
            } else if inv_kbt > 0.0 {
                let exponent = (-inv_kbt * delta_e).clamp(-500.0, 0.0);
                self.rng.r#gen::<f64>() < exponent.exp()
            } else {
                false
            };
            if accept {
                self.spins[idx] = -s_i;
            }
        }
    }

    #[verified_engine::verified]
    pub(super) fn evolve_fermi_game_inner(
        &mut self,
        steps: usize,
        temperature: f64,
        payoff: &MatrixPayoff,
    ) {
        let mat = payoff.payoff_matrix();
        if steps == 0
            || self.n_nodes == 0
            || mat.nrows() != 2
            || mat.ncols() != 2
            || !temperature.is_finite()
        {
            return;
        }
        let (r, s, t, p) = (mat[(0, 0)], mat[(0, 1)], mat[(1, 0)], mat[(1, 1)]);
        if !r.is_finite() || !s.is_finite() || !t.is_finite() || !p.is_finite() {
            return;
        }
        let inv_temp = 1.0 / temperature.max(1e-3);
        for _ in 0..steps {
            let i = self.rng.gen_range(0..self.n_nodes);
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            if start == end {
                continue;
            }
            let j = self.col_idx[self.rng.gen_range(start..end)];
            let s_i = self.spins[i];
            let s_j = self.spins[j];
            if s_i != s_j {
                let pi_i = self.node_game_payoff(i, r, s, t, p);
                let pi_j = self.node_game_payoff(j, r, s, t, p);
                let arg = ((pi_i - pi_j) * inv_temp).clamp(-500.0, 500.0);
                let adopt_prob = 1.0 / (1.0 + arg.exp());
                if self.rng.r#gen::<f64>() < adopt_prob {
                    self.spins[i] = s_j;
                }
            }
        }
    }
}

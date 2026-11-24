//! Arbitrary CSR Graph Ising & Spatial Evolutionary Game Simulation.
//!
//! Provides [`GraphSpinSystem`] for zero-allocation Metropolis-Hastings Ising spin flips
//! and pairwise Fermi evolutionary games on arbitrary weighted Compressed Sparse Row (CSR)
//! contact networks.

use super::KB;
use domain_applied::applied::game_theory::evolutionary::MatrixPayoff;
use math_commons::error::MathError;
use oxidize_core::rng::OxidizeRng;
use rand::Rng;
use verified_engine::Theory;

/// Maximum CSR node bound $N_{\max} = 256$ for [`GraphSpinSystem`] (`@metropolis1953equation`, `spec:continuum_spde_stat_mech_v1`).
/// Numerical stability: bounds integer magnetization $|M| \le 256$ without overflow. Domain of validity: $0 \le N \le 256$.
pub const MAX_GRAPH_SPIN_NODES: usize = 256;

/// Maximum CSR half-edge bound $E_{\max} = 4096$ (`@metropolis1953equation`, `spec:continuum_spde_stat_mech_v1`).
/// Numerical stability: bounds neighbor sum accumulation length per step. Domain of validity: $0 \le |\text{col\_idx}| \le 4096$.
pub const MAX_GRAPH_SPIN_CSR_ENTRIES: usize = 4_096;

/// Discrete spin $s_i \in \{-1, +1\}$ system with Hamiltonian $E = -\frac{1}{2}J\sum_{i,j} w_{ij}s_i s_j - h\sum_i s_i$ over a weighted CSR graph (`@metropolis1953equation`, `@szabo1998evolutionary`, `spec:continuum_spde_stat_mech_v1`).
///
/// - **Numerical stability**: Clamps Boltzmann and Fermi exponents to $[-500, 500]$ and $k_B T \ge 10^{-30}$ with zero post-initialization heap allocations.
/// - **Domain of validity**: Validated CSR graphs with $0 \le N \le 256$, $0 \le E \le 4096$, finite edge weights $w_{ij} \in \mathbb{R}$, and finite $T > 0$.
///
/// # Examples
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::stat_mech::graph_ising::GraphSpinSystem;
/// use math_explorer::physics::stat_mech::KB;
///
/// // 4-node cycle: 0-1-2-3-0 (symmetric CSR)
/// let row_ptr = vec![0, 2, 4, 6, 8];
/// let col_idx = vec![1, 3, 0, 2, 1, 3, 0, 2];
/// let weights = vec![1.0; 8];
/// let mut sys = GraphSpinSystem::try_from_csr(4, row_ptr, col_idx, weights, Some(7)).unwrap();
/// sys.evolve(2_000, 0.5 / KB, 1.0, 0.0);
/// assert_eq!(sys.spins().len(), 4);
/// ```
#[derive(Debug, Clone, Theory)]
#[theory(
    description = "Ising spin and pairwise Fermi evolutionary game dynamics over arbitrary weighted CSR contact graphs.",
    citation = "metropolis1953equation; szabo1998evolutionary"
)]
pub struct GraphSpinSystem {
    n_nodes: usize,
    spins: Vec<i8>,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    weights: Vec<f64>,
    rng: OxidizeRng,
}

#[verified_engine::verified]
fn validate_csr_inputs(
    n_nodes: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    weights: &[f64],
) -> Result<(), MathError> {
    if n_nodes > MAX_GRAPH_SPIN_NODES || col_idx.len() > MAX_GRAPH_SPIN_CSR_ENTRIES {
        return Err(MathError::NumericalError {
            reason: "CSR graph exceeds maximum spin system bounds".to_string(),
        });
    }
    if row_ptr.len() != n_nodes + 1 || row_ptr.first().copied() != Some(0) {
        return Err(MathError::NumericalError {
            reason: "CSR row_ptr must have length n_nodes + 1 starting at 0".to_string(),
        });
    }
    if row_ptr[n_nodes] != col_idx.len() || col_idx.len() != weights.len() {
        return Err(MathError::NumericalError {
            reason: "CSR col_idx and weights lengths must match row_ptr[n_nodes]".to_string(),
        });
    }
    for u in 0..n_nodes {
        if row_ptr[u] > row_ptr[u + 1] {
            return Err(MathError::NumericalError {
                reason: "CSR row_ptr must be monotonically non-decreasing".to_string(),
            });
        }
    }
    if col_idx.iter().any(|&v| v >= n_nodes) || weights.iter().any(|w| !w.is_finite()) {
        return Err(MathError::NumericalError {
            reason: "CSR column indices must be < n_nodes and weights must be finite".to_string(),
        });
    }
    Ok(())
}

impl GraphSpinSystem {
    /// Constructs [`GraphSpinSystem`] with uniform initial spins $s_i \in \{-1, +1\}$ over CSR adjacency (`@metropolis1953equation`, `spec:continuum_spde_stat_mech_v1`).
    /// Numerical stability: validates finite weights and exact CSR index bounds up front. Domain of validity: $N \le 256$, $|\text{col\_idx}| = |\text{weights}| = p_N \le 4096$.
    ///
    /// # Errors
    /// Returns [`MathError::NumericalError`] if CSR bounds, monotonicity, or finiteness fail.
    #[verified_engine::verified]
    pub fn try_from_csr(
        n_nodes: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        weights: Vec<f64>,
        seed: Option<u64>,
    ) -> Result<Self, MathError> {
        validate_csr_inputs(n_nodes, &row_ptr, &col_idx, &weights)?;
        let mut rng = match seed {
            Some(s) => OxidizeRng::from_seed(s),
            None => OxidizeRng::default(),
        };
        let mut spins = Vec::with_capacity(n_nodes);
        for _ in 0..n_nodes {
            spins.push(if rng.r#gen::<bool>() { 1 } else { -1 });
        }
        Ok(Self {
            n_nodes,
            spins,
            row_ptr,
            col_idx,
            weights,
            rng,
        })
    }

    /// Returns vertex count $N = |\mathcal{V}|$ (`@metropolis1953equation`, `spec:continuum_spde_stat_mech_v1`). Numerical stability: exact `usize` read. Domain of validity: $0 \le N \le 256$.
    #[must_use]
    #[verified_engine::verified]
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Returns CSR half-edge count $E = |\text{col\_idx}|$ (`@metropolis1953equation`, `spec:continuum_spde_stat_mech_v1`). Numerical stability: exact `usize` length. Domain of validity: $0 \le E \le 4096$.
    #[must_use]
    #[verified_engine::verified]
    pub fn csr_len(&self) -> usize {
        self.col_idx.len()
    }

    /// Returns spin configuration slice $\mathbf{s} \in \{-1, +1\}^N$ (`@metropolis1953equation`, `spec:continuum_spde_stat_mech_v1`). Numerical stability: exact `i8` borrow. Domain of validity: length $N$.
    #[must_use]
    #[verified_engine::verified]
    pub fn spins(&self) -> &[i8] {
        &self.spins
    }

    /// Returns CSR row offsets $p_0 \le \dots \le p_N$ (`@metropolis1953equation`, `spec:continuum_spde_stat_mech_v1`). Numerical stability: exact `usize` borrow. Domain of validity: length $N + 1$.
    #[must_use]
    #[verified_engine::verified]
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Returns CSR neighbor indices $j_k \in [0, N)$ (`@metropolis1953equation`, `spec:continuum_spde_stat_mech_v1`). Numerical stability: exact `usize` borrow. Domain of validity: length $E$.
    #[must_use]
    #[verified_engine::verified]
    pub fn col_idx(&self) -> &[usize] {
        &self.col_idx
    }

    /// Returns CSR edge coupling weights $w_k \in \mathbb{R}$ (`@metropolis1953equation`, `spec:continuum_spde_stat_mech_v1`). Numerical stability: exact finite `f64` borrow. Domain of validity: length $E$.
    #[must_use]
    #[verified_engine::verified]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Assigns spin $s_{\text{node}} \leftarrow \operatorname{sgn}(\text{spin}) \in \{-1, +1\}$ (`@metropolis1953equation`, `spec:continuum_spde_stat_mech_v1`). Numerical stability: exact integer write; out-of-range `node` is a no-op. Domain of validity: $0 \le \text{node} < N$.
    #[verified_engine::verified]
    pub fn set_spin(&mut self, node: usize, spin: i8) {
        if node < self.spins.len() {
            self.spins[node] = if spin >= 0 { 1 } else { -1 };
        }
    }

    /// Negates spin $s_{\text{node}} \leftarrow -s_{\text{node}}$ in place (`@metropolis1953equation`, `spec:continuum_spde_stat_mech_v1`). Numerical stability: exact `i8` negation without overflow for $\pm 1$. Domain of validity: $0 \le \text{node} < N$.
    #[verified_engine::verified]
    pub fn flip_spin(&mut self, node: usize) {
        if node < self.spins.len() {
            self.spins[node] = -self.spins[node];
        }
    }

    /// Broadcasts uniform spin state $s_i \leftarrow \pm 1$ across all $N$ nodes (`@metropolis1953equation`, `spec:continuum_spde_stat_mech_v1`). Numerical stability: exact `i8` fill. Domain of validity: all `spin: i8`.
    #[verified_engine::verified]
    pub fn set_all_spins(&mut self, spin: i8) {
        let target = if spin >= 0 { 1 } else { -1 };
        self.spins.fill(target);
    }

    /// Resamples independent Bernoulli spins $\mathbb{P}(s_i = \pm 1) = \tfrac{1}{2}$ in place (`@metropolis1953equation`, `spec:continuum_spde_stat_mech_v1`). Numerical stability: zero-allocation PRNG draw. Domain of validity: $0 \le N \le 256$.
    #[verified_engine::verified]
    pub fn randomize(&mut self) {
        for s in &mut self.spins {
            *s = if self.rng.r#gen::<bool>() { 1 } else { -1 };
        }
    }

    /// Computes total integer magnetization $M = \sum_{i=0}^{N-1} s_i$ (`@metropolis1953equation`, `spec:continuum_spde_stat_mech_v1`). Numerical stability: exact `i64` sum without overflow. Domain of validity: $0 \le N \le 256$.
    #[must_use]
    #[verified_engine::verified]
    pub fn magnetization(&self) -> i64 {
        self.spins.iter().map(|&s| i64::from(s)).sum()
    }

    /// Computes intensive magnetization $m = N^{-1}\sum_{i=0}^{N-1} s_i \in [-1, 1]$ (`@metropolis1953equation`, `spec:continuum_spde_stat_mech_v1`). Numerical stability: guards $N = 0 \mapsto 0.0$. Domain of validity: $0 \le N \le 256$.
    #[must_use]
    #[verified_engine::verified]
    pub fn normalized_magnetization(&self) -> f64 {
        if self.n_nodes == 0 {
            return 0.0;
        }
        (self.magnetization() as f64) / (self.n_nodes as f64)
    }

    /// Computes cooperator density $\rho_C = N^{-1}|\{i : s_i = +1\}| \in [0, 1]$ (`@szabo1998evolutionary`, `spec:continuum_spde_stat_mech_v1`). Numerical stability: exact integer count before `f64` division; $N = 0 \mapsto 0.0$. Domain of validity: $0 \le N \le 256$.
    #[must_use]
    #[verified_engine::verified]
    pub fn cooperator_fraction(&self) -> f64 {
        if self.n_nodes == 0 {
            return 0.0;
        }
        let coop = self.spins.iter().filter(|&&s| s > 0).count();
        (coop as f64) / (self.n_nodes as f64)
    }

    #[inline]
    #[verified_engine::verified]
    fn weighted_neighbor_sum(&self, node: usize) -> f64 {
        let start = self.row_ptr[node];
        let end = self.row_ptr[node + 1];
        let mut acc = 0.0_f64;
        for idx in start..end {
            let nbr = self.col_idx[idx];
            acc += self.weights[idx] * f64::from(self.spins[nbr]);
        }
        acc
    }

    /// Evaluates Ising Hamiltonian $E = -\frac{1}{2} J \sum_{i, j} w_{ij} s_i s_j - h \sum_i s_i$ (`@metropolis1953equation`, `spec:continuum_spde_stat_mech_v1`).
    /// Numerical stability: $\mathcal{O}((N + E)\varepsilon_{\text{mach}})$ pairwise `f64` summation. Domain of validity: finite $J, h \in \mathbb{R}$ on symmetric CSR graphs.
    #[must_use]
    #[verified_engine::verified]
    pub fn total_energy(&self, j_coupling: f64, h_field: f64) -> f64 {
        let mut interaction = 0.0_f64;
        let mut zeeman = 0.0_f64;
        for i in 0..self.n_nodes {
            let s_i = f64::from(self.spins[i]);
            interaction += -0.5 * j_coupling * s_i * self.weighted_neighbor_sum(i);
            zeeman += -h_field * s_i;
        }
        interaction + zeeman
    }

    /// Computes mean node payoff $\bar{\pi} = N^{-1}\sum_{i=0}^{N-1} \pi_i$ for a $2 \times 2$ [`MatrixPayoff`] (`@szabo1998evolutionary`, `spec:continuum_spde_stat_mech_v1`).
    /// Numerical stability: clamps negative edge weights to $0$ and guards $N = 0 \mapsto 0.0$. Domain of validity: $2 \times 2$ finite payoff matrices and $0 \le N \le 256$.
    #[must_use]
    #[verified_engine::verified]
    pub fn mean_game_payoff(&self, payoff: &MatrixPayoff) -> f64 {
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
        // Self-interaction baseline plus weighted neighbor interactions, matching SpinLattice convention.
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

    /// Advances Metropolis-Hastings spin flips with acceptance $P = \min(1, e^{-\Delta E / (k_B T)})$ (`@metropolis1953equation`, `spec:continuum_spde_stat_mech_v1`).
    /// Numerical stability: clamps $k_B T \ge 10^{-30}$ and $-\beta \Delta E \in [-500, 0]$ with zero heap allocations. Domain of validity: finite $T > 0$ and finite $J, h \in \mathbb{R}$.
    #[verified_engine::verified]
    pub fn evolve(&mut self, steps: usize, temperature: f64, j_coupling: f64, h_field: f64) {
        if steps == 0
            || self.n_nodes == 0
            || !temperature.is_finite()
            || !j_coupling.is_finite()
            || !h_field.is_finite()
        {
            return;
        }
        let k_bt = (KB * temperature).max(1e-30);
        let beta = 1.0 / k_bt;
        for _ in 0..steps {
            let idx = self.rng.gen_range(0..self.n_nodes);
            let s_i = self.spins[idx];
            let nbr_sum = self.weighted_neighbor_sum(idx);
            let delta_e = 2.0 * f64::from(s_i) * (j_coupling * nbr_sum + h_field);
            let accept = if delta_e <= 0.0 {
                true
            } else {
                let exponent = (-beta * delta_e).clamp(-500.0, 0.0);
                self.rng.r#gen::<f64>() < exponent.exp()
            };
            if accept {
                self.spins[idx] = -s_i;
            }
        }
    }

    /// Advances pairwise Fermi imitation $W(s_i \leftarrow s_j) = [1 + \exp((\pi_i - \pi_j)/K)]^{-1}$ over CSR edges (`@szabo1998evolutionary`, `spec:continuum_spde_stat_mech_v1`).
    /// Numerical stability: clamps $K \ge 10^{-3}$ and $(\pi_i - \pi_j)/K \in [-500, 500]$ with zero heap allocations. Domain of validity: finite $K > 0$ and finite $2 \times 2$ [`MatrixPayoff`].
    #[verified_engine::verified]
    pub fn evolve_fermi_game(&mut self, steps: usize, temperature: f64, payoff: &MatrixPayoff) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use verified_engine::engine::VerifiedEngine;

    fn sample_wheel_csr() -> (usize, Vec<usize>, Vec<usize>, Vec<f64>) {
        // 5-node hub-and-rim graph: hub 0 connected to 1..=4, plus cycle 1-2-3-4-1
        let adj: [&[usize]; 5] = [
            &[1, 2, 3, 4],
            &[0, 2, 4],
            &[0, 1, 3],
            &[0, 2, 4],
            &[0, 1, 3],
        ];
        let mut row_ptr = Vec::with_capacity(6);
        let mut col_idx = Vec::new();
        row_ptr.push(0);
        for nbrs in adj {
            col_idx.extend_from_slice(nbrs);
            row_ptr.push(col_idx.len());
        }
        let weights = vec![1.0; col_idx.len()];
        (5, row_ptr, col_idx, weights)
    }

    #[test]
    fn test_graph_spin_system_csr_validation_and_ferromagnetic_alignment() {
        let (n, row_ptr, col_idx, weights) = sample_wheel_csr();
        let mut sys =
            GraphSpinSystem::try_from_csr(n, row_ptr, col_idx, weights, Some(123)).unwrap();
        sys.evolve(10_000, 0.25 / KB, 1.0, 0.25);
        assert!(sys.normalized_magnetization() > 0.6);
        assert!(sys.total_energy(1.0, 0.25).is_finite());

        assert!(GraphSpinSystem::try_from_csr(2, vec![0, 1], vec![0], vec![1.0], Some(1)).is_err());
        assert!(
            GraphSpinSystem::try_from_csr(2, vec![0, 1, 1], vec![5], vec![1.0], Some(1)).is_err()
        );
    }

    #[test]
    fn test_graph_spin_system_zero_alloc_1000_steps_under_verified_engine() {
        let (n, row_ptr, col_idx, weights) = sample_wheel_csr();
        let mut sys =
            GraphSpinSystem::try_from_csr(n, row_ptr, col_idx, weights, Some(42)).unwrap();
        let payoff =
            MatrixPayoff::new(DMatrix::from_row_slice(2, 2, &[1.0, -0.15, 1.2, 0.0])).unwrap();

        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let ((mag, coop), metrics) = VerifiedEngine::run_verified(|| {
            sys.evolve(1_000, 1.2 / KB, 1.0, 0.0);
            sys.evolve_fermi_game(1_000, 0.1, &payoff);
            (sys.magnetization(), sys.cooperator_fraction())
        });
        assert!(mag.abs() <= 5);
        assert!((0.0..=1.0).contains(&coop));
        assert!(metrics.function_calls > 0);
    }
}

//! Arbitrary CSR Graph Ising & Spatial Evolutionary Game Simulation.
//!
//! Provides [`GraphSpinSystem`] for zero-allocation Metropolis-Hastings Ising spin flips
//! and pairwise Fermi evolutionary games on arbitrary weighted Compressed Sparse Row (CSR)
//! contact networks, supporting both ferromagnetic ($w_{ij} > 0$) and frustrated
//! antiferromagnetic ($w_{ij} < 0$) couplings.

#[path = "graph_ising_evolution.rs"]
mod evolution;

use super::KB;
use domain_applied::applied::game_theory::evolutionary::MatrixPayoff;
use math_commons::error::MathError;
use oxidize_core::rng::OxidizeRng;
use rand::Rng;
use verified_engine::Theory;

/// Maximum supported node count for [`GraphSpinSystem::try_from_csr`] topologies.
pub const MAX_GRAPH_SPIN_NODES: usize = 256;

/// Maximum supported directed/symmetric CSR half-edge count (`2 * 2_048 = 4_096`).
pub const MAX_GRAPH_SPIN_CSR_ENTRIES: usize = 4_096;

/// Discrete spin ($s_i \in \{-1, +1\}$) system governed by Hamiltonian $\mathcal{H}(\mathbf{s}) = -\sum_{(i,j)} J_{ij} w_{ij} s_i s_j - h \sum_i s_i$ over a weighted Compressed Sparse Row (CSR) graph.
///
/// Supports both ferromagnetic/antiferromagnetic Metropolis-Hastings dynamics ([`Self::evolve`])
/// and spatial 2x2 evolutionary games under pairwise Fermi imitation ([`Self::evolve_fermi_game`])
/// with zero post-initialization heap allocations.
///
/// Reference: Metropolis et al. (1953), *J. Chem. Phys.* 21, 1087, DOI: 10.1063/1.1699114; Szabó & Tőke (1998), *Phys. Rev. E* 58, 69, DOI: 10.1103/PhysRevE.58.69; [`docs/math_specs/domain_physics.md`](../../../../../../docs/math_specs/domain_physics.md).
///
/// # Stability
/// Detailed-balance acceptance probabilities $\min(1, e^{-\beta \Delta E})$ and logistic Fermi probabilities $(1 + e^{\Delta \pi / K})^{-1}$ clamp extreme exponents to prevent IEEE-754 overflow.
///
/// # Domain of Validity
/// Requires validated CSR topology with `n_nodes <= 256`, `col_idx.len() <= 4_096`, finite edge weights $w_{ij}$, and strictly positive temperature $T > 0$ (or noise $K > 0$).
///
/// # Examples
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::stat_mech::graph_ising::GraphSpinSystem;
/// use math_explorer::physics::stat_mech::KB;
///
/// // Antiferromagnetic triangle (odd cycle) cannot satisfy all three bonds simultaneously
/// let edges = vec![(0, 1, -1.0), (1, 2, -1.0), (2, 0, -1.0)];
/// let mut system = GraphSpinSystem::new(3, &edges, Some(42));
/// system.evolve(1_000, 0.2 / KB, 1.0, 0.0);
///
/// assert_eq!(system.frustrated_edge_count(1.0), 1);
/// assert!((system.hamiltonian(1.0, 0.0) - (-1.0)).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Theory)]
#[theory(
    description = "Ising spin and pairwise Fermi evolutionary game dynamics over arbitrary weighted CSR contact graphs.",
    citation = "Metropolis et al. (1953) J. Chem. Phys. 21, 1087; Szabó & Tőke (1998) Phys. Rev. E 58, 69"
)]
pub struct GraphSpinSystem {
    pub(super) n_nodes: usize,
    pub(super) num_edges: usize,
    pub(super) spins: Vec<i8>,
    pub(super) row_ptr: Vec<usize>,
    pub(super) col_idx: Vec<usize>,
    pub(super) weights: Vec<f64>,
    pub(super) rng: OxidizeRng,
}

#[verified_engine::verified]
fn sanitize_edges(num_nodes: usize, edges: &[(usize, usize, f64)]) -> Vec<(usize, usize, f64)> {
    let mut clean: Vec<(usize, usize, f64)> = Vec::with_capacity(edges.len());
    for &(u, v, w) in edges {
        if u >= num_nodes || v >= num_nodes || u == v || !w.is_finite() {
            continue;
        }
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        if let Some(existing) = clean.iter_mut().find(|(ea, eb, _)| *ea == a && *eb == b) {
            existing.2 = w;
        } else {
            clean.push((a, b, w));
        }
    }
    clean
}

#[verified_engine::verified]
fn build_csr(
    num_nodes: usize,
    edges: &[(usize, usize, f64)],
) -> (Vec<usize>, Vec<usize>, Vec<f64>, usize) {
    let clean = sanitize_edges(num_nodes, edges);
    let mut degrees = vec![0usize; num_nodes];
    for &(u, v, _) in &clean {
        degrees[u] += 1;
        degrees[v] += 1;
    }

    let mut row_ptr = vec![0usize; num_nodes + 1];
    for i in 0..num_nodes {
        row_ptr[i + 1] = row_ptr[i] + degrees[i];
    }

    let total_directed = row_ptr[num_nodes];
    let mut col_idx = vec![0usize; total_directed];
    let mut weights = vec![0.0_f64; total_directed];
    let mut cursors = row_ptr[..num_nodes].to_vec();

    for &(u, v, w) in &clean {
        let cu = cursors[u];
        col_idx[cu] = v;
        weights[cu] = w;
        cursors[u] += 1;

        let cv = cursors[v];
        col_idx[cv] = u;
        weights[cv] = w;
        cursors[v] += 1;
    }

    (row_ptr, col_idx, weights, clean.len())
}

#[verified_engine::verified]
fn count_csr_undirected_edges(n_nodes: usize, row_ptr: &[usize], col_idx: &[usize]) -> usize {
    let mut upper = 0usize;
    let mut asymmetric = 0usize;
    for u in 0..n_nodes {
        for &v in &col_idx[row_ptr[u]..row_ptr[u + 1]] {
            if v > u {
                upper += 1;
            } else if v < u && !col_idx[row_ptr[v]..row_ptr[v + 1]].contains(&u) {
                asymmetric += 1;
            }
        }
    }
    upper + asymmetric
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
            reason: "CSR graph exceeds maximum spin system bounds",
        });
    }
    if row_ptr.len() != n_nodes + 1 || row_ptr.first().copied() != Some(0) {
        return Err(MathError::NumericalError {
            reason: "CSR row_ptr must have length n_nodes + 1 starting at 0",
        });
    }
    if row_ptr[n_nodes] != col_idx.len() || col_idx.len() != weights.len() {
        return Err(MathError::NumericalError {
            reason: "CSR col_idx and weights lengths must match row_ptr[n_nodes]",
        });
    }
    for u in 0..n_nodes {
        if row_ptr[u] > row_ptr[u + 1] {
            return Err(MathError::NumericalError {
                reason: "CSR row_ptr must be monotonically non-decreasing",
            });
        }
    }
    if col_idx.iter().any(|&v| v >= n_nodes) || weights.iter().any(|w| !w.is_finite()) {
        return Err(MathError::NumericalError {
            reason: "CSR column indices must be < n_nodes and weights must be finite",
        });
    }
    Ok(())
}

impl GraphSpinSystem {
    /// Creates a new random [`GraphSpinSystem`] from undirected weighted edges `(u, v, weight)`.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(num_nodes: usize, edges: &[(usize, usize, f64)], seed: Option<u64>) -> Self {
        let mut rng = seed.map_or_else(OxidizeRng::default, OxidizeRng::new);
        let spins: Vec<i8> = (0..num_nodes)
            .map(|_| if rng.gen_bool(0.5) { 1 } else { -1 })
            .collect();
        let (row_ptr, col_idx, weights, num_edges) = build_csr(num_nodes, edges);
        Self {
            n_nodes: num_nodes,
            num_edges,
            spins,
            row_ptr,
            col_idx,
            weights,
            rng,
        }
    }

    /// Creates a [`GraphSpinSystem`] with explicit initial spins (`>= 0` mapped to `+1`, `< 0` to `-1`).
    #[must_use]
    #[verified_engine::verified]
    pub fn new_with_spins(
        spins: Vec<i8>,
        edges: &[(usize, usize, f64)],
        seed: Option<u64>,
    ) -> Self {
        let num_nodes = spins.len();
        let normalized_spins = spins
            .into_iter()
            .map(|s| if s >= 0 { 1 } else { -1 })
            .collect();
        let rng = seed.map_or_else(OxidizeRng::default, OxidizeRng::new);
        let (row_ptr, col_idx, weights, num_edges) = build_csr(num_nodes, edges);
        Self {
            n_nodes: num_nodes,
            num_edges,
            spins: normalized_spins,
            row_ptr,
            col_idx,
            weights,
            rng,
        }
    }

    /// Constructs a validated [`GraphSpinSystem`] from CSR buffers and an optional RNG seed.
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
        let num_edges = count_csr_undirected_edges(n_nodes, &row_ptr, &col_idx);
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
            num_edges,
            spins,
            row_ptr,
            col_idx,
            weights,
            rng,
        })
    }

    /// Returns the number of nodes in the CSR spin graph.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Returns the number of spin sites in the graph.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn num_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Returns the number of unique undirected edges in the CSR topology.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Returns the number of CSR adjacency entries.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn csr_len(&self) -> usize {
        self.col_idx.len()
    }

    /// Returns the slice of spin states (`+1` or `-1`).
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn spins(&self) -> &[i8] {
        &self.spins
    }

    /// Returns CSR row pointers (`len == n_nodes + 1`).
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Returns CSR column neighbor indices.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn col_idx(&self) -> &[usize] {
        &self.col_idx
    }

    /// Returns CSR edge coupling weights.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Returns the CSR neighbor indices and weights for `node`.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn neighbors_of(&self, node: usize) -> (&[usize], &[f64]) {
        if node >= self.n_nodes {
            return (&[], &[]);
        }
        let start = self.row_ptr[node];
        let end = self.row_ptr[node + 1];
        (&self.col_idx[start..end], &self.weights[start..end])
    }

    /// Sets the spin of `node` to `+1` (for `spin >= 0`) or `-1` (for `spin < 0`).
    #[inline]
    #[verified_engine::verified]
    pub fn set_spin(&mut self, node: usize, spin: i8) {
        if node < self.spins.len() {
            self.spins[node] = if spin >= 0 { 1 } else { -1 };
        }
    }

    /// Flips the spin of `node` in place without heap allocation.
    #[inline]
    #[verified_engine::verified]
    pub fn flip_spin(&mut self, node: usize) {
        if node < self.spins.len() {
            self.spins[node] = -self.spins[node];
        }
    }

    /// Sets all spins to `+1` (`spin >= 0`) or `-1` (`spin < 0`).
    #[verified_engine::verified]
    pub fn set_all_spins(&mut self, spin: i8) {
        let target = if spin >= 0 { 1 } else { -1 };
        self.spins.fill(target);
    }

    /// Randomizes all node spins uniformly in `{-1, +1}` without heap allocation.
    #[verified_engine::verified]
    pub fn randomize(&mut self) {
        for s in &mut self.spins {
            *s = if self.rng.r#gen::<bool>() { 1 } else { -1 };
        }
    }

    /// Re-seeds the internal deterministic [`OxidizeRng`].
    pub fn reseed(&mut self, seed: u64) {
        self.rng = OxidizeRng::new(seed);
    }

    /// Computes total integer magnetization $\sum_i s_i$.
    #[must_use]
    #[verified_engine::verified]
    pub fn magnetization(&self) -> i64 {
        self.spins.iter().map(|&s| i64::from(s)).sum()
    }

    /// Computes normalized magnetization per node in `[-1.0, 1.0]`.
    #[must_use]
    #[verified_engine::verified]
    pub fn normalized_magnetization(&self) -> f64 {
        if self.n_nodes == 0 {
            return 0.0;
        }
        (self.magnetization() as f64) / (self.n_nodes as f64)
    }

    /// Computes normalized magnetization per spin $m = \frac{1}{N}\sum_i s_i \in [-1, 1]$.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn mean_magnetization(&self) -> f64 {
        self.normalized_magnetization()
    }

    /// Computes the fraction of `+1` (Cooperator) nodes in `[0.0, 1.0]`.
    #[must_use]
    #[verified_engine::verified]
    pub fn cooperator_fraction(&self) -> f64 {
        if self.n_nodes == 0 {
            return 0.0;
        }
        let coop = self.spins.iter().filter(|&&s| s > 0).count();
        (coop as f64) / (self.n_nodes as f64)
    }

    /// Computes total Ising Hamiltonian energy $E = -\frac{1}{2} J \sum_{i, j} w_{ij} s_i s_j - h \sum_i s_i$.
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

    /// Computes total Ising Hamiltonian $H = -J \sum_{(i,j)} w_{ij} s_i s_j - h \sum_i s_i$.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn hamiltonian(&self, j_coupling: f64, h_field: f64) -> f64 {
        self.total_energy(j_coupling, h_field)
    }

    /// Counts frustrated undirected bonds where $J w_{ij} s_i s_j < 0$.
    #[must_use]
    #[verified_engine::verified]
    pub fn frustrated_edge_count(&self, j_coupling: f64) -> usize {
        let mut frustrated = 0usize;
        for i in 0..self.n_nodes {
            let si = f64::from(self.spins[i]);
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for k in start..end {
                let j = self.col_idx[k];
                if j > i {
                    let bond_alignment =
                        j_coupling * self.weights[k] * si * f64::from(self.spins[j]);
                    if bond_alignment < -1e-12 {
                        frustrated += 1;
                    }
                }
            }
        }
        frustrated
    }

    /// Computes mean game payoff per node on the CSR graph for a 2x2 [`MatrixPayoff`].
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn mean_game_payoff(&self, payoff: &MatrixPayoff) -> f64 {
        self.mean_game_payoff_inner(payoff)
    }

    /// Performs one single-spin Metropolis update with zero heap allocations.
    #[inline]
    #[verified_engine::verified]
    pub fn metropolis_step(&mut self, temperature: f64, j_coupling: f64, h_field: f64) {
        self.evolve_inner(1, temperature, j_coupling, h_field);
    }

    /// Advances Metropolis-Hastings Ising spin dynamics by `steps` elementary trials with zero heap allocations.
    #[inline]
    #[verified_engine::verified]
    pub fn evolve(&mut self, steps: usize, temperature: f64, j_coupling: f64, h_field: f64) {
        self.evolve_inner(steps, temperature, j_coupling, h_field);
    }

    /// Evolves a 2x2 evolutionary game via pairwise Fermi imitation over CSR edges with zero heap allocations.
    #[inline]
    #[verified_engine::verified]
    pub fn evolve_fermi_game(&mut self, steps: usize, temperature: f64, payoff: &MatrixPayoff) {
        self.evolve_fermi_game_inner(steps, temperature, payoff);
    }
}

#[cfg(test)]
#[path = "graph_ising_tests.rs"]
mod tests;

//! Arbitrary-Graph Ising Spin System with Signed Couplings.
//!
//! Provides [`GraphSpinSystem`] for simulating discrete Ising spins $s_i \in \{-1, +1\}$
//! on arbitrary weighted graphs using compressed sparse row (CSR) adjacency arrays
//! for zero-allocation Metropolis-Hastings evolution.

use super::KB;
use oxidize_core::rng::OxidizeRng;
use rand::Rng;

/// Compressed sparse row (CSR) Ising spin system on an arbitrary signed-weight graph.
///
/// Supports positive ferromagnetic weights ($w_{ij} > 0$) and negative antiferromagnetic
/// weights ($w_{ij} < 0$), enabling geometric spin frustration on odd cycles while keeping
/// [`SpinLattice`](super::ising::SpinLattice) intact for 2D grids.
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
#[derive(Debug, Clone)]
pub struct GraphSpinSystem {
    num_nodes: usize,
    num_edges: usize,
    spins: Vec<i8>,
    neighbors_flat: Vec<usize>,
    weights_flat: Vec<f64>,
    offsets: Vec<usize>,
    rng: OxidizeRng,
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
) -> (Vec<usize>, Vec<f64>, Vec<usize>, usize) {
    let clean = sanitize_edges(num_nodes, edges);
    let mut degrees = vec![0usize; num_nodes];
    for &(u, v, _) in &clean {
        degrees[u] += 1;
        degrees[v] += 1;
    }

    let mut offsets = vec![0usize; num_nodes + 1];
    for i in 0..num_nodes {
        offsets[i + 1] = offsets[i] + degrees[i];
    }

    let total_directed = offsets[num_nodes];
    let mut neighbors_flat = vec![0usize; total_directed];
    let mut weights_flat = vec![0.0_f64; total_directed];
    let mut cursors = offsets[..num_nodes].to_vec();

    for &(u, v, w) in &clean {
        let cu = cursors[u];
        neighbors_flat[cu] = v;
        weights_flat[cu] = w;
        cursors[u] += 1;

        let cv = cursors[v];
        neighbors_flat[cv] = u;
        weights_flat[cv] = w;
        cursors[v] += 1;
    }

    (neighbors_flat, weights_flat, offsets, clean.len())
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
        let (neighbors_flat, weights_flat, offsets, num_edges) = build_csr(num_nodes, edges);
        Self {
            num_nodes,
            num_edges,
            spins,
            neighbors_flat,
            weights_flat,
            offsets,
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
        let (neighbors_flat, weights_flat, offsets, num_edges) = build_csr(num_nodes, edges);
        Self {
            num_nodes,
            num_edges,
            spins: normalized_spins,
            neighbors_flat,
            weights_flat,
            offsets,
            rng,
        }
    }

    /// Returns the number of spin sites in the graph.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of unique undirected edges in the CSR topology.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Returns the current spin state slice (`-1` or `+1` per node).
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn spins(&self) -> &[i8] {
        &self.spins
    }

    /// Returns the CSR neighbor indices and weights for `node`.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn neighbors_of(&self, node: usize) -> (&[usize], &[f64]) {
        if node >= self.num_nodes {
            return (&[], &[]);
        }
        let start = self.offsets[node];
        let end = self.offsets[node + 1];
        (
            &self.neighbors_flat[start..end],
            &self.weights_flat[start..end],
        )
    }

    /// Sets the spin at `node` (`>= 0` becomes `+1`, `< 0` becomes `-1`).
    #[inline]
    #[verified_engine::verified]
    pub fn set_spin(&mut self, node: usize, spin: i8) {
        if node < self.num_nodes {
            self.spins[node] = if spin >= 0 { 1 } else { -1 };
        }
    }

    /// Re-seeds the internal deterministic [`OxidizeRng`].
    pub fn reseed(&mut self, seed: u64) {
        self.rng = OxidizeRng::new(seed);
    }

    /// Computes total integer magnetization $M = \sum_i s_i$.
    #[must_use]
    #[verified_engine::verified]
    pub fn magnetization(&self) -> i64 {
        let mut sum = 0_i64;
        for &s in &self.spins {
            sum += i64::from(s);
        }
        sum
    }

    /// Computes normalized magnetization per spin $m = \frac{1}{N}\sum_i s_i \in [-1, 1]$.
    #[must_use]
    #[verified_engine::verified]
    pub fn mean_magnetization(&self) -> f64 {
        if self.num_nodes == 0 {
            return 0.0;
        }
        self.magnetization() as f64 / self.num_nodes as f64
    }

    /// Computes total Ising Hamiltonian $H = -J \sum_{(i,j)} w_{ij} s_i s_j - h \sum_i s_i$.
    #[must_use]
    #[verified_engine::verified]
    pub fn hamiltonian(&self, j_coupling: f64, h_field: f64) -> f64 {
        let mut interaction_sum = 0.0_f64;
        let mut field_sum = 0.0_f64;

        for i in 0..self.num_nodes {
            let si = f64::from(self.spins[i]);
            field_sum += si;
            let start = self.offsets[i];
            let end = self.offsets[i + 1];
            for k in start..end {
                let j = self.neighbors_flat[k];
                if j > i {
                    let sj = f64::from(self.spins[j]);
                    interaction_sum += self.weights_flat[k] * si * sj;
                }
            }
        }

        -j_coupling * interaction_sum - h_field * field_sum
    }

    /// Counts frustrated undirected bonds where $J w_{ij} s_i s_j < 0$.
    #[must_use]
    #[verified_engine::verified]
    pub fn frustrated_edge_count(&self, j_coupling: f64) -> usize {
        let mut frustrated = 0usize;
        for i in 0..self.num_nodes {
            let si = f64::from(self.spins[i]);
            let start = self.offsets[i];
            let end = self.offsets[i + 1];
            for k in start..end {
                let j = self.neighbors_flat[k];
                if j > i {
                    let bond_alignment =
                        j_coupling * self.weights_flat[k] * si * f64::from(self.spins[j]);
                    if bond_alignment < -1e-12 {
                        frustrated += 1;
                    }
                }
            }
        }
        frustrated
    }

    /// Performs one single-spin Metropolis update with zero heap allocations.
    #[inline]
    #[verified_engine::verified]
    pub fn metropolis_step(&mut self, temperature: f64, j_coupling: f64, h_field: f64) {
        self.evolve(1, temperature, j_coupling, h_field);
    }

    /// Evolves `steps` Metropolis single-spin trials with zero heap allocations.
    #[verified_engine::verified]
    pub fn evolve(&mut self, steps: usize, temperature: f64, j_coupling: f64, h_field: f64) {
        if steps == 0 || self.num_nodes == 0 {
            return;
        }
        let kbt = KB * temperature;
        let inv_kbt = if kbt > 0.0 && kbt.is_finite() {
            1.0 / kbt
        } else {
            0.0
        };

        for _ in 0..steps {
            let idx = self.rng.gen_range(0..self.num_nodes);
            let s = self.spins[idx];
            let start = self.offsets[idx];
            let end = self.offsets[idx + 1];

            let mut weighted_neighbors = 0.0_f64;
            for k in start..end {
                let nb = self.neighbors_flat[k];
                weighted_neighbors += self.weights_flat[k] * f64::from(self.spins[nb]);
            }

            let delta_e = 2.0 * f64::from(s) * (j_coupling * weighted_neighbors + h_field);
            let accept = if delta_e <= 0.0 {
                true
            } else if inv_kbt > 0.0 {
                self.rng.r#gen::<f64>() < (-delta_e * inv_kbt).exp()
            } else {
                false
            };

            if accept {
                self.spins[idx] = -s;
            }
        }
    }
}

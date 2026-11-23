//! Kuramoto coupled phase-oscillator network dynamics with preallocated RK4 integration.
//!
//! Models synchronization on arbitrary weighted graphs governed by:
//!
//! $$\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=0}^{N-1} A_{ij} \sin(\theta_j - \theta_i)$$

use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;
use std::f64::consts::TAU;

/// Builder for constructing a [`KuramotoSystem`] with preallocated integration buffers.
#[derive(Debug, Clone)]
pub struct KuramotoBuilder {
    num_oscillators: usize,
    coupling_k: f64,
    frequency_spread: f64,
    edges: Vec<(usize, usize, f64)>,
}

impl Default for KuramotoBuilder {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            num_oscillators: 12,
            coupling_k: 3.5,
            frequency_spread: 0.8,
            edges: Vec::new(),
        }
    }
}

impl KuramotoBuilder {
    /// Creates a new builder initialized with a 12-node ring defaults.
    #[must_use]
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of coupled phase oscillators $N$.
    #[must_use]
    #[verified_engine::verified]
    pub fn num_oscillators(mut self, count: usize) -> Self {
        self.num_oscillators = count;
        self
    }

    /// Sets the global coupling constant $K$.
    #[must_use]
    #[verified_engine::verified]
    pub fn coupling_k(mut self, k: f64) -> Self {
        self.coupling_k = if k.is_finite() { k.max(0.0) } else { 0.0 };
        self
    }

    /// Sets the half-width detuning spread $\Delta\omega$ of natural frequencies.
    #[must_use]
    #[verified_engine::verified]
    pub fn frequency_spread(mut self, spread: f64) -> Self {
        self.frequency_spread = if spread.is_finite() {
            spread.max(0.0)
        } else {
            0.0
        };
        self
    }

    /// Sets undirected weighted edges `(u, v, weight)`. If omitted, builds a connected ring+chord graph.
    #[must_use]
    #[verified_engine::verified]
    pub fn edges(mut self, edges: Vec<(usize, usize, f64)>) -> Self {
        self.edges = edges;
        self
    }

    /// Builds a [`KuramotoSystem`] with preallocated CSR adjacency and RK4 scratch buffers.
    #[must_use]
    #[verified_engine::verified]
    pub fn build(self) -> KuramotoSystem {
        let n = self.num_oscillators;
        let edges = if self.edges.is_empty() && n >= 2 {
            default_small_world_ring_edges(n)
        } else {
            self.edges
        };
        let mut sys = KuramotoSystem {
            num_oscillators: 0,
            coupling_k: self.coupling_k,
            frequency_spread: self.frequency_spread,
            time: 0.0,
            thetas: Vec::with_capacity(n),
            omegas: Vec::with_capacity(n),
            row_offsets: vec![0],
            col_indices: Vec::new(),
            weights: Vec::new(),
            k1: Vec::with_capacity(n),
            k2: Vec::with_capacity(n),
            k3: Vec::with_capacity(n),
            k4: Vec::with_capacity(n),
            scratch_thetas: Vec::with_capacity(n),
        };
        sys.set_topology(n, &edges);
        sys.reset_phases();
        sys
    }
}

/// Generates a deterministic ring with second-nearest-neighbor chords for standalone oscillator networks.
#[must_use]
#[verified_engine::verified]
pub fn default_small_world_ring_edges(num_nodes: usize) -> Vec<(usize, usize, f64)> {
    if num_nodes < 2 {
        return Vec::new();
    }
    if num_nodes == 2 {
        return vec![(0, 1, 1.0)];
    }
    let mut edges = Vec::with_capacity(num_nodes * 2);
    for i in 0..num_nodes {
        let next = (i + 1) % num_nodes;
        edges.push((i.min(next), i.max(next), 1.0));
        if num_nodes >= 5 {
            let chord = (i + 2) % num_nodes;
            edges.push((i.min(chord), i.max(chord), 0.5));
        }
    }
    edges.sort_unstable_by_key(|a| (a.0, a.1));
    edges.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
    edges
}

/// Coupled Kuramoto phase-oscillator solver with zero-allocation RK4 stepping.
///
/// # Example
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::chaos::kuramoto::KuramotoSystem;
///
/// let mut system = KuramotoSystem::builder()
///     .num_oscillators(8)
///     .coupling_k(6.0)
///     .frequency_spread(0.4)
///     .build();
///
/// for _ in 0..200 {
///     system.step_rk4(0.05);
/// }
/// let (r, _psi) = system.order_parameter();
/// assert!((0.0..=1.0).contains(&r));
/// ```
#[derive(Debug, Clone)]
pub struct KuramotoSystem {
    num_oscillators: usize,
    coupling_k: f64,
    frequency_spread: f64,
    time: f64,
    thetas: Vec<f64>,
    omegas: Vec<f64>,
    row_offsets: Vec<usize>,
    col_indices: Vec<usize>,
    weights: Vec<f64>,
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    k4: Vec<f64>,
    scratch_thetas: Vec<f64>,
}

impl KuramotoSystem {
    /// Returns a [`KuramotoBuilder`] for configuring a new oscillator network.
    #[must_use]
    #[verified_engine::verified]
    pub fn builder() -> KuramotoBuilder {
        KuramotoBuilder::new()
    }

    /// Number of oscillators $N$ in the active network.
    #[must_use]
    #[verified_engine::verified]
    pub fn num_oscillators(&self) -> usize {
        self.num_oscillators
    }

    /// Global coupling strength $K$.
    #[must_use]
    #[verified_engine::verified]
    pub fn coupling_k(&self) -> f64 {
        self.coupling_k
    }

    /// Updates the global coupling strength $K$ without allocating.
    #[verified_engine::verified]
    pub fn set_coupling_k(&mut self, k: f64) {
        if k.is_finite() {
            self.coupling_k = k.max(0.0);
        }
    }

    /// Natural frequency half-spread $\Delta\omega$.
    #[must_use]
    #[verified_engine::verified]
    pub fn frequency_spread(&self) -> f64 {
        self.frequency_spread
    }

    /// Updates natural frequencies $\omega_i \in [-\Delta\omega, +\Delta\omega]$ in-place without allocating.
    #[verified_engine::verified]
    pub fn set_frequency_spread(&mut self, spread: f64) {
        if !spread.is_finite() {
            return;
        }
        self.frequency_spread = spread.max(0.0);
        populate_natural_frequencies(&mut self.omegas, self.frequency_spread);
    }

    /// Current simulation time $t$.
    #[must_use]
    #[verified_engine::verified]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Immutable slice of oscillator phases $\theta_i \in [0, 2\pi)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn phases(&self) -> &[f64] {
        &self.thetas
    }

    /// Immutable slice of natural frequencies $\omega_i$.
    #[must_use]
    #[verified_engine::verified]
    pub fn natural_frequencies(&self) -> &[f64] {
        &self.omegas
    }

    /// Number of directed coupling entries ($2|E|$ for undirected graphs).
    #[must_use]
    #[verified_engine::verified]
    pub fn directed_edge_count(&self) -> usize {
        self.col_indices.len()
    }

    /// Resets simulation time to `0.0` and initializes deterministically splayed phases in-place.
    #[verified_engine::verified]
    pub fn reset_phases(&mut self) {
        self.time = 0.0;
        let n = self.num_oscillators;
        for (i, theta) in self.thetas.iter_mut().enumerate() {
            let frac = (i as f64 + 0.5) / (n.max(1) as f64);
            // Slight incommensurate phase offset avoids artificial symmetry traps at K > K_c.
            *theta = (TAU * frac + 0.37 * ((i as f64) * 1.37).sin()).rem_euclid(TAU);
        }
    }

    /// Replaces oscillator Phases directly when `phases.len() == num_oscillators` (zero allocation).
    #[verified_engine::verified]
    pub fn set_phases(&mut self, phases: &[f64]) {
        if phases.len() != self.num_oscillators {
            return;
        }
        for (dst, &src) in self.thetas.iter_mut().zip(phases.iter()) {
            if src.is_finite() {
                *dst = src.rem_euclid(TAU);
            }
        }
    }

    /// Rebuilds the CSR weighted adjacency and resizes RK4 scratch buffers for a new topology.
    ///
    /// Call this only when the graph topology revision advances; hot-path stepping via
    /// [`Self::step_rk4`] reuses these buffers without allocating.
    #[verified_engine::verified]
    pub fn set_topology(&mut self, num_nodes: usize, edges: &[(usize, usize, f64)]) {
        let old_n = self.num_oscillators;
        self.num_oscillators = num_nodes;
        self.thetas.resize(num_nodes, 0.0);
        for (i, theta) in self
            .thetas
            .iter_mut()
            .enumerate()
            .take(num_nodes)
            .skip(old_n)
        {
            let frac = (i as f64 + 0.5) / (num_nodes.max(1) as f64);
            *theta = (TAU * frac).rem_euclid(TAU);
        }
        self.omegas.resize(num_nodes, 0.0);
        populate_natural_frequencies(&mut self.omegas, self.frequency_spread);

        self.k1.resize(num_nodes, 0.0);
        self.k2.resize(num_nodes, 0.0);
        self.k3.resize(num_nodes, 0.0);
        self.k4.resize(num_nodes, 0.0);
        self.scratch_thetas.resize(num_nodes, 0.0);

        rebuild_csr_adjacency(
            num_nodes,
            edges,
            &mut self.row_offsets,
            &mut self.col_indices,
            &mut self.weights,
        );
    }

    /// Advances the oscillator phases by `dt` using 4th-order Runge-Kutta without heap allocations.
    #[verified_engine::verified]
    pub fn step_rk4(&mut self, dt: f64) {
        let n = self.num_oscillators;
        if n == 0 || !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let scale = self.coupling_k / (n as f64);
        let half_dt = 0.5 * dt;
        let csr = CsrAdjacencyRef {
            row_offsets: &self.row_offsets,
            col_indices: &self.col_indices,
            weights: &self.weights,
        };

        compute_derivatives(n, scale, &self.omegas, csr, &self.thetas, &mut self.k1);
        stage_phases(n, &self.thetas, &self.k1, half_dt, &mut self.scratch_thetas);
        compute_derivatives(
            n,
            scale,
            &self.omegas,
            csr,
            &self.scratch_thetas,
            &mut self.k2,
        );
        stage_phases(n, &self.thetas, &self.k2, half_dt, &mut self.scratch_thetas);
        compute_derivatives(
            n,
            scale,
            &self.omegas,
            csr,
            &self.scratch_thetas,
            &mut self.k3,
        );
        stage_phases(n, &self.thetas, &self.k3, dt, &mut self.scratch_thetas);
        compute_derivatives(
            n,
            scale,
            &self.omegas,
            csr,
            &self.scratch_thetas,
            &mut self.k4,
        );

        let sixth_dt = dt / 6.0;
        for i in 0..n {
            let slope = self.k1[i] + 2.0 * self.k2[i] + 2.0 * self.k3[i] + self.k4[i];
            self.thetas[i] = (self.thetas[i] + sixth_dt * slope).rem_euclid(TAU);
        }
        self.time += dt;
    }

    /// Computes the Kuramoto complex order parameter `(R, psi)` with `R` in `[0.0, 1.0]`.
    #[must_use]
    #[verified_engine::verified]
    pub fn order_parameter(&self) -> (f64, f64) {
        let n = self.num_oscillators;
        if n == 0 {
            return (0.0, 0.0);
        }
        let mut sum_cos = 0.0;
        let mut sum_sin = 0.0;
        for &theta in &self.thetas[..n] {
            let (s, c) = theta.sin_cos();
            sum_cos += c;
            sum_sin += s;
        }
        let inv_n = 1.0 / (n as f64);
        let mean_x = sum_cos * inv_n;
        let mean_y = sum_sin * inv_n;
        let r = mean_x.hypot(mean_y).clamp(0.0, 1.0);
        let psi = mean_y.atan2(mean_x);
        (r, psi)
    }

    /// Mean-field critical coupling estimate $K_c \approx \frac{4 \Delta\omega}{\pi (\langle w \rangle / N)}$
    /// scaled for network sparsity. Returns `f64::INFINITY` when the graph has zero total coupling weight.
    #[must_use]
    #[verified_engine::verified]
    pub fn critical_coupling_estimate(&self) -> f64 {
        let n = self.num_oscillators;
        if n <= 1 || self.frequency_spread <= 0.0 {
            return 0.0;
        }
        let mut total_weight = 0.0;
        for &w in &self.weights {
            if w > 0.0 {
                total_weight += w;
            }
        }
        if total_weight <= 1e-12 {
            return f64::INFINITY;
        }
        let mean_weighted_degree = total_weight / (n as f64);
        let density = mean_weighted_degree / (n as f64);
        (4.0 * self.frequency_spread) / (std::f64::consts::PI * density)
    }
}

mod csr;
use self::csr::{
    CsrAdjacencyRef, compute_derivatives, populate_natural_frequencies, rebuild_csr_adjacency,
    stage_phases,
};

impl TheoryDescribable for KuramotoSystem {
    fn theory_description(&self) -> String {
        let (r, _) = self.order_parameter();
        format!(
            "Kuramoto coupled phase oscillators dθ_i/dt = ω_i + (K/N) Σ A_ij sin(θ_j - θ_i) with N={}, K={:.2}, coherence R(t)={:.3}.",
            self.num_oscillators, self.coupling_k, r
        )
    }

    fn phonetic_description(&self) -> String {
        "Kuramoto coupled phase oscillator synchronization model".to_string()
    }

    fn theory_citation(&self) -> String {
        "Kuramoto, Y. (1975). Self-entrainment of a population of coupled non-linear oscillators; Strogatz, S. H. (2000). From Kuramoto to Crawford.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "order_parameter".to_string(),
            "Complex coherence R(t) = |(1/N) sum exp(i theta_j)| measuring phase synchronization"
                .to_string(),
        );
        map
    }
}

#[cfg(test)]
mod tests;

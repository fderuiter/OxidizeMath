//! Spatial Evolutionary Game Dynamics on Arbitrary Weighted Graphs.
//!
//! Provides [`GraphReplicatorSystem`] for evolving per-node strategy simplexes on
//! arbitrary graphs with compressed sparse row (CSR) neighbor arrays and preallocated
//! scratch slices so numerical time-stepping performs zero heap allocations.

use crate::error::GameTheoryError;
use nalgebra::DMatrix;

const LOCAL_SELECTION_GAIN: f64 = 0.05;

/// Spatial evolutionary game system over a weighted graph using CSR adjacency buffers.
///
/// Each node $i \in \{0, \dots, N-1\}$ carries a strategy distribution $\mathbf{x}_i \in \Delta^{S-1}$
/// and accumulates payoffs across its CSR neighborhood while imitating higher-earning
/// positively coupled neighbors without heap allocations during [`Self::step`] or [`Self::evolve`].
///
/// # Examples
///
/// ```
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::game_theory::evolutionary::GraphReplicatorSystem;
/// use math_explorer::nalgebra::DMatrix;
///
/// // Prisoner's Dilemma payoff matrix: R=3, S=0, T=4, P=1
/// let payoff = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 4.0, 1.0]);
/// let edges = vec![(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (0, 4, 1.0)];
/// let mut sys = GraphReplicatorSystem::from_population_mean(5, &payoff, &edges, &[0.6, 0.4])
///     .expect("valid spatial game");
///
/// sys.evolve(100, 0.05);
/// let mut mean = [0.0; 2];
/// sys.mean_frequencies(&mut mean);
/// assert!((mean[0] + mean[1] - 1.0).abs() < 1e-9);
/// ```
#[derive(Debug, Clone)]
pub struct GraphReplicatorSystem {
    num_nodes: usize,
    num_strategies: usize,
    payoff_flat: Vec<f64>,
    neighbors_flat: Vec<usize>,
    weights_flat: Vec<f64>,
    offsets: Vec<usize>,
    states_flat: Vec<f64>,
    fitness_scratch: Vec<f64>,
    node_payoff_scratch: Vec<f64>,
    next_states_scratch: Vec<f64>,
}

#[verified_engine::verified]
fn build_csr(
    num_nodes: usize,
    edges: &[(usize, usize, f64)],
) -> (Vec<usize>, Vec<f64>, Vec<usize>) {
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

    let mut degrees = vec![0usize; num_nodes];
    for &(u, v, _) in &clean {
        degrees[u] += 1;
        degrees[v] += 1;
    }

    let mut offsets = vec![0usize; num_nodes + 1];
    for i in 0..num_nodes {
        offsets[i + 1] = offsets[i] + degrees[i];
    }

    let total = offsets[num_nodes];
    let mut neighbors_flat = vec![0usize; total];
    let mut weights_flat = vec![0.0_f64; total];
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

    (neighbors_flat, weights_flat, offsets)
}

#[verified_engine::verified]
fn project_simplex_row(row: &mut [f64]) {
    let len = row.len();
    if len == 0 {
        return;
    }
    let mut sum = 0.0_f64;
    for val in row.iter_mut() {
        if !val.is_finite() || *val < 0.0 {
            *val = 0.0;
        }
        sum += *val;
    }
    if sum > 1e-15 {
        let inv = 1.0 / sum;
        for val in row.iter_mut() {
            *val *= inv;
        }
    } else {
        let uniform = 1.0 / len as f64;
        row.fill(uniform);
    }
}

#[verified_engine::verified]
fn validate_inputs(
    num_nodes: usize,
    payoff_matrix: &DMatrix<f64>,
    initial_states: &[f64],
) -> Result<usize, GameTheoryError> {
    let rows = payoff_matrix.nrows();
    let cols = payoff_matrix.ncols();
    if rows == 0 || rows != cols {
        return Err(GameTheoryError::NonSquarePayoffMatrix { rows, cols });
    }
    if num_nodes == 0 {
        return Err(GameTheoryError::InvalidParameter {
            name: "num_nodes".to_string(),
            value: 0.0,
        });
    }
    if initial_states.len() != num_nodes * rows {
        return Err(GameTheoryError::InvalidParameter {
            name: "initial_states_len".to_string(),
            value: initial_states.len() as f64,
        });
    }
    Ok(rows)
}

impl GraphReplicatorSystem {
    /// Creates a [`GraphReplicatorSystem`] from an explicit flattened `num_nodes * S` state buffer.
    #[verified_engine::verified]
    pub fn new(
        num_nodes: usize,
        payoff_matrix: &DMatrix<f64>,
        edges: &[(usize, usize, f64)],
        initial_states: &[f64],
    ) -> Result<Self, GameTheoryError> {
        let num_strategies = validate_inputs(num_nodes, payoff_matrix, initial_states)?;
        let mut payoff_flat = vec![0.0_f64; num_strategies * num_strategies];
        for (r, row_out) in payoff_flat.chunks_exact_mut(num_strategies).enumerate() {
            for (c, cell) in row_out.iter_mut().enumerate() {
                *cell = payoff_matrix[(r, c)];
            }
        }

        let (neighbors_flat, weights_flat, offsets) = build_csr(num_nodes, edges);
        let total_state_len = num_nodes * num_strategies;
        let mut states_flat = initial_states.to_vec();
        for i in 0..num_nodes {
            let slice = &mut states_flat[i * num_strategies..(i + 1) * num_strategies];
            project_simplex_row(slice);
        }

        Ok(Self {
            num_nodes,
            num_strategies,
            payoff_flat,
            neighbors_flat,
            weights_flat,
            offsets,
            states_flat,
            fitness_scratch: vec![0.0_f64; total_state_len],
            node_payoff_scratch: vec![0.0_f64; num_nodes],
            next_states_scratch: vec![0.0_f64; total_state_len],
        })
    }

    /// Seeds node states around `mean_state` weighted by positive degree so hub clusters can nucleate.
    #[verified_engine::verified]
    pub fn from_population_mean(
        num_nodes: usize,
        payoff_matrix: &DMatrix<f64>,
        edges: &[(usize, usize, f64)],
        mean_state: &[f64],
    ) -> Result<Self, GameTheoryError> {
        let s = payoff_matrix.nrows();
        if s == 0 || s != payoff_matrix.ncols() {
            return Err(GameTheoryError::NonSquarePayoffMatrix {
                rows: s,
                cols: payoff_matrix.ncols(),
            });
        }
        if mean_state.len() != s {
            return Err(GameTheoryError::InvalidParameter {
                name: "mean_state_len".to_string(),
                value: mean_state.len() as f64,
            });
        }

        let (_, weights_flat, offsets) = build_csr(num_nodes, edges);
        let mut pos_deg = vec![0.0_f64; num_nodes];
        let mut mean_deg = 0.0_f64;
        for i in 0..num_nodes {
            let mut d = 0.0_f64;
            for &w in &weights_flat[offsets[i]..offsets[i + 1]] {
                if w > 0.0 {
                    d += w;
                }
            }
            pos_deg[i] = d;
            mean_deg += d;
        }
        mean_deg /= num_nodes.max(1) as f64;

        let mut base = mean_state.to_vec();
        project_simplex_row(&mut base);
        let mut initial = vec![0.0_f64; num_nodes * s];
        let interior = base.iter().all(|&x| (1e-4..=1.0 - 1e-4).contains(&x));

        for i in 0..num_nodes {
            let row = &mut initial[i * s..(i + 1) * s];
            row.copy_from_slice(&base);
            if interior && s >= 2 && mean_deg > 1e-9 {
                let rel = ((pos_deg[i] - mean_deg) / mean_deg.max(1.0)).clamp(-0.85, 0.85);
                let shift = 0.35 * rel * base[0].min(1.0 - base[0]);
                row[0] = (row[0] + shift).clamp(0.01, 0.99);
                row[1] = (row[1] - shift).clamp(0.01, 0.99);
                project_simplex_row(row);
            }
        }
        Self::new(num_nodes, payoff_matrix, edges, &initial)
    }

    /// Returns the number of graph nodes.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of game strategies per node.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn num_strategies(&self) -> usize {
        self.num_strategies
    }

    /// Returns the strategy simplex slice of `node`.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn node_state(&self, node: usize) -> &[f64] {
        if node >= self.num_nodes {
            return &[];
        }
        let s = self.num_strategies;
        &self.states_flat[node * s..(node + 1) * s]
    }

    /// Writes the population-mean strategy frequencies into `out` with zero heap allocations.
    #[verified_engine::verified]
    pub fn mean_frequencies(&self, out: &mut [f64]) {
        let s = self.num_strategies.min(out.len());
        out[..s].fill(0.0);
        if self.num_nodes == 0 || s == 0 {
            return;
        }
        for i in 0..self.num_nodes {
            let base = i * self.num_strategies;
            for strat in 0..s {
                out[strat] += self.states_flat[base + strat];
            }
        }
        let inv_n = 1.0 / self.num_nodes as f64;
        for val in &mut out[..s] {
            *val *= inv_n;
        }
    }

    #[verified_engine::verified]
    fn compute_local_payoffs(&mut self) {
        let s = self.num_strategies;
        self.fitness_scratch.fill(0.0);

        for i in 0..self.num_nodes {
            let start = self.offsets[i];
            let end = self.offsets[i + 1];
            let f_base = i * s;

            if start == end {
                let x_row = &self.states_flat[f_base..f_base + s];
                for (r, pay_row) in self.payoff_flat.chunks_exact(s).enumerate() {
                    let sum: f64 = pay_row.iter().zip(x_row).map(|(&a, &x)| a * x).sum();
                    self.fitness_scratch[f_base + r] = sum;
                }
            } else {
                for k in start..end {
                    let nb = self.neighbors_flat[k];
                    let w = self.weights_flat[k];
                    let nb_base = nb * s;
                    let nb_row = &self.states_flat[nb_base..nb_base + s];
                    for (r, pay_row) in self.payoff_flat.chunks_exact(s).enumerate() {
                        let pair_payoff: f64 =
                            pay_row.iter().zip(nb_row).map(|(&a, &x)| a * x).sum();
                        self.fitness_scratch[f_base + r] += w * pair_payoff;
                    }
                }
            }

            let mut pi_i = 0.0_f64;
            for r in 0..s {
                pi_i += self.states_flat[f_base + r] * self.fitness_scratch[f_base + r];
            }
            self.node_payoff_scratch[i] = pi_i;
        }
    }

    /// Advances all node strategy distributions by one step `dt` with zero heap allocations.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        if !dt.is_finite() || dt <= 0.0 || self.num_nodes == 0 {
            return;
        }
        self.compute_local_payoffs();
        let s = self.num_strategies;

        for i in 0..self.num_nodes {
            let start = self.offsets[i];
            let end = self.offsets[i + 1];
            let base = i * s;
            let pi_i = self.node_payoff_scratch[i];

            if start == end {
                for r in 0..s {
                    let xi = self.states_flat[base + r];
                    let dx = xi * (self.fitness_scratch[base + r] - pi_i);
                    self.next_states_scratch[base + r] = xi + dt * dx;
                }
            } else {
                let mut abs_weight_sum = 0.0_f64;
                let mut min_pi = pi_i;
                let mut max_pi = pi_i;
                for k in start..end {
                    abs_weight_sum += self.weights_flat[k].abs();
                    let pi_j = self.node_payoff_scratch[self.neighbors_flat[k]];
                    min_pi = min_pi.min(pi_j);
                    max_pi = max_pi.max(pi_j);
                }
                let norm = abs_weight_sum.max(1e-9);
                let span = (max_pi - min_pi).max(1.0);

                for r in 0..s {
                    let xi = self.states_flat[base + r];
                    let mut imitation_flux = 0.0_f64;
                    for k in start..end {
                        let w = self.weights_flat[k];
                        if w > 0.0 {
                            let nb = self.neighbors_flat[k];
                            let nb_base = nb * s;
                            let gain = (self.node_payoff_scratch[nb] - pi_i).max(0.0) / span;
                            let xj = self.states_flat[nb_base + r];
                            imitation_flux += (w / norm) * gain * (xj - xi);
                        }
                    }
                    let local_sel = (LOCAL_SELECTION_GAIN / norm)
                        * xi
                        * (self.fitness_scratch[base + r] - pi_i);
                    self.next_states_scratch[base + r] = xi + dt * (imitation_flux + local_sel);
                }
            }

            project_simplex_row(&mut self.next_states_scratch[base..base + s]);
        }

        self.states_flat.copy_from_slice(&self.next_states_scratch);
    }

    /// Advances the spatial replicator system for `steps` iterations with zero heap allocations.
    #[verified_engine::verified]
    pub fn evolve(&mut self, steps: usize, dt: f64) {
        for _ in 0..steps {
            self.step(dt);
        }
    }
}

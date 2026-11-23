mod builder;
pub use builder::{NetworkEpidemicModelBuilder, NetworkModelError};

use rand::Rng;
use std::f32::consts::TAU;

/// Calculates R0 for a heterogeneous network.
///
/// $R_0 = \frac{\beta}{\gamma} \frac{\langle k^2 \rangle - \langle k \rangle}{\langle k \rangle}$
#[verified_engine::verified]
pub fn heterogeneous_r0(beta: f64, gamma: f64, mean_degree: f64, degree_variance: f64) -> f64 {
    if mean_degree == 0.0 || gamma == 0.0 {
        return 0.0;
    }

    let mean_k_sq = degree_variance + mean_degree.powi(2);
    let factor = (mean_k_sq - mean_degree) / mean_degree;

    (beta / gamma) * factor
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(missing_docs)]
pub enum NodeState {
    #[allow(missing_docs)]
    Susceptible,
    #[allow(missing_docs)]
    Infected,
    #[allow(missing_docs)]
    Recovered,
    /// Pre-emptively or reactively immunized vertex that neither acquires nor transmits contagion.
    Vaccinated,
}

#[allow(missing_docs)]
#[derive(Debug, Clone)]
pub struct NetworkEpidemicModel {
    pub(super) num_nodes: usize,
    pub(super) states: Vec<NodeState>,
    pub(super) scratch_states: Vec<NodeState>,
    pub(super) positions: Vec<[f32; 2]>,
    pub(super) adjacency: Vec<Vec<usize>>,
    pub(super) edge_quarantined: Vec<Vec<bool>>,
    pub(super) edges: Vec<[usize; 2]>,
    pub(super) patient_zero: Option<usize>,
    pub(super) beta: f64,
    pub(super) gamma: f64,
}

impl PartialEq for NetworkEpidemicModel {
    fn eq(&self, other: &Self) -> bool {
        self.num_nodes == other.num_nodes
            && self.states == other.states
            && self.positions == other.positions
            && self.adjacency == other.adjacency
            && self.edge_quarantined == other.edge_quarantined
            && self.edges == other.edges
            && self.patient_zero == other.patient_zero
            && self.beta == other.beta
            && self.gamma == other.gamma
    }
}

#[verified_engine::verified]
pub(super) fn canonical_edges_from_adjacency(
    num_nodes: usize,
    adjacency: &[Vec<usize>],
) -> Vec<[usize; 2]> {
    let mut edges = Vec::new();
    for (u, nbrs) in adjacency.iter().enumerate().take(num_nodes) {
        for &v in nbrs {
            if v < num_nodes && u != v {
                let pair = if u < v { [u, v] } else { [v, u] };
                edges.push(pair);
            }
        }
    }
    edges.sort_unstable();
    edges.dedup();
    edges
}

impl NetworkEpidemicModel {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn builder() -> NetworkEpidemicModelBuilder {
        NetworkEpidemicModelBuilder::new()
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn states(&self) -> &[NodeState] {
        &self.states
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn states_mut(&mut self) -> &mut Vec<NodeState> {
        &mut self.states
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn positions(&self) -> &[[f32; 2]] {
        &self.positions
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn adjacency(&self) -> &[Vec<usize>] {
        &self.adjacency
    }

    /// Returns canonical undirected contact edges `[u, v]` (`u < v`) in deterministic order.
    #[verified_engine::verified]
    pub fn edges(&self) -> &[[usize; 2]] {
        &self.edges
    }

    /// Returns the index of the initial patient-zero vertex, if present.
    #[verified_engine::verified]
    pub fn patient_zero(&self) -> Option<usize> {
        self.patient_zero
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn beta(&self) -> f64 {
        self.beta
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn set_num_nodes(&mut self, num: usize) -> Result<(), NetworkModelError> {
        if num == 0 {
            return Err(NetworkModelError::InvalidNodeCount);
        }
        self.num_nodes = num;
        Ok(())
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn set_beta(&mut self, beta: f64) -> Result<(), NetworkModelError> {
        if beta < 0.0 {
            return Err(NetworkModelError::InvalidParameters);
        }
        self.beta = beta;
        Ok(())
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn set_gamma(&mut self, gamma: f64) -> Result<(), NetworkModelError> {
        if gamma < 0.0 {
            return Err(NetworkModelError::InvalidParameters);
        }
        self.gamma = gamma;
        Ok(())
    }

    /// Sets or clears vaccination on `node` without heap allocation.
    ///
    /// Rejects out-of-bounds indices and nodes currently in `Infected` or `Recovered` states.
    #[verified_engine::verified]
    pub fn set_node_vaccinated(
        &mut self,
        node: usize,
        vaccinated: bool,
    ) -> Result<(), NetworkModelError> {
        if node >= self.num_nodes || node >= self.states.len() {
            return Err(NetworkModelError::InvalidNodeIndex);
        }
        match (self.states[node], vaccinated) {
            (NodeState::Susceptible | NodeState::Vaccinated, true) => {
                self.states[node] = NodeState::Vaccinated;
                Ok(())
            }
            (NodeState::Vaccinated | NodeState::Susceptible, false) => {
                self.states[node] = NodeState::Susceptible;
                Ok(())
            }
            (NodeState::Infected | NodeState::Recovered, _) => {
                Err(NetworkModelError::InvalidIntervention)
            }
        }
    }

    /// Symmetrically quarantines or restores the undirected edge `(u, v)` without heap allocation.
    #[verified_engine::verified]
    pub fn set_edge_quarantined(
        &mut self,
        u: usize,
        v: usize,
        quarantined: bool,
    ) -> Result<(), NetworkModelError> {
        if u >= self.num_nodes
            || v >= self.num_nodes
            || u == v
            || u >= self.adjacency.len()
            || v >= self.adjacency.len()
        {
            return Err(NetworkModelError::InvalidEdge);
        }
        let Some(uv_idx) = self.adjacency[u].iter().position(|&nbr| nbr == v) else {
            return Err(NetworkModelError::InvalidEdge);
        };
        let Some(vu_idx) = self.adjacency[v].iter().position(|&nbr| nbr == u) else {
            return Err(NetworkModelError::InvalidEdge);
        };
        self.edge_quarantined[u][uv_idx] = quarantined;
        self.edge_quarantined[v][vu_idx] = quarantined;
        Ok(())
    }

    /// Returns `true` if the undirected contact edge `(u, v)` is currently quarantined.
    #[verified_engine::verified]
    pub fn is_edge_quarantined(&self, u: usize, v: usize) -> bool {
        if u >= self.num_nodes || v >= self.num_nodes || u >= self.adjacency.len() {
            return false;
        }
        self.adjacency[u]
            .iter()
            .position(|&nbr| nbr == v)
            .and_then(|idx| self.edge_quarantined.get(u)?.get(idx).copied())
            .unwrap_or(false)
    }

    #[verified_engine::verified]
    fn is_node_active(&self, node: usize) -> bool {
        node < self.num_nodes
            && node < self.states.len()
            && matches!(
                self.states[node],
                NodeState::Susceptible | NodeState::Infected
            )
    }

    /// Returns the number of unquarantined active (`Susceptible` or `Infected`) neighbors of `node`.
    #[verified_engine::verified]
    pub fn active_degree(&self, node: usize) -> usize {
        if !self.is_node_active(node) || node >= self.adjacency.len() {
            return 0;
        }
        let masks = self
            .edge_quarantined
            .get(node)
            .map_or(&[][..], Vec::as_slice);
        self.adjacency[node]
            .iter()
            .enumerate()
            .filter(|&(idx, &nbr)| {
                !masks.get(idx).copied().unwrap_or(false) && self.is_node_active(nbr)
            })
            .count()
    }

    /// Computes `(mean_degree, degree_variance)` across active edges without heap allocation.
    #[verified_engine::verified]
    pub fn active_degree_moments(&self) -> (f64, f64) {
        let n = self.num_nodes.min(self.states.len());
        if n == 0 {
            return (0.0, 0.0);
        }
        let mut sum_k = 0.0_f64;
        let mut sum_k_sq = 0.0_f64;
        for i in 0..n {
            let k = self.active_degree(i) as f64;
            sum_k += k;
            sum_k_sq += k * k;
        }
        let inv_n = 1.0 / (n as f64);
        let mean = sum_k * inv_n;
        let variance = (sum_k_sq * inv_n - mean * mean).max(0.0);
        (mean, variance)
    }

    /// Evaluates the effective heterogeneous basic reproduction number on the residual active graph.
    #[verified_engine::verified]
    pub fn effective_r0(&self) -> f64 {
        let (mean_k, var_k) = self.active_degree_moments();
        heterogeneous_r0(self.beta, self.gamma, mean_k, var_k).max(0.0)
    }

    /// Creates a new network epidemic model and initializes it with a geometric graph.
    #[verified_engine::verified]
    pub fn new(num_nodes: usize, beta: f64, gamma: f64) -> Self {
        let mut rng = oxidize_core::rng::OxidizeRng::default();
        Self::new_with_rng(num_nodes, beta, gamma, &mut rng)
    }

    /// Creates a new network epidemic model and initializes it with a geometric graph
    /// using an injected RNG.
    #[verified_engine::verified]
    pub fn new_with_rng<R: Rng>(num_nodes: usize, beta: f64, gamma: f64, rng: &mut R) -> Self {
        let mut model = Self {
            num_nodes,
            states: Vec::new(),
            scratch_states: Vec::new(),
            positions: Vec::new(),
            adjacency: Vec::new(),
            edge_quarantined: Vec::new(),
            edges: Vec::new(),
            patient_zero: None,
            beta,
            gamma,
        };
        model.initialize_geometric_graph_with_rng(rng);
        model
    }

    /// Initializes a geometric graph using the default deterministic RNG.
    #[verified_engine::verified]
    pub fn initialize_geometric_graph(&mut self) {
        let mut rng = oxidize_core::rng::OxidizeRng::default();
        self.initialize_geometric_graph_with_rng(&mut rng);
    }

    #[verified_engine::verified]
    fn connect_geometric_edges(&mut self, connection_radius: f32) {
        let radius_sq = connection_radius * connection_radius;
        for i in 0..self.num_nodes {
            for j in (i + 1)..self.num_nodes {
                let dx = self.positions[i][0] - self.positions[j][0];
                let dy = self.positions[i][1] - self.positions[j][1];
                if dx * dx + dy * dy < radius_sq {
                    self.edges.push([i, j]);
                    self.adjacency[i].push(j);
                    self.adjacency[j].push(i);
                    self.edge_quarantined[i].push(false);
                    self.edge_quarantined[j].push(false);
                }
            }
        }
    }

    /// Initializes a geometric graph using an injected RNG.
    #[verified_engine::verified]
    pub fn initialize_geometric_graph_with_rng<R: Rng>(&mut self, rng: &mut R) {
        self.states = vec![NodeState::Susceptible; self.num_nodes];
        self.scratch_states = vec![NodeState::Susceptible; self.num_nodes];
        self.positions = vec![[0.0, 0.0]; self.num_nodes];
        self.adjacency = vec![vec![]; self.num_nodes];
        self.edge_quarantined = vec![vec![]; self.num_nodes];
        self.edges.clear();
        self.patient_zero = None;

        let radius = 200.0;
        for i in 0..self.num_nodes {
            let angle = rng.r#gen_range(0.0..TAU);
            let r = radius * rng.r#gen_range(0.0f32..1.0f32).sqrt();
            self.positions[i] = [r * angle.cos(), r * angle.sin()];
        }

        self.connect_geometric_edges(60.0);

        if self.num_nodes > 0 {
            let start_idx = rng.r#gen_range(0..self.num_nodes);
            self.states[start_idx] = NodeState::Infected;
            self.scratch_states[start_idx] = NodeState::Infected;
            self.patient_zero = Some(start_idx);
        }
    }

    /// Steps the simulation forward using the default deterministic RNG.
    #[verified_engine::verified]
    pub fn step(&mut self) {
        let mut rng = oxidize_core::rng::OxidizeRng::default();
        self.step_with_rng(&mut rng);
    }

    #[verified_engine::verified]
    fn count_active_infected_neighbors(&self, node: usize) -> usize {
        if node >= self.adjacency.len() {
            return 0;
        }
        let masks = self
            .edge_quarantined
            .get(node)
            .map_or(&[][..], Vec::as_slice);
        self.adjacency[node]
            .iter()
            .enumerate()
            .filter(|&(idx, &nbr)| {
                !masks.get(idx).copied().unwrap_or(false)
                    && self.states.get(nbr).copied() == Some(NodeState::Infected)
            })
            .count()
    }

    /// Steps the simulation forward using an injected RNG without heap allocations.
    #[verified_engine::verified]
    pub fn step_with_rng<R: Rng>(&mut self, rng: &mut R) {
        let n = self.num_nodes.min(self.states.len());
        if self.scratch_states.len() < n {
            self.scratch_states.resize(n, NodeState::Susceptible);
        }
        self.scratch_states[..n].copy_from_slice(&self.states[..n]);

        for i in 0..n {
            match self.states[i] {
                NodeState::Susceptible => {
                    let infected_neighbors = self.count_active_infected_neighbors(i);
                    for _ in 0..infected_neighbors {
                        if rng.r#gen::<f64>() < self.beta {
                            self.scratch_states[i] = NodeState::Infected;
                            break;
                        }
                    }
                }
                NodeState::Infected => {
                    if rng.r#gen::<f64>() < self.gamma {
                        self.scratch_states[i] = NodeState::Recovered;
                    }
                }
                NodeState::Recovered | NodeState::Vaccinated => {}
            }
        }
        self.states[..n].copy_from_slice(&self.scratch_states[..n]);
    }
}

#[cfg(test)]
mod tests;

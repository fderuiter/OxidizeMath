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

    // Var(k) = E[k^2] - (E[k])^2
    // E[k^2] = Var(k) + (E[k])^2

    let mean_k_sq = degree_variance + mean_degree.powi(2);
    let factor = (mean_k_sq - mean_degree) / mean_degree;

    (beta / gamma) * factor
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub enum NodeState {
    #[allow(missing_docs)]
    Susceptible,
    #[allow(missing_docs)]
    Infected,
    #[allow(missing_docs)]
    Recovered,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[allow(missing_docs)]
pub struct NetworkEpidemicModel {
    num_nodes: usize,
    states: Vec<NodeState>,
    positions: Vec<[f32; 2]>,
    adjacency: Vec<Vec<usize>>,
    beta: f64,
    gamma: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub enum NetworkModelError {
    #[allow(missing_docs)]
    InvalidNodeCount,
    #[allow(missing_docs)]
    InvalidParameters,
    #[allow(missing_docs)]
    StateMismatch,
    #[allow(missing_docs)]
    PositionMismatch,
    #[allow(missing_docs)]
    AdjacencyMismatch,
}

impl std::fmt::Display for NetworkModelError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidNodeCount => write!(f, "Number of nodes must be greater than zero"),
            Self::InvalidParameters => write!(f, "Beta and gamma parameters must be non-negative"),
            Self::StateMismatch => write!(f, "States vector length must match num_nodes"),
            Self::PositionMismatch => write!(f, "Positions vector length must match num_nodes"),
            Self::AdjacencyMismatch => write!(f, "Adjacency list length must match num_nodes"),
        }
    }
}

impl std::error::Error for NetworkModelError {}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub struct NetworkEpidemicModelBuilder {
    num_nodes: Option<usize>,
    beta: Option<f64>,
    gamma: Option<f64>,
    states: Option<Vec<NodeState>>,
    positions: Option<Vec<[f32; 2]>>,
    adjacency: Option<Vec<Vec<usize>>>,
}

impl Default for NetworkEpidemicModelBuilder {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkEpidemicModelBuilder {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self {
            num_nodes: None,
            beta: None,
            gamma: None,
            states: None,
            positions: None,
            adjacency: None,
        }
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn num_nodes(mut self, num: usize) -> Self {
        self.num_nodes = Some(num);
        self
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn parameters(mut self, beta: f64, gamma: f64) -> Self {
        self.beta = Some(beta);
        self.gamma = Some(gamma);
        self
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn states(mut self, states: Vec<NodeState>) -> Self {
        self.states = Some(states);
        self
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn positions(mut self, positions: Vec<[f32; 2]>) -> Self {
        self.positions = Some(positions);
        self
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn adjacency(mut self, adjacency: Vec<Vec<usize>>) -> Self {
        self.adjacency = Some(adjacency);
        self
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn build(self) -> Result<NetworkEpidemicModel, NetworkModelError> {
        let num_nodes = self.num_nodes.ok_or(NetworkModelError::InvalidNodeCount)?;
        let beta = self.beta.unwrap_or(0.0);
        let gamma = self.gamma.unwrap_or(0.0);

        if num_nodes == 0 {
            return Err(NetworkModelError::InvalidNodeCount);
        }

        if !beta.is_finite() || !gamma.is_finite() || beta < 0.0 || gamma < 0.0 {
            return Err(NetworkModelError::InvalidParameters);
        }

        let states = self
            .states
            .unwrap_or_else(|| vec![NodeState::Susceptible; num_nodes]);
        if states.len() != num_nodes {
            return Err(NetworkModelError::StateMismatch);
        }

        let positions = self
            .positions
            .unwrap_or_else(|| vec![[0.0, 0.0]; num_nodes]);
        if positions.len() != num_nodes {
            return Err(NetworkModelError::PositionMismatch);
        }

        let adjacency = self.adjacency.unwrap_or_else(|| vec![vec![]; num_nodes]);
        if adjacency.len() != num_nodes {
            return Err(NetworkModelError::AdjacencyMismatch);
        }

        Ok(NetworkEpidemicModel {
            num_nodes,
            states,
            positions,
            adjacency,
            beta,
            gamma,
        })
    }
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

    /// Creates a new network epidemic model and initializes it with a geometric graph.
    ///
    /// This method uses the default thread-local RNG. For deterministic behavior,
    /// use `new_with_rng`.
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
            states: vec![NodeState::Susceptible; num_nodes],
            positions: vec![[0.0, 0.0]; num_nodes],
            adjacency: vec![vec![]; num_nodes],
            beta,
            gamma,
        };
        model.initialize_geometric_graph_with_rng(rng);
        model
    }

    /// Initializes a geometric graph. This method uses the default thread-local RNG.
    #[verified_engine::verified]
    pub fn initialize_geometric_graph(&mut self) {
        let mut rng = oxidize_core::rng::OxidizeRng::default();
        self.initialize_geometric_graph_with_rng(&mut rng);
    }

    /// Initializes a geometric graph using an injected RNG.
    #[verified_engine::verified]
    pub fn initialize_geometric_graph_with_rng<R: Rng>(&mut self, rng: &mut R) {
        self.states = vec![NodeState::Susceptible; self.num_nodes];
        self.positions = vec![[0.0, 0.0]; self.num_nodes];
        self.adjacency = vec![vec![]; self.num_nodes];

        // Random geometric graph
        let radius = 200.0;
        for i in 0..self.num_nodes {
            let angle = rng.r#gen_range(0.0..TAU);
            let r = radius * rng.r#gen_range(0.0f32..1.0f32).sqrt();
            self.positions[i] = [r * angle.cos(), r * angle.sin()];
        }

        let connection_radius = 60.0;
        for i in 0..self.num_nodes {
            for j in (i + 1)..self.num_nodes {
                let dx = self.positions[i][0] - self.positions[j][0];
                let dy = self.positions[i][1] - self.positions[j][1];
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < connection_radius {
                    self.adjacency[i].push(j);
                    self.adjacency[j].push(i);
                }
            }
        }

        // Start with one infected
        if self.num_nodes > 0 {
            let start_idx = rng.r#gen_range(0..self.num_nodes);
            self.states[start_idx] = NodeState::Infected;
        }
    }

    /// Steps the simulation forward using the default thread-local RNG.
    #[verified_engine::verified]
    pub fn step(&mut self) {
        let mut rng = oxidize_core::rng::OxidizeRng::default();
        self.step_with_rng(&mut rng);
    }

    /// Steps the simulation forward using an injected RNG.
    #[verified_engine::verified]
    pub fn step_with_rng<R: Rng>(&mut self, rng: &mut R) {
        let mut next_states = self.states.clone();

        for (i, next_state) in next_states.iter_mut().enumerate().take(self.num_nodes) {
            match self.states[i] {
                NodeState::Susceptible => {
                    // Check infected neighbors
                    let infected_neighbors = self.adjacency[i]
                        .iter()
                        .filter(|&&j| self.states[j] == NodeState::Infected)
                        .count();
                    for _ in 0..infected_neighbors {
                        if rng.r#gen::<f64>() < self.beta {
                            *next_state = NodeState::Infected;
                            break;
                        }
                    }
                }
                NodeState::Infected => {
                    if rng.r#gen::<f64>() < self.gamma {
                        *next_state = NodeState::Recovered;
                    }
                }
                NodeState::Recovered => {}
            }
        }
        self.states = next_states;
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for NetworkEpidemicModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct RawNetworkEpidemicModel {
            num_nodes: usize,
            states: Vec<NodeState>,
            positions: Vec<[f32; 2]>,
            adjacency: Vec<Vec<usize>>,
            beta: f64,
            gamma: f64,
        }

        let raw = RawNetworkEpidemicModel::deserialize(deserializer)?;
        if raw
            .positions
            .iter()
            .any(|p| !p[0].is_finite() || !p[1].is_finite())
        {
            return Err(serde::de::Error::custom(
                "Node positions must be finite floats",
            ));
        }
        if raw
            .adjacency
            .iter()
            .any(|neighbors| neighbors.iter().any(|&idx| idx >= raw.num_nodes))
        {
            return Err(serde::de::Error::custom(
                "Adjacency indices must be < num_nodes",
            ));
        }
        NetworkEpidemicModelBuilder::new()
            .num_nodes(raw.num_nodes)
            .parameters(raw.beta, raw.gamma)
            .states(raw.states)
            .positions(raw.positions)
            .adjacency(raw.adjacency)
            .build()
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_heterogeneous_r0() {
        let beta = 0.5;
        let gamma = 0.1;
        // Homogeneous network: Variance = 0. Factor = (k^2 - k)/k = (k^2 - k)/k = k - 1?
        // Wait, if Var=0, then <k^2> = <k>^2.
        // Factor = (<k>^2 - <k>)/<k> = <k> - 1.
        // Standard formula usually assumes contact rate is proportional to k.

        // Using provided formula:
        let mean_k = 4.0;
        let var_k = 0.0;
        let r0 = heterogeneous_r0(beta, gamma, mean_k, var_k);

        // R0 = (beta/gamma) * (16 - 4)/4 = 5 * 3 = 15.
        assert!((r0 - 15.0).abs() < math_commons::registry::TOLERANCE_FAST);
    }
}

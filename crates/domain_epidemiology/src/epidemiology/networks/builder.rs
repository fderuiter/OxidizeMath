use super::{NetworkEpidemicModel, NodeState, canonical_edges_from_adjacency};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// Node index exceeds the model's `num_nodes`.
    InvalidNodeIndex,
    /// Edge endpoints are out of bounds or not connected in the contact graph.
    InvalidEdge,
    /// Attempted to vaccinate a node that is already infected or recovered.
    InvalidIntervention,
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
            Self::InvalidNodeIndex => write!(f, "Node index is out of bounds"),
            Self::InvalidEdge => write!(f, "Edge does not exist in contact adjacency graph"),
            Self::InvalidIntervention => {
                write!(f, "Cannot vaccinate an infected or recovered node")
            }
        }
    }
}

impl std::error::Error for NetworkModelError {}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
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
        if beta < 0.0 || gamma < 0.0 {
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
        if adjacency.len() != num_nodes
            || adjacency.iter().any(|n| n.iter().any(|&v| v >= num_nodes))
        {
            return Err(NetworkModelError::AdjacencyMismatch);
        }

        let scratch_states = states.clone();
        let edge_quarantined = adjacency.iter().map(|n| vec![false; n.len()]).collect();
        let edges = canonical_edges_from_adjacency(num_nodes, &adjacency);
        let patient_zero = states.iter().position(|&s| s == NodeState::Infected);

        Ok(NetworkEpidemicModel {
            num_nodes,
            states,
            scratch_states,
            positions,
            adjacency,
            edge_quarantined,
            edges,
            patient_zero,
            beta,
            gamma,
        })
    }
}

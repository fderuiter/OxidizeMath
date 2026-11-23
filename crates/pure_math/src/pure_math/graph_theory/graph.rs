// Using petgraph for graph representations.
// The Graph struct will be a wrapper around a petgraph::Graph.
use petgraph::Undirected;
use petgraph::graph::{Graph as PetgraphGraph, NodeIndex};

/// A wrapper around petgraph::Graph to represent an undirected graph.
/// The nodes (N) and edges (E) are generic.
#[derive(Debug, Clone)]
pub struct Graph<N, E> {
    #[allow(missing_docs)]
    pub graph: PetgraphGraph<N, E, Undirected>,
}

impl<N, E> Default for Graph<N, E> {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new()
    }
}

impl<N, E> Graph<N, E> {
    /// Creates a new empty graph.
    #[verified_engine::verified]
    pub fn new() -> Self {
        Graph {
            graph: PetgraphGraph::new_undirected(),
        }
    }

    /// Adds a node to the graph.
    #[verified_engine::verified]
    pub fn add_node(&mut self, weight: N) -> NodeIndex {
        self.graph.add_node(weight)
    }

    /// Adds an edge between two nodes.
    #[verified_engine::verified]
    pub fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, weight: E) {
        self.graph.add_edge(a, b, weight);
    }
}

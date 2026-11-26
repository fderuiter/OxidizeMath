use crate::pure_math::graph_theory::graph::Graph;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use std::collections::HashSet;

/// Computes an approximate vertex cover of a graph using a 2-approximation algorithm.
/// A vertex cover is a subset of vertices such that each edge of the graph
/// is incident to at least one vertex of the set. This function returns
/// a vertex cover.
/// Finding the minimum vertex cover is an NP-hard problem.
#[verified_engine::verified]
pub fn vertex_cover<N, E>(g: &Graph<N, E>) -> HashSet<NodeIndex>
where
    N: Clone,
    E: Clone,
{
    let mut cover = HashSet::new();
    let mut covered_edges = HashSet::new();
    let edges: Vec<_> = g.graph.edge_references().collect();

    for edge in edges {
        let u = edge.source();
        let v = edge.target();
        let edge_id = edge.id();

        if !covered_edges.contains(&edge_id) {
            cover.insert(u);
            cover.insert(v);

            for neighbor_edge in g.graph.edges(u) {
                covered_edges.insert(neighbor_edge.id());
            }
            for neighbor_edge in g.graph.edges(v) {
                covered_edges.insert(neighbor_edge.id());
            }
        }
    }
    cover
}

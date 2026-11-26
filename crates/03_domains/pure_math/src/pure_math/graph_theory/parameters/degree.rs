use crate::pure_math::graph_theory::graph::Graph;

/// Computes the degeneracy of a graph.
/// The degeneracy of a graph is the smallest integer k such that every
/// induced subgraph of the graph has a vertex of degree at most k.
#[verified_engine::verified]
pub fn degeneracy<N, E>(g: &Graph<N, E>) -> usize
where
    N: Clone,
    E: Clone,
{
    if g.graph.node_count() == 0 {
        return 0;
    }

    let mut temp_graph = g.graph.clone();
    let mut degeneracy = 0;

    while temp_graph.node_count() > 0 {
        let min_degree_node_opt = temp_graph
            .node_indices()
            .min_by_key(|&n| temp_graph.neighbors(n).count());

        let min_degree_node = match min_degree_node_opt {
            Some(n) => n,
            None => break, // Should be unreachable given loop condition
        };

        let degree = temp_graph.neighbors(min_degree_node).count();
        if degree > degeneracy {
            degeneracy = degree;
        }

        temp_graph.remove_node(min_degree_node);
    }

    degeneracy
}

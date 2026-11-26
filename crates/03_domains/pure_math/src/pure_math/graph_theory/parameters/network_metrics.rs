use crate::pure_math::graph_theory::graph::Graph;
use petgraph::algo::dijkstra;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

/// Calculates the degree centrality for each node in the graph.
#[verified_engine::verified]
pub fn degree_centrality<N, E>(g: &Graph<N, E>) -> HashMap<NodeIndex, f64> {
    let mut centrality = HashMap::new();
    let node_count = g.graph.node_count();

    if node_count <= 1 {
        for node in g.graph.node_indices() {
            centrality.insert(node, 0.0);
        }
        return centrality;
    }

    let denominator = (node_count - 1) as f64;
    for node in g.graph.node_indices() {
        let degree = g.graph.neighbors(node).count() as f64;
        centrality.insert(node, degree / denominator);
    }

    centrality
}

/// Calculates the closeness centrality for each node in the graph.
/// Assumes unweighted edges (weight = 1.0) for simplicity.
#[verified_engine::verified]
pub fn closeness_centrality<N, E>(g: &Graph<N, E>) -> HashMap<NodeIndex, f64> {
    let mut centrality = HashMap::new();
    let node_count = g.graph.node_count();

    if node_count <= 1 {
        for node in g.graph.node_indices() {
            centrality.insert(node, 0.0);
        }
        return centrality;
    }

    let numerator = (node_count - 1) as f64;
    for node in g.graph.node_indices() {
        let dists = dijkstra(&g.graph, node, None, |_| 1);
        let sum_dist: usize = dists.values().sum();

        let c = if sum_dist > 0 {
            numerator / (sum_dist as f64)
        } else {
            0.0
        };
        centrality.insert(node, c);
    }

    centrality
}

/// Calculates the global diameter of the graph (maximum shortest path between any two nodes).
#[verified_engine::verified]
pub fn diameter<N, E>(g: &Graph<N, E>) -> usize {
    let mut max_dist = 0;
    for node in g.graph.node_indices() {
        let dists = dijkstra(&g.graph, node, None, |_| 1);
        for &d in dists.values() {
            if d > max_dist {
                max_dist = d;
            }
        }
    }
    max_dist
}

/// Calculates the local clustering coefficient for each node.
#[verified_engine::verified]
pub fn clustering_coefficients<N, E>(g: &Graph<N, E>) -> HashMap<NodeIndex, f64> {
    let mut clustering = HashMap::new();

    for node in g.graph.node_indices() {
        let neighbors: Vec<NodeIndex> = g.graph.neighbors(node).collect();
        let k = neighbors.len();

        if k < 2 {
            clustering.insert(node, 0.0);
            continue;
        }

        let mut links = 0;
        for i in 0..k {
            for j in (i + 1)..k {
                if g.graph.contains_edge(neighbors[i], neighbors[j]) {
                    links += 1;
                }
            }
        }

        let possible_links = (k * (k - 1)) / 2;
        let c = (links as f64) / (possible_links as f64);
        clustering.insert(node, c);
    }

    clustering
}

/// Calculates the average clustering coefficient of the graph.
#[verified_engine::verified]
pub fn average_clustering_coefficient<N, E>(g: &Graph<N, E>) -> f64 {
    let coeffs = clustering_coefficients(g);
    if coeffs.is_empty() {
        return 0.0;
    }
    let sum: f64 = coeffs.values().sum();
    sum / (coeffs.len() as f64)
}

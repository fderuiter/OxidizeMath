//! Augmenting-path and minimum-cut trace reconstruction helpers for [`super::max_flow_min_cut`].

use super::EdmondsKarpScratch;

#[verified_engine::verified]
pub(super) fn reconstruct_augmenting_path(
    scratch: &EdmondsKarpScratch,
    source: usize,
    sink: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut rev_nodes = Vec::new();
    let mut rev_edges = Vec::new();
    let mut cur = sink;
    rev_nodes.push(sink);
    let mut guard = 0usize;
    while cur != source && guard < scratch.num_nodes {
        let arc = scratch.parent_arc[cur];
        rev_edges.push(arc / 2);
        let (u, _) = scratch.arc_endpoints(arc);
        rev_nodes.push(u);
        cur = u;
        guard += 1;
    }
    rev_nodes.reverse();
    rev_edges.reverse();
    (rev_nodes, rev_edges)
}

#[verified_engine::verified]
pub(super) fn collect_cut_edges(scratch: &EdmondsKarpScratch, source_side: &[bool]) -> Vec<usize> {
    let mut cut_edges = Vec::new();
    let mut e = 0usize;
    while e < scratch.num_edges {
        let u = scratch.edge_u[e];
        let v = scratch.edge_v[e];
        let crosses = if scratch.directed {
            source_side[u] && !source_side[v]
        } else {
            source_side[u] != source_side[v]
        };
        if crosses {
            cut_edges.push(e);
        }
        e += 1;
    }
    cut_edges
}

// [cite:graph_parameters_rust]

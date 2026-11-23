//! Weighted graph skeleton extraction and network benchmarks (`wire_cost`,
//! [`pure_math::pure_math::graph_theory::dijkstra::dijkstra`] route stretch, and single-edge fault
//! survival) compared against static minimum spanning trees.

use super::engine::PhysarumEngine;
use super::params::PhysarumTerminal;
use super::skeleton::{
    MaskView, RawChain, build_foreground_mask, build_pinned_terminal_mask, cluster_keypoints,
    fill_small_interior_holes, prune_short_spurs, simplify_polyline_rdp, trace_skeleton_chains,
    zhang_suen_thin,
};
use pure_math::__doctest_facade::petgraph::graph::NodeIndex;
use pure_math::pure_math::graph_theory::dijkstra::dijkstra;
use pure_math::pure_math::graph_theory::graph::Graph;

/// Extracted undirected edge segment in continuous grid coordinates `(x, y)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysarumSkeletonEdge {
    /// Start node index in [`PhysarumNetworkBenchmark::node_positions`].
    pub u: usize,
    /// End node index in [`PhysarumNetworkBenchmark::node_positions`].
    pub v: usize,
    /// Start coordinate `[x, y]` in grid space.
    pub p0: [f64; 2],
    /// End coordinate `[x, y]` in grid space.
    pub p1: [f64; 2],
    /// Euclidean edge length $\|p_1 - p_0\|_2$.
    pub length: f64,
}

/// Complete topological and graph-theoretic benchmark comparing an extracted Physarum network
/// against [`dijkstra`] shortest paths and the static terminal Minimum Spanning Tree (MST).
pub struct PhysarumNetworkBenchmark {
    /// Extracted weighted undirected skeleton graph (`N = [x, y]`, `E = euclidean_length`).
    pub graph: Graph<[f64; 2], f64>,
    /// Coordinates `[x, y]` for each node in `graph`.
    pub node_positions: Vec<[f64; 2]>,
    /// Node index (`0..node_positions.len()`) mapped to each nutrient terminal `0..terminal_count`.
    pub terminal_nodes: Vec<usize>,
    /// Detected non-terminal Steiner junction coordinates (degree $\ge 3$ in pruned skeleton).
    pub steiner_junctions: Vec<[f64; 2]>,
    /// Extracted simplified skeleton edges.
    pub edges: Vec<PhysarumSkeletonEdge>,
    /// Static Euclidean Minimum Spanning Tree reference edges `[p0, p1]`.
    pub mst_segments: Vec<([f64; 2], [f64; 2])>,
    /// Total Euclidean wire length of the extracted Physarum skeleton.
    pub wire_cost: f64,
    /// Total Euclidean wire length of the static terminal Minimum Spanning Tree.
    pub mst_cost: f64,
    /// Normalized wire cost ratio `wire_cost / mst_cost`.
    pub cost_ratio: f64,
    /// Percentage wire-length reduction `(mst_cost - wire_cost) / mst_cost * 100.0`.
    pub length_reduction_pct: f64,
    /// Mean `dijkstra` shortest-path stretch $d_G(i, j) / \|T_i - T_j\|_2$ across connected terminal pairs.
    pub mean_route_stretch: f64,
    /// Mean `dijkstra` shortest-path stretch on the static terminal MST baseline.
    pub mst_mean_stretch: f64,
    /// Fraction of unordered terminal pairs mutually connected in `graph` (`0.0..=1.0`).
    pub connected_pair_fraction: f64,
    /// Fraction of single-edge deletions in `graph` after which all terminals remain connected (`0.0..=1.0`).
    pub fault_survival: f64,
    /// Mean fraction of terminal pairs remaining connected across single-edge deletions (`0.0..=1.0`).
    pub pairwise_fault_survival: f64,
}

/// Extracts a 1-pixel medial skeleton from `engine.trail()`, constructs a weighted
/// [`Graph`], and benchmarks wire cost, [`dijkstra`] stretch, and fault survival against the static MST.
#[verified_engine::verified]
pub fn extract_physarum_benchmark(engine: &PhysarumEngine) -> PhysarumNetworkBenchmark {
    let w = engine.width();
    let h = engine.height();
    let terms = engine.terminals();
    assert!(terms.len() >= 2);
    debug_assert!(w >= 64 && h >= 64);

    let (mst_graph, mst_nodes, mst_segments, mst_cost) = build_terminal_mst(terms);
    let mst_mean_stretch = compute_mean_dijkstra_stretch(&mst_graph, &mst_nodes, terms).0;

    let mut mask = build_foreground_mask(engine);
    let pinned = build_pinned_terminal_mask(terms, w, h, &mut mask);
    fill_small_interior_holes(&mut mask, engine.obstacles(), w, h, 14);
    zhang_suen_thin(&mut mask, &pinned, w, h);
    prune_short_spurs(&mut mask, &pinned, w, h, 10);

    let view = MaskView { mask: &mask, w, h };
    let (graph, node_positions, pg_term_nodes, steiner_junctions, edges, wire_cost) =
        skeleton_to_weighted_graph(view, terms);

    let (mean_route_stretch, connected_pair_fraction) =
        compute_mean_dijkstra_stretch(&graph, &pg_term_nodes, terms);
    let terminal_nodes: Vec<usize> = pg_term_nodes.iter().map(|n| n.index()).collect();
    let (fault_survival, pairwise_fault_survival) =
        compute_single_edge_fault_survival(node_positions.len(), &edges, &terminal_nodes);

    let safe_mst = mst_cost.max(1e-9);
    let cost_ratio = if wire_cost > 0.0 {
        wire_cost / safe_mst
    } else {
        0.0
    };
    let length_reduction_pct = if wire_cost > 0.0 {
        ((mst_cost - wire_cost) / safe_mst) * 100.0
    } else {
        0.0
    };

    PhysarumNetworkBenchmark {
        graph,
        node_positions,
        terminal_nodes,
        steiner_junctions,
        edges,
        mst_segments,
        wire_cost,
        mst_cost,
        cost_ratio,
        length_reduction_pct,
        mean_route_stretch,
        mst_mean_stretch,
        connected_pair_fraction,
        fault_survival,
        pairwise_fault_survival,
    }
}

type WeightedSkeletonParts = (
    Graph<[f64; 2], f64>,
    Vec<[f64; 2]>,
    Vec<NodeIndex>,
    Vec<[f64; 2]>,
    Vec<PhysarumSkeletonEdge>,
    f64,
);

#[verified_engine::verified]
fn skeleton_to_weighted_graph(
    view: MaskView<'_>,
    terms: &[PhysarumTerminal],
) -> WeightedSkeletonParts {
    let (mut node_positions, pixel_to_cluster) = cluster_keypoints(view, terms);
    let raw_chains = trace_skeleton_chains(view, &pixel_to_cluster);
    let clean_chains = prune_and_contract_chains(raw_chains, node_positions.len(), terms.len());

    let mut deg = vec![0usize; node_positions.len()];
    for &(u, v, _) in &clean_chains {
        deg[u] += 1;
        deg[v] += 1;
    }
    let steiner_junctions: Vec<[f64; 2]> = (terms.len()..node_positions.len())
        .filter(|&cid| deg[cid] >= 3)
        .map(|cid| node_positions[cid])
        .collect();

    let mut graph = Graph::<[f64; 2], f64>::new();
    let mut pg_nodes = Vec::with_capacity(node_positions.len() + clean_chains.len() * 2);
    for &pos in &node_positions {
        pg_nodes.push(graph.add_node(pos));
    }
    let terminal_nodes: Vec<NodeIndex> = (0..terms.len()).map(|i| pg_nodes[i]).collect();

    let mut edges = Vec::new();
    let mut wire_cost = 0.0;
    for (u_cluster, v_cluster, poly) in clean_chains {
        wire_cost += append_simplified_chain(
            &poly,
            (u_cluster, v_cluster),
            &mut graph,
            &mut node_positions,
            &mut pg_nodes,
            &mut edges,
        );
    }
    assert_eq!(terminal_nodes.len(), terms.len());
    debug_assert!(wire_cost >= 0.0);
    (
        graph,
        node_positions,
        terminal_nodes,
        steiner_junctions,
        edges,
        wire_cost,
    )
}

#[verified_engine::verified]
fn prune_and_contract_chains(
    raw_chains: Vec<RawChain>,
    cluster_count: usize,
    term_count: usize,
) -> Vec<RawChain> {
    let mut chains = dedup_undirected_chains(raw_chains);
    for _round in 0..cluster_count {
        let mut deg = vec![0usize; cluster_count];
        for &(u, v, _) in &chains {
            deg[u] += 1;
            deg[v] += 1;
        }
        let before = chains.len();
        chains.retain(|&(u, v, _)| {
            let u_dead = u >= term_count && deg[u] <= 1;
            let v_dead = v >= term_count && deg[v] <= 1;
            !(u_dead || v_dead)
        });
        if chains.len() == before {
            break;
        }
    }
    contract_degree2_nonterminals(&mut chains, cluster_count, term_count);
    assert!(term_count <= cluster_count);
    debug_assert!(chains.iter().all(|(u, v, _)| u != v));
    chains
}

#[verified_engine::verified]
fn dedup_undirected_chains(raw_chains: Vec<RawChain>) -> Vec<RawChain> {
    let mut out: Vec<RawChain> = Vec::with_capacity(raw_chains.len());
    for (u, v, poly) in raw_chains {
        if u == v {
            continue;
        }
        let a = u.min(v);
        let b = u.max(v);
        let duplicate = out.iter().any(|(ou, ov, opoly)| {
            (*ou).min(*ov) == a
                && (*ou).max(*ov) == b
                && (opoly.len().abs_diff(poly.len()) <= 3 || poly.len().min(opoly.len()) < 16)
        });
        if !duplicate {
            out.push((u, v, poly));
        }
    }
    debug_assert!(out.iter().all(|(u, v, _)| u != v));
    assert!(out.len() <= 4096);
    out
}

#[verified_engine::verified]
fn contract_degree2_nonterminals(
    chains: &mut Vec<RawChain>,
    cluster_count: usize,
    term_count: usize,
) {
    for cid in term_count..cluster_count {
        let inc: Vec<usize> = chains
            .iter()
            .enumerate()
            .filter_map(|(i, &(u, v, _))| (u == cid || v == cid).then_some(i))
            .collect();
        if inc.len() != 2 {
            continue;
        }
        let (i0, i1) = (inc[0].min(inc[1]), inc[0].max(inc[1]));
        let (u1, v1, mut p1) = chains.remove(i1);
        let (u0, v0, mut p0) = chains.remove(i0);
        let a = if u0 == cid { v0 } else { u0 };
        let b = if u1 == cid { v1 } else { u1 };
        if a == b {
            continue;
        }
        if v0 != cid {
            p0.reverse();
        }
        if u1 != cid {
            p1.reverse();
        }
        p0.extend_from_slice(&p1[1..]);
        chains.push((a, b, p0));
    }
    assert!(term_count <= cluster_count);
    debug_assert!(chains.len() <= 4096);
}

#[verified_engine::verified]
fn append_simplified_chain(
    raw_poly: &[[f64; 2]],
    endpoints: (usize, usize),
    graph: &mut Graph<[f64; 2], f64>,
    node_positions: &mut Vec<[f64; 2]>,
    pg_nodes: &mut Vec<NodeIndex>,
    edges: &mut Vec<PhysarumSkeletonEdge>,
) -> f64 {
    let (u_cluster, v_cluster) = endpoints;
    let mut anchored = raw_poly.to_vec();
    anchored[0] = node_positions[u_cluster];
    let last_idx = anchored.len() - 1;
    anchored[last_idx] = node_positions[v_cluster];
    let simp = simplify_polyline_rdp(&anchored, 2.10);

    let mut prev_id = u_cluster;
    let mut chain_len = 0.0;
    for seg in 1..simp.len() {
        let next_id = if seg + 1 == simp.len() {
            v_cluster
        } else {
            let id = node_positions.len();
            node_positions.push(simp[seg]);
            pg_nodes.push(graph.add_node(simp[seg]));
            id
        };
        let p0 = node_positions[prev_id];
        let p1 = node_positions[next_id];
        let len = (p1[0] - p0[0]).hypot(p1[1] - p0[1]).max(1e-6);
        graph.add_edge(pg_nodes[prev_id], pg_nodes[next_id], len);
        edges.push(PhysarumSkeletonEdge {
            u: prev_id,
            v: next_id,
            p0,
            p1,
            length: len,
        });
        chain_len += len;
        prev_id = next_id;
    }
    debug_assert!(chain_len.is_finite());
    assert!(simp.len() >= 2);
    chain_len
}

type MstBuildOutput = (
    Graph<[f64; 2], f64>,
    Vec<NodeIndex>,
    Vec<([f64; 2], [f64; 2])>,
    f64,
);

#[verified_engine::verified]
fn build_terminal_mst(terms: &[PhysarumTerminal]) -> MstBuildOutput {
    let k = terms.len();
    let mut graph = Graph::<[f64; 2], f64>::new();
    let nodes: Vec<NodeIndex> = terms.iter().map(|t| graph.add_node([t.x, t.y])).collect();
    let mut in_tree = vec![false; k];
    in_tree[0] = true;
    let mut segments = Vec::with_capacity(k.saturating_sub(1));
    let mut total_cost = 0.0;

    for _step in 1..k {
        let mut best_pair = (0usize, 0usize);
        let mut best_dist = f64::INFINITY;
        for (u, &u_in) in in_tree.iter().enumerate() {
            if !u_in {
                continue;
            }
            for (v, &v_in) in in_tree.iter().enumerate() {
                if !v_in {
                    let d = (terms[u].x - terms[v].x).hypot(terms[u].y - terms[v].y);
                    if d < best_dist {
                        best_dist = d;
                        best_pair = (u, v);
                    }
                }
            }
        }
        if best_dist.is_finite() {
            let (u, v) = best_pair;
            in_tree[v] = true;
            let w_uv = best_dist.max(1e-6);
            graph.add_edge(nodes[u], nodes[v], w_uv);
            segments.push(([terms[u].x, terms[u].y], [terms[v].x, terms[v].y]));
            total_cost += w_uv;
        }
    }
    assert_eq!(nodes.len(), k);
    debug_assert!(total_cost >= 0.0);
    (graph, nodes, segments, total_cost)
}

#[verified_engine::verified]
fn compute_mean_dijkstra_stretch(
    graph: &Graph<[f64; 2], f64>,
    term_nodes: &[NodeIndex],
    terms: &[PhysarumTerminal],
) -> (f64, f64) {
    let k = terms.len();
    if k < 2 {
        return (0.0, 1.0);
    }
    let total_pairs = (k * (k - 1)) / 2;
    let mut connected_pairs = 0usize;
    let mut stretch_sum = 0.0;

    for i in 0..k {
        let res = dijkstra(&graph.graph, term_nodes[i]);
        for j in (i + 1)..k {
            if let Some(&dist_g) = res.distances.get(&term_nodes[j]) {
                let euclid = (terms[i].x - terms[j].x)
                    .hypot(terms[i].y - terms[j].y)
                    .max(1.0);
                stretch_sum += (dist_g / euclid).max(1.0);
                connected_pairs += 1;
            }
        }
    }
    let mean_stretch = if connected_pairs > 0 {
        stretch_sum / (connected_pairs as f64)
    } else {
        0.0
    };
    let frac = (connected_pairs as f64) / (total_pairs as f64);
    debug_assert!((0.0..=1.0).contains(&frac));
    assert!(mean_stretch.is_finite());
    (mean_stretch, frac)
}

#[verified_engine::verified]
fn compute_single_edge_fault_survival(
    node_count: usize,
    edges: &[PhysarumSkeletonEdge],
    term_nodes: &[usize],
) -> (f64, f64) {
    let k = term_nodes.len();
    if edges.is_empty() || k < 2 || node_count == 0 {
        return (0.0, 0.0);
    }
    let total_pairs = ((k * (k - 1)) / 2) as f64;
    let mut full_survived = 0usize;
    let mut pair_ratio_sum = 0.0;
    let mut parent = vec![0usize; node_count];

    for skip_idx in 0..edges.len() {
        for (idx, slot) in parent.iter_mut().enumerate() {
            *slot = idx;
        }
        for (e_idx, edge) in edges.iter().enumerate() {
            if e_idx != skip_idx {
                union_nodes(&mut parent, edge.u, edge.v);
            }
        }
        let mut reachable_pairs = 0usize;
        for i in 0..k {
            let ri = find_root(&mut parent, term_nodes[i]);
            for &tj in &term_nodes[(i + 1)..k] {
                if ri == find_root(&mut parent, tj) {
                    reachable_pairs += 1;
                }
            }
        }
        if (reachable_pairs as f64) >= total_pairs {
            full_survived += 1;
        }
        pair_ratio_sum += (reachable_pairs as f64) / total_pairs;
    }
    let denom = edges.len() as f64;
    debug_assert!(denom >= 1.0);
    assert!(full_survived <= edges.len());
    ((full_survived as f64) / denom, pair_ratio_sum / denom)
}

#[verified_engine::verified]
fn find_root(parent: &mut [usize], mut x: usize) -> usize {
    let limit = parent.len();
    for _ in 0..limit {
        let px = parent[x];
        if px == x {
            return x;
        }
        parent[x] = parent[px];
        x = px;
    }
    debug_assert!(x < parent.len());
    assert!(!parent.is_empty());
    x
}

#[verified_engine::verified]
fn union_nodes(parent: &mut [usize], a: usize, b: usize) {
    let ra = find_root(parent, a);
    let rb = find_root(parent, b);
    if ra != rb {
        parent[rb] = ra;
    }
    debug_assert!(ra < parent.len());
    assert!(rb < parent.len());
}

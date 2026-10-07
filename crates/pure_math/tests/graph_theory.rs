#![allow(missing_docs)]
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use pure_math::pure_math::graph_theory::{
    graph::Graph,
    layout::{circular_layout, force_directed_layout, grid_layout},
    parameters::{degree::degeneracy, modulator::vertex_cover, treewidth::treewidth},
};

#[test]
#[verified_engine::verified]
fn test_degeneracy_k5() {
    // Create a K5 graph (a clique of 5 vertices).
    // The degeneracy of a K5 graph is 4.
    let mut g: Graph<(), ()> = Graph::new();
    let nodes: Vec<NodeIndex> = (0..5).map(|_| g.add_node(())).collect();

    for i in 0..5 {
        for j in (i + 1)..5 {
            g.add_edge(nodes[i], nodes[j], ());
        }
    }
    assert_eq!(degeneracy(&g), 4);
}

#[test]
#[verified_engine::verified]
fn test_degeneracy_path() {
    // Create a path graph P5: 0 -- 1 -- 2 -- 3 -- 4
    // The degeneracy of a path graph is 1.
    let mut g: Graph<(), ()> = Graph::new();
    let nodes: Vec<NodeIndex> = (0..5).map(|_| g.add_node(())).collect();
    g.add_edge(nodes[0], nodes[1], ());
    g.add_edge(nodes[1], nodes[2], ());
    g.add_edge(nodes[2], nodes[3], ());
    g.add_edge(nodes[3], nodes[4], ());
    assert_eq!(degeneracy(&g), 1);
}

#[test]
#[verified_engine::verified]
fn test_degeneracy_empty() {
    let g: Graph<(), ()> = Graph::new();
    assert_eq!(degeneracy(&g), 0);
}

#[test]
#[verified_engine::verified]
fn test_vertex_cover() {
    // Create a star graph with a center and 3 leaves.
    // The minimum vertex cover is {center}, size 1.
    let mut g: Graph<(), ()> = Graph::new();
    let center = g.add_node(());
    let n1 = g.add_node(());
    let n2 = g.add_node(());
    let n3 = g.add_node(());
    g.add_edge(center, n1, ());
    g.add_edge(center, n2, ());
    g.add_edge(center, n3, ());

    let cover = vertex_cover(&g);

    // Verify that it is a valid vertex cover.
    for edge in g.graph.edge_references() {
        let u = edge.source();
        let v = edge.target();
        assert!(cover.contains(&u) || cover.contains(&v));
    }
}

#[test]
#[verified_engine::verified]
fn test_treewidth_placeholder() {
    let mut g: Graph<(), ()> = Graph::new();
    let n1 = g.add_node(());
    let n2 = g.add_node(());
    g.add_edge(n1, n2, ());
    assert_eq!(treewidth(&g), 0);
}

#[test]
#[verified_engine::verified]
fn test_circular_layout() {
    let bounds = (0.0, 0.0, 200.0, 200.0);

    // Empty graph
    let g_empty: Graph<(), ()> = Graph::new();
    let res_empty = circular_layout(&g_empty, bounds);
    assert!(res_empty.is_empty());

    // Single node
    let mut g_single: Graph<(), ()> = Graph::new();
    let n_single = g_single.add_node(());
    let res_single = circular_layout(&g_single, bounds);
    assert_eq!(res_single.len(), 1);
    let pos_single = res_single[&n_single];
    assert!((pos_single.0 - 100.0).abs() < 1e-4);
    assert!((pos_single.1 - 100.0).abs() < 1e-4);

    // 5 nodes
    let mut g_multi: Graph<(), ()> = Graph::new();
    let nodes: Vec<_> = (0..5).map(|_| g_multi.add_node(())).collect();
    let res_multi = circular_layout(&g_multi, bounds);
    assert_eq!(res_multi.len(), 5);

    // Verify all nodes are within bounds and non-overlapping
    for i in 0..nodes.len() {
        let p_i = res_multi[&nodes[i]];
        assert!(p_i.0 >= 0.0 && p_i.0 <= 200.0);
        assert!(p_i.1 >= 0.0 && p_i.1 <= 200.0);

        for j in (i + 1)..nodes.len() {
            let p_j = res_multi[&nodes[j]];
            let dist = ((p_i.0 - p_j.0).powi(2) + (p_i.1 - p_j.1).powi(2)).sqrt();
            assert!(dist > 1.0, "Nodes should not overlap");
        }
    }
}

#[test]
#[verified_engine::verified]
fn test_grid_layout() {
    let bounds = (0.0, 0.0, 300.0, 300.0);

    // Empty graph
    let g_empty: Graph<(), ()> = Graph::new();
    assert!(grid_layout(&g_empty, bounds).is_empty());

    // Single node
    let mut g_single: Graph<(), ()> = Graph::new();
    let n_single = g_single.add_node(());
    let res_single = grid_layout(&g_single, bounds);
    assert_eq!(res_single.len(), 1);
    assert!(res_single.contains_key(&n_single));

    // 6 nodes (grid layout 3x2)
    let mut g_multi: Graph<(), ()> = Graph::new();
    let nodes: Vec<_> = (0..6).map(|_| g_multi.add_node(())).collect();
    let res_multi = grid_layout(&g_multi, bounds);
    assert_eq!(res_multi.len(), 6);

    for i in 0..nodes.len() {
        let p_i = res_multi[&nodes[i]];
        assert!(p_i.0 >= 0.0 && p_i.0 <= 300.0);
        assert!(p_i.1 >= 0.0 && p_i.1 <= 300.0);

        for j in (i + 1)..nodes.len() {
            let p_j = res_multi[&nodes[j]];
            let dist = ((p_i.0 - p_j.0).powi(2) + (p_i.1 - p_j.1).powi(2)).sqrt();
            assert!(dist > 1.0, "Grid nodes should not overlap");
        }
    }
}

#[test]
#[verified_engine::verified]
fn test_force_directed_layout() {
    let bounds = (0.0, 0.0, 400.0, 400.0);

    // Empty graph
    let g_empty: Graph<(), ()> = Graph::new();
    assert!(force_directed_layout(&g_empty, bounds, 50).is_empty());

    // Triangle + 1 disconnected node
    let mut g: Graph<(), ()> = Graph::new();
    let n0 = g.add_node(());
    let n1 = g.add_node(());
    let n2 = g.add_node(());
    let n3 = g.add_node(());

    g.add_edge(n0, n1, ());
    g.add_edge(n1, n2, ());
    g.add_edge(n2, n0, ());

    let res = force_directed_layout(&g, bounds, 50);
    assert_eq!(res.len(), 4);

    let nodes = [n0, n1, n2, n3];
    for i in 0..nodes.len() {
        let p_i = res[&nodes[i]];
        assert!(p_i.0.is_finite() && p_i.1.is_finite());
        assert!(p_i.0 >= 0.0 && p_i.0 <= 400.0);
        assert!(p_i.1 >= 0.0 && p_i.1 <= 400.0);

        for j in (i + 1)..nodes.len() {
            let p_j = res[&nodes[j]];
            let dist = ((p_i.0 - p_j.0).powi(2) + (p_i.1 - p_j.1).powi(2)).sqrt();
            assert!(dist > 0.1, "Force directed nodes should not overlap");
        }
    }
}

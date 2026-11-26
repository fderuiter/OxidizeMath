//! Integration tests for zero-allocation network flow, Kruskal MST, and graph interchange codecs.

use pure_math::pure_math::graph_theory::{
    GraphTheoryError,
    interchange::{
        DotEdgeStyle, DotNodeStyle, DotStyleOverlay, InterchangeGraph, InterchangeNode,
        MAX_INTERCHANGE_EDGES, MAX_INTERCHANGE_VERTICES, parse_dot, parse_edge_csv, parse_graphml,
        to_dot, to_edge_csv, to_graphml, to_styled_dot,
    },
    max_flow_min_cut::{EdmondsKarpScratch, max_flow_min_cut, solve_max_flow_min_cut_in_place},
    minimum_spanning_tree::{
        KruskalScratch, minimum_spanning_tree, solve_minimum_spanning_tree_in_place,
    },
};
use verified_engine::engine::VerifiedEngine;

#[test]
#[verified_engine::verified]
fn test_edmonds_karp_max_flow_equals_min_cut_zero_alloc() {
    // Classic CLRS 6-vertex directed network: s=0, t=5, max flow = 23.
    let edges = [
        (0, 1, 16.0),
        (0, 2, 13.0),
        (1, 2, 10.0),
        (2, 1, 4.0),
        (1, 3, 12.0),
        (3, 2, 9.0),
        (2, 4, 14.0),
        (4, 3, 7.0),
        (3, 5, 20.0),
        (4, 5, 4.0),
    ];
    let mut scratch = EdmondsKarpScratch::with_capacity(6, edges.len());
    scratch
        .prepare(6, &edges, true)
        .expect("valid flow topology");
    let mut flows = [0.0; 10];
    let mut cut_mask = [false; 6];

    let (res, _) = VerifiedEngine::run_verified(|| {
        solve_max_flow_min_cut_in_place(&mut scratch, 0, 5, &mut flows, &mut cut_mask)
    });
    let summary = res.expect("max-flow succeeded");
    assert!((summary.max_flow - 23.0).abs() < 1e-9);
    assert!((summary.min_cut_capacity - 23.0).abs() < 1e-9);
    assert!(cut_mask[0] && !cut_mask[5]);

    let full = max_flow_min_cut(6, &edges, 0, 5, true).expect("ergonomic max-flow");
    assert!((full.max_flow - 23.0).abs() < 1e-9);
    assert!((full.min_cut_capacity - 23.0).abs() < 1e-9);
    assert!(!full.steps.is_empty());
    let cut_sum: f64 = full.cut_edge_indices.iter().map(|&idx| edges[idx].2).sum();
    assert!((cut_sum - full.max_flow).abs() < 1e-9);
}

#[test]
#[verified_engine::verified]
fn test_edmonds_karp_undirected_and_validation() {
    let edges = [(0, 1, 3.0), (1, 2, 2.0), (0, 2, 4.0), (2, 1, 1.0)];
    let res = max_flow_min_cut(3, &edges, 0, 1, false).expect("undirected flow");
    // Direct (0,1)=3.0 + via node 2: min(4.0, 2.0 + 1.0) = 3.0 -> total 6.0.
    assert!((res.max_flow - 6.0).abs() < 1e-9);
    assert!((res.min_cut_capacity - 6.0).abs() < 1e-9);

    assert!(matches!(
        max_flow_min_cut(3, &edges, 1, 1, true),
        Err(GraphTheoryError::SourceEqualsSink { node: 1 })
    ));
    assert!(matches!(
        max_flow_min_cut(3, &[(0, 1, -2.0)], 0, 2, true),
        Err(GraphTheoryError::InvalidCapacity { edge_index: 0, .. })
    ));
}

#[test]
#[verified_engine::verified]
fn test_kruskal_mst_exact_weight_and_zero_alloc() {
    let edges = [
        (0, 1, 7.0),
        (0, 3, 5.0),
        (1, 2, 8.0),
        (1, 3, 9.0),
        (1, 4, 7.0),
        (2, 4, 5.0),
        (3, 4, 15.0),
        (3, 5, 6.0),
        (4, 5, 8.0),
        (4, 6, 9.0),
        (5, 6, 11.0),
    ];
    let mut scratch = KruskalScratch::with_capacity(7, edges.len());
    scratch.reserve_for(7, edges.len());
    let mut selected = [false; 11];

    let (res, _) = VerifiedEngine::run_verified(|| {
        solve_minimum_spanning_tree_in_place(&mut scratch, 7, &edges, &mut selected)
    });
    let summary = res.expect("kruskal in place");
    // Classic 7-vertex MST weight: 5 + 5 + 6 + 7 + 7 + 9 = 39.
    assert!((summary.total_weight - 39.0).abs() < 1e-9);
    assert_eq!(summary.selected_edge_count, 6);
    assert_eq!(summary.component_count, 1);

    let full = minimum_spanning_tree(7, &edges).expect("ergonomic mst");
    assert!((full.total_weight - 39.0).abs() < 1e-9);
    assert_eq!(full.tree_edge_indices.len(), 6);
    assert_eq!(full.steps.len(), edges.len());
}

#[test]
#[verified_engine::verified]
fn test_interchange_dot_graphml_csv_round_trip_and_styled_dot() {
    let mut g = InterchangeGraph::new(true);
    g.insert_node(InterchangeNode {
        id: 0,
        pos: Some((120.5, 240.0)),
        label: Some("s".to_string()),
    })
    .expect("node 0");
    g.insert_node(InterchangeNode {
        id: 1,
        pos: Some((300.0, 120.25)),
        label: Some("u".to_string()),
    })
    .expect("node 1");
    g.insert_node(InterchangeNode {
        id: 2,
        pos: Some((480.0, 240.0)),
        label: Some("t".to_string()),
    })
    .expect("node 2");
    g.push_edge(0, 1, 3.5).expect("e0");
    g.push_edge(1, 2, 2.25).expect("e1");
    g.push_edge(0, 2, 1.75).expect("e2");

    let dot = to_dot(&g).expect("to_dot");
    let from_dot = parse_dot(&dot).expect("parse_dot");
    assert!(from_dot.directed);
    assert_eq!(from_dot.nodes.len(), 3);
    assert_eq!(from_dot.edges, g.edges);
    assert_eq!(from_dot.nodes[&0].pos, Some((120.5, 240.0)));

    let xml = to_graphml(&g).expect("to_graphml");
    let from_xml = parse_graphml(&xml).expect("parse_graphml");
    assert!(from_xml.directed);
    assert_eq!(from_xml.nodes.len(), 3);
    assert_eq!(from_xml.edges, g.edges);
    assert_eq!(from_xml.nodes[&1].pos, Some((300.0, 120.25)));

    let csv = to_edge_csv(&g).expect("to_edge_csv");
    let from_csv = parse_edge_csv(&csv).expect("parse_edge_csv");
    assert!(from_csv.directed);
    assert_eq!(from_csv.edges, g.edges);
    assert_eq!(from_csv.nodes[&0].pos, Some((120.5, 240.0)));

    let mut overlay = DotStyleOverlay {
        graph_label: Some("Min-Cut Overlay".to_string()),
        ..Default::default()
    };
    overlay.node_styles.insert(
        0,
        DotNodeStyle {
            color: Some("#1f77b4".to_string()),
            fillcolor: Some("#aec7e8".to_string()),
            penwidth: Some(2.5),
            xlabel: Some("S".to_string()),
        },
    );
    overlay.edge_styles.insert(
        1,
        DotEdgeStyle {
            color: Some("#d62728".to_string()),
            penwidth: Some(4.0),
            xlabel: Some("cut (2.25)".to_string()),
        },
    );
    let styled = to_styled_dot(&g, &overlay).expect("styled dot");
    assert!(styled.contains("color=\"#d62728\""));
    assert!(styled.contains("penwidth=4.0"));
    assert!(styled.contains("xlabel=\"cut (2.25)\""));
    assert_eq!(
        parse_dot(&styled).expect("reparse styled dot").edges,
        g.edges
    );
}

#[test]
#[verified_engine::verified]
fn test_interchange_rejects_non_finite_and_oversized_inputs() {
    let mut g = InterchangeGraph::new(false);
    assert!(matches!(
        g.push_edge(0, 1, f64::NAN),
        Err(GraphTheoryError::InvalidWeight { .. })
    ));
    assert!(matches!(
        g.insert_node(InterchangeNode {
            id: 0,
            pos: Some((f64::INFINITY, 0.0)),
            label: None,
        }),
        Err(GraphTheoryError::InvalidCoordinate { node: 0, .. })
    ));

    let mut big = InterchangeGraph::new(false);
    for id in 0..MAX_INTERCHANGE_VERTICES {
        big.ensure_node(id).expect("within vertex cap");
    }
    assert!(matches!(
        big.ensure_node(MAX_INTERCHANGE_VERTICES),
        Err(GraphTheoryError::VertexLimitExceeded {
            max: MAX_INTERCHANGE_VERTICES,
            ..
        })
    ));

    let mut dense = InterchangeGraph::new(true);
    for e in 0..MAX_INTERCHANGE_EDGES {
        dense
            .push_edge(e % 64, (e * 7 + 1) % 64, 1.0)
            .expect("within edge cap");
    }
    assert!(matches!(
        dense.push_edge(0, 1, 1.0),
        Err(GraphTheoryError::EdgeLimitExceeded {
            max: MAX_INTERCHANGE_EDGES,
            ..
        })
    ));
}

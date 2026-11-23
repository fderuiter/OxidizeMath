#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::pure_math::graph_theory::parameters::degree::degeneracy;
use crate::pure_math::graph_theory::parameters::modulator::vertex_cover;
use petgraph::visit::EdgeRef;

fn sample_weighted_wheel_plus_isolate() -> GraphTopologyDocument {
    let mut doc = GraphTopologyDocument::new("Wheel7WithIsolate", false);
    doc.attributes.insert(
        "source_suite".to_string(),
        "pure_math_roundtrip".to_string(),
    );
    let coords = [
        ("0", "Hub 0", 300.0, 220.0, TopologyNodeShape::DoubleCircle),
        ("1", "Rim 1", 420.0, 220.0, TopologyNodeShape::Circle),
        ("2", "Rim 2", 360.0, 324.0, TopologyNodeShape::Circle),
        ("3", "Rim 3", 240.0, 324.0, TopologyNodeShape::Circle),
        ("4", "Rim 4", 180.0, 220.0, TopologyNodeShape::Circle),
        ("5", "Rim 5", 240.0, 116.0, TopologyNodeShape::Circle),
        ("6", "Rim 6", 360.0, 116.0, TopologyNodeShape::Circle),
        ("7", "Isolate 7", 500.0, 110.0, TopologyNodeShape::Box),
    ];
    for (id, label, x, y, shape) in coords {
        let mut node = TopologyNode::new(id, label).with_position(x, y);
        node.shape = shape;
        node.fill_hex = Some("#1E3A8A".to_string());
        node.stroke_hex = Some("#60A5FA".to_string());
        doc.nodes.push(node);
    }
    for rim in 1..=6 {
        let mut spoke = TopologyEdge::new("0", rim.to_string(), 1.5 + f64::from(rim) * 0.25);
        spoke.label = Some(format!("s{rim}"));
        doc.edges.push(spoke);
        let next = if rim == 6 { 1 } else { rim + 1 };
        let mut cycle = TopologyEdge::new(rim.to_string(), next.to_string(), 2.0);
        cycle.style = TopologyEdgeStyle::Dashed;
        doc.edges.push(cycle);
    }
    doc
}

fn assert_valid_vertex_cover<N, E>(
    g: &crate::pure_math::graph_theory::graph::Graph<N, E>,
    cover: &std::collections::HashSet<petgraph::graph::NodeIndex>,
) {
    for edge in g.graph.edge_references() {
        assert!(
            cover.contains(&edge.source()) || cover.contains(&edge.target()),
            "Edge ({:?}, {:?}) uncovered by returned vertex_cover",
            edge.source(),
            edge.target()
        );
    }
}

fn assert_parsed_matches_wheel(
    fmt_name: &str,
    parsed: &GraphTopologyDocument,
    expected_nodes: usize,
    expected_weights: &[f64],
    expected_deg: usize,
) {
    assert_eq!(
        parsed.nodes.len(),
        expected_nodes,
        "{fmt_name} node count mismatch"
    );
    assert_eq!(
        parsed.edges.len(),
        expected_weights.len(),
        "{fmt_name} edge count mismatch"
    );
    let weights: Vec<f64> = parsed.edges.iter().map(|e| e.weight).collect();
    assert_eq!(weights, expected_weights, "{fmt_name} weights mismatch");

    let (pg, _) = parsed
        .to_undirected_petgraph()
        .unwrap_or_else(|e| panic!("{fmt_name} petgraph conversion failed: {e}"));
    assert_eq!(
        degeneracy(&pg),
        expected_deg,
        "{fmt_name} degeneracy mismatch"
    );
    let vc = vertex_cover(&pg);
    assert_valid_vertex_cover(&pg, &vc);
    assert!(!vc.is_empty() && vc.len() <= expected_nodes);
}

#[test]
fn test_dot_graphml_csv_roundtrips_preserve_counts_weights_degeneracy_and_vertex_cover() {
    let original = sample_weighted_wheel_plus_isolate();
    let (orig_pg, _) = original.to_undirected_petgraph().expect("valid original");
    let expected_deg = degeneracy(&orig_pg);
    assert_eq!(expected_deg, 3, "6-wheel + isolate has degeneracy 3");
    assert_valid_vertex_cover(&orig_pg, &vertex_cover(&orig_pg));
    let orig_weights: Vec<f64> = original.edges.iter().map(|e| e.weight).collect();

    let dot_text = original
        .serialize(TopologyFormat::Dot)
        .expect("DOT serialize");
    let graphml_text = original
        .serialize(TopologyFormat::GraphMl)
        .expect("GraphML serialize");
    let csv_text = original
        .serialize(TopologyFormat::Csv)
        .expect("CSV serialize");

    let from_dot =
        GraphTopologyDocument::parse_with_filename("wheel.dot", &format!("\u{feff}{dot_text}"))
            .expect("DOT parse");
    let from_graphml = GraphTopologyDocument::parse_with_filename(
        "wheel.graphml",
        &format!("\u{feff}{graphml_text}"),
    )
    .expect("GraphML parse");
    let from_csv =
        GraphTopologyDocument::parse_with_filename("wheel.csv", &format!("\u{feff}{csv_text}"))
            .expect("CSV parse");

    for (fmt_name, parsed) in [
        ("DOT", &from_dot),
        ("GraphML", &from_graphml),
        ("CSV", &from_csv),
    ] {
        assert_parsed_matches_wheel(
            fmt_name,
            parsed,
            original.nodes.len(),
            &orig_weights,
            expected_deg,
        );
    }

    assert_eq!(dot_text, original.serialize(TopologyFormat::Dot).unwrap());
    assert_eq!(
        graphml_text,
        original.serialize(TopologyFormat::GraphMl).unwrap()
    );
    assert_eq!(csv_text, original.serialize(TopologyFormat::Csv).unwrap());
}

#[test]
fn test_tikz_and_svg_exports_include_directed_multiedges_and_self_loops() {
    let mut doc = GraphTopologyDocument::new("Minsky_PrimeGame & Special_Chars", true);
    doc.nodes.push(TopologyNode::new("q0", "q_0 (Start)"));
    doc.nodes.push(TopologyNode::new("q2", "q_2 #100%"));
    let mut e1 = TopologyEdge::new("q0", "q2", 1.0);
    e1.label = Some("R1: 17/91".to_string());
    let mut e2 = TopologyEdge::new("q2", "q0", 1.0);
    e2.label = Some("R2: 78/85".to_string());
    let mut loop_edge = TopologyEdge::new("q2", "q2", 2.5);
    loop_edge.label = Some("R3: 19/51".to_string());
    doc.edges.extend([e1, e2, loop_edge]);

    let tikz = doc.serialize(TopologyFormat::Tikz).expect("TikZ serialize");
    assert!(tikz.contains("\\documentclass[tikz,border=10pt]{standalone}"));
    assert!(tikz.contains("\\begin{tikzpicture}"));
    assert!(tikz.contains("bend left="));
    assert!(tikz.contains("loop above"));
    assert!(tikz.contains("\\#100\\%"));

    let svg = doc.serialize(TopologyFormat::Svg).expect("SVG serialize");
    assert!(svg.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\""));
    assert!(svg.contains("marker-end=\"url(#topo-arrow)\""));
    assert!(svg.contains("<path d=\"M "));
    assert!(svg.contains("&amp;"));
}

#[test]
fn test_automatic_layout_fallback_for_coordinate_free_tsv_and_dot() {
    let tsv = "u\tv\tw\nA\tB\t3.5\nB\tC\t1.25\nC\tA\t2.0\n";
    let parsed_tsv =
        GraphTopologyDocument::parse_with_filename("edges.tsv", tsv).expect("TSV parse");
    assert_eq!(parsed_tsv.nodes.len(), 3);
    assert_eq!(parsed_tsv.edges.len(), 3);
    for node in &parsed_tsv.nodes {
        let [x, y] = node.position.expect("fallback position must be assigned");
        assert!(x.is_finite() && y.is_finite());
    }

    let compact_dot = "strict digraph Compact { A->B->C [weight=4.0]; }";
    let parsed_dot =
        GraphTopologyDocument::parse_with_filename("compact.dot", compact_dot).expect("DOT parse");
    assert!(parsed_dot.directed);
    assert_eq!(parsed_dot.nodes.len(), 3);
    assert_eq!(parsed_dot.edges.len(), 2);
    assert_eq!(parsed_dot.edges[0].weight, 4.0);
}

#[test]
fn test_security_bounds_reject_oversized_payloads_nodes_edges_and_xml_entities() {
    let xxe_doctype = "<?xml version=\"1.0\"?>\n<!DOCTYPE graphml [\n<!ENTITY lol \"boom\">\n]>\n<graphml><graph id=\"G\"><node id=\"0\"/></graph></graphml>";
    assert_eq!(
        parse_graphml_topology(xxe_doctype),
        Err(TopologyCodecError::ForbiddenXmlDeclaration)
    );

    let entity_only = "<graphml><!entity secret SYSTEM \"file:///etc/passwd\"><graph id=\"G\"><node id=\"0\"/></graph></graphml>";
    assert_eq!(
        GraphTopologyDocument::parse_with_filename("evil.graphml", entity_only),
        Err(TopologyCodecError::ForbiddenXmlDeclaration)
    );

    let oversized_bytes = "0,1\n".repeat((MAX_TOPOLOGY_IMPORT_BYTES / 4) + 10);
    assert!(matches!(
        GraphTopologyDocument::parse_with_filename("huge.csv", &oversized_bytes),
        Err(TopologyCodecError::PayloadTooLarge { .. })
    ));

    let mut too_many_nodes_csv = String::from("source,target\n");
    for i in 0..=MAX_TOPOLOGY_NODES {
        use std::fmt::Write as _;
        let _ = writeln!(too_many_nodes_csv, "n{i},n0");
    }
    assert!(matches!(
        parse_csv_topology(&too_many_nodes_csv),
        Err(TopologyCodecError::TooManyNodes { .. })
    ));

    let mut too_many_edges_dot = String::from("digraph Dense {\n");
    for i in 0..=MAX_TOPOLOGY_EDGES {
        let u = i % 80;
        let v = (i * 7 + 1) % 80;
        use std::fmt::Write as _;
        let _ = writeln!(too_many_edges_dot, "  \"n{u}\" -> \"n{v}\";");
    }
    too_many_edges_dot.push_str("}\n");
    assert!(matches!(
        parse_dot_topology(&too_many_edges_dot),
        Err(TopologyCodecError::TooManyEdges { .. })
    ));
}

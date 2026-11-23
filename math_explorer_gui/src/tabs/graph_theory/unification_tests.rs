#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{
    GraphTheoryTab, TopologyHandoffTarget,
    algorithm_visualizer::{Algorithm, AlgorithmVisualizerTool},
    graph_editor::GraphEditorTool,
    network_metrics::NetworkMetricsTool,
    parse_topology_document,
    shared_topology::queue_topology_handoff,
};
use crate::framework::InteractiveTool;
use crate::tabs::ExplorerTab;
use crate::url_state::{MAX_URL_FRAGMENT_BYTES, URL_JSON_BOUNDS};
use eframe::egui;

#[test]
fn test_cross_tool_topology_handoff_editor_visualizer_metrics() {
    let mut editor = GraphEditorTool::default();
    assert!(!editor.can_undo() && !editor.can_redo());
    editor.load_braess_preset(true);
    let shared = editor.export_shared_topology();
    assert_eq!(shared.nodes.len(), 4);
    assert_eq!(shared.edges.len(), 5);
    assert!(shared.directed);

    let mut visualizer = AlgorithmVisualizerTool::default();
    visualizer.apply_shared_topology(&shared);
    visualizer.set_algorithm(Algorithm::MaxFlowMinCut);
    visualizer.set_animation_step(visualizer.total_steps());
    let flow = visualizer.flow_visualization().expect("flow solved");
    // Braess diamond edges: (0,1,0), (1,3,1), (0,2,1), (2,3,0), (1,2,0) -> max flow 0 along zero-cap edges,
    // verify min_cut_capacity == max_flow exact equality.
    assert!((flow.max_flow - flow.min_cut_capacity).abs() < 1e-9);

    let mut metrics = NetworkMetricsTool::default();
    metrics.apply_shared_topology(&shared);
    // 4-node diamond + diagonal = K_4 minus 1 edge (5 edges on 4 nodes): treewidth == 2, degeneracy == 2.
    assert_eq!(metrics.degeneracy_value(), 2);
    assert_eq!(metrics.treewidth_value(), 2);
}

#[test]
fn test_graph_theory_tab_processes_pending_handoff() {
    let ctx = egui::Context::default();
    let mut tab = GraphTheoryTab::default();
    assert!(tab.select_tool_by_name("Graph Editor"));

    let mut editor = GraphEditorTool::default();
    editor.load_braess_preset(false);
    let topo = editor.export_shared_topology();

    queue_topology_handoff(&ctx, TopologyHandoffTarget::NetworkMetrics, topo.clone());
    tab.process_pending_handoff(&ctx);

    assert_eq!(tab.shared_topology.nodes.len(), 4);
    let active = tab
        .framework()
        .and_then(|fw| fw.active_tool.as_ref())
        .expect("active tool");
    assert_eq!(active.name(), "Network Metrics");
    let metrics = active
        .as_any()
        .and_then(|a| a.downcast_ref::<NetworkMetricsTool>())
        .expect("downcast NetworkMetricsTool");
    // 4-cycle C_4 has treewidth 2 and degeneracy 2.
    assert_eq!(metrics.treewidth_value(), 2);
    assert_eq!(metrics.degeneracy_value(), 2);
}

#[test]
fn test_network_metrics_default_wheel_invariants_and_exports() {
    let tool = NetworkMetricsTool::default();
    assert_eq!(tool.diameter_value(), Some(2));
    assert_eq!(tool.degeneracy_value(), 3);
    assert_eq!(tool.treewidth_value(), 3);
    assert!(!tool.vertex_cover_nodes().is_empty());

    let dot = tool.to_styled_dot_string().expect("styled dot");
    assert!(dot.contains("Treewidth=3") && dot.contains("Degeneracy=3"));
    assert!(dot.contains("fillcolor=\"#"));

    let xml = tool.to_graphml_string().expect("graphml");
    assert!(xml.contains("<graphml") && xml.contains("<edge"));

    let csv = tool.to_invariants_csv_string().expect("invariants csv");
    assert!(csv.contains("treewidth=3") && csv.contains("in_vertex_cover"));
}

#[test]
fn test_file_import_parsers_and_url_bounds_round_trip() {
    let dot_src = "digraph Net { 0 -> 1 [weight=4.0]; 0 -> 2 [weight=2.0]; 1 -> 2 [weight=3.0]; 1 -> 3 [weight=1.0]; 2 -> 3 [weight=5.0]; }";
    let topo = parse_topology_document("capacity.dot", dot_src).expect("parse dot");
    assert!(topo.directed);
    assert_eq!(topo.nodes.len(), 4);
    assert_eq!(topo.edges.len(), 5);

    let mut vis = AlgorithmVisualizerTool::default();
    vis.apply_shared_topology(&topo);
    vis.set_algorithm(Algorithm::MaxFlowMinCut);
    let url_json = vis.save_url_state().expect("url state");
    assert!(url_json.len() < MAX_URL_FRAGMENT_BYTES);
    URL_JSON_BOUNDS
        .validate_json(&url_json)
        .expect("fits URL_JSON_BOUNDS");

    let mut restored = AlgorithmVisualizerTool::default();
    restored
        .try_load_state(&url_json)
        .expect("restore custom url state");
    let flow = restored.flow_visualization().expect("restored max flow");
    // Paths: 0->1->3 (cap 1) + 0->1->2->3 (cap 3) + 0->2->3 (cap 2) = 6.0.
    assert!((flow.max_flow - 6.0).abs() < 1e-9);
    assert!((flow.min_cut_capacity - 6.0).abs() < 1e-9);

    let json_doc = serde_json::to_string(&topo.to_compact_state()).expect("json");
    let from_json = parse_topology_document("custom.json", &json_doc).expect("parse json");
    assert_eq!(from_json.edges.len(), 5);
}

// [cite:graph_parameters_rust]

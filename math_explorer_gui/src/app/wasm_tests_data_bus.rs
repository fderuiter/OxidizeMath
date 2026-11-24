//! Headless browser `#[wasm_bindgen_test]` suite for multi-pane `WorkspaceDataBus` coupling.

use super::{MathExplorerApp, SharedGraph, SharedNode, ViewportLayout};
use crate::app::data_bus::{publish_shared_graph, read_epidemic_trace};
use crate::framework::InteractiveTool;
use crate::tabs::epidemiology::network_propagation::NetworkPropagationTool;
use crate::tabs::epidemiology::sir::SirTool;
use crate::tabs::game_theory::replicator::ReplicatorDynamicsTool;
use crate::tabs::graph_theory::algorithm_visualizer::AlgorithmVisualizerTool;
use crate::tabs::graph_theory::graph_editor::{EditorMode, GraphEditorTool};
use crate::tabs::graph_theory::network_metrics::NetworkMetricsTool;
use crate::tabs::neuroscience::neural_network_viz::NeuralNetworkVizTool;
use crate::tabs::solid_state::ising::IsingModelTool;
use eframe::egui;
use eframe::App;
use wasm_bindgen_test::wasm_bindgen_test;

fn find_wasm_tab(app: &MathExplorerApp, label: &str) -> usize {
    app.tabs
        .iter()
        .position(|t| t.name().contains(label))
        .unwrap_or_else(|| panic!("Missing tab {label}"))
}

#[wasm_bindgen_test]
fn wasm_quad_grid_and_split_multi_pane_graph_bus_propagation() {
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    let gt_idx = find_wasm_tab(&app, "Graph Theory");
    let epi_idx = find_wasm_tab(&app, "Epidemiology");
    let ss_idx = find_wasm_tab(&app, "Solid State");
    let game_idx = find_wasm_tab(&app, "Game Theory");

    app.set_layout(ViewportLayout::QuadGrid);
    app.set_pane_tab(0, epi_idx);
    app.set_pane_tab(1, ss_idx);
    app.set_pane_tab(2, game_idx);
    app.set_pane_tab(3, gt_idx);

    assert!(app
        .pane_tab_mut(0)
        .unwrap()
        .select_tool_by_name("Network Propagation"));
    assert!(app
        .pane_tab_mut(1)
        .unwrap()
        .select_tool_by_name("Ising Model"));
    assert!(app
        .pane_tab_mut(2)
        .unwrap()
        .select_tool_by_name("Replicator Dynamics"));
    let pane3_gt = app.pane_tab_mut(3).expect("Pane 3 Graph Theory tab");
    assert!(pane3_gt.select_tool_by_name("Graph Editor"));
    let triangle_json = r#"{"node_positions":{"0":[120.0,140.0],"1":[260.0,140.0],"2":[190.0,260.0]},"edges":[[0,1,1.5],[1,2,-1.0],[0,2,1.0]],"selected_node":null,"next_node_id":3}"#;
    pane3_gt
        .framework_mut()
        .and_then(|fw| fw.active_tool.as_mut())
        .expect("Graph Editor active")
        .try_load_state(triangle_json)
        .expect("valid sculpted graph");

    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1200.0_f32, 800.0_f32),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw.clone(), |c| app.update(c, &mut frame));
    assert!(app.data_bus.graph_revision() >= 1);

    // Second frame delivers Pane 3's published graph to Panes 0..2 and settles
    let _ = ctx.run(raw, |c| app.update(c, &mut frame));
    assert!(app.data_bus.shared_graph.is_some());
}

#[wasm_bindgen_test]
fn wasm_cross_domain_consumers_and_opt_out_isolation() {
    let ctx = egui::Context::default();
    let mut editor = GraphEditorTool::with_brush_config(EditorMode::HubStamp, 3, 1.5);
    editor.stamp_hub_at(egui::Pos2::new(180.0_f32, 180.0_f32));
    let _ = editor.upsert_signed_edge(1, 2, -1.0);

    let mut metrics = NetworkMetricsTool::default();
    let mut algo = AlgorithmVisualizerTool::default();
    let mut net_prop = NetworkPropagationTool::default();
    let mut sir = SirTool::default();
    let mut hh_viz = NeuralNetworkVizTool::default();
    let mut ising = IsingModelTool::default();
    let mut replicator = ReplicatorDynamicsTool::default();

    let _ = ctx.run(egui::RawInput::default(), |c| {
        egui::CentralPanel::default().show(c, |ui| {
            editor.show_inside(ui);
            metrics.show_inside(ui);
            algo.show_inside(ui);
            net_prop.show_inside(ui);
            sir.show_inside(ui);
            hh_viz.show_inside(ui);
            ising.show_inside(ui);
            replicator.show_inside(ui);
        });
    });

    assert_eq!(metrics.node_count(), 4);
    assert_eq!(metrics.diameter_value(), Some(2));
    assert_eq!(algo.visit_count(), 4);
    assert_eq!(hh_viz.neuron_count(), 4);
    assert!((hh_viz.synaptic_weight(1, 2) + 1.0).abs() < 1e-9);
    assert!(ising.is_graph_spin_mode());
    assert!(ising.follow_workspace_graph());
    assert!(replicator.follow_workspace_graph());
    assert!(!replicator.spatial_trajectory().is_empty());
    assert!(read_epidemic_trace(&ctx).is_some());

    // Isolate NetworkMetricsTool while bus advances to 6 nodes
    metrics.follow_workspace_graph = false;
    let next_graph = SharedGraph {
        revision: 10,
        nodes: (0..6)
            .map(|id| SharedNode {
                id,
                norm_pos: [0.5_f32, 0.5_f32],
            })
            .collect(),
        edges: vec![(0, 1, 1.0), (1, 2, 1.0)],
    };
    let _ = publish_shared_graph(&ctx, next_graph);
    let _ = ctx.run(egui::RawInput::default(), |c| {
        egui::CentralPanel::default().show(c, |ui| {
            metrics.show_inside(ui);
            algo.show_inside(ui);
        });
    });
    assert_eq!(metrics.node_count(), 4);
    assert_eq!(algo.visit_count(), 3);
}

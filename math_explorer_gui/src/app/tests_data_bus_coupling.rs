//! Multi-pane cross-domain coupling, isolation, node-budget, and `.mws` session tests.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::app::data_bus::{
    publish_shared_graph, read_epidemic_trace, read_pane_sync_meta, write_pane_sync_meta,
    SharedGraph, SharedNode, MAX_EPIDEMIC_NODE_BUDGET, MAX_GAME_NODE_BUDGET, MAX_HH_NEURON_BUDGET,
    MAX_SPIN_NODE_BUDGET,
};
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

fn find_tab_index(app: &MathExplorerApp, title: &str) -> usize {
    app.tabs
        .iter()
        .position(|t| t.name().contains(title))
        .unwrap_or_else(|| panic!("Missing tab '{title}'"))
}

fn run_app_frame(app: &mut MathExplorerApp, ctx: &egui::Context, frame: &mut eframe::Frame) {
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0_f32, 820.0_f32),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw, |c| app.update(c, frame));
}

fn make_signed_test_graph(revision: u64, num_nodes: usize) -> SharedGraph {
    let nodes = (0..num_nodes)
        .map(|i| {
            let frac = i as f32 / (num_nodes.max(2) - 1) as f32;
            SharedNode {
                id: i,
                norm_pos: [0.1_f32 + 0.8_f32 * frac, 0.2_f32 + 0.6_f32 * frac],
            }
        })
        .collect();
    let mut edges = Vec::new();
    for i in 0..num_nodes.saturating_sub(1) {
        let w = if i % 2 == 0 { 2.0 } else { -1.5 };
        edges.push((i, i + 1, w));
    }
    if num_nodes >= 3 {
        edges.push((0, 2, -1.0));
    }
    SharedGraph {
        revision,
        nodes,
        edges,
    }
}

#[test]
fn test_graph_editor_couples_all_six_tools_in_one_frame() {
    let ctx = egui::Context::default();
    let mut editor = GraphEditorTool::with_brush_config(EditorMode::HubStamp, 4, 2.0);
    editor.stamp_hub_at(egui::Pos2::new(220.0_f32, 220.0_f32));
    let _ = editor.upsert_signed_edge(1, 2, -1.25);

    let mut metrics = NetworkMetricsTool::default();
    let mut algo = AlgorithmVisualizerTool::default();
    let mut net_prop = NetworkPropagationTool::default();
    let mut hh_viz = NeuralNetworkVizTool::default();
    let mut ising = IsingModelTool::default();
    let mut replicator = ReplicatorDynamicsTool::default();

    let _ = ctx.run(egui::RawInput::default(), |c| {
        egui::CentralPanel::default().show(c, |ui| {
            editor.show_inside(ui);
            metrics.show_inside(ui);
            algo.show_inside(ui);
            net_prop.show_inside(ui);
            hh_viz.show_inside(ui);
            ising.show_inside(ui);
            replicator.show_inside(ui);
        });
    });

    assert_eq!(metrics.node_count(), 5);
    assert_eq!(metrics.diameter_value(), Some(2));
    assert_eq!(algo.visit_count(), 5);
    assert_eq!(hh_viz.neuron_count(), 5);
    assert!((hh_viz.synaptic_weight(1, 2) + 1.25).abs() < 1e-9);
    assert!(ising.is_graph_spin_mode());
    assert!(ising.frustrated_edge_count() >= 1);
    assert!(!replicator.spatial_trajectory().is_empty());
}

#[test]
fn test_quad_grid_and_split_layouts_propagate_reverse_pane_order() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    let gt_idx = find_tab_index(&app, "Graph Theory");
    let epi_idx = find_tab_index(&app, "Epidemiology");
    let ss_idx = find_tab_index(&app, "Solid State");
    let game_idx = find_tab_index(&app, "Game Theory");

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
    let triangle_json = r#"{"node_positions":{"0":[120.0,140.0],"1":[260.0,140.0],"2":[190.0,260.0]},"edges":[[0,1,1.5],[1,2,-1.0],[0,2,1.0]],"selected_node":null,"next_node_id":3}"#;
    let pane3 = app.pane_tab_mut(3).unwrap();
    assert!(pane3.select_tool_by_name("Graph Editor"));
    pane3
        .framework_mut()
        .and_then(|fw| fw.active_tool.as_mut())
        .unwrap()
        .try_load_state(triangle_json)
        .unwrap();

    run_app_frame(&mut app, &ctx, &mut frame);
    assert!(
        app.data_bus.graph_revision() >= 1,
        "Publisher in Pane 3 of QuadGrid must commit SharedGraph to MathExplorerApp.data_bus"
    );

    for layout in [
        ViewportLayout::SplitHorizontal,
        ViewportLayout::SplitVertical,
        ViewportLayout::Single,
    ] {
        app.set_layout(layout);
        run_app_frame(&mut app, &ctx, &mut frame);
    }
    let rev_before = app.data_bus.graph_revision();
    run_app_frame(&mut app, &ctx, &mut frame);
    assert_eq!(app.data_bus.graph_revision(), rev_before);
}

#[test]
fn test_network_propagation_heterogeneous_r0_and_sir_overlay() {
    let ctx = egui::Context::default();
    // Star graph K_{1,4}: hub degree 4, four leaves degree 1 -> <k> = 8/5 = 1.6, <k^2> = 20/5 = 4.0, Var(k) = 1.44
    let star = SharedGraph {
        revision: 3,
        nodes: (0..5)
            .map(|id| SharedNode {
                id,
                norm_pos: [0.5_f32, 0.5_f32],
            })
            .collect(),
        edges: vec![(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0)],
    };
    let _ = publish_shared_graph(&ctx, star);

    let mut net_prop = NetworkPropagationTool::default();
    let mut sir = SirTool::default();
    let _ = ctx.run(egui::RawInput::default(), |c| {
        egui::CentralPanel::default().show(c, |ui| {
            net_prop.show_inside(ui);
            net_prop.step_once();
            net_prop.show_inside(ui);
            sir.show_inside(ui);
        });
    });

    let (mean_k, var_k, r0_het) = net_prop.degree_moments_and_r0();
    assert!((mean_k - 1.6).abs() < 1e-9);
    assert!((var_k - 1.44).abs() < 1e-9);
    let expected_r0 = (0.05 / 0.02) * ((1.44 + 1.6_f64.powi(2) - 1.6) / 1.6);
    assert!((r0_het - expected_r0).abs() < 1e-9);

    let trace = read_epidemic_trace(&ctx).expect("EpidemicTraceSignal streamed to bus");
    assert_eq!(trace.population, 5);
    assert!(trace.infected.len() >= 2);
}

#[test]
fn test_follow_workspace_graph_opt_out_isolates_pane() {
    let ctx = egui::Context::default();
    let _ = publish_shared_graph(&ctx, make_signed_test_graph(1, 4));

    let mut metrics = NetworkMetricsTool::default();
    let mut ising = IsingModelTool::default();
    let mut replicator = ReplicatorDynamicsTool::default();

    let _ = ctx.run(egui::RawInput::default(), |c| {
        egui::CentralPanel::default().show(c, |ui| {
            metrics.show_inside(ui);
            ising.show_inside(ui);
            replicator.show_inside(ui);
        });
    });
    assert_eq!(metrics.node_count(), 4);
    assert!(ising.follow_workspace_graph());
    assert!(replicator.follow_workspace_graph());

    // Simulate pane-header Follow Workspace Graph toggle off on pane 0
    let mut meta = read_pane_sync_meta(&ctx);
    meta.pending_toggle[0] = Some(false);
    write_pane_sync_meta(&ctx, meta);
    metrics.follow_workspace_graph = false;

    let _ = publish_shared_graph(&ctx, make_signed_test_graph(2, 7));
    let _ = ctx.run(egui::RawInput::default(), |c| {
        egui::CentralPanel::default().show(c, |ui| {
            ising.show_inside(ui);
            metrics.show_inside(ui);
        });
    });

    assert!(!ising.follow_workspace_graph());
    assert_eq!(ising.graph_spin_count(), Some(4));
    assert_eq!(
        metrics.node_count(),
        4,
        "Opted-out pane must freeze local topology while bus advances to 7 nodes"
    );
}

#[test]
fn test_mws_envelope_roundtrip_and_tamper_rejection() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let mut app = MathExplorerApp::default();
    let gt_idx = find_tab_index(&app, "Graph Theory");
    app.set_layout(ViewportLayout::SplitHorizontal);
    app.set_pane_tab(1, gt_idx);
    app.data_bus.shared_graph = Some(std::sync::Arc::new(make_signed_test_graph(9, 6)));

    let pane1_tool = app
        .pane_tab_mut(1)
        .and_then(|t| {
            let _ = t.select_tool_by_name("Network Metrics");
            t.framework_mut()?.active_tool.as_mut()
        })
        .unwrap();
    pane1_tool
        .try_load_state(r#"{"selected_metric":"DegreeCentrality","follow_workspace_graph":false}"#)
        .unwrap();

    let mws_json = app.session_manager.create_envelope(app.get_app_state());
    let mut restored_app = MathExplorerApp::default();
    let parsed = restored_app
        .session_manager
        .parse_envelope(&mws_json)
        .expect("valid .mws envelope must parse");
    restored_app.apply_app_state(parsed);

    assert_eq!(
        restored_app.viewport_layout,
        ViewportLayout::SplitHorizontal
    );
    assert_eq!(restored_app.data_bus.graph_revision(), 9);
    assert_eq!(
        restored_app
            .data_bus
            .shared_graph
            .as_ref()
            .unwrap()
            .nodes
            .len(),
        6
    );
    let restored_tool_json = restored_app
        .pane_tab_mut(1)
        .and_then(|t| t.framework_mut()?.active_tool.as_ref()?.save_state())
        .unwrap();
    assert!(restored_tool_json.contains("\"follow_workspace_graph\":false"));

    // Tampered unknown field inside data_bus must fail deny_unknown_fields validation
    let tampered = mws_json.replace("\"revision\": 9", "\"revision\": 9, \"injected\": true");
    assert!(restored_app
        .session_manager
        .parse_envelope(&tampered)
        .is_err());
}

#[test]
fn test_node_budget_clamps_emit_polite_banners_without_panicking() {
    let ctx = egui::Context::default();
    let huge = std::sync::Arc::new(make_signed_test_graph(5, MAX_SPIN_NODE_BUDGET + 25));
    let _ = publish_shared_graph(&ctx, (*huge).clone());

    let mut hh_viz = NeuralNetworkVizTool::default();
    let mut net_prop = NetworkPropagationTool::default();
    let mut replicator = ReplicatorDynamicsTool::default();
    let mut ising = IsingModelTool::default();

    let _ = ctx.run(egui::RawInput::default(), |c| {
        egui::CentralPanel::default().show(c, |ui| {
            hh_viz.show_inside(ui);
            net_prop.show_inside(ui);
            replicator.show_inside(ui);
            ising.show_inside(ui);
        });
    });

    assert_eq!(hh_viz.neuron_count(), MAX_HH_NEURON_BUDGET);
    let (clamped_epi, orig_epi) = huge.clamped_to_budget(MAX_EPIDEMIC_NODE_BUDGET);
    assert_eq!(clamped_epi.nodes.len(), MAX_EPIDEMIC_NODE_BUDGET);
    assert_eq!(orig_epi, Some(MAX_SPIN_NODE_BUDGET + 25));
    let (clamped_game, _) = huge.clamped_to_budget(MAX_GAME_NODE_BUDGET);
    assert_eq!(clamped_game.nodes.len(), MAX_GAME_NODE_BUDGET);
    assert!(ising.is_graph_spin_mode());
}

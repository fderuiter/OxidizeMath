#![cfg_attr(any(), verified(opt_out = "tests"))]

#[path = "tests_sim_bus/isolation_and_bounds.rs"]
mod isolation_and_bounds;

use super::*;
use crate::framework::InteractiveTool;
use eframe::App;
use eframe::egui::{self, Pos2};

pub(super) fn run_app_frame(ctx: &egui::Context, app: &mut MathExplorerApp) {
    let mut frame = eframe::Frame::_new_kittest();
    let screen = egui::Rect::from_min_size(Pos2::ZERO, egui::vec2(1440.0, 900.0));
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            ..Default::default()
        },
        |c| app.update(c, &mut frame),
    );
}

pub(super) fn assign_pane_tool(
    app: &mut MathExplorerApp,
    pane: usize,
    tab_name: &str,
    tool_name: &str,
) {
    let idx = app
        .tabs
        .iter()
        .position(|t| t.name() == tab_name)
        .unwrap_or_else(|| panic!("Missing tab '{tab_name}'"));
    app.set_pane_tab(pane, idx);
    let tab = app.pane_tab_mut(pane).expect("pane tab exists");
    assert!(
        tab.select_tool_by_name(tool_name),
        "Failed to select tool '{tool_name}' on tab '{tab_name}'"
    );
}

pub(super) fn pane_tool_ref<T: 'static>(app: &mut MathExplorerApp, pane: usize) -> &T {
    app.pane_tab_mut(pane)
        .and_then(|t| t.framework())
        .and_then(|fw| fw.active_tool.as_deref())
        .and_then(InteractiveTool::as_any)
        .and_then(|a| a.downcast_ref::<T>())
        .expect("downcast pane tool ref")
}

pub(super) fn pane_tool_mut<T: 'static>(app: &mut MathExplorerApp, pane: usize) -> &mut T {
    app.pane_tab_mut(pane)
        .and_then(|t| t.framework_mut())
        .and_then(|fw| fw.active_tool.as_deref_mut())
        .and_then(InteractiveTool::as_any_mut)
        .and_then(|a| a.downcast_mut::<T>())
        .expect("downcast pane tool mut")
}

#[cfg(feature = "pure_math")]
pub(super) fn seed_two_community_bridge_graph(
    editor: &mut crate::tabs::graph_theory::graph_editor::GraphEditorTool,
) -> (usize, usize) {
    let mut left_hub = 0;
    let mut right_hub = 0;
    for i in 0..5 {
        let id = editor.insert_node_at(Pos2::new(140.0 + i as f32 * 25.0, 180.0));
        if i == 0 {
            left_hub = id;
        } else {
            assert!(editor.upsert_weighted_edge(left_hub, id, 1.0));
        }
    }
    for i in 0..5 {
        let id = editor.insert_node_at(Pos2::new(420.0 + i as f32 * 25.0, 320.0));
        if i == 0 {
            right_hub = id;
        } else {
            assert!(editor.upsert_weighted_edge(right_hub, id, 1.0));
        }
    }
    for u in 1..5 {
        assert!(editor.upsert_weighted_edge(u, (u % 4) + 1, 1.0));
        assert!(editor.upsert_weighted_edge(5 + u, 5 + (u % 4) + 1, 1.0));
    }
    assert!(editor.upsert_weighted_edge(left_hub, right_hub, 2.0));
    (left_hub, right_hub)
}

#[cfg(all(feature = "pure_math", feature = "epidemiology"))]
fn export_r0_intervention_plot(r0_bridged: f64, r0_spokes: f64, r0_severed: f64) {
    let datasets = vec![(
        "R0 Trajectory (Bridged vs Severed Hub)".to_string(),
        vec![[0.0, r0_bridged], [1.0, r0_spokes], [2.0, r0_severed]],
    )];
    let opts = egui_plot::export::ExportOptions::default();
    if let Ok(png) = egui_plot::export::export_custom_png(&datasets, &opts) {
        let _ = std::fs::write("/tmp/shared_topology_intervention_comparison.png", png);
    }
}

#[cfg(all(feature = "pure_math", feature = "epidemiology"))]
fn assert_quad_panes_synced(app: &mut MathExplorerApp, expected_rev: u64, expected_nodes: usize) {
    use crate::tabs::epidemiology::network_propagation::NetworkPropagationTool;
    use crate::tabs::graph_theory::algorithm_visualizer::AlgorithmVisualizerTool;
    use crate::tabs::graph_theory::network_metrics::NetworkMetricsTool;

    assert_eq!(
        pane_tool_ref::<NetworkMetricsTool>(app, 1).active_node_count(),
        expected_nodes
    );
    assert_eq!(
        pane_tool_ref::<AlgorithmVisualizerTool>(app, 2).synced_bus_revision(),
        Some(expected_rev)
    );
    assert_eq!(
        pane_tool_ref::<AlgorithmVisualizerTool>(app, 2).active_node_count(),
        expected_nodes
    );
    assert_eq!(
        pane_tool_ref::<NetworkPropagationTool>(app, 3).synced_bus_revision(),
        Some(expected_rev)
    );
    assert_eq!(
        pane_tool_ref::<NetworkPropagationTool>(app, 3).active_node_count(),
        expected_nodes
    );
}

#[test]
#[cfg(all(feature = "pure_math", feature = "epidemiology"))]
fn test_live_graph_edit_propagates_across_layouts_and_interventions() {
    use crate::tabs::epidemiology::network_propagation::NetworkPropagationTool;
    use crate::tabs::graph_theory::graph_editor::GraphEditorTool;
    use crate::tabs::graph_theory::network_metrics::NetworkMetricsTool;

    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.set_link_topology(Some(&ctx), true);
    app.set_layout(ViewportLayout::QuadGrid);
    assign_pane_tool(&mut app, 0, "Graph Theory", "Graph Editor");
    assign_pane_tool(&mut app, 1, "Graph Theory", "Network Metrics");
    assign_pane_tool(&mut app, 2, "Graph Theory", "Algorithm Visualizer");
    assign_pane_tool(&mut app, 3, "Epidemiology", "Network Propagation");

    let (hub_a, hub_b) =
        seed_two_community_bridge_graph(pane_tool_mut::<GraphEditorTool>(&mut app, 0));
    run_app_frame(&ctx, &mut app);

    let rev1 = pane_tool_ref::<NetworkMetricsTool>(&mut app, 1)
        .synced_bus_revision()
        .expect("metrics synced in 1 frame");
    assert_quad_panes_synced(&mut app, rev1, 10);
    let r0_bridged = pane_tool_ref::<NetworkPropagationTool>(&mut app, 3).live_heterogeneous_r0();

    for layout in [
        ViewportLayout::SplitHorizontal,
        ViewportLayout::SplitVertical,
        ViewportLayout::QuadGrid,
    ] {
        app.set_layout(layout);
        let added =
            pane_tool_mut::<GraphEditorTool>(&mut app, 0).insert_node_at(Pos2::new(260.0, 260.0));
        assert!(
            pane_tool_mut::<GraphEditorTool>(&mut app, 0).upsert_weighted_edge(hub_a, added, 1.0)
        );
        run_app_frame(&ctx, &mut app);
        let expected_n = app.sim_bus.latest_graph().expect("bus graph").node_count();
        assert_eq!(
            pane_tool_ref::<NetworkMetricsTool>(&mut app, 1).active_node_count(),
            expected_n
        );
    }

    let r0_spokes = pane_tool_ref::<NetworkPropagationTool>(&mut app, 3).live_heterogeneous_r0();
    assert!(pane_tool_mut::<GraphEditorTool>(&mut app, 0).remove_edge_between(hub_a, hub_b));
    run_app_frame(&ctx, &mut app);
    let r0_severed = pane_tool_ref::<NetworkPropagationTool>(&mut app, 3).live_heterogeneous_r0();
    assert!(r0_severed < r0_spokes && r0_bridged > 0.0);
    export_r0_intervention_plot(r0_bridged, r0_spokes, r0_severed);
}

#[cfg(feature = "pure_math")]
fn seed_k6_and_sparsify_to_cycle(
    ed: &mut crate::tabs::graph_theory::graph_editor::GraphEditorTool,
    sparsify: bool,
) {
    if !sparsify {
        for i in 0..6 {
            ed.insert_node_at(Pos2::new(100.0 + i as f32 * 40.0, 200.0));
        }
        for u in 0..6 {
            for v in (u + 1)..6 {
                assert!(ed.upsert_weighted_edge(u, v, 1.0));
            }
        }
    } else {
        for u in 0..6 {
            for v in (u + 1)..6 {
                if (v - u) != 1 && !(u == 0 && v == 5) {
                    assert!(ed.remove_edge_between(u, v));
                }
            }
        }
    }
}

#[test]
#[cfg(all(feature = "pure_math", feature = "physics", feature = "applied"))]
fn test_ising_and_replicator_sync_with_live_bridge_removal() {
    use crate::tabs::game_theory::replicator::ReplicatorDynamicsTool;
    use crate::tabs::graph_theory::graph_editor::GraphEditorTool;
    use crate::tabs::solid_state::ising::IsingModelTool;

    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.set_link_topology(Some(&ctx), true);
    app.set_layout(ViewportLayout::QuadGrid);
    assign_pane_tool(&mut app, 0, "Graph Theory", "Graph Editor");
    assign_pane_tool(&mut app, 1, "Solid State Physics", "Ising Model");
    assign_pane_tool(&mut app, 2, "Game Theory", "Replicator Dynamics");

    pane_tool_mut::<ReplicatorDynamicsTool>(&mut app, 2)
        .try_load_state(
            r#"{"payoff_matrix":[[3.0,-1.0],[4.0,0.0]],"initial_population":[0.5,0.5],"time_horizon":15.0,"dt":0.05,"strategy_names":["Cooperate","Defect"]}"#,
        )
        .expect("valid donation PD state");
    seed_k6_and_sparsify_to_cycle(pane_tool_mut::<GraphEditorTool>(&mut app, 0), false);
    run_app_frame(&ctx, &mut app);

    assert!(pane_tool_ref::<IsingModelTool>(&mut app, 1).is_graph_mode());
    assert_eq!(
        pane_tool_ref::<IsingModelTool>(&mut app, 1).active_graph_node_count(),
        Some(6)
    );
    assert!(
        pane_tool_ref::<IsingModelTool>(&mut app, 1)
            .synced_bus_revision()
            .is_some()
    );
    assert_eq!(
        pane_tool_ref::<ReplicatorDynamicsTool>(&mut app, 2).active_graph_mean_degree(),
        Some(5.0)
    );
    assert!(
        pane_tool_ref::<ReplicatorDynamicsTool>(&mut app, 2)
            .synced_bus_revision()
            .is_some()
    );
    let dense_coop = pane_tool_ref::<ReplicatorDynamicsTool>(&mut app, 2)
        .final_population()
        .expect("trajectory")[0];

    seed_k6_and_sparsify_to_cycle(pane_tool_mut::<GraphEditorTool>(&mut app, 0), true);
    run_app_frame(&ctx, &mut app);
    let sparse_coop = pane_tool_ref::<ReplicatorDynamicsTool>(&mut app, 2)
        .final_population()
        .expect("trajectory")[0];
    assert!(dense_coop < 0.15 && sparse_coop > 0.85);
}

#[test]
#[cfg(all(feature = "pure_math", feature = "biology"))]
fn test_neural_clamping_badge_directed_synapses_and_local_preset_opt_out() {
    use crate::tabs::graph_theory::graph_editor::GraphEditorTool;
    use crate::tabs::neuroscience::neural_network_viz::NeuralNetworkVizTool;

    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.set_link_topology(Some(&ctx), true);
    app.set_layout(ViewportLayout::SplitHorizontal);
    assign_pane_tool(&mut app, 0, "Graph Theory", "Graph Editor");
    assign_pane_tool(&mut app, 1, "Neuroscience", "Neural Network Viz");

    {
        let ed = pane_tool_mut::<GraphEditorTool>(&mut app, 0);
        ed.directed_edges = true;
        for i in 0..40 {
            ed.insert_node_at(Pos2::new(
                80.0 + (i % 10) as f32 * 30.0,
                90.0 + (i / 10) as f32 * 40.0,
            ));
        }
        assert!(ed.upsert_weighted_edge(0, 1, 2.75));
    }
    run_app_frame(&ctx, &mut app);

    {
        let nn = pane_tool_mut::<NeuralNetworkVizTool>(&mut app, 1);
        assert_eq!(nn.neuron_count(), 32);
        assert_eq!(
            nn.clamp_badge_text().as_deref(),
            Some("Clamped 40 -> 32 neurons")
        );
        assert!((nn.synaptic_weight(0, 1) - 2.75).abs() < 1e-9);
        assert_eq!(nn.synaptic_weight(1, 0), 0.0);
        assert!(nn.synced_bus_revision().is_some());
        nn.topology_source = TopologySource::LocalPreset;
    }
    run_app_frame(&ctx, &mut app);

    {
        let nn = pane_tool_mut::<NeuralNetworkVizTool>(&mut app, 1);
        assert_eq!(nn.neuron_count(), 5);
        assert!(nn.clamp_badge_text().is_none());
        assert!(nn.synced_bus_revision().is_none());
        nn.topology_source = TopologySource::SharedBus;
    }
    run_app_frame(&ctx, &mut app);
    let nn = pane_tool_ref::<NeuralNetworkVizTool>(&mut app, 1);
    assert_eq!(nn.neuron_count(), 32);
    assert_eq!(
        nn.clamp_badge_text().as_deref(),
        Some("Clamped 40 -> 32 neurons")
    );
}

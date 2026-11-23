//! End-to-end multi-pane frame tests for live cross-domain pipelines and zero-allocation GUI solvers.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{MathExplorerApp, ViewportLayout};
use crate::framework::InteractiveTool;
use crate::tabs::chaos::attractors::AttractorPlotter;
use crate::tabs::epidemiology::network_propagation::NetworkPropagationTool;
use crate::tabs::geometry_topology::surface_viewer::SurfaceViewer;
use crate::tabs::geometry_topology::vietoris_rips::VietorisRipsTool;
use crate::tabs::graph_theory::graph_editor::SerializedGraphState;
use crate::tabs::neuroscience::neural_network_viz::{NeuralDynamicsMode, NeuralNetworkVizTool};
use eframe::App;
use eframe::egui;
use std::collections::BTreeMap;

fn run_frame(ctx: &egui::Context, app: &mut MathExplorerApp) {
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 900.0));
    let mut frame = eframe::Frame::_new_kittest();
    let input = egui::RawInput {
        screen_rect: Some(screen),
        ..Default::default()
    };
    let _ = ctx.run(input, |c| app.update(c, &mut frame));
}

fn assign_tool(app: &mut MathExplorerApp, pane: usize, tool_id: &str) {
    let idx = app
        .tabs
        .iter()
        .position(|t| {
            t.framework()
                .is_some_and(|fw| fw.available_tools.iter().any(|m| m.id == tool_id))
        })
        .unwrap_or_else(|| panic!("No tab hosts tool {tool_id}"));
    app.set_pane_tab(pane, idx);
    let fw = app
        .pane_tab_mut(pane)
        .and_then(|t| t.framework_mut())
        .unwrap_or_else(|| panic!("Missing framework on pane {pane}"));
    assert!(fw.select_tool_by_name(tool_id), "Missing tool {tool_id}");
}

fn load_pane_graph(
    app: &mut MathExplorerApp,
    pane: usize,
    n: usize,
    edges: Vec<(usize, usize, f64)>,
) {
    let mut node_positions = BTreeMap::new();
    for i in 0..n {
        node_positions.insert(i, (i as f32 * 24.0, (i % 3) as f32 * 20.0));
    }
    let json = serde_json::to_string(&SerializedGraphState {
        node_positions,
        edges,
        selected_node: None,
        next_node_id: n,
        edge_latencies: Vec::new(),
        congestion: Default::default(),
    })
    .unwrap();
    app.pane_tab_mut(pane)
        .unwrap()
        .framework_mut()
        .unwrap()
        .active_tool
        .as_mut()
        .unwrap()
        .try_load_state(&json)
        .unwrap();
}

fn pane_tool_ref<T: 'static>(app: &MathExplorerApp, pane: usize) -> &T {
    app.pane_tab(pane)
        .unwrap()
        .framework()
        .unwrap()
        .active_tool
        .as_ref()
        .unwrap()
        .as_any()
        .unwrap()
        .downcast_ref::<T>()
        .unwrap()
}

fn pane_tool_mut<T: 'static>(app: &mut MathExplorerApp, pane: usize) -> &mut T {
    app.pane_tab_mut(pane)
        .unwrap()
        .framework_mut()
        .unwrap()
        .active_tool
        .as_mut()
        .unwrap()
        .as_any_mut()
        .unwrap()
        .downcast_mut::<T>()
        .unwrap()
}

#[test]
fn test_live_graph_editor_updates_epidemic_effective_r0_in_pane_1() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::SplitHorizontal);
    assign_tool(&mut app, 0, "GraphEditorTool");
    assign_tool(&mut app, 1, "NetworkPropagationTool");

    // Sparse dyads: 8 vertices with only two isolated edges (low mean degree & variance).
    load_pane_graph(&mut app, 0, 8, vec![(0, 1, 1.0), (2, 3, 1.0)]);
    assert!(
        app.connect_pane_pipeline_interactive(Some(&ctx), 0, 1)
            .is_ok()
    );

    run_frame(&ctx, &mut app);
    run_frame(&ctx, &mut app);

    let sparse_r0 = {
        let tool = pane_tool_ref::<NetworkPropagationTool>(&app, 1);
        assert_eq!(tool.active_node_count(), 8);
        tool.current_effective_r0()
    };

    // Edit hub vertex 0 + bridge edges across all 8 vertices in Pane 0.
    let mut hub_edges: Vec<(usize, usize, f64)> = (1..8).map(|v| (0, v, 1.0)).collect();
    hub_edges.push((1, 2, 1.0));
    load_pane_graph(&mut app, 0, 8, hub_edges);
    run_frame(&ctx, &mut app);
    run_frame(&ctx, &mut app);

    let hub_r0 = pane_tool_ref::<NetworkPropagationTool>(&app, 1).current_effective_r0();
    assert!(
        hub_r0 > sparse_r0 + 0.05,
        "Adding hub/bridge edges in Pane 0 must increase Pane 1 effective_r0 ({sparse_r0:.4} -> {hub_r0:.4})"
    );
}

#[test]
fn test_live_graph_editor_updates_kuramoto_synchronization_in_pane_1() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::SplitHorizontal);
    assign_tool(&mut app, 0, "GraphEditorTool");
    assign_tool(&mut app, 1, "NeuralNetworkVizTool");

    {
        let viz = pane_tool_mut::<NeuralNetworkVizTool>(&mut app, 1);
        viz.mode = NeuralDynamicsMode::Kuramoto;
        viz.kuramoto.set_coupling(8.0);
        viz.is_running = true;
    }

    load_pane_graph(&mut app, 0, 6, Vec::new());
    assert!(
        app.connect_pane_pipeline_interactive(Some(&ctx), 0, 1)
            .is_ok()
    );
    run_frame(&ctx, &mut app);
    run_frame(&ctx, &mut app);

    let uncoupled_r = {
        let viz = pane_tool_mut::<NeuralNetworkVizTool>(&mut app, 1);
        assert_eq!(viz.kuramoto.phases().len(), 6);
        for _ in 0..40 {
            viz.step();
        }
        viz.kuramoto.order_parameter().0
    };

    let complete_edges: Vec<(usize, usize, f64)> = (0..6)
        .flat_map(|u| ((u + 1)..6).map(move |v| (u, v, 3.0)))
        .collect();
    load_pane_graph(&mut app, 0, 6, complete_edges);
    run_frame(&ctx, &mut app);
    run_frame(&ctx, &mut app);

    let coupled_r = {
        let viz = pane_tool_mut::<NeuralNetworkVizTool>(&mut app, 1);
        assert_eq!(viz.kuramoto.phases().len(), 6);
        viz.kuramoto.set_coupling(8.0);
        for _ in 0..80 {
            viz.step();
        }
        viz.kuramoto.order_parameter().0
    };

    assert!(
        coupled_r > uncoupled_r && coupled_r > 0.75,
        "Coupled graph from Pane 0 must synchronize Kuramoto r(t) ({uncoupled_r:.3} -> {coupled_r:.3})"
    );
}

#[test]
fn test_gui_hodgkin_huxley_and_kuramoto_step_zero_alloc_under_verified_engine() {
    let mut hh_tool = NeuralNetworkVizTool::default();
    hh_tool.mode = NeuralDynamicsMode::HodgkinHuxley;
    hh_tool.step();

    let mut km_tool = NeuralNetworkVizTool::default();
    km_tool.mode = NeuralDynamicsMode::Kuramoto;
    km_tool.step();

    let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ((), _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        for _ in 0..32 {
            hh_tool.step();
        }
    });
    let ((r, psi), _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        for _ in 0..32 {
            km_tool.step();
        }
        km_tool.kuramoto.order_parameter()
    });
    assert!(r.is_finite() && (0.0..=1.0 + 1e-9).contains(&r));
    assert!(psi.is_finite());
}

#[test]
fn test_live_attractor_streams_landmarks_into_vietoris_rips() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::SplitHorizontal);
    assign_tool(&mut app, 0, "AttractorPlotter");
    assign_tool(&mut app, 1, "VietorisRipsTool");
    assert!(
        app.connect_pane_pipeline_interactive(Some(&ctx), 0, 1)
            .is_ok()
    );

    run_frame(&ctx, &mut app);
    {
        let plotter = pane_tool_mut::<AttractorPlotter>(&mut app, 0);
        for _ in 0..6 {
            plotter.step_simulation();
        }
    }
    run_frame(&ctx, &mut app);
    run_frame(&ctx, &mut app);

    let vr = pane_tool_ref::<VietorisRipsTool>(&app, 1);
    assert!(vr.mode.is_3d());
    assert_eq!(vr.points_3d.len(), 64);
    assert!(vr.chaos.live_stream_seq > 0);
}

#[test]
fn test_live_morphogenesis_streams_field_into_surface_viewer_and_vietoris_rips() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::QuadGrid);
    assign_tool(&mut app, 0, "MorphogenesisTool");
    assign_tool(&mut app, 1, "SurfaceViewer");
    assign_tool(&mut app, 2, "VietorisRipsTool");
    assert!(
        app.connect_pane_pipeline_interactive(Some(&ctx), 0, 1)
            .is_ok()
    );
    assert!(
        app.connect_pane_pipeline_interactive(Some(&ctx), 0, 2)
            .is_ok()
    );

    run_frame(&ctx, &mut app);
    app.pane_tab_mut(0)
        .unwrap()
        .framework_mut()
        .unwrap()
        .active_tool
        .as_mut()
        .unwrap()
        .step_simulation();
    run_frame(&ctx, &mut app);
    run_frame(&ctx, &mut app);

    let mesh = pane_tool_ref::<SurfaceViewer>(&app, 1)
        .export_mesh_3d()
        .expect("mesh export available")
        .expect("valid field surface mesh");
    assert!(!mesh.vertices.is_empty());

    let vr2 = pane_tool_ref::<VietorisRipsTool>(&app, 2);
    assert!(vr2.mode.is_3d());
    assert!(!vr2.points_3d.is_empty());
}

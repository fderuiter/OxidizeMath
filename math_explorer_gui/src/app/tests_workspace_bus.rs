#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::app::workspace_bus::KURAMOTO_COHERENCE_SERIES;
use crate::framework::InteractiveTool;
use crate::tabs::chaos::kuramoto::KuramotoOscillatorTool;
use crate::tabs::epidemiology::network_propagation::NetworkPropagationTool;
use crate::tabs::graph_theory::graph_editor::GraphEditorTool;
use crate::tabs::graph_theory::network_metrics::NetworkMetricsTool;
use eframe::egui;
use eframe::App;

fn run_frame(ctx: &egui::Context, app: &mut MathExplorerApp) {
    let screen = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1280.0, 900.0));
    let mut frame = eframe::Frame::_new_kittest();
    let input = egui::RawInput {
        screen_rect: Some(screen),
        ..Default::default()
    };
    let _ = ctx.run(input, |c| app.update(c, &mut frame));
}

fn find_tab_idx(app: &MathExplorerApp, slug: &str) -> usize {
    app.tabs
        .iter()
        .position(|t| t.url_slug() == slug)
        .unwrap_or_else(|| panic!("Missing tab slug '{slug}'"))
}

fn configure_pane_tool(app: &mut MathExplorerApp, pane: usize, slug: &str, tool_id: &str) {
    let idx = find_tab_idx(app, slug);
    app.set_pane_tab(pane, idx);
    let tab = app.pane_tab_mut(pane).expect("pane tab");
    assert!(tab.select_tool_by_name(tool_id));
}

fn load_barbell_graph_json(with_bridge: bool) -> String {
    let edges = if with_bridge {
        serde_json::json!([
            [0, 1, 1.0],
            [1, 2, 1.0],
            [0, 2, 1.0],
            [2, 3, 1.0],
            [3, 4, 1.0],
            [4, 5, 1.0],
            [3, 5, 1.0]
        ])
    } else {
        serde_json::json!([
            [0, 1, 1.0],
            [1, 2, 1.0],
            [0, 2, 1.0],
            [3, 4, 1.0],
            [4, 5, 1.0],
            [3, 5, 1.0]
        ])
    };
    serde_json::json!({
        "node_positions": {
            "0": [100.0, 140.0],
            "1": [100.0, 260.0],
            "2": [200.0, 200.0],
            "3": [360.0, 200.0],
            "4": [460.0, 140.0],
            "5": [460.0, 260.0]
        },
        "edges": edges,
        "selected_node": 2,
        "next_node_id": 6
    })
    .to_string()
}

fn set_graph_editor_state(app: &mut MathExplorerApp, pane: usize, state_json: &str) {
    let tab = app.pane_tab_mut(pane).unwrap();
    let fw = tab.framework_mut().unwrap();
    let editor = fw
        .active_tool
        .as_mut()
        .unwrap()
        .as_any_mut()
        .unwrap()
        .downcast_mut::<GraphEditorTool>()
        .unwrap();
    editor.try_load_state(state_json).unwrap();
}

fn pane_tool_ref<T: 'static>(app: &mut MathExplorerApp, pane: usize) -> &T {
    app.pane_tab_mut(pane)
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

#[test]
fn test_quad_grid_one_frame_coupling_and_bridge_cut_emergent_drop() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::QuadGrid);
    configure_pane_tool(&mut app, 0, "graph_theory", "GraphEditorTool");
    configure_pane_tool(&mut app, 1, "epidemiology", "NetworkPropagationTool");
    configure_pane_tool(&mut app, 2, "graph_theory", "NetworkMetricsTool");
    configure_pane_tool(&mut app, 3, "chaos", "KuramotoOscillatorTool");

    set_graph_editor_state(&mut app, 0, &load_barbell_graph_json(true));
    for p in 0..4 {
        app.set_pane_bus_linked(p, true);
    }

    // One rendered frame delivers Pane 0 topology to Panes 1, 2, and 3!
    run_frame(&ctx, &mut app);
    assert_eq!(app.workspace_bus.topology.revision(), 1);

    let r0_connected =
        pane_tool_ref::<NetworkPropagationTool>(&mut app, 1).current_heterogeneous_r0();
    let m = pane_tool_ref::<NetworkMetricsTool>(&mut app, 2);
    let (diam_connected, recomputes_after_f1) = (m.current_diameter(), m.recompute_count());
    assert_eq!(diam_connected, Some(3));

    // Idle frame without graph edit must NOT recompute NetworkMetricsTool.
    run_frame(&ctx, &mut app);
    let recomputes_after_idle = pane_tool_ref::<NetworkMetricsTool>(&mut app, 2).recompute_count();
    assert_eq!(recomputes_after_idle, recomputes_after_f1);

    // Cut bridge edge (2, 3) in Pane 0 and render one frame: R0 drops and diameter changes.
    set_graph_editor_state(&mut app, 0, &load_barbell_graph_json(false));
    run_frame(&ctx, &mut app);
    assert_eq!(app.workspace_bus.topology.revision(), 2);
    let r0_cut = pane_tool_ref::<NetworkPropagationTool>(&mut app, 1).current_heterogeneous_r0();
    assert!(r0_cut < r0_connected);
}

#[test]
fn test_split_horizontal_and_vertical_coupling_while_unlinked_stays_isolated() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    for layout in [
        ViewportLayout::SplitHorizontal,
        ViewportLayout::SplitVertical,
    ] {
        let mut app = MathExplorerApp::default();
        app.set_layout(layout);
        configure_pane_tool(&mut app, 0, "graph_theory", "GraphEditorTool");
        configure_pane_tool(&mut app, 1, "graph_theory", "NetworkMetricsTool");
        set_graph_editor_state(&mut app, 0, &load_barbell_graph_json(true));

        // Unlinked by default: Pane 1 stays on its 7-node sample wheel graph (diameter 2).
        run_frame(&ctx, &mut app);
        let diam_unlinked = app
            .pane_tab_mut(1)
            .unwrap()
            .framework()
            .unwrap()
            .active_tool
            .as_ref()
            .unwrap()
            .as_any()
            .unwrap()
            .downcast_ref::<NetworkMetricsTool>()
            .unwrap()
            .current_diameter();
        assert_eq!(diam_unlinked, Some(2));

        // Link both panes: next rendered frame updates Pane 1 to barbell diameter 3.
        app.set_pane_bus_linked(0, true);
        app.set_pane_bus_linked(1, true);
        run_frame(&ctx, &mut app);
        let diam_linked = app
            .pane_tab_mut(1)
            .unwrap()
            .framework()
            .unwrap()
            .active_tool
            .as_ref()
            .unwrap()
            .as_any()
            .unwrap()
            .downcast_ref::<NetworkMetricsTool>()
            .unwrap()
            .current_diameter();
        assert_eq!(diam_linked, Some(3));
    }
}

#[test]
fn test_kuramoto_streams_coherence_to_bus_and_mws_plus_url_roundtrip() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::SplitHorizontal);
    configure_pane_tool(&mut app, 0, "graph_theory", "GraphEditorTool");
    configure_pane_tool(&mut app, 1, "chaos", "KuramotoOscillatorTool");
    set_graph_editor_state(&mut app, 0, &load_barbell_graph_json(true));
    app.set_pane_bus_linked(0, true);
    app.set_pane_bus_linked(1, true);

    run_frame(&ctx, &mut app);
    {
        let kuramoto = app
            .pane_tab_mut(1)
            .unwrap()
            .framework_mut()
            .unwrap()
            .active_tool
            .as_mut()
            .unwrap()
            .as_any_mut()
            .unwrap()
            .downcast_mut::<KuramotoOscillatorTool>()
            .unwrap();
        let [_, r] = kuramoto.step_once();
        assert!((0.0..=1.0).contains(&r));
    }
    run_frame(&ctx, &mut app);
    assert!(app
        .workspace_bus
        .signals
        .series(KURAMOTO_COHERENCE_SERIES)
        .is_some());

    // 1. Verify .mws envelope round-trip restores layout, pane states, and bus links.
    let mws_json = app.session_manager.create_envelope(app.get_app_state());
    let mut mws_restored = MathExplorerApp::default();
    mws_restored.load_state_from_json(&mws_json);
    assert_eq!(
        mws_restored.viewport_layout,
        ViewportLayout::SplitHorizontal
    );
    assert_eq!(mws_restored.pane_bus_links, [true, true, false, false]);
    assert_eq!(mws_restored.workspace_bus.topology.positions().len(), 6);

    // 2. Verify share URL round-trip restores layout, pane states, and bus links.
    let share_frag = app.encode_current_share_fragment().unwrap();
    assert!(share_frag.contains("&ws="));
    let mut url_restored = MathExplorerApp::default();
    assert!(url_restored
        .apply_share_location_strings(&share_frag, "", None)
        .unwrap());
    assert_eq!(
        url_restored.viewport_layout,
        ViewportLayout::SplitHorizontal
    );
    assert_eq!(url_restored.pane_bus_links, [true, true, false, false]);
    assert_eq!(url_restored.workspace_bus.topology.positions().len(), 6);
}

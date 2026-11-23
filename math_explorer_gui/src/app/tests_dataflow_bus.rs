//! Tests for `PanePipelineDag` acyclicity, pane slot remapping, `.mws`/URL persistence,
//! `TwinSlotLiveRegion` ARIA announcements, and `OnceLock`-free landmark handoffs.

#![cfg_attr(any(), verified(opt_out = "tests"))]

#[cfg(all(feature = "pure_math", feature = "epidemiology", feature = "biology"))]
use super::dataflow_bus::{DagLinkError, MAX_DATAFLOW_GRAPH_NODES, PaneLink};
use super::dataflow_bus::{
    DataflowStreamKind, GraphTopologyPayload, LandmarkCloud3DPayload, PaneDataflowBus,
    PanePipelineDag, ScalarField2DPayload, commit_pane_outputs, poll_pane_graph,
    poll_pane_landmarks, poll_pane_scalar_field, publish_pane_graph, publish_pane_landmarks,
    publish_pane_scalar_field, stage_pane_inputs,
};
use crate::app::viewport::focus::CURRENT_RENDERING_PANE_KEY;
use eframe::egui;

#[cfg(all(feature = "pure_math", feature = "epidemiology", feature = "biology"))]
use super::{MathExplorerApp, ViewportLayout};
#[cfg(all(feature = "pure_math", feature = "epidemiology", feature = "biology"))]
use crate::accessibility::with_live_region;
#[cfg(feature = "pure_math")]
use crate::tabs::geometry_topology::vietoris_rips::handoff::{
    ChaosLandmarkHandoff, submit_chaos_landmark_handoff, take_chaos_landmark_handoff,
};
#[cfg(all(feature = "pure_math", feature = "epidemiology", feature = "biology"))]
use crate::tabs::graph_theory::graph_editor::SerializedGraphState;
#[cfg(all(feature = "pure_math", feature = "epidemiology", feature = "biology"))]
use crate::url_state::pipeline::{
    PipelineUrlEnvelope, build_pane_url_entry, encode_pipeline_share_fragment,
};
#[cfg(all(feature = "pure_math", feature = "epidemiology", feature = "biology"))]
use crate::url_state::{MAX_DECODED_URL_STATE_BYTES, encode_base64url};
#[cfg(feature = "pure_math")]
use math_explorer::pure_math::statistics::tda::Point3D;
#[cfg(all(feature = "pure_math", feature = "epidemiology", feature = "biology"))]
use std::collections::BTreeMap;

#[test]
fn test_core_dag_and_bus_staging_roundtrip() {
    let ctx = egui::Context::default();
    let mut dag = PanePipelineDag::default();
    let mut bus = PaneDataflowBus::default();
    assert!(dag.connect(0, 1, DataflowStreamKind::GraphTopology).is_ok());
    assert!(dag.connect(0, 2, DataflowStreamKind::Landmarks3D).is_ok());
    assert!(dag.connect(0, 3, DataflowStreamKind::ScalarField2D).is_ok());

    ctx.data_mut(|d| d.insert_temp(egui::Id::new(CURRENT_RENDERING_PANE_KEY), 0usize));
    stage_pane_inputs(&ctx, 0, &dag, &bus);
    let g = GraphTopologyPayload::sanitized(
        7,
        vec![[-10.0, 0.0], [10.0, 20.0]],
        vec![(0, 1, 1.0)],
        false,
    );
    assert_eq!(g.normalized_positions(2, 50.0).len(), 2);
    publish_pane_graph(&ctx, g);
    publish_pane_landmarks(
        &ctx,
        LandmarkCloud3DPayload::sanitized(8, vec![[1.0, 2.0, 3.0]], 10.0, 28.0, 2.66, 0.01),
    );
    publish_pane_scalar_field(
        &ctx,
        ScalarField2DPayload::from_grid_downsampled(9, 4, 4, &[0.5; 16], [-1.0, 1.0], [-1.0, 1.0])
            .expect("valid 4x4 grid"),
    );
    commit_pane_outputs(&ctx, 0, &dag, &mut bus);

    for (pane, expected_rev) in [(1usize, 7u64), (2, 8), (3, 9)] {
        ctx.data_mut(|d| d.insert_temp(egui::Id::new(CURRENT_RENDERING_PANE_KEY), pane));
        stage_pane_inputs(&ctx, pane, &dag, &bus);
        let rev = poll_pane_graph(&ctx)
            .map(|p| p.revision)
            .or_else(|| poll_pane_landmarks(&ctx).map(|p| p.revision))
            .or_else(|| poll_pane_scalar_field(&ctx).map(|p| p.revision));
        assert_eq!(rev, Some(expected_rev));
        commit_pane_outputs(&ctx, pane, &dag, &mut bus);
    }
}

#[cfg(all(feature = "pure_math", feature = "epidemiology", feature = "biology"))]
fn select_pane_tool(app: &mut MathExplorerApp, pane: usize, tool_id: &str) {
    let tab_idx = app
        .tabs
        .iter()
        .position(|t| {
            t.framework()
                .is_some_and(|fw| fw.available_tools.iter().any(|m| m.id == tool_id))
        })
        .unwrap_or_else(|| panic!("No tab hosts tool {tool_id}"));
    app.set_pane_tab(pane, tab_idx);
    let fw = app
        .pane_tab_mut(pane)
        .and_then(|t| t.framework_mut())
        .unwrap_or_else(|| panic!("Missing framework on pane {pane}"));
    assert!(fw.select_tool_by_name(tool_id), "Missing tool {tool_id}");
}

#[cfg(all(feature = "pure_math", feature = "epidemiology", feature = "biology"))]
#[test]
fn test_pipeline_dag_rejects_self_loops_and_cycles_with_twin_slot_aria() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut dag = PanePipelineDag::default();
    assert_eq!(
        dag.connect(0, 0, DataflowStreamKind::GraphTopology),
        Err(DagLinkError::SelfLoop)
    );
    assert_eq!(
        dag.connect(0, 4, DataflowStreamKind::GraphTopology),
        Err(DagLinkError::OutOfBounds)
    );
    assert!(dag.connect(0, 1, DataflowStreamKind::GraphTopology).is_ok());
    assert_eq!(
        dag.connect(1, 0, DataflowStreamKind::GraphTopology),
        Err(DagLinkError::CycleDetected)
    );
    assert!(dag.connect(1, 2, DataflowStreamKind::Landmarks3D).is_ok());
    assert_eq!(
        dag.connect(2, 0, DataflowStreamKind::ScalarField2D),
        Err(DagLinkError::CycleDetected)
    );

    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::SplitHorizontal);
    select_pane_tool(&mut app, 0, "GraphEditorTool");
    select_pane_tool(&mut app, 1, "NetworkPropagationTool");

    let slot_before = with_live_region(|r| r.active_slot);
    assert!(
        app.connect_pane_pipeline_interactive(Some(&ctx), 0, 1)
            .is_ok()
    );
    with_live_region(|r| {
        assert_ne!(r.active_slot, slot_before);
        assert!(r.slots[1 - r.active_slot].is_empty());
        assert!(r.slots[r.active_slot].contains("Connected"));
    });

    assert_eq!(
        app.connect_pane_pipeline_interactive(Some(&ctx), 1, 0),
        Err(DagLinkError::CycleDetected)
    );
    let assertive =
        ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_assertive_message")));
    assert!(
        assertive
            .as_deref()
            .is_some_and(|m| m.contains("cycle") || m.contains("rejected"))
    );

    assert!(app.disconnect_pane_pipeline_interactive(Some(&ctx), 1));
    assert!(app.pipeline_dag.is_empty());
    with_live_region(|r| assert!(r.slots[r.active_slot].contains("Disconnected")));
}

#[cfg(all(feature = "pure_math", feature = "epidemiology", feature = "biology"))]
#[test]
fn test_pipeline_dag_remaps_swap_rotate_clone_and_prune_inactive() {
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::QuadGrid);
    select_pane_tool(&mut app, 0, "GraphEditorTool");
    select_pane_tool(&mut app, 1, "NetworkPropagationTool");
    select_pane_tool(&mut app, 2, "MorphogenesisTool");
    select_pane_tool(&mut app, 3, "SurfaceViewer");

    assert!(app.connect_pane_pipeline_interactive(None, 0, 1).is_ok());
    assert!(app.connect_pane_pipeline_interactive(None, 2, 3).is_ok());

    app.swap_pane_pair(0, 1);
    assert_eq!(
        app.pipeline_dag.upstream_for(0),
        Some(PaneLink {
            from_pane: 1,
            to_pane: 0,
            kind: DataflowStreamKind::GraphTopology
        })
    );
    app.swap_pane_pair(0, 1);

    // Clockwise QuadGrid rotation moves 0 -> 1 -> 3 -> 2 -> 0, so edge 0 -> 1 becomes 1 -> 3.
    app.rotate_panes(true);
    assert_eq!(app.pipeline_dag.links().len(), 2);
    assert_eq!(
        app.pipeline_dag.upstream_for(3).map(|l| l.from_pane),
        Some(1)
    );

    app.set_layout(ViewportLayout::QuadGrid);
    select_pane_tool(&mut app, 0, "GraphEditorTool");
    select_pane_tool(&mut app, 1, "NetworkPropagationTool");
    app.pipeline_dag.clear();
    assert!(app.connect_pane_pipeline_interactive(None, 0, 1).is_ok());
    assert!(app.clone_pane_to(1, 3));
    assert_eq!(
        app.pipeline_dag.upstream_for(3),
        Some(PaneLink {
            from_pane: 0,
            to_pane: 3,
            kind: DataflowStreamKind::GraphTopology
        })
    );

    app.set_layout(ViewportLayout::SplitHorizontal);
    assert_eq!(app.pipeline_dag.links().len(), 1);
    assert!(app.pipeline_dag.upstream_for(3).is_none());

    app.set_layout(ViewportLayout::Single);
    assert!(app.pipeline_dag.is_empty());
}

#[cfg(all(feature = "pure_math", feature = "epidemiology", feature = "biology"))]
fn seed_pane_six_node_graph(app: &mut MathExplorerApp, pane: usize) {
    let mut pos = BTreeMap::new();
    for i in 0..6 {
        pos.insert(i, (i as f32 * 25.0, 10.0));
    }
    let graph_json = serde_json::to_string(&SerializedGraphState {
        node_positions: pos,
        edges: vec![(0, 1, 1.5), (1, 2, 2.0), (0, 3, 1.0)],
        selected_node: None,
        next_node_id: 6,
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
        .try_load_state(&graph_json)
        .unwrap();
}

#[cfg(all(feature = "pure_math", feature = "epidemiology", feature = "biology"))]
#[test]
fn test_pipeline_mws_envelope_and_share_url_roundtrip_and_200_node_cap() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::SplitHorizontal);
    select_pane_tool(&mut app, 0, "GraphEditorTool");
    select_pane_tool(&mut app, 1, "NeuralNetworkVizTool");
    seed_pane_six_node_graph(&mut app, 0);

    let unlinked_frag = app.encode_current_share_fragment().unwrap();
    assert!(!unlinked_frag.contains("&pipe="));

    assert!(
        app.connect_pane_pipeline_interactive(Some(&ctx), 0, 1)
            .is_ok()
    );

    let mws_json = app.save_state_to_json();
    let mut from_mws = MathExplorerApp::default();
    from_mws.load_state_from_json(&mws_json);
    assert_eq!(from_mws.viewport_layout, ViewportLayout::SplitHorizontal);
    assert_eq!(from_mws.pipeline_dag.links(), app.pipeline_dag.links());

    let pipe_frag = app.encode_current_share_fragment().unwrap();
    assert!(pipe_frag.contains("&pipe="));
    assert!(pipe_frag.len() <= MAX_DECODED_URL_STATE_BYTES);

    let mut from_url = MathExplorerApp::default();
    assert_eq!(
        from_url.apply_share_location_strings(&pipe_frag, "", Some(&ctx)),
        Ok(true)
    );
    assert_eq!(from_url.viewport_layout, ViewportLayout::SplitHorizontal);
    assert_eq!(from_url.pipeline_dag.links(), app.pipeline_dag.links());

    let sanitized = GraphTopologyPayload::sanitized(
        1,
        vec![[1.0, 2.0]; MAX_DATAFLOW_GRAPH_NODES + 25],
        vec![(0, MAX_DATAFLOW_GRAPH_NODES + 5, 1.0), (0, 1, 2.0)],
        false,
    );
    assert_eq!(sanitized.node_positions.len(), MAX_DATAFLOW_GRAPH_NODES);
    assert_eq!(
        sanitized.truncated_from,
        Some(MAX_DATAFLOW_GRAPH_NODES + 25)
    );
    assert_eq!(sanitized.edges, vec![(0, 1, 2.0)]);
}

#[cfg(all(feature = "pure_math", feature = "epidemiology", feature = "biology"))]
#[test]
fn test_tampered_cyclic_pipeline_url_rejected_atomically() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let baseline_tab = app.selected_tab;

    let cyclic_env = PipelineUrlEnvelope {
        layout: ViewportLayout::SplitHorizontal,
        focused_pane: 0,
        panes: vec![
            build_pane_url_entry("graph-theory", Some("GraphEditorTool"), None).unwrap(),
            build_pane_url_entry("epidemiology", Some("NetworkPropagationTool"), None).unwrap(),
        ],
        links: vec![
            PaneLink {
                from_pane: 0,
                to_pane: 1,
                kind: DataflowStreamKind::GraphTopology,
            },
            PaneLink {
                from_pane: 1,
                to_pane: 0,
                kind: DataflowStreamKind::GraphTopology,
            },
        ],
    };
    assert!(encode_pipeline_share_fragment(
        "graph-theory",
        Some("GraphEditorTool"),
        cyclic_env.clone(),
    )
    .is_err());

    let cyclic_b64 = encode_base64url(&serde_json::to_vec(&cyclic_env).unwrap());
    let tampered_cyclic_frag = format!("#tab=graph-theory&tool=GraphEditorTool&pipe={cyclic_b64}");
    assert!(
        app.apply_share_location_strings(&tampered_cyclic_frag, "", Some(&ctx))
            .is_err()
    );
    assert_eq!(app.selected_tab, baseline_tab);
    assert!(app.pipeline_dag.is_empty());
}

#[cfg(all(feature = "pure_math", feature = "epidemiology", feature = "biology"))]
#[test]
fn test_over_200_node_pipeline_url_rejected_atomically() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();

    let mut huge_pos = BTreeMap::new();
    for i in 0..(MAX_DATAFLOW_GRAPH_NODES + 5) {
        huge_pos.insert(i, (0.0, 0.0));
    }
    let huge_json = serde_json::to_string(&SerializedGraphState {
        node_positions: huge_pos,
        edges: Vec::new(),
        selected_node: None,
        next_node_id: MAX_DATAFLOW_GRAPH_NODES + 5,
        edge_latencies: Vec::new(),
        congestion: Default::default(),
    })
    .unwrap();
    let over_cap_env = PipelineUrlEnvelope {
        layout: ViewportLayout::SplitHorizontal,
        focused_pane: 0,
        panes: vec![
            build_pane_url_entry("graph-theory", Some("GraphEditorTool"), Some(&huge_json))
                .unwrap(),
            build_pane_url_entry("epidemiology", Some("NetworkPropagationTool"), None).unwrap(),
        ],
        links: vec![PaneLink {
            from_pane: 0,
            to_pane: 1,
            kind: DataflowStreamKind::GraphTopology,
        }],
    };
    let over_b64 = encode_base64url(&serde_json::to_vec(&over_cap_env).unwrap());
    let tampered_over_frag = format!("#tab=graph-theory&tool=GraphEditorTool&pipe={over_b64}");
    assert!(
        app.apply_share_location_strings(&tampered_over_frag, "", Some(&ctx))
            .is_err()
    );
    assert!(app.pipeline_dag.is_empty());
}

#[cfg(feature = "pure_math")]
#[test]
fn test_vietoris_rips_handoff_contains_zero_oncelock_and_isolates_contexts() {
    let handoff_src = include_str!("../tabs/geometry_topology/vietoris_rips/handoff.rs");
    assert!(
        !handoff_src.contains("OnceLock") && !handoff_src.contains("Mutex"),
        "vietoris_rips/handoff.rs must not contain process-global OnceLock or Mutex statics"
    );

    let ctx_a = egui::Context::default();
    let ctx_b = egui::Context::default();
    let sample = ChaosLandmarkHandoff {
        points: vec![Point3D::new(1.0, 2.0, 3.0)],
        sigma: 10.0,
        rho: 28.0,
        beta: 8.0 / 3.0,
        dt: 0.01,
    };
    submit_chaos_landmark_handoff(&ctx_a, sample.clone());
    assert!(take_chaos_landmark_handoff(&ctx_b).is_none());
    assert_eq!(take_chaos_landmark_handoff(&ctx_a), Some(sample));
    assert!(take_chaos_landmark_handoff(&ctx_a).is_none());
}

#[cfg(all(feature = "pure_math", feature = "epidemiology", feature = "biology"))]
#[test]
fn test_pipeline_links_undo_redo_and_challenge_lock_guard() {
    use crate::tabs::epidemiology::network_propagation::challenge::CHALLENGE_EPI_HUB_VACCINATION;
    use eframe::App;

    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::SplitHorizontal);
    select_pane_tool(&mut app, 0, "GraphEditorTool");
    select_pane_tool(&mut app, 1, "NetworkPropagationTool");

    let _ = ctx.run(egui::RawInput::default(), |c| app.update(c, &mut frame));
    assert!(app.pipeline_dag.is_empty());

    assert!(
        app.connect_pane_pipeline_interactive(Some(&ctx), 0, 1)
            .is_ok()
    );
    let _ = ctx.run(egui::RawInput::default(), |c| app.update(c, &mut frame));
    assert_eq!(app.pipeline_dag.links().len(), 1);

    app.undo();
    let _ = ctx.run(egui::RawInput::default(), |c| app.update(c, &mut frame));
    assert!(app.pipeline_dag.is_empty());

    app.redo();
    let _ = ctx.run(egui::RawInput::default(), |c| app.update(c, &mut frame));
    assert_eq!(app.pipeline_dag.links().len(), 1);

    if let Some(fw) = app.pane_tab_mut(1).and_then(|t| t.framework_mut()) {
        assert!(
            fw.launch_challenge_by_id(CHALLENGE_EPI_HUB_VACCINATION, Some(&ctx))
                .is_ok()
        );
    }
    let _ = ctx.run(egui::RawInput::default(), |c| app.update(c, &mut frame));
    let locked_nodes = app
        .pane_tab(1)
        .and_then(|t| t.framework())
        .and_then(|fw| fw.active_tool.as_ref()?.as_any())
        .and_then(|a| {
            a.downcast_ref::<crate::tabs::epidemiology::network_propagation::NetworkPropagationTool>()
        })
        .unwrap()
        .active_node_count();
    assert_eq!(locked_nodes, 72);
}

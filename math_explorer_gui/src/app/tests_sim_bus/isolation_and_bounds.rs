#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::run_app_frame;
#[cfg(feature = "pure_math")]
use super::{assign_pane_tool, pane_tool_mut, pane_tool_ref, seed_two_community_bridge_graph};
use crate::accessibility::{all_announcements, clear_test_announcements, with_live_region};
use crate::app::*;
#[cfg(all(
    feature = "pure_math",
    feature = "physics",
    feature = "biology",
    feature = "applied",
    feature = "epidemiology"
))]
use crate::framework::InteractiveTool;
#[cfg(feature = "pure_math")]
use crate::framework::ToolStateBounds;
#[cfg(all(
    feature = "pure_math",
    feature = "physics",
    feature = "biology",
    feature = "applied",
    feature = "epidemiology"
))]
use crate::url_state::{encode_share_fragment, parse_share_location, validate_url_preset_json};
use eframe::egui;

#[test]
#[cfg(all(feature = "pure_math", feature = "physics"))]
fn test_two_workspaces_isolated_3d_landmark_streams_and_zero_globals() {
    use crate::tabs::geometry_topology::vietoris_rips::handoff::{
        ChaosLandmarkHandoff, poll_live_attractor_landmarks, publish_live_attractor_landmarks,
        request_open_vietoris_rips, submit_chaos_landmark_handoff, take_chaos_landmark_handoff,
        take_open_vietoris_rips_request,
    };
    use math_explorer::pure_math::statistics::tda::Point3D;

    let handoff_src = include_str!("../../tabs/geometry_topology/vietoris_rips/handoff.rs");
    let non_comment_code: String = handoff_src
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !non_comment_code.contains("OnceLock") && !non_comment_code.contains("Mutex"),
        "handoff.rs must have zero OnceLock or Mutex statics"
    );

    let (ctx_a, mut app_a) = (egui::Context::default(), MathExplorerApp::default());
    let (ctx_b, mut app_b) = (egui::Context::default(), MathExplorerApp::default());
    assert!(!app_a.sim_bus.ptr_eq(&app_b.sim_bus));
    run_app_frame(&ctx_a, &mut app_a);
    run_app_frame(&ctx_b, &mut app_b);

    let cloud_a = ChaosLandmarkHandoff {
        points: (0..24)
            .map(|i| Point3D::new(i as f64, i as f64 * 0.5, 1.0))
            .collect(),
        sigma: 10.0,
        rho: 28.0,
        beta: 8.0 / 3.0,
        dt: 0.01,
    };
    let cloud_b = ChaosLandmarkHandoff {
        points: (0..16)
            .map(|i| Point3D::new(-i as f64, 2.0, i as f64 * 0.25))
            .collect(),
        sigma: 14.0,
        rho: 45.0,
        beta: 3.0,
        dt: 0.02,
    };

    publish_live_attractor_landmarks(&ctx_a, cloud_a);
    submit_chaos_landmark_handoff(&ctx_b, cloud_b.clone());
    request_open_vietoris_rips(&ctx_b);

    let (_, polled_a) = poll_live_attractor_landmarks(&ctx_a, 0).expect("workspace A live frame");
    let (_, polled_b) = poll_live_attractor_landmarks(&ctx_b, 0).expect("workspace B live frame");
    assert_eq!(polled_a.points.len(), 24);
    assert!((polled_a.rho - 28.0).abs() < 1e-9);
    assert_eq!(polled_b.points.len(), 16);
    assert!((polled_b.rho - 45.0).abs() < 1e-9);
    assert!(take_chaos_landmark_handoff(&ctx_a).is_none());
    assert_eq!(take_chaos_landmark_handoff(&ctx_b), Some(cloud_b));
    assert!(!take_open_vietoris_rips_request(&ctx_a));
    assert!(take_open_vietoris_rips_request(&ctx_b));
}

#[test]
#[cfg(feature = "pure_math")]
fn test_mws_envelope_preserves_link_topology_and_pane_source() {
    use crate::tabs::graph_theory::algorithm_visualizer::AlgorithmVisualizerTool;
    use crate::tabs::graph_theory::graph_editor::GraphEditorTool;

    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.set_link_topology(Some(&ctx), true);
    app.set_layout(ViewportLayout::SplitHorizontal);
    assign_pane_tool(&mut app, 0, "Graph Theory", "Graph Editor");
    assign_pane_tool(&mut app, 1, "Graph Theory", "Algorithm Visualizer");
    seed_two_community_bridge_graph(pane_tool_mut::<GraphEditorTool>(&mut app, 0));
    pane_tool_mut::<AlgorithmVisualizerTool>(&mut app, 1).topology_source =
        TopologySource::LocalPreset;
    run_app_frame(&ctx, &mut app);

    let mws_json = app.session_manager.create_envelope(AppState::extract(&app));
    ToolStateBounds::new(8 * 1024 * 1024, 32, 100_000)
        .validate_json(&mws_json)
        .expect("MWS within 8 MiB session bounds");
    let parsed_state = app
        .session_manager
        .parse_envelope(&mws_json)
        .expect("valid MWS");
    assert!(parsed_state.compare_sync.link_topology);
    let mut restored = MathExplorerApp::default();
    parsed_state.apply(&mut restored);
    assert!(restored.compare_sync.link_topology);
    assert_eq!(
        pane_tool_ref::<AlgorithmVisualizerTool>(&mut restored, 1).topology_source,
        TopologySource::LocalPreset
    );
}

#[test]
#[cfg(all(
    feature = "pure_math",
    feature = "physics",
    feature = "biology",
    feature = "applied",
    feature = "epidemiology"
))]
fn test_url_fragment_bounds_and_round_trips() {
    use crate::tabs::epidemiology::network_propagation::NetworkPropagationTool;
    use crate::tabs::game_theory::replicator::ReplicatorDynamicsTool;
    use crate::tabs::graph_theory::algorithm_visualizer::AlgorithmVisualizerTool;
    use crate::tabs::graph_theory::network_metrics::NetworkMetricsTool;
    use crate::tabs::neuroscience::neural_network_viz::NeuralNetworkVizTool;
    use crate::tabs::solid_state::ising::IsingModelTool;

    let tools: Vec<(&str, &str, Box<dyn InteractiveTool>)> = vec![
        (
            "graph_theory",
            "NetworkMetricsTool",
            Box::new(NetworkMetricsTool::default()),
        ),
        (
            "graph_theory",
            "AlgorithmVisualizerTool",
            Box::new(AlgorithmVisualizerTool::default()),
        ),
        (
            "epidemiology",
            "NetworkPropagationTool",
            Box::new(NetworkPropagationTool::default()),
        ),
        (
            "neuroscience",
            "NeuralNetworkVizTool",
            Box::new(NeuralNetworkVizTool::default()),
        ),
        (
            "solid_state",
            "IsingModelTool",
            Box::new(IsingModelTool::default()),
        ),
        (
            "game_theory",
            "ReplicatorDynamicsTool",
            Box::new(ReplicatorDynamicsTool::default()),
        ),
    ];
    for (slug, id, mut tool) in tools {
        let saved = tool.save_state().expect("saves JSON");
        validate_url_preset_json(&saved).expect("fits inside 4 KiB URL_JSON_BOUNDS");
        let frag = encode_share_fragment(slug, Some(id), Some(&saved)).expect("URL fragment");
        let decoded = parse_share_location(&frag, "")
            .expect("parses URL fragment")
            .expect("share params present");
        tool.try_load_state(decoded.state_json.as_deref().unwrap())
            .expect("round-trips state");
    }
}

#[test]
#[cfg(all(
    feature = "pure_math",
    feature = "physics",
    feature = "biology",
    feature = "applied",
    feature = "epidemiology"
))]
fn test_consumer_tools_reject_unknown_json_fields() {
    use crate::tabs::epidemiology::network_propagation::NetworkPropagationTool;
    use crate::tabs::game_theory::replicator::ReplicatorDynamicsTool;
    use crate::tabs::graph_theory::algorithm_visualizer::AlgorithmVisualizerTool;
    use crate::tabs::graph_theory::network_metrics::NetworkMetricsTool;
    use crate::tabs::neuroscience::neural_network_viz::NeuralNetworkVizTool;
    use crate::tabs::solid_state::ising::IsingModelTool;

    let mut tools: Vec<Box<dyn InteractiveTool>> = vec![
        Box::new(NetworkMetricsTool::default()),
        Box::new(AlgorithmVisualizerTool::default()),
        Box::new(NetworkPropagationTool::default()),
        Box::new(NeuralNetworkVizTool::default()),
        Box::new(IsingModelTool::default()),
        Box::new(ReplicatorDynamicsTool::default()),
    ];
    for tool in &mut tools {
        let base = tool.save_state().expect("valid JSON");
        let mut obj: serde_json::Value = serde_json::from_str(&base).unwrap();
        obj.as_object_mut()
            .unwrap()
            .insert("rogue_unknown_field".to_string(), serde_json::json!(42));
        let tampered = serde_json::to_string(&obj).unwrap();
        assert!(
            tool.try_load_state(&tampered).is_err(),
            "{} must reject unknown JSON fields under #[serde(deny_unknown_fields)]",
            tool.name()
        );
    }
}

#[test]
fn test_wcag_contrast_and_twin_slot_live_announcements() {
    for theme in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
        let palette = theme.palette();
        for bg in [
            palette.surface_fill,
            palette.info,
            palette.warning,
            palette.success,
        ] {
            let fg = contrasting_text_color(bg);
            let ratio = contrast_ratio(fg, bg);
            assert!(
                ratio >= 4.5,
                "Theme {theme:?} pill bg {bg:?} / fg {fg:?} contrast {ratio:.2} < 4.5"
            );
        }
    }

    clear_test_announcements();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.set_link_topology(Some(&ctx), true);
    crate::accessibility::sync_context_live_messages(&ctx, false);
    let slot_after_enable = with_live_region(|r| r.active_slot);
    crate::accessibility::sync_context_live_messages(&ctx, true);

    app.set_link_topology(Some(&ctx), false);
    crate::accessibility::sync_context_live_messages(&ctx, false);
    let slot_after_disable = with_live_region(|r| r.active_slot);
    assert_ne!(
        slot_after_enable, slot_after_disable,
        "TwinSlotLiveRegion must ping-pong slots"
    );

    let log = all_announcements();
    assert!(
        log.iter()
            .any(|(msg, prio)| prio == "polite" && msg.contains("Shared topology linking enabled"))
    );
    assert!(log
        .iter()
        .any(|(msg, prio)| prio == "polite" && msg.contains("Shared topology linking disabled")));
}

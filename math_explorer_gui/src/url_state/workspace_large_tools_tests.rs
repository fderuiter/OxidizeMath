//! Contract tests for the 5 large-state tools (`4 KiB < state <= 32 KiB`) compressing into
//! `#ws=<base64url>` (`<= 8 KiB` fragment) and `> 32 KiB` overflow clearing + `.mws` guidance.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{
    last_replaced_browser_fragment, take_last_copied_share_url, UrlSyncController,
    MAX_DECODED_URL_STATE_BYTES, MAX_DECODED_WORKSPACE_URL_BYTES, MAX_URL_FRAGMENT_BYTES,
    URL_SYNC_DEBOUNCE_MS,
};
use crate::app::{MathExplorerApp, ViewportLayout};
use crate::tabs::favoritism::state::{ChildInputsState, ChildState, FavoritismToolState};
use crate::tabs::fluid_dynamics::potential_flow::{ElementType, SerializedFlowElement};
use crate::tabs::graph_theory::graph_editor::SerializedGraphState;
use eframe::egui;
use math_explorer::applied::favoritism::FavoritismInputs;
use std::collections::BTreeMap;

fn build_large_graph_state(node_count: usize) -> String {
    let mut node_positions = BTreeMap::new();
    let mut edges = Vec::new();
    for i in 0..node_count {
        node_positions.insert(i, (i as f32 * 12.345, (i % 17) as f32 * 9.875));
        if i > 0 {
            edges.push((i - 1, i, 1.0 + (i % 11) as f64 * 0.375));
        }
        if i > 2 {
            edges.push((i - 3, i, 2.5 + (i % 7) as f64 * 0.125));
        }
    }
    serde_json::to_string(&SerializedGraphState {
        node_positions,
        edges,
        selected_node: Some(0),
        next_node_id: node_count,
        edge_latencies: Vec::new(),
        congestion: Default::default(),
    })
    .unwrap()
}

fn build_large_potential_flow_state() -> String {
    let elements: Vec<SerializedFlowElement> = (0..60)
        .map(|i| SerializedFlowElement {
            element_type: if i % 2 == 0 {
                ElementType::Vortex
            } else {
                ElementType::Source
            },
            strength: 2.5 + (i % 9) as f64 * 0.75,
            x: -8.0 + (i % 10) as f64 * 1.6,
            y: -6.0 + (i / 10) as f64 * 2.1,
            angle: 0.25,
        })
        .collect();
    serde_json::json!({
        "elements": elements,
        "grid_size": 24,
        "view_range": 10.0,
        "selected_element_type": "Source",
        "new_element_strength": 10.0,
        "new_element_x": 0.0,
        "new_element_y": 0.0,
        "new_element_angle": 0.0
    })
    .to_string()
}

fn build_large_favoritism_state() -> String {
    let base = ChildInputsState::from_inputs(&FavoritismInputs::default());
    let children: Vec<ChildState> = (0..18)
        .map(|i| ChildState {
            name: format!("Family Member Candidate #{i:02}"),
            inputs: ChildInputsState {
                days_integrated: 365.0 + f64::from(i) * 30.0,
                distance_km: 12.5 * f64::from(i),
                g_practical: (i % 10) as f64,
                ..base.clone()
            },
        })
        .collect();
    serde_json::to_string(&FavoritismToolState {
        children,
        selected_child_index: Some(3),
        new_child_name: "Sibling Draft".to_string(),
    })
    .unwrap()
}

fn build_large_replicator_state() -> String {
    let n = 11_usize;
    let payoff_matrix: Vec<Vec<f64>> = (0..n)
        .map(|r| {
            (0..n)
                .map(|c| ((r as f64 * 1.37) - (c as f64 * 0.83)).sin() * 3.125)
                .collect()
        })
        .collect();
    let symbolic_payoffs: Vec<Vec<String>> = (0..n)
        .map(|r| {
            (0..n)
                .map(|c| {
                    format!(
                        "(v - c) / {} + {:.6}",
                        (r + c + 2),
                        (r * 7 + c) as f64 * 0.11
                    )
                })
                .collect()
        })
        .collect();
    let strategy_names: Vec<String> = (0..n)
        .map(|i| format!("Evolutionary_Strategy_Lineage_{i:02}"))
        .collect();
    let initial_population = vec![1.0 / (n as f64); n];
    serde_json::json!({
        "payoff_matrix": payoff_matrix,
        "initial_population": initial_population,
        "time_horizon": 25.0,
        "dt": 0.05,
        "strategy_names": strategy_names,
        "symbolic_payoffs": symbolic_payoffs,
        "symbolic_params": { "c": 4.0, "v": 2.5 }
    })
    .to_string()
}

fn build_large_neural_network_viz_state() -> String {
    let n = 28_usize;
    let external_input: Vec<f64> = (0..n).map(|i| (i % 12) as f64 * 2.25).collect();
    let weights: Vec<Vec<f64>> = (0..n)
        .map(|r| {
            (0..n)
                .map(|c| {
                    if r == c {
                        0.0
                    } else {
                        ((r + c) % 7) as f64 * 0.45 - 1.2
                    }
                })
                .collect()
        })
        .collect();
    serde_json::json!({
        "external_input": external_input,
        "weights": weights,
        "dt": 0.05,
        "is_running": false
    })
    .to_string()
}

fn assert_large_tool_ws_roundtrip(ctx: &egui::Context, slug: &str, tool_id: &str, raw_state: &str) {
    let mut app = MathExplorerApp::default();
    let tab_idx = app
        .tabs
        .iter()
        .position(|t| t.url_slug() == slug)
        .unwrap_or_else(|| panic!("Missing tab {slug}"));
    app.selected_tab = tab_idx;
    assert!(app.tabs[tab_idx].select_tool_by_name(tool_id));
    app.tabs[tab_idx]
        .framework_mut()
        .unwrap()
        .active_tool
        .as_mut()
        .unwrap()
        .try_load_state(raw_state)
        .unwrap_or_else(|e| panic!("{tool_id} failed try_load_state: {e}"));

    let exported_preset = app.tabs[tab_idx].export_url_preset_json().unwrap().unwrap();
    assert!(
        exported_preset.len() > MAX_DECODED_URL_STATE_BYTES
            && exported_preset.len() <= MAX_DECODED_WORKSPACE_URL_BYTES,
        "{tool_id} preset len {} must be in 4097..=32768",
        exported_preset.len()
    );
    let expected_saved = app.tabs[tab_idx]
        .framework()
        .unwrap()
        .active_tool
        .as_ref()
        .unwrap()
        .save_state();

    let global_frag = app.encode_current_share_fragment().unwrap();
    assert!(global_frag.starts_with("#ws=") && global_frag.len() <= MAX_URL_FRAGMENT_BYTES);

    app.tabs[tab_idx]
        .framework_mut()
        .unwrap()
        .copy_active_share_link(ctx);
    let pane_copied = take_last_copied_share_url().unwrap();
    let pane_frag = format!("#{}", pane_copied.split_once('#').unwrap().1);
    assert!(pane_frag.starts_with("#ws=") && pane_frag.len() <= MAX_URL_FRAGMENT_BYTES);

    for frag in [&global_frag, &pane_frag] {
        let mut restored = MathExplorerApp::default();
        assert!(restored
            .apply_share_location_strings(frag, "", None)
            .unwrap());
        assert_eq!(restored.viewport_layout, ViewportLayout::Single);
        assert_eq!(restored.selected_tab, tab_idx);
        let actual_saved = restored.tabs[tab_idx]
            .framework()
            .unwrap()
            .active_tool
            .as_ref()
            .unwrap()
            .save_state();
        assert_eq!(
            actual_saved, expected_saved,
            "{tool_id} lossless #ws roundtrip"
        );
    }
}

#[test]
fn test_five_large_state_tools_exceeding_4kib_roundtrip_via_ws_within_8kib_fragment() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let cases = [
        (
            "graph_theory",
            "GraphEditorTool",
            build_large_graph_state(90),
        ),
        (
            "fluid_dynamics",
            "PotentialFlowTool",
            build_large_potential_flow_state(),
        ),
        (
            "favoritism",
            "FavoritismTool",
            build_large_favoritism_state(),
        ),
        (
            "game_theory",
            "ReplicatorDynamicsTool",
            build_large_replicator_state(),
        ),
        (
            "neuroscience",
            "NeuralNetworkVizTool",
            build_large_neural_network_viz_state(),
        ),
    ];
    let ctx = egui::Context::default();
    for (slug, tool_id, raw_state) in cases {
        assert_large_tool_ws_roundtrip(&ctx, slug, tool_id, &raw_state);
    }
}

#[test]
fn test_oversized_32kib_workspace_clears_url_hash_and_recommends_mws_export() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut app = MathExplorerApp::default();
    let ctx = egui::Context::default();
    let mut sync = UrlSyncController::default();

    // Establish initial baseline signature before switching to `graph_theory`
    let _ = sync.poll_at_millis(
        app.current_url_signature(),
        false,
        0,
        URL_SYNC_DEBOUNCE_MS,
        || unreachable!(),
    );
    let graph_idx = app
        .tabs
        .iter()
        .position(|t| t.url_slug() == "graph_theory")
        .unwrap();
    app.selected_tab = graph_idx;
    assert!(app.tabs[graph_idx].select_tool_by_name("GraphEditorTool"));
    let sig0 = app.current_url_signature();
    let _ = sync.poll_at_millis(sig0, false, 100, URL_SYNC_DEBOUNCE_MS, || unreachable!());
    let _ = sync.poll_at_millis(sig0, false, 450, URL_SYNC_DEBOUNCE_MS, || {
        app.encode_current_share_fragment()
    });
    assert!(last_replaced_browser_fragment()
        .as_deref()
        .unwrap_or("")
        .starts_with("#tab=graph_theory"));

    // Load a > 32,768-byte graph state: debounced URL sync must clear `location.hash` to `""`
    let huge_graph = build_large_graph_state(850);
    assert!(huge_graph.len() > MAX_DECODED_WORKSPACE_URL_BYTES);
    app.tabs[graph_idx]
        .framework_mut()
        .unwrap()
        .active_tool
        .as_mut()
        .unwrap()
        .try_load_state(&huge_graph)
        .unwrap();

    let sig1 = app.current_url_signature();
    let _ = sync.poll_at_millis(sig1, false, 1000, URL_SYNC_DEBOUNCE_MS, || unreachable!());
    let _ = sync.poll_at_millis(sig1, false, 1350, URL_SYNC_DEBOUNCE_MS, || {
        app.encode_current_share_fragment()
    });
    assert_eq!(sync.last_synced_fragment(), Some(""));
    assert_eq!(last_replaced_browser_fragment().as_deref(), Some(""));

    // Explicit `Copy Share Link` must surface `.mws` export guidance
    app.copy_current_share_link(&ctx);
    let (toast_msg, is_err) = app
        .session_manager
        .toast()
        .expect("overflow toast expected");
    assert!(*is_err && toast_msg.contains(".mws"));
}

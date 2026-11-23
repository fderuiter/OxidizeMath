//! Multi-pane workspace link (`&ws=<base64url>`) URL share unit tests.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[cfg(all(feature = "physics", feature = "pure_math"))]
#[test]
fn test_workspace_share_url_clamps_high_res_pane_presets_and_preserves_links() {
    let flow_preset: serde_json::Value = serde_json::from_str(
        &crate::presets::serialize_compact_preset_envelope(
            "Flow",
            "fluid_dynamics",
            serde_json::json!({
                "elements": [{"element_type": "Uniform", "strength": 10.0, "x": 0.0, "y": 0.0, "angle": 0.0}],
                "grid_size": 1000,
                "view_range": 10.0
            }),
        )
        .unwrap(),
    )
    .unwrap();
    let ws_json = serde_json::json!({
        "layout": "h",
        "links": [true, false],
        "focus": 1,
        "ratio": 0.5,
        "quad": [0.5, 0.5],
        "panes": [
            { "tab": "graph_theory", "tool": "GraphEditorTool" },
            { "tab": "fluid_dynamics", "tool": "PotentialFlowTool", "preset": flow_preset }
        ]
    })
    .to_string();
    let frag =
        encode_share_fragment_with_workspace("fluid_dynamics", None, None, Some(&ws_json)).unwrap();
    let mut app = crate::app::MathExplorerApp::default();
    assert_eq!(app.apply_share_location_strings(&frag, "", None), Ok(true));
    assert_eq!(app.pane_bus_links, [true, false, false, false]);
    let exported = app
        .pane_tab(1)
        .unwrap()
        .export_url_preset_json()
        .unwrap()
        .unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&exported).unwrap();
    assert_eq!(parsed["config"]["grid_size"], 100);
}

#[test]
fn test_hud_story_and_workspace_share_url_roundtrip() {
    let base_hud = encode_share_fragment_with_hud(
        "game_theory",
        Some("ReplicatorDynamicsTool"),
        Some("prisoners_dilemma_tragedy"),
        None,
        Some(1),
        None,
    )
    .unwrap();
    let ws_json = serde_json::json!({
        "layout": "h",
        "links": [true, true],
        "focus": 0,
        "ratio": 0.5,
        "quad": [0.5, 0.5],
        "panes": [
            { "tab": "game_theory", "tool": "ReplicatorDynamicsTool" },
            { "tab": "chaos", "tool": "KuramotoOscillatorTool" }
        ]
    })
    .to_string();
    let combined = append_workspace_to_share_fragment(&base_hud, Some(&ws_json)).unwrap();
    let parsed = parse_share_location(&combined, "").unwrap().unwrap();
    assert_eq!(parsed.tab, "game_theory");
    assert_eq!(parsed.preset.as_deref(), Some("prisoners_dilemma_tragedy"));
    assert_eq!(parsed.step, Some(1));
    assert_eq!(parsed.workspace_json.as_deref(), Some(ws_json.as_str()));
}

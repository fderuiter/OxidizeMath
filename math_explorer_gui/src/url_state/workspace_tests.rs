//! Unit tests for DEFLATE `#ws=<base64url>` v2 workspace codec, zip-bomb defense,
//! non-share anchor handling, and multi-pane `UrlStateSignature` debounced sync.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{
    decode_workspace_share_param, encode_base64url, encode_single_pane_smart_fragment,
    encode_workspace_share_fragment, last_replaced_browser_fragment, parse_share_location,
    parse_workspace_or_share_location, PaneUrlSignature, ParsedShareLocation, ShareUrlParams,
    UrlStateSignature, UrlSyncController, WorkspaceShareParams, MAX_DECODED_URL_STATE_BYTES,
    MAX_DECODED_WORKSPACE_URL_BYTES, MAX_URL_FRAGMENT_BYTES,
};
use crate::app::{ComparisonSyncConfig, ViewportLayout};
use flate2::write::DeflateEncoder;
use flate2::Compression;
use std::io::Write;

fn sample_pane(tab: &str, tool: Option<&str>, x0: f64) -> ShareUrlParams {
    let state_json = serde_json::json!({
        "metadata": {
            "schema_version": "1.0",
            "title": "Test",
            "domain": tab,
            "timestamp": "2026-11-22T00:00:00Z"
        },
        "config": {
            "x0": x0,
            "grid_size": 120
        }
    })
    .to_string();
    ShareUrlParams {
        tab: tab.to_string(),
        tool: tool.map(str::to_string),
        state_json: Some(state_json),
    }
}

fn assert_layout_ws_roundtrip(layout: ViewportLayout, count: usize, focused: usize) {
    let panes: Vec<ShareUrlParams> = (0..count)
        .map(|i| {
            sample_pane(
                "fluid_dynamics",
                Some("PotentialFlowTool"),
                1.5 + f64::from(i as u32),
            )
        })
        .collect();
    let sync_cfg = ComparisonSyncConfig {
        link_axes: true,
        link_cursors: count == 4,
        sync_playback: count > 1,
    };
    let input = WorkspaceShareParams {
        layout,
        split_ratio: 0.64,
        quad_split_ratios: (0.35, 0.72),
        focused_pane: focused,
        compare_sync: sync_cfg,
        panes,
    };
    let frag = encode_workspace_share_fragment(&input).expect("v2 ws fragment must encode");
    assert!(frag.starts_with("#ws=") && frag.len() <= MAX_URL_FRAGMENT_BYTES);

    let parsed = parse_workspace_or_share_location(&frag, "")
        .expect("v2 ws fragment must parse")
        .expect("must return Some(Workspace)");
    let ParsedShareLocation::Workspace(decoded) = parsed else {
        panic!("Expected Workspace variant");
    };
    assert_eq!(decoded.layout, layout);
    assert_eq!(decoded.focused_pane, focused);
    assert_eq!(decoded.panes.len(), count);
    if matches!(
        layout,
        ViewportLayout::SplitHorizontal | ViewportLayout::SplitVertical
    ) {
        assert!((decoded.split_ratio - 0.64).abs() < 1e-5);
    }
    if layout == ViewportLayout::QuadGrid {
        assert!((decoded.quad_split_ratios.0 - 0.35).abs() < 1e-5);
        assert!((decoded.quad_split_ratios.1 - 0.72).abs() < 1e-5);
    }
    if layout != ViewportLayout::Single {
        assert_eq!(decoded.compare_sync, sync_cfg);
    }
    for pane in &decoded.panes {
        let v: serde_json::Value =
            serde_json::from_str(pane.state_json.as_deref().unwrap()).unwrap();
        assert_eq!(
            v["config"]["grid_size"].as_u64(),
            Some(crate::presets::MAX_INTERACTIVE_GRID_SIZE as u64)
        );
    }
}

#[test]
fn test_workspace_v2_deflate_roundtrip_all_layouts_ratios_and_compare_sync() {
    for (layout, count, focused) in [
        (ViewportLayout::Single, 1, 0),
        (ViewportLayout::SplitHorizontal, 2, 1),
        (ViewportLayout::SplitVertical, 2, 0),
        (ViewportLayout::QuadGrid, 4, 3),
    ] {
        assert_layout_ws_roundtrip(layout, count, focused);
    }
}

#[test]
fn test_workspace_zip_bomb_capped_at_32kib_before_json_parsing() {
    // 256 KiB of repetitive JSON compresses down to < 600 bytes raw DEFLATE
    let repeated = format!(
        r#"{{"v":2,"layout":"single","focused_pane":0,"panes":[{{"tab":"chaos","state":{{"metadata":{{"schema_version":"1.0"}},"config":{{"pad":"{}"}}}}}}]}}"#,
        "A".repeat(256 * 1024)
    );
    assert!(repeated.len() > MAX_DECODED_WORKSPACE_URL_BYTES * 4);

    let mut enc = DeflateEncoder::new(Vec::new(), Compression::best());
    enc.write_all(repeated.as_bytes()).unwrap();
    let compressed = enc.finish().unwrap();
    let ws_token = encode_base64url(&compressed);
    assert!(
        ws_token.len() < 2048,
        "Compressed zip-bomb fits under 8 KiB URL cap"
    );

    let err = decode_workspace_share_param(&ws_token).unwrap_err();
    assert!(
        err.contains("32768") || err.contains("exceeds"),
        "Zip-bomb must abort at 32 KiB decompression cap, got: {err}"
    );
}

#[test]
fn test_workspace_rejects_wrong_version_mixed_params_and_invalid_pane_counts() {
    let wrong_ver = r#"{"v":99,"layout":"Single","focused_pane":0,"panes":[{"tab":"chaos"}]}"#;
    let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
    enc.write_all(wrong_ver.as_bytes()).unwrap();
    let tok = encode_base64url(&enc.finish().unwrap());
    let err = decode_workspace_share_param(&tok).unwrap_err();
    assert!(err.contains("Unsupported workspace URL schema version"));

    let bad_count = r#"{"v":2,"layout":"QuadGrid","focused_pane":0,"panes":[{"tab":"chaos"},{"tab":"physics"}]}"#;
    let mut enc2 = DeflateEncoder::new(Vec::new(), Compression::default());
    enc2.write_all(bad_count.as_bytes()).unwrap();
    let tok2 = encode_base64url(&enc2.finish().unwrap());
    assert!(decode_workspace_share_param(&tok2).is_err());

    let valid_ws = encode_workspace_share_fragment(&WorkspaceShareParams {
        layout: ViewportLayout::Single,
        split_ratio: 0.5,
        quad_split_ratios: (0.5, 0.5),
        focused_pane: 0,
        compare_sync: ComparisonSyncConfig::default(),
        panes: vec![sample_pane("chaos", Some("AttractorPlotter"), 1.0)],
    })
    .unwrap();
    let mixed = format!("{valid_ws}&tab=physics");
    assert!(parse_workspace_or_share_location(&mixed, "").is_err());
    assert!(parse_workspace_or_share_location("#ws=", "").is_err());
    assert!(parse_workspace_or_share_location("#ws=***not_base64***", "").is_err());
}

#[test]
fn test_non_share_anchors_return_ok_none_while_malformed_share_params_error() {
    for (hash, search) in [
        ("#overview", ""),
        ("#section-methods", ""),
        ("#utm_source=docs", ""),
        ("#utm_source=docs&utm_medium=referral", ""),
        ("", "?utm_source=github&keep=1"),
        ("#overview", "?utm_campaign=spring"),
    ] {
        assert_eq!(
            parse_share_location(hash, search),
            Ok(None),
            "parse_share_location should ignore non-share anchor ({hash}, {search})"
        );
        assert_eq!(
            parse_workspace_or_share_location(hash, search),
            Ok(None),
            "parse_workspace_or_share_location should ignore non-share anchor ({hash}, {search})"
        );
    }

    for malformed in [
        "#tab=chaos&unknown=1",
        "#tool=AttractorPlotter",
        "#state=e30",
        "#tab=",
        "#ws=bad!token",
    ] {
        assert!(
            parse_workspace_or_share_location(malformed, "").is_err(),
            "Malformed share fragment `{malformed}` must still fail validation"
        );
    }
}

#[test]
fn test_single_pane_smart_fragment_uses_v1_under_4kib_and_ws_above_4kib() {
    let small = sample_pane("chaos", Some("AttractorPlotter"), 2.0);
    assert!(small.state_json.as_ref().unwrap().len() <= MAX_DECODED_URL_STATE_BYTES);
    let v1_frag = encode_single_pane_smart_fragment(
        &small.tab,
        small.tool.as_deref(),
        small.state_json.as_deref(),
    )
    .unwrap();
    assert!(v1_frag.starts_with("#tab=chaos&tool=AttractorPlotter&state="));

    let entries: Vec<serde_json::Value> = (0..220)
        .map(|i| serde_json::json!([i, (i + 1) % 220, 1.234567 + f64::from(i) * 0.01]))
        .collect();
    let large_json = serde_json::json!({
        "metadata": { "schema_version": "1.0", "title": "Large", "domain": "graph_theory" },
        "config": { "edges": entries }
    })
    .to_string();
    assert!(large_json.len() > MAX_DECODED_URL_STATE_BYTES);
    assert!(large_json.len() <= MAX_DECODED_WORKSPACE_URL_BYTES);

    let ws_frag = encode_single_pane_smart_fragment(
        "graph_theory",
        Some("GraphEditorTool"),
        Some(&large_json),
    )
    .unwrap();
    assert!(ws_frag.starts_with("#ws="));
    assert!(ws_frag.len() <= MAX_URL_FRAGMENT_BYTES);

    let oversize_json = serde_json::json!({
        "metadata": { "schema_version": "1.0" },
        "config": { "blob": "X".repeat(MAX_DECODED_WORKSPACE_URL_BYTES + 256) }
    })
    .to_string();
    let err = encode_single_pane_smart_fragment(
        "graph_theory",
        Some("GraphEditorTool"),
        Some(&oversize_json),
    )
    .unwrap_err();
    assert!(
        err.contains(".mws"),
        "Expected .mws export guidance, got: {err}"
    );
}

#[test]
fn test_multi_pane_url_signature_debounces_and_clears_hash_on_overflow() {
    let mut ctrl = UrlSyncController::default();
    let mut sig = UrlStateSignature {
        viewport_layout: ViewportLayout::SplitHorizontal,
        split_ratio_bits: 0.5_f32.to_bits(),
        focused_pane: 0,
        pane_revisions: [
            PaneUrlSignature {
                tab_idx: 1,
                active_tool_id: Some("AttractorPlotter"),
                state_revision: 10,
            },
            PaneUrlSignature {
                tab_idx: 3,
                active_tool_id: Some("WaveSimulator"),
                state_revision: 20,
            },
            PaneUrlSignature::default(),
            PaneUrlSignature::default(),
        ],
        ..UrlStateSignature::default()
    };
    ctrl.mark_synced(sig, Some("#ws=initial".to_string()));

    // Secondary pane parameter change (pane 1 revision 20 -> 21) triggers 300 ms debounce
    sig.pane_revisions[1].state_revision = 21;
    assert_eq!(
        ctrl.poll_at_millis(sig, false, 1000, 300, || unreachable!()),
        Some(std::time::Duration::from_millis(300))
    );
    assert_eq!(
        ctrl.poll_at_millis(
            sig,
            false,
            1300,
            300,
            || Ok("#ws=updated_pane1".to_string())
        ),
        None
    );
    assert_eq!(ctrl.last_synced_fragment(), Some("#ws=updated_pane1"));
    assert_eq!(
        last_replaced_browser_fragment().as_deref(),
        Some("#ws=updated_pane1")
    );

    // Focused pane change (0 -> 1) and overflow (> 32 KiB) clears browser fragment to ""
    sig.focused_pane = 1;
    let _ = ctrl.poll_at_millis(sig, false, 2000, 300, || unreachable!());
    let _ = ctrl.poll_at_millis(sig, false, 2350, 300, || {
        Err("exceeds 32768-byte limit; use .mws".to_string())
    });
    assert_eq!(ctrl.last_synced_fragment(), Some(""));
    assert_eq!(last_replaced_browser_fragment().as_deref(), Some(""));
}

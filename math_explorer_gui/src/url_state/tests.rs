#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::web_io::PlatformInstant;
use std::time::Duration;

#[test]
fn test_base64url_rfc4648_roundtrip_and_url_safe_alphabet() {
    assert_eq!(encode_base64url(b""), "");
    assert_eq!(decode_base64url("", 64).unwrap(), b"");

    for len in [1_usize, 2, 3, 4, 5, 7, 31, 64, 255, 512] {
        let payload: Vec<u8> = (0..len)
            .map(|i| ((i.wrapping_mul(73).wrapping_add(19)) & 0xff) as u8)
            .collect();
        let encoded = encode_base64url(&payload);
        assert!(
            !encoded.contains('+') && !encoded.contains('/') && !encoded.contains('='),
            "base64url must be unpadded and URL-safe: {encoded}"
        );
        let decoded = decode_base64url(&encoded, 1024).unwrap();
        assert_eq!(decoded, payload);
    }

    // Force 0xfb / 0xff bytes so '-' (62) and '_' (63) appear in output
    let high_bits = [0xfb_u8, 0xff, 0xfe];
    let enc_high = encode_base64url(&high_bits);
    assert!(enc_high.contains('-') && enc_high.contains('_'));
    assert_eq!(decode_base64url(&enc_high, 16).unwrap(), high_bits);
}

#[test]
fn test_base64url_rejects_corrupted_padding_noncanonical_bits_and_oversized_decode() {
    assert!(decode_base64url("A", 64).is_err());
    assert!(decode_base64url("abc+def_", 64).is_err());
    assert!(decode_base64url("abc/def_", 64).is_err());
    assert!(decode_base64url("ab=c", 64).is_err());
    assert!(decode_base64url("YQ=", 64).is_err());
    assert_eq!(decode_base64url("YQ==", 64).unwrap(), b"a");

    // 'YR' has non-zero low 4 bits in the 2nd sextet for a 1-byte payload
    assert!(decode_base64url("YR", 64).is_err());

    let five_kib = vec![b'x'; MAX_DECODED_URL_STATE_BYTES + 1];
    let enc = encode_base64url(&five_kib);
    let err = decode_base64url(&enc, MAX_DECODED_URL_STATE_BYTES).unwrap_err();
    assert!(err.contains("exceeds maximum allowed limit"));
}

#[test]
fn test_parse_share_location_hash_precedence_and_query_fallback() {
    assert_eq!(parse_share_location("", "").unwrap(), None);
    assert_eq!(parse_share_location("#the_canvas_id", "").unwrap(), None);
    assert_eq!(
        parse_share_location("", "?stall_timeout_ms=8000&debug=1").unwrap(),
        None
    );

    let json = crate::presets::serialize_compact_preset_envelope(
        "Test",
        "traceability",
        serde_json::json!({ "repo_path": null }),
    )
    .unwrap();
    let frag = encode_share_fragment("traceability", None, Some(&json)).unwrap();

    // Hash takes precedence over query string
    let from_hash = parse_share_location(&frag, "?tab=other_tab&stall_timeout_ms=5000")
        .unwrap()
        .unwrap();
    assert_eq!(from_hash.tab, "traceability");
    assert_eq!(from_hash.tool, None);
    assert_eq!(from_hash.state_json.as_deref(), Some(json.as_str()));

    // Query fallback works and ignores unrelated parameters like stall_timeout_ms
    let query = format!("?stall_timeout_ms=5000&{}", frag.strip_prefix('#').unwrap());
    let from_query = parse_share_location("", &query).unwrap().unwrap();
    assert_eq!(from_query, from_hash);
}

#[test]
fn test_parse_share_location_enforces_size_caps_and_syntax_rules() {
    let huge_hash = format!("#tab=chaos&state={}", "A".repeat(MAX_URL_FRAGMENT_BYTES));
    assert!(parse_share_location(&huge_hash, "")
        .unwrap_err()
        .contains("8192"));

    let huge_query = format!("?tab=chaos&state={}", "A".repeat(MAX_URL_FRAGMENT_BYTES));
    assert!(parse_share_location("", &huge_query)
        .unwrap_err()
        .contains("8192"));

    let oversized_json = format!(r#"{{"pad":"{}"}}"#, "x".repeat(MAX_DECODED_URL_STATE_BYTES));
    assert!(encode_share_fragment("chaos", None, Some(&oversized_json)).is_err());
    let oversized_b64 = encode_base64url(oversized_json.as_bytes());
    let oversized_frag = format!("#tab=chaos&state={oversized_b64}");
    assert!(parse_share_location(&oversized_frag, "")
        .unwrap_err()
        .contains("4096"));

    assert!(parse_share_location("#tool=AttractorPlotter", "").is_err());
    assert!(parse_share_location("#tab=&tool=AttractorPlotter", "").is_err());
    assert!(parse_share_location("#tab=chaos&tool=", "").is_err());
    assert!(parse_share_location("#tab=chaos&state=", "").is_err());
    assert!(parse_share_location("#tab=chaos&tab=ai", "").is_err());
    assert!(parse_share_location("#tab=chaos&unknown_key=1", "").is_err());
    assert!(parse_share_location("#tab=%ZZ", "").is_err());

    // Overflow float 1e309 in JSON must be rejected as non-finite
    let overflow_b64 = encode_base64url(br#"{"x":1e309}"#);
    let overflow_frag = format!("#tab=chaos&state={overflow_b64}");
    assert!(parse_share_location(&overflow_frag, "").is_err());
}

#[test]
fn test_url_sync_controller_debounces_300ms_and_defers_pointer_drag() {
    let mut sync = UrlSyncController::default();
    let t0 = PlatformInstant::now();
    let debounce = Duration::from_millis(URL_SYNC_DEBOUNCE_MS);
    let mut encode_calls = 0_usize;
    let sig = |rev| UrlStateSignature {
        selected_tab: 0,
        active_tool_id: Some("ToolA"),
        state_revision: rev,
        ..UrlStateSignature::default()
    };

    assert_eq!(
        sync.poll_with_now(sig(1), false, t0, debounce, || {
            encode_calls += 1;
            Ok("#tab=a".to_string())
        }),
        None
    );
    assert_eq!(encode_calls, 0);

    for (rev, dragging, now_ms) in [(2, false, 50), (3, true, 400), (3, false, 650)] {
        assert!(sync
            .poll_at_millis(sig(rev), dragging, now_ms, URL_SYNC_DEBOUNCE_MS, || {
                encode_calls += 1;
                Ok("#tab=a&state=2".to_string())
            })
            .is_some());
        assert_eq!(encode_calls, 0);
    }

    for now_ms in [705_u64, 1200] {
        assert_eq!(
            sync.poll_at_millis(sig(3), false, now_ms, URL_SYNC_DEBOUNCE_MS, || {
                encode_calls += 1;
                Ok("#tab=a&state=2".to_string())
            }),
            None
        );
    }
    assert_eq!(encode_calls, 1);
    assert_eq!(sync.last_synced_fragment(), Some("#tab=a&state=2"));
}

#[test]
fn test_cross_platform_share_url_origin_desktop_and_web() {
    let desktop_url = build_share_url_from_fragment("#tab=traceability");
    assert_eq!(
        desktop_url,
        "https://fderuiter.github.io/OxidizeMath/#tab=traceability"
    );

    let web_url = with_mock_web_base_url(
        "https://sim.example.edu/OxidizeMath/?stall_timeout_ms=9000&compile_timeout_ms=30000&tab=stale&state=old#prev",
        || build_share_url_from_fragment("#tab=chaos&tool=AttractorPlotter"),
    );
    assert_eq!(
        web_url,
        "https://sim.example.edu/OxidizeMath/?stall_timeout_ms=9000&compile_timeout_ms=30000#tab=chaos&tool=AttractorPlotter"
    );
}

#[test]
fn test_tick_url_sync_defers_during_active_animated_slider_sweeps() {
    let ctx = eframe::egui::Context::default();
    let mut app = crate::app::MathExplorerApp::default();
    let _ = ctx.run(eframe::egui::RawInput::default(), |egui_ctx| {
        app.tick_url_sync(egui_ctx);
    });
    assert_eq!(app.url_sync_last_fragment_for_test(), None);

    // Mark state dirty at t = 0
    app.selected_tab = 1;
    let _ = ctx.run(eframe::egui::RawInput::default(), |egui_ctx| {
        app.tick_url_sync(egui_ctx);
    });
    assert_eq!(app.url_sync_last_fragment_for_test(), None);

    // Advance past 300 ms while an AnimatedSlider sweep is active: flush must stay deferred
    std::thread::sleep(Duration::from_millis(URL_SYNC_DEBOUNCE_MS + 20));
    let _ = ctx.run(eframe::egui::RawInput::default(), |egui_ctx| {
        let flags = crate::reflective_ui::slider::SweepFrameFlags {
            pass_nr: Some(egui_ctx.cumulative_pass_nr()),
            active_sweep_count: 1,
            mutation_count: 1,
        };
        egui_ctx.data_mut(|d| {
            d.insert_temp(eframe::egui::Id::new("ANIMATED_SLIDER_SWEEP_FLAGS"), flags)
        });
        app.tick_url_sync(egui_ctx);
    });
    assert_eq!(app.url_sync_last_fragment_for_test(), None);

    // Once the sweep stops and another 300 ms elapses, the deferred URL fragment flushes cleanly
    std::thread::sleep(Duration::from_millis(URL_SYNC_DEBOUNCE_MS + 20));
    let _ = ctx.run(eframe::egui::RawInput::default(), |egui_ctx| {
        crate::reflective_ui::slider::reset_frame_sweep_flags(egui_ctx);
        app.tick_url_sync(egui_ctx);
    });
    assert!(app
        .url_sync_last_fragment_for_test()
        .is_some_and(|frag| frag.starts_with("#tab=")));
}

#[test]
fn test_preset_challenge_and_step_url_codec_roundtrip_and_bounds() {
    let ch_frag = encode_share_fragment_with_hud(
        "financial_math",
        Some("BankrollGrowthTool"),
        None,
        Some("kelly_ruin_trap"),
        None,
        None,
    )
    .unwrap();
    assert!(ch_frag.len() <= MAX_URL_FRAGMENT_BYTES);
    let parsed_ch = parse_share_location(&ch_frag, "").unwrap().unwrap();
    assert_eq!(parsed_ch.tab, "financial_math");
    assert_eq!(parsed_ch.tool.as_deref(), Some("BankrollGrowthTool"));
    assert_eq!(parsed_ch.challenge.as_deref(), Some("kelly_ruin_trap"));
    assert_eq!(parsed_ch.preset, None);
    assert_eq!(parsed_ch.step, None);

    let st_frag = encode_share_fragment_with_hud(
        "game_theory",
        Some("ReplicatorDynamicsTool"),
        Some("prisoners_dilemma_tragedy"),
        None,
        Some(2),
        None,
    )
    .unwrap();
    let parsed_st = parse_share_location(&st_frag, "").unwrap().unwrap();
    assert_eq!(
        parsed_st.preset.as_deref(),
        Some("prisoners_dilemma_tragedy")
    );
    assert_eq!(parsed_st.step, Some(2));

    // Reject simultaneous preset + challenge, step without preset, or malformed step
    assert!(parse_share_location("#tab=chaos&preset=a&challenge=b", "").is_err());
    assert!(parse_share_location("#tab=chaos&challenge=b&step=1", "").is_err());
    assert!(parse_share_location("#tab=chaos&preset=a&step=999", "").is_err());
    assert!(parse_share_location("#tab=chaos&preset=", "").is_err());
}

#[test]
fn test_shared_url_clamps_high_res_surface_and_flow_grids_to_interactive_caps() {
    let curv_json = crate::presets::serialize_compact_preset_envelope(
        "Curvature",
        "geometry_topology",
        serde_json::json!({
            "surface_type": "Torus",
            "curvature_type": "Gaussian",
            "sphere_radius": 1.0,
            "torus_major_radius": 2.0,
            "torus_minor_radius": 0.5,
            "klein_radius": 2.0,
            "u_resolution": 1000,
            "v_resolution": 1000,
            "pitch": 0.5,
            "yaw": 0.5,
            "zoom": 1.0
        }),
    )
    .unwrap();
    let curv_frag = encode_share_fragment(
        "geometry_topology",
        Some("CurvatureHeatmap"),
        Some(&curv_json),
    )
    .unwrap();
    let parsed_curv = parse_share_location(&curv_frag, "").unwrap().unwrap();
    let curv_cfg: serde_json::Value =
        serde_json::from_str(parsed_curv.state_json.as_deref().unwrap()).unwrap();
    assert_eq!(curv_cfg["config"]["u_resolution"], 120);
    assert_eq!(curv_cfg["config"]["v_resolution"], 120);

    let flow_json = crate::presets::serialize_compact_preset_envelope(
        "Flow",
        "fluid_dynamics",
        serde_json::json!({
            "elements": [{"element_type": "Uniform", "strength": 10.0, "x": 0.0, "y": 0.0, "angle": 0.0}],
            "grid_size": 1000,
            "view_range": 10.0
        }),
    )
    .unwrap();
    let flow_frag = encode_share_fragment(
        "fluid_dynamics",
        Some("PotentialFlowTool"),
        Some(&flow_json),
    )
    .unwrap();
    let mut app = crate::app::MathExplorerApp::default();
    assert_eq!(
        app.apply_share_location_strings(&flow_frag, "", None),
        Ok(true)
    );
    let parsed_flow = parse_share_location(&flow_frag, "").unwrap().unwrap();
    let flow_cfg: serde_json::Value =
        serde_json::from_str(parsed_flow.state_json.as_deref().unwrap()).unwrap();
    assert_eq!(flow_cfg["config"]["grid_size"], 100);
}

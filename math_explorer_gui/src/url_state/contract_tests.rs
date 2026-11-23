#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::app::{MathExplorerApp, ThemeMode, ViewportLayout};
use crate::framework::{StarterPresetMetadata, ToolMetadata};
use eframe::egui;
use std::collections::HashSet;

fn perturb_json_numbers(val: &mut serde_json::Value) -> bool {
    match val {
        serde_json::Value::Number(num) => {
            if let Some(f) = num.as_f64() {
                if f.is_finite() && num.as_i64().is_none() && num.as_u64().is_none() {
                    let candidate = if f.abs() > 1e-3 { f * 1.05 } else { 0.125 };
                    if let Some(n) = serde_json::Number::from_f64(candidate) {
                        *num = n;
                        return true;
                    }
                }
            }
            false
        }
        serde_json::Value::Object(map) => {
            for (k, v) in map.iter_mut() {
                if k == "schema_version" || k == "domain" || k == "timestamp" {
                    continue;
                }
                if perturb_json_numbers(v) {
                    return true;
                }
            }
            false
        }
        serde_json::Value::Array(arr) => arr.iter_mut().any(perturb_json_numbers),
        _ => false,
    }
}

fn split_share_url(url: &str) -> (&str, &str) {
    let idx = url.find('#').expect("share URL must include '#' fragment");
    (&url[idx..], "")
}

fn seed_custom_tab_states(app: &mut MathExplorerApp) {
    for tab in &mut app.tabs {
        match tab.url_slug() {
            "morphogenesis" => {
                let mut val: serde_json::Value =
                    serde_json::from_str(&tab.export_url_preset_json().unwrap().unwrap()).unwrap();
                val["config"]["a"] = serde_json::json!(0.17);
                val["config"]["b"] = serde_json::json!(0.83);
                val["config"]["d_u"] = serde_json::json!(1.25);
                val["config"]["d_v"] = serde_json::json!(64.0);
                val["config"]["dt"] = serde_json::json!(0.03);
                val["config"]["width"] = serde_json::json!(72);
                val["config"]["height"] = serde_json::json!(72);
                tab.import_url_preset_json(Some("MorphogenesisTool"), Some(&val.to_string()))
                    .unwrap();
            }
            "mri" => {
                let mut val: serde_json::Value =
                    serde_json::from_str(&tab.export_url_preset_json().unwrap().unwrap()).unwrap();
                val["config"]["t1"] = serde_json::json!(2.25);
                val["config"]["t2"] = serde_json::json!(0.35);
                val["config"]["bx"] = serde_json::json!(0.02);
                val["config"]["by"] = serde_json::json!(-0.01);
                val["config"]["bz"] = serde_json::json!(1.5);
                val["config"]["time_scale"] = serde_json::json!(4.5);
                tab.import_url_preset_json(Some("MriTool"), Some(&val.to_string()))
                    .unwrap();
            }
            "favoritism" => {
                let orig = tab.export_url_preset_json().unwrap().unwrap();
                let mut val: serde_json::Value = serde_json::from_str(&orig).unwrap();
                val["config"]["new_child_name"] = serde_json::json!("Shared Sibling");
                val["config"]["children"][0]["inputs"]["wealth"] = serde_json::json!(6.75);
                tab.import_url_preset_json(Some("FavoritismTool"), Some(&val.to_string()))
                    .unwrap();
            }
            _ => {}
        }
    }
}

#[test]
fn test_contract_all_21_tabs_unique_slugs_and_session_preserving_roundtrip() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut source_app = MathExplorerApp::default();
    assert_eq!(source_app.tabs.len(), 21, "Expected all 21 ExplorerTabs");

    let mut slugs = HashSet::new();
    for tab in &source_app.tabs {
        assert!(
            slugs.insert(tab.url_slug()),
            "Duplicate tab URL slug '{}'",
            tab.url_slug()
        );
    }
    seed_custom_tab_states(&mut source_app);

    for target_idx in 0..source_app.tabs.len() {
        source_app.selected_tab = target_idx;
        let share_url = source_app.encode_current_share_url().unwrap();
        let expected_preset = source_app.tabs[target_idx]
            .export_url_preset_json()
            .unwrap();
        let (hash, search) = split_share_url(&share_url);

        // Build destination app with HighContrast theme, SplitHorizontal layout, and cached tab state
        let mut dest_app = MathExplorerApp::default();
        dest_app.theme = ThemeMode::HighContrast;
        dest_app.set_layout(ViewportLayout::SplitHorizontal);
        let other_idx = (target_idx + 1) % dest_app.tabs.len();
        let cached_other_before = dest_app.tabs[other_idx].save_state();

        let hydrated = dest_app
            .apply_share_location_strings(hash, search, None)
            .unwrap();
        assert!(hydrated);
        assert_eq!(dest_app.selected_tab, target_idx);
        assert_eq!(dest_app.viewport_layout, ViewportLayout::Single);
        assert_eq!(dest_app.theme, ThemeMode::HighContrast);
        assert_eq!(dest_app.tabs[other_idx].save_state(), cached_other_before);
        assert_eq!(
            dest_app.tabs[target_idx].export_url_preset_json().unwrap(),
            expected_preset,
            "Tab '{}' failed lossless URL round-trip",
            dest_app.tabs[target_idx].name()
        );
    }
}

#[test]
fn test_contract_all_55_tools_lossless_url_roundtrip() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let tools: Vec<&'static ToolMetadata> = inventory::iter::<ToolMetadata>.into_iter().collect();
    assert_eq!(
        tools.len(),
        61,
        "Expected all 61 registered InteractiveTools"
    );

    let mut source_app = MathExplorerApp::default();
    for meta in tools {
        let tab_idx = source_app
            .tabs
            .iter()
            .position(|t| matches_tab_slug(meta.domain, t.url_slug(), t.name()))
            .unwrap_or_else(|| panic!("No tab found for tool domain '{}'", meta.domain));

        source_app.selected_tab = tab_idx;
        assert!(
            source_app.tabs[tab_idx].select_tool_by_name(meta.id),
            "Failed to select tool '{}'",
            meta.id
        );

        // Perturb float parameter when accepted by tool validator so non-default state is tested
        if let Some(orig_envelope) = source_app.tabs[tab_idx].export_url_preset_json().unwrap() {
            let mut parsed: serde_json::Value = serde_json::from_str(&orig_envelope).unwrap();
            if perturb_json_numbers(&mut parsed) {
                let candidate = parsed.to_string();
                let _ = source_app.tabs[tab_idx]
                    .import_url_preset_json(Some(meta.id), Some(&candidate));
            }
        }

        let expected_tool_state = source_app.tabs[tab_idx]
            .framework()
            .and_then(|fw| fw.active_tool.as_ref())
            .and_then(|t| t.save_state());
        let share_url = source_app.encode_current_share_url().unwrap();
        let (hash, _) = split_share_url(&share_url);
        assert!(hash.len() <= MAX_URL_FRAGMENT_BYTES);

        let mut dest_app = MathExplorerApp::default();
        assert!(dest_app
            .apply_share_location_strings(hash, "", None)
            .unwrap());
        assert_eq!(dest_app.selected_tab, tab_idx);
        assert_eq!(dest_app.tabs[tab_idx].active_tool_id(), Some(meta.id));
        let restored_tool_state = dest_app.tabs[tab_idx]
            .framework()
            .and_then(|fw| fw.active_tool.as_ref())
            .and_then(|t| t.save_state());
        assert_eq!(
            restored_tool_state, expected_tool_state,
            "Tool '{}' in domain '{}' failed lossless URL round-trip",
            meta.id, meta.domain
        );
    }
}

#[test]
fn test_contract_all_59_starter_presets_lossless_url_roundtrip() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let presets: Vec<&'static StarterPresetMetadata> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .collect();
    assert_eq!(
        presets.len(),
        73,
        "Expected all 73 StarterPresetMetadata entries"
    );

    for preset in presets {
        let mut source_app = MathExplorerApp::default();
        let tab_idx = source_app
            .tabs
            .iter()
            .position(|t| matches_tab_slug(preset.domain, t.url_slug(), t.name()))
            .unwrap_or_else(|| panic!("No tab found for starter preset '{}'", preset.id));

        source_app.selected_tab = tab_idx;
        let raw_preset = (preset.preset_generator)();
        if let Some(fw) = source_app.tabs[tab_idx].framework_mut() {
            fw.apply_starter_card(preset.title, preset.tool_name, Some(raw_preset))
                .unwrap_or_else(|e| panic!("Starter '{}' failed to apply: {e}", preset.id));
        } else {
            source_app.tabs[tab_idx]
                .import_url_preset_json(None, Some(raw_preset))
                .unwrap_or_else(|e| panic!("Standalone starter '{}' failed: {e}", preset.id));
        }

        let expected_json = source_app.tabs[tab_idx].export_url_preset_json().unwrap();
        if let Some(ref j) = expected_json {
            assert!(j.len() <= MAX_DECODED_URL_STATE_BYTES);
        }
        let share_url = source_app.encode_current_share_url().unwrap();
        let (hash, _) = split_share_url(&share_url);
        assert!(hash.len() <= MAX_URL_FRAGMENT_BYTES);

        // Test both hash and query-parameter fallback round-trips
        let query = format!("?stall_timeout_ms=6000&{}", hash.strip_prefix('#').unwrap());
        for (h, q) in [(hash, ""), ("", query.as_str())] {
            let mut dest_app = MathExplorerApp::default();
            assert!(dest_app.apply_share_location_strings(h, q, None).unwrap());
            assert_eq!(dest_app.selected_tab, tab_idx);
            assert_eq!(
                dest_app.tabs[tab_idx].export_url_preset_json().unwrap(),
                expected_json,
                "Starter preset '{}' failed lossless URL round-trip",
                preset.id
            );
        }
    }
}

fn build_invalid_share_url_cases() -> Vec<String> {
    let unknown_field_b64 = encode_base64url(
        br#"{"metadata":{"schema_version":"1.0","title":"Bad","domain":"chaos","timestamp":"0"},"config":{"evil":123}}"#,
    );
    let wrong_domain_b64 = encode_base64url(
        br#"{"metadata":{"schema_version":"1.0","title":"Bad","domain":"quantum","timestamp":"0"},"config":{}}"#,
    );
    let wrong_schema_b64 = encode_base64url(
        br#"{"metadata":{"schema_version":"2.0","title":"Bad","domain":"chaos","timestamp":"0"},"config":{}}"#,
    );
    let bad_morpho_b64 = encode_base64url(
        br#"{"metadata":{"schema_version":"1.0","title":"Bad","domain":"biology","timestamp":"0"},"config":{"a":0.1,"b":0.9,"d_u":1.0,"d_v":50.0,"dt":-0.5,"width":64,"height":64}}"#,
    );
    let bad_fractal_b64 = encode_base64url(
        br#"{"metadata":{"schema_version":"1.0","title":"Bad","domain":"chaos","timestamp":"0"},"config":{"center_re":0.0,"center_im":0.0,"zoom":-5.0,"max_iter":100,"mode":"Mandelbrot","julia_re":0.0,"julia_im":0.0}}"#,
    );
    let traversal_trace_b64 = encode_base64url(
        br#"{"metadata":{"schema_version":"1.0","title":"Bad","domain":"traceability","timestamp":"0"},"config":{"repo_path":"../../etc/passwd"}}"#,
    );
    vec![
        "#tab=chaos&tool=AttractorPlotter&state=***invalid_base64***".to_string(),
        format!("#tab=chaos&tool=AttractorPlotter&state={unknown_field_b64}"),
        format!("#tab=chaos&tool=AttractorPlotter&state={wrong_domain_b64}"),
        format!("#tab=chaos&tool=AttractorPlotter&state={wrong_schema_b64}"),
        format!("#tab=morphogenesis&state={bad_morpho_b64}"),
        format!("#tab=chaos&tool=FractalViewer&state={bad_fractal_b64}"),
        format!("#tab=traceability&state={traversal_trace_b64}"),
        "#tab=nonexistent_domain".to_string(),
        "#tab=chaos&tool=NonExistentTool".to_string(),
        format!("#tab=chaos&state={}", "A".repeat(MAX_URL_FRAGMENT_BYTES)),
    ]
}

#[test]
fn test_invalid_share_urls_fail_atomically_and_emit_toast_telemetry_and_assertive_aria() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let chaos_idx = app
        .tabs
        .iter()
        .position(|t| t.url_slug() == "chaos")
        .unwrap();
    app.selected_tab = chaos_idx;
    let baseline_json = app.tabs[chaos_idx].export_url_preset_json().unwrap();
    let cases = build_invalid_share_url_cases();

    let _ = crate::telemetry::poll_app_telemetry();
    for bad_hash in &cases {
        app.session_manager.clear_toast();
        let res = app.apply_share_location_strings(bad_hash, "", Some(&ctx));
        assert!(res.is_err(), "Expected error for '{bad_hash}'");

        // Active simulation parameters must remain 100% untouched
        assert_eq!(app.selected_tab, chaos_idx);
        assert_eq!(
            app.tabs[chaos_idx].export_url_preset_json().unwrap(),
            baseline_json
        );

        let (toast_msg, is_err) = app
            .session_manager
            .toast()
            .expect("Warning toast must be set on invalid shared URL");
        assert!(*is_err);
        assert!(toast_msg.contains("Invalid shared simulation link ignored"));

        let aria_msg =
            ctx.data_mut(|d| d.remove_temp::<String>(egui::Id::new("aria_live_assertive_message")));
        assert!(
            aria_msg
                .as_deref()
                .is_some_and(|m| m.contains("Invalid shared simulation link ignored")),
            "Expected assertive ARIA status announcement for '{bad_hash}'"
        );
    }

    let events = crate::telemetry::poll_app_telemetry();
    assert!(
        events
            .iter()
            .any(|ev| ev.source == "UrlState"
                && ev.severity == federated_registry::Severity::Warning),
        "Expected UrlState Warning telemetry event"
    );
}

#[test]
fn test_copy_share_link_populates_clipboard_and_confirmation_toast() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let _ = take_last_copied_share_url();

    let cc = eframe::CreationContext::_new_kittest(ctx.clone());
    let mut app = MathExplorerApp::new(&cc);
    let mri_idx = app.tabs.iter().position(|t| t.url_slug() == "mri").unwrap();
    app.selected_tab = mri_idx;
    app.copy_current_share_link(&ctx);

    let copied = take_last_copied_share_url().expect("Share link must be copied to clipboard");
    assert!(
        copied.starts_with("https://fderuiter.github.io/OxidizeMath/#tab=mri&tool=MriTool&state="),
        "Unexpected MRI share URL: {copied}"
    );
    let (toast_msg, is_err) = app
        .session_manager
        .toast()
        .expect("Confirmation toast expected");
    assert!(!*is_err);
    assert!(toast_msg.contains("Copied shareable simulation link"));

    let trace_idx = app
        .tabs
        .iter()
        .position(|t| t.url_slug() == "traceability")
        .unwrap();
    app.selected_tab = trace_idx;
    app.copy_current_share_link(&ctx);
    let trace_copied =
        take_last_copied_share_url().expect("Traceability share link must be copied");
    assert!(trace_copied
        .starts_with("https://fderuiter.github.io/OxidizeMath/#tab=traceability&state="));

    #[cfg(not(target_arch = "wasm32"))]
    {
        let subdir_preset = crate::presets::serialize_compact_preset_envelope(
            "Traceability Portal",
            "traceability",
            serde_json::json!({ "repo_path": "papers" }),
        )
        .unwrap();
        app.tabs[trace_idx]
            .import_url_preset_json(None, Some(&subdir_preset))
            .unwrap();
        let exported = app.tabs[trace_idx]
            .export_url_preset_json()
            .unwrap()
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&exported).unwrap();
        assert_eq!(parsed["config"]["repo_path"], serde_json::json!("papers"));
    }
}

#[test]
fn test_focused_secondary_pane_share_link_and_command_registry_registration() {
    use eframe::App;

    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let _ = take_last_copied_share_url();
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    let mut app = MathExplorerApp::default();

    let chaos_idx = app
        .tabs
        .iter()
        .position(|t| t.url_slug() == "chaos")
        .unwrap();
    let quantum_idx = app
        .tabs
        .iter()
        .position(|t| t.url_slug() == "quantum")
        .unwrap();

    app.selected_tab = chaos_idx;
    app.set_layout(ViewportLayout::SplitHorizontal);
    app.set_pane_tab(1, quantum_idx);
    app.focus_pane(1);
    assert!(app
        .pane_tab_mut(1)
        .unwrap()
        .select_tool_by_name("SpinVisualizer"));

    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw, |egui_ctx| app.update(egui_ctx, &mut frame));

    let registry = ctx
        .data(|d| {
            d.get_temp::<egui_plot::commands::CommandRegistryData>(egui::Id::new("CMD_REGISTRY"))
        })
        .expect("CMD_REGISTRY must be populated during update");
    assert!(
        registry
            .commands
            .iter()
            .any(|cmd| cmd.name == "Copy Share Link"),
        "Copy Share Link shortcut must be registered in CMD_REGISTRY for Keyboard Shortcuts modal"
    );

    app.copy_current_share_link(&ctx);
    let copied = take_last_copied_share_url().expect("Focused Pane 1 share URL copied");
    assert!(
        copied.starts_with(
            "https://fderuiter.github.io/OxidizeMath/#tab=quantum&tool=SpinVisualizer&state="
        ) && !copied.contains("SplitHorizontal"),
        "Focused secondary pane share URL must encode Pane 1 without split layout metadata: {copied}"
    );
}

//! End-to-end multi-pane contract tests for `#ws=<base64url>` round-trips across all 8 focus
//! configurations, `localStorage` + WebGL recovery reload precedence, atomic error rollback,
//! non-share anchors, and `hashchange` / `popstate` undo history.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{
    encode_workspace_share_fragment, enqueue_browser_navigation_for_test,
    reset_history_bridge_for_test, take_last_copied_share_url, with_mock_browser_location,
    with_mock_self_synced_fragment, ShareUrlParams, WorkspaceShareParams, MAX_URL_FRAGMENT_BYTES,
};
use crate::app::session::SessionLifecycleState;
use crate::app::{ComparisonSyncConfig, MathExplorerApp, ViewportLayout};
use eframe::{egui, App as _};
use federated_registry::Severity;
use std::collections::HashMap;

#[derive(Default)]
struct MemoryStorage {
    entries: HashMap<String, String>,
}

impl eframe::Storage for MemoryStorage {
    fn get_string(&self, key: &str) -> Option<String> {
        self.entries.get(key).cloned()
    }

    fn set_string(&mut self, key: &str, value: String) {
        self.entries.insert(key.to_string(), value);
    }

    fn flush(&mut self) {}
}

const ALL_EIGHT_MULTI_PANE_FOCUS_CASES: [(ViewportLayout, usize); 8] = [
    (ViewportLayout::SplitHorizontal, 0),
    (ViewportLayout::SplitHorizontal, 1),
    (ViewportLayout::SplitVertical, 0),
    (ViewportLayout::SplitVertical, 1),
    (ViewportLayout::QuadGrid, 0),
    (ViewportLayout::QuadGrid, 1),
    (ViewportLayout::QuadGrid, 2),
    (ViewportLayout::QuadGrid, 3),
];

fn find_tab_idx(app: &MathExplorerApp, slug: &str) -> usize {
    app.tabs
        .iter()
        .position(|t| t.url_slug() == slug)
        .unwrap_or_else(|| panic!("Missing tab slug {slug}"))
}

fn configure_multi_pane_case(app: &mut MathExplorerApp, layout: ViewportLayout, focused: usize) {
    let chaos_idx = find_tab_idx(app, "chaos");
    let quantum_idx = find_tab_idx(app, "quantum");
    let climate_idx = find_tab_idx(app, "climate");

    // Duplicate `chaos` across pane 0 and pane 1 with distinct tool parameters!
    app.set_layout(layout);
    app.set_pane_tab(0, chaos_idx);
    app.set_pane_tab(1, chaos_idx);
    if layout == ViewportLayout::QuadGrid {
        app.set_pane_tab(2, quantum_idx);
        app.set_pane_tab(3, climate_idx);
        app.quad_split_ratios = (0.38, 0.66);
    } else {
        app.split_ratio = 0.63;
    }
    app.focused_pane = focused;
    app.compare_sync = ComparisonSyncConfig {
        link_axes: true,
        link_cursors: focused % 2 == 1,
        sync_playback: true,
    };

    assert!(app.tabs[chaos_idx].select_tool_by_name("FractalViewer"));
    app.tabs[chaos_idx]
        .framework_mut()
        .unwrap()
        .active_tool
        .as_mut()
        .unwrap()
        .try_load_state(
            r#"{"center_re":-0.65,"center_im":0.12,"zoom":2.25,"max_iter":180,"mode":"Mandelbrot","julia_re":-0.4,"julia_im":0.6}"#,
        )
        .unwrap();

    let pane1 = app.pane_tab_mut(1).expect("pane 1 active");
    assert!(pane1.select_tool_by_name("FractalViewer"));
    pane1
        .framework_mut()
        .unwrap()
        .active_tool
        .as_mut()
        .unwrap()
        .try_load_state(
            r#"{"center_re":0.15,"center_im":-0.35,"zoom":4.5,"max_iter":260,"mode":"Julia","julia_re":-0.8,"julia_im":0.156}"#,
        )
        .unwrap();
}

fn pane_tool_state(app: &MathExplorerApp, pane_idx: usize) -> Option<String> {
    if pane_idx == 0 {
        let idx = app.selected_tab;
        return app.tabs[idx]
            .framework()
            .and_then(|fw| fw.active_tool.as_ref())
            .and_then(|t| t.save_state());
    }
    app.pane_tab(pane_idx)
        .and_then(|t| t.framework())
        .and_then(|fw| fw.active_tool.as_ref())
        .and_then(|t| t.save_state())
}

#[test]
fn test_all_8_multi_pane_focus_cases_roundtrip_url_storage_and_webgl_recovery() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    for (layout, focused) in ALL_EIGHT_MULTI_PANE_FOCUS_CASES {
        let mut source_app = MathExplorerApp::default();
        let baseline_single = source_app.get_app_state();
        source_app.session_manager.push_undo(baseline_single);
        configure_multi_pane_case(&mut source_app, layout, focused);

        let expected_pane0 = pane_tool_state(&source_app, 0);
        let expected_pane1 = pane_tool_state(&source_app, 1);
        assert_ne!(
            expected_pane0, expected_pane1,
            "Duplicate domain panes must stay isolated"
        );

        // 1. `#ws=...` URL round-trip
        let frag = source_app.encode_current_share_fragment().unwrap();
        assert!(frag.starts_with("#ws="));
        assert!(frag.len() <= MAX_URL_FRAGMENT_BYTES);

        let mut url_app = MathExplorerApp::default();
        assert!(url_app
            .apply_share_location_strings(&frag, "", None)
            .unwrap());
        assert_eq!(url_app.viewport_layout, layout);
        assert_eq!(url_app.focused_pane, focused);
        assert_eq!(url_app.compare_sync, source_app.compare_sync);
        assert_eq!(pane_tool_state(&url_app, 0), expected_pane0);
        assert_eq!(pane_tool_state(&url_app, 1), expected_pane1);

        // 2. `eframe::Storage` (`eframe::APP_KEY`) + self-synced URL reload preserves undo stack
        let mut storage = MemoryStorage::default();
        source_app.save(&mut storage);
        let ctx = egui::Context::default();
        let mut cc = eframe::CreationContext::_new_kittest(ctx.clone());
        cc.storage = Some(&storage);
        let reloaded = with_mock_browser_location(&frag, "", || {
            with_mock_self_synced_fragment(&frag, || MathExplorerApp::new(&cc))
        });
        assert_eq!(reloaded.viewport_layout, layout);
        assert_eq!(reloaded.focused_pane, focused);
        assert_eq!(pane_tool_state(&reloaded, 0), expected_pane0);
        assert_eq!(pane_tool_state(&reloaded, 1), expected_pane1);
        assert_eq!(reloaded.session_manager.undo_depth(), 1);

        // 3. WebGL recovery snapshot (`capture_recovery_snapshot` + `restore_recovery_snapshot`)
        let snapshot = source_app.capture_recovery_snapshot();
        let mut recovered = MathExplorerApp::default();
        recovered.restore_recovery_snapshot(&snapshot, &ctx);
        assert_eq!(recovered.viewport_layout, layout);
        assert_eq!(recovered.focused_pane, focused);
        assert_eq!(pane_tool_state(&recovered, 0), expected_pane0);
        assert_eq!(pane_tool_state(&recovered, 1), expected_pane1);
        assert_eq!(recovered.session_manager.undo_depth(), 1);
    }
}

#[test]
fn test_atomic_multi_pane_ws_failure_preserves_layout_focus_and_panes() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut app = MathExplorerApp::default();
    configure_multi_pane_case(&mut app, ViewportLayout::QuadGrid, 2);
    let before_pane0 = pane_tool_state(&app, 0);
    let before_pane1 = pane_tool_state(&app, 1);

    let invalid_ws = encode_workspace_share_fragment(&WorkspaceShareParams {
        layout: ViewportLayout::QuadGrid,
        split_ratio: 0.5,
        quad_split_ratios: (0.4, 0.6),
        focused_pane: 1,
        compare_sync: ComparisonSyncConfig::default(),
        panes: vec![
            ShareUrlParams {
                tab: "chaos".to_string(),
                tool: Some("FractalViewer".to_string()),
                challenge: None,
                state_json: None,
            },
            ShareUrlParams {
                tab: "physics".to_string(),
                tool: None,
                challenge: None,
                state_json: None,
            },
            ShareUrlParams {
                tab: "quantum".to_string(),
                tool: None,
                challenge: None,
                state_json: None,
            },
            ShareUrlParams {
                tab: "chaos".to_string(),
                tool: Some("NonExistentFourthPaneTool".to_string()),
                challenge: None,
                state_json: None,
            },
        ],
    })
    .unwrap();

    let _ = crate::telemetry::poll_app_telemetry();
    let ctx = egui::Context::default();
    assert!(app
        .apply_share_location_strings(&invalid_ws, "", Some(&ctx))
        .is_err());

    // Layout, focus, ratios, and all 4 panes must remain 100% untouched!
    assert_eq!(app.viewport_layout, ViewportLayout::QuadGrid);
    assert_eq!(app.focused_pane, 2);
    assert_eq!(app.quad_split_ratios, (0.38, 0.66));
    assert_eq!(pane_tool_state(&app, 0), before_pane0);
    assert_eq!(pane_tool_state(&app, 1), before_pane1);

    let (toast_msg, is_err) = app
        .session_manager
        .toast()
        .expect("error toast must be shown");
    assert!(*is_err && toast_msg.contains("Invalid shared simulation link ignored"));
    assert!(crate::telemetry::poll_app_telemetry()
        .iter()
        .any(|ev| ev.source == "UrlState" && ev.severity == Severity::Warning));
}

#[test]
fn test_non_share_anchors_preserve_multi_pane_layout_without_warnings() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut app = MathExplorerApp::default();
    configure_multi_pane_case(&mut app, ViewportLayout::SplitVertical, 1);

    let _ = crate::telemetry::poll_app_telemetry();
    let ctx = egui::Context::default();
    for (hash, search) in [
        ("#overview", ""),
        ("#utm_source=docs", ""),
        ("#section-results", "?utm_medium=web"),
    ] {
        assert_eq!(
            app.apply_share_location_strings(hash, search, Some(&ctx)),
            Ok(false)
        );
        app.apply_external_browser_navigation(hash, search, &ctx);
        assert_eq!(app.viewport_layout, ViewportLayout::SplitVertical);
        assert_eq!(app.focused_pane, 1);
        assert!(app.session_manager.toast().is_none());
    }
    assert!(crate::telemetry::poll_app_telemetry().is_empty());
}

#[test]
fn test_external_url_navigation_records_undo_on_startup_and_hashchange_popstate() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    reset_history_bridge_for_test();

    // 1. Startup external `#tab=...` URL overriding a saved multi-pane `localStorage` workspace
    let mut saved_app = MathExplorerApp::default();
    configure_multi_pane_case(&mut saved_app, ViewportLayout::SplitHorizontal, 1);
    let expected_pane0 = pane_tool_state(&saved_app, 0);
    let expected_pane1 = pane_tool_state(&saved_app, 1);
    let mut storage = MemoryStorage::default();
    saved_app.save(&mut storage);

    let ctx = egui::Context::default();
    let mut cc = eframe::CreationContext::_new_kittest(ctx.clone());
    cc.storage = Some(&storage);
    let mut booted = with_mock_browser_location("#tab=quantum&tool=WaveSimulator", "", || {
        MathExplorerApp::new(&cc)
    });
    assert_eq!(booted.viewport_layout, ViewportLayout::Single);
    assert_eq!(booted.tabs[booted.selected_tab].url_slug(), "quantum");
    assert_eq!(booted.session_manager.undo_depth(), 1);

    booted.undo();
    assert_eq!(booted.viewport_layout, ViewportLayout::SplitHorizontal);
    assert_eq!(booted.focused_pane, 1);
    assert!((booted.split_ratio - 0.63).abs() < 1e-5);
    assert_eq!(pane_tool_state(&booted, 0), expected_pane0);
    assert_eq!(pane_tool_state(&booted, 1), expected_pane1);

    // 2. In-tab `hashchange`/`popstate` navigation even when `RestoringHistory` was active
    assert_eq!(
        booted.session_manager.lifecycle_state(),
        SessionLifecycleState::RestoringHistory
    );
    let quad_ws = {
        let mut q = MathExplorerApp::default();
        configure_multi_pane_case(&mut q, ViewportLayout::QuadGrid, 3);
        q.copy_current_share_link(&ctx);
        let copied = take_last_copied_share_url().unwrap();
        assert!(copied.contains("#ws="));
        q.encode_current_share_fragment().unwrap()
    };
    enqueue_browser_navigation_for_test(&quad_ws, "");
    booted.poll_share_shortcut_and_clipboard(&ctx);
    assert_eq!(booted.viewport_layout, ViewportLayout::QuadGrid);
    assert_eq!(booted.focused_pane, 3);
    assert!(booted.session_manager.can_undo());

    booted.undo();
    assert_eq!(booted.viewport_layout, ViewportLayout::SplitHorizontal);
    assert_eq!(booted.focused_pane, 1);
}

#[test]
fn test_multi_pane_workspace_url_preserves_inverse_challenge_and_clamps_locks() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut app = MathExplorerApp::default();
    let mri_idx = find_tab_idx(&app, "mri");
    let pk_idx = find_tab_idx(&app, "clinical_trials");

    app.set_layout(ViewportLayout::SplitHorizontal);
    app.set_pane_tab(0, mri_idx);
    app.set_pane_tab(1, pk_idx);
    app.focus_pane(1);

    app.tabs[mri_idx]
        .import_url_share_params(Some("MriTool"), Some("bloch_pi_inversion"), None)
        .unwrap();
    app.pane_tab_mut(1)
        .unwrap()
        .import_url_share_params(
            Some("PharmacokineticsTool"),
            Some("pk_therapeutic_window"),
            None,
        )
        .unwrap();

    let ws_frag = app.encode_current_share_fragment().unwrap();
    assert!(ws_frag.starts_with("#ws="));
    let ws_token = ws_frag.strip_prefix("#ws=").unwrap();
    let mut decoded = super::decode_workspace_share_param(ws_token).unwrap();
    assert_eq!(
        decoded.panes[0].challenge.as_deref(),
        Some("bloch_pi_inversion")
    );
    assert_eq!(
        decoded.panes[1].challenge.as_deref(),
        Some("pk_therapeutic_window")
    );

    // Tamper with locked `mtc` and `ke` fields inside secondary pane `#ws=...` envelope
    let mut env1: serde_json::Value =
        serde_json::from_str(decoded.panes[1].state_json.as_ref().unwrap()).unwrap();
    env1["config"]["dose_mg"] = serde_json::json!(165.0);
    env1["config"]["mtc"] = serde_json::json!(99.0);
    env1["config"]["ke"] = serde_json::json!(0.45);
    decoded.panes[1].state_json = Some(env1.to_string());
    let tampered_ws = encode_workspace_share_fragment(&decoded).unwrap();

    let mut restored = MathExplorerApp::default();
    assert!(restored
        .apply_share_location_strings(&tampered_ws, "", None)
        .unwrap());
    assert_eq!(restored.viewport_layout, ViewportLayout::SplitHorizontal);
    assert_eq!(restored.focused_pane, 1);
    assert_eq!(
        restored.tabs[mri_idx].active_challenge_id(),
        Some("bloch_pi_inversion")
    );
    let pane1_fw = restored.pane_tab(1).unwrap().framework().unwrap();
    assert_eq!(
        pane1_fw.active_challenge_id(),
        Some("pk_therapeutic_window")
    );
    let cfg: serde_json::Value =
        serde_json::from_str(&pane1_fw.active_tool.as_ref().unwrap().save_state().unwrap())
            .unwrap();
    assert_eq!(cfg["dose_mg"], serde_json::json!(165.0));
    assert_eq!(cfg["mtc"], serde_json::json!(12.0));
    assert_eq!(cfg["ke"], serde_json::json!(0.12));
}

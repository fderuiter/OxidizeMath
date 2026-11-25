//! Unit tests for `sessionStorage` boot sentinel bracketing, reload quarantine, theme
//! preservation, `replace_browser_url_fragment` share stripping, and safe-mode backup export.

use super::boot_sentinel::{
    complete_first_frame, dismiss_safe_mode_banner, extract_theme_token_lenient,
    hydrate_with_boot_sentinel, set_mock_boot_storage, snapshot_mock_boot_storage,
    trigger_quarantined_backup_download, MockBootStorage, BOOT_SENTINEL_SESSION_KEY,
    PRESERVED_THEME_STORAGE_KEY, QUARANTINED_BACKUP_FILENAME, QUARANTINED_WORKSPACE_STORAGE_KEY,
};
use super::{MathExplorerApp, ThemeMode};
use eframe::egui;

struct ScopedMockBootGuard;

impl ScopedMockBootGuard {
    fn install(mock: MockBootStorage) -> Self {
        set_mock_boot_storage(Some(mock));
        Self
    }
}

impl Drop for ScopedMockBootGuard {
    fn drop(&mut self) {
        set_mock_boot_storage(None);
    }
}

#[test]
fn normal_startup_arms_sentinel_before_hydration_and_clears_after_frame_one() {
    let _guard = ScopedMockBootGuard::install(MockBootStorage::default());
    let cc = eframe::CreationContext::_new_kittest(egui::Context::default());
    let mut app = MathExplorerApp::default();

    hydrate_with_boot_sentinel(&mut app, &cc);
    assert!(app.boot_sentinel.sentinel_armed);
    assert!(!app.boot_sentinel.safe_mode_banner_open);
    let during_frame_1 = snapshot_mock_boot_storage().expect("mock storage active");
    assert_eq!(
        during_frame_1
            .session
            .get(BOOT_SENTINEL_SESSION_KEY)
            .map(String::as_str),
        Some("hydrating")
    );

    complete_first_frame(&mut app);
    assert!(!app.boot_sentinel.sentinel_armed);
    let after_frame_1 = snapshot_mock_boot_storage().expect("mock storage active");
    assert!(!after_frame_1
        .session
        .contains_key(BOOT_SENTINEL_SESSION_KEY));
}

#[test]
fn active_sentinel_on_reload_quarantines_state_strips_url_and_preserves_theme() {
    let poisoned_json = r#"{
        "schema_version": "1.0",
        "app_version": "0.1.0",
        "timestamp": "2026-11-25T00:00:00Z",
        "app_state": {
            "selected_tab": 999,
            "theme": "Light",
            "poisoned_solver_payload": true
        }
    }"#;
    let mut mock = MockBootStorage::default();
    mock.session.insert(
        BOOT_SENTINEL_SESSION_KEY.to_string(),
        "hydrating".to_string(),
    );
    mock.local
        .insert(eframe::APP_KEY.to_string(), poisoned_json.to_string());
    let _guard = ScopedMockBootGuard::install(mock);

    let _ = crate::url_state::replace_browser_url_fragment("#ws=poisoned");
    let cc = eframe::CreationContext::_new_kittest(egui::Context::default());
    let mut app = MathExplorerApp::default();

    hydrate_with_boot_sentinel(&mut app, &cc);

    assert_eq!(app.theme, ThemeMode::Light);
    assert_eq!(cc.egui_ctx.theme(), egui::Theme::Light);
    assert!(app.boot_sentinel.safe_mode_banner_open);
    assert!(!app.show_starter_gallery);
    assert_eq!(app.boot_sentinel.quarantined_backup_json, poisoned_json);
    assert_eq!(
        crate::url_state::last_replaced_browser_fragment().as_deref(),
        Some("")
    );

    let after_quarantine = snapshot_mock_boot_storage().expect("mock storage active");
    assert!(!after_quarantine.local.contains_key(eframe::APP_KEY));
    assert_eq!(
        after_quarantine
            .local
            .get(QUARANTINED_WORKSPACE_STORAGE_KEY)
            .map(String::as_str),
        Some(poisoned_json)
    );
    assert_eq!(
        after_quarantine
            .local
            .get(PRESERVED_THEME_STORAGE_KEY)
            .map(String::as_str),
        Some("light")
    );

    assert!(trigger_quarantined_backup_download(&mut app, &cc.egui_ctx));
    let exported = snapshot_mock_boot_storage()
        .and_then(|s| s.last_downloaded_backup)
        .expect("backup exported");
    assert_eq!(exported.0, QUARANTINED_BACKUP_FILENAME);
    assert_eq!(exported.1, poisoned_json);

    dismiss_safe_mode_banner(&mut app, &cc.egui_ctx);
    assert!(!app.boot_sentinel.safe_mode_banner_open);

    complete_first_frame(&mut app);
    let after_frame_1 = snapshot_mock_boot_storage().expect("mock storage active");
    assert!(!after_frame_1
        .session
        .contains_key(BOOT_SENTINEL_SESSION_KEY));
}

#[test]
fn share_url_only_crash_still_enters_safe_mode_and_provides_valid_mws_backup() {
    let mut mock = MockBootStorage::default();
    mock.session.insert(
        BOOT_SENTINEL_SESSION_KEY.to_string(),
        "hydrating".to_string(),
    );
    mock.dom_theme = Some("high-contrast".to_string());
    let _guard = ScopedMockBootGuard::install(mock);

    let _ = crate::url_state::replace_browser_url_fragment("#tab=crashing_tool");
    let cc = eframe::CreationContext::_new_kittest(egui::Context::default());
    let mut app = MathExplorerApp::default();

    hydrate_with_boot_sentinel(&mut app, &cc);

    assert_eq!(app.theme, ThemeMode::HighContrast);
    assert!(app.boot_sentinel.safe_mode_banner_open);
    assert!(!app.boot_sentinel.quarantined_backup_json.is_empty());
    assert!(app
        .session_manager
        .parse_envelope(&app.boot_sentinel.quarantined_backup_json)
        .is_ok());
    assert_eq!(
        crate::url_state::last_replaced_browser_fragment().as_deref(),
        Some("")
    );
}

#[test]
fn preserved_theme_from_manual_reset_restores_on_clean_reload() {
    let mut mock = MockBootStorage::default();
    mock.local.insert(
        PRESERVED_THEME_STORAGE_KEY.to_string(),
        "high-contrast".to_string(),
    );
    let _guard = ScopedMockBootGuard::install(mock);

    let cc = eframe::CreationContext::_new_kittest(egui::Context::default());
    let mut app = MathExplorerApp::default();
    hydrate_with_boot_sentinel(&mut app, &cc);

    assert_eq!(app.theme, ThemeMode::HighContrast);
    assert!(!app.boot_sentinel.safe_mode_banner_open);
    let snapshot = snapshot_mock_boot_storage().expect("mock storage active");
    assert!(!snapshot.local.contains_key(PRESERVED_THEME_STORAGE_KEY));
}

#[test]
fn lenient_theme_extractor_handles_envelope_legacy_and_corrupt_json() {
    assert_eq!(
        extract_theme_token_lenient(r#"{"schema_version":"1.0","app_state":{"theme":"Light"}}"#),
        Some(ThemeMode::Light)
    );
    assert_eq!(
        extract_theme_token_lenient(r#"{"theme":"HighContrast","broken":123}"#),
        Some(ThemeMode::HighContrast)
    );
    assert_eq!(extract_theme_token_lenient("not valid json {{{"), None);
}

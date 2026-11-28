use super::*;
use crate::app::MathExplorerApp;
use eframe::{App, Storage};
use std::collections::HashMap;

#[derive(Default)]
pub(super) struct TestStorage {
    entries: HashMap<String, String>,
    pub(super) flushes: usize,
}

impl Storage for TestStorage {
    fn get_string(&self, key: &str) -> Option<String> {
        self.entries.get(key).cloned()
    }
    fn set_string(&mut self, key: &str, value: String) {
        self.entries.insert(key.to_string(), value);
    }
    fn flush(&mut self) {
        self.flushes += 1;
    }
}

#[test]
fn test_theme_mode_roundtrips_through_visuals_and_dom_tokens() {
    let cases = [
        (ThemeMode::Dark, "dark", "dark", "#121214"),
        (ThemeMode::Light, "light", "light", "#f8f8f8"),
        (ThemeMode::HighContrast, "high-contrast", "dark", "#000000"),
    ];
    for (mode, data_tok, scheme, hex) in cases {
        let visuals = mode.visuals();
        assert_eq!(ThemeMode::from_visuals(&visuals), mode);
        assert_eq!(SemanticPalette::from_visuals(&visuals), mode.palette());
        assert_eq!(mode.data_theme(), data_tok);
        assert_eq!(mode.color_scheme(), scheme);
        assert_eq!(mode.theme_color_hex(), hex);
        assert_eq!(ThemeMode::from_token(data_tok), Some(mode));
        assert_eq!(ThemeMode::from_token(&format!("{mode:?}")), Some(mode));
    }
    assert_eq!(ThemeMode::from_token("unknown"), None);
}

#[test]
fn test_startup_media_query_precedence() {
    use egui::Theme::{Dark, Light};
    use ThemeMode::{Dark as TD, HighContrast as THC, Light as TL};
    for (fc, mc, pl, sys, expected) in [
        (true, false, true, Some(Light), THC),
        (false, true, true, Some(Light), THC),
        (false, false, true, None, TL),
        (false, false, false, Some(Light), TL),
        (false, false, false, Some(Dark), TD),
        (false, false, false, None, TD),
    ] {
        assert_eq!(resolve_startup_theme(fc, mc, pl, sys), expected);
    }
}

#[test]
fn test_forced_colors_document_metadata_preserves_light_canvas() {
    assert_eq!(
        resolve_document_theme_metadata(ThemeMode::HighContrast, false, true),
        ("high-contrast", Some("dark"), "dark", "#000000")
    );
    assert_eq!(
        resolve_document_theme_metadata(ThemeMode::HighContrast, true, true),
        ("high-contrast", None, "dark light", "#f8f8f8")
    );
    assert_eq!(
        resolve_document_theme_metadata(ThemeMode::HighContrast, true, false),
        ("high-contrast", None, "dark light", "#000000")
    );
    assert_eq!(
        resolve_document_theme_metadata(ThemeMode::Light, false, true),
        ("light", Some("light"), "light", "#f8f8f8")
    );
}

#[test]
fn test_immediate_storage_writes_on_switch_undo_redo_and_session_open() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();
    let mut storage = TestStorage::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| app.update(ctx, &mut frame));
    let saved_theme =
        |s: &TestStorage| extract_explicit_theme_from_json(&s.get_string(eframe::APP_KEY).unwrap());

    app.theme = ThemeMode::Light;
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        let mut mgr = std::mem::take(&mut app.session_manager);
        mgr.track_frame_changes(ctx, &app);
        app.session_manager = mgr;
        sync_theme_with_storage(&mut app, ctx, Some(&mut storage));
    });
    assert_eq!(
        (saved_theme(&storage), storage.flushes),
        (Some(ThemeMode::Light), 1)
    );

    app.split_ratio = 0.62;
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        let mut mgr = std::mem::take(&mut app.session_manager);
        mgr.track_frame_changes(ctx, &app);
        app.session_manager = mgr;
    });
    app.undo();
    sync_theme_with_storage(&mut app, &ctx, Some(&mut storage));
    assert_eq!(saved_theme(&storage), Some(ThemeMode::Light));

    app.redo();
    sync_theme_with_storage(&mut app, &ctx, Some(&mut storage));
    assert_eq!(saved_theme(&storage), Some(ThemeMode::Light));

    let (tx, rx) = std::sync::mpsc::channel();
    app.session_manager.open_rx = Some(rx);
    app.session_manager.lifecycle_state = crate::app::SessionLifecycleState::AwaitingFileDialog;
    let mut hc_state = app.get_app_state();
    hc_state.theme = ThemeMode::HighContrast;
    tx.send(Ok((
        "hc.mws".into(),
        app.session_manager.create_envelope(hc_state),
    )))
    .unwrap();
    let mut mgr = std::mem::take(&mut app.session_manager);
    mgr.poll(&mut app, &ctx);
    app.session_manager = mgr;
    sync_theme_with_storage(&mut app, &ctx, Some(&mut storage));
    assert_eq!(saved_theme(&storage), Some(ThemeMode::HighContrast));
    assert_eq!(ctx.style().visuals, ThemeMode::HighContrast.visuals());
}

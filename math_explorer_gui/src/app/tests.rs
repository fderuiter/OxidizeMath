#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
#![allow(clippy::field_reassign_with_default)]

use super::*;
use eframe::{App, Storage};

struct MockStorage {
    data: std::collections::HashMap<String, String>,
}
impl eframe::Storage for MockStorage {
    fn get_string(&self, key: &str) -> Option<String> {
        self.data.get(key).cloned()
    }
    fn set_string(&mut self, key: &str, value: String) {
        self.data.insert(key.to_string(), value);
    }
    fn flush(&mut self) {}
}

#[test]
fn test_theme_mode_default() {
    assert_eq!(ThemeMode::default(), ThemeMode::Dark);
    assert_eq!(ThemeMode::Dark.name(), "Dark");
    assert_eq!(ThemeMode::Light.name(), "Light");
    assert_eq!(ThemeMode::HighContrast.name(), "High Contrast");
    assert_eq!(ThemeMode::Dark.to_string(), "Dark");

    let app = MathExplorerApp::default();
    assert_eq!(app.theme, ThemeMode::Dark);
}

#[test]
fn test_theme_mode_state_transitions_and_persistence() {
    let mut app1 = MathExplorerApp::default();
    app1.theme = ThemeMode::Light;
    let json = app1.save_state_to_json();
    let mut app2 = MathExplorerApp::default();
    app2.load_state_from_json(&json);
    assert_eq!(app2.theme, ThemeMode::Light);

    app1.theme = ThemeMode::HighContrast;
    let json_hc = app1.save_state_to_json();
    let mut app3 = MathExplorerApp::default();
    app3.load_state_from_json(&json_hc);
    assert_eq!(app3.theme, ThemeMode::HighContrast);
}

#[test]
fn test_theme_mode_visuals_update() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    // Initial frame: Dark visuals
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        app.update(ctx, &mut frame);
    });
    assert_eq!(ctx.style().visuals, ThemeMode::Dark.visuals());

    // Switch to Light mode
    app.theme = ThemeMode::Light;
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        app.update(ctx, &mut frame);
    });
    assert_eq!(ctx.style().visuals, ThemeMode::Light.visuals());

    // Switch to High Contrast mode
    app.theme = ThemeMode::HighContrast;
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        app.update(ctx, &mut frame);
    });
    assert_eq!(ctx.style().visuals, ThemeMode::HighContrast.visuals());
}

#[test]
fn test_app_state_save_and_restore() {
    let mut app1 = MathExplorerApp::default();
    if app1.tabs.len() > 1 {
        app1.selected_tab = 1;
    }
    app1.show_info = false;
    app1.show_warnings = true;
    app1.theme = ThemeMode::Light;

    let json = app1.save_state_to_json();
    assert!(!json.is_empty());

    let mut app2 = MathExplorerApp::default();
    app2.load_state_from_json(&json);

    if app1.tabs.len() > 1 {
        assert_eq!(app2.selected_tab, 1);
    }
    assert!(!app2.show_info);
    assert!(app2.show_warnings);
    assert_eq!(app2.theme, ThemeMode::Light);
}

#[test]
fn test_eframe_storage_persistence() {
    let mut storage = MockStorage {
        data: std::collections::HashMap::new(),
    };
    let mut app1 = MathExplorerApp::default();
    if app1.tabs.len() > 1 {
        app1.selected_tab = 1;
    }
    app1.theme = ThemeMode::HighContrast;
    app1.save(&mut storage);

    let saved_str = storage
        .get_string(eframe::APP_KEY)
        .expect("State must be saved in storage");
    assert!(!saved_str.is_empty());

    let mut app2 = MathExplorerApp::default();
    app2.load_state_from_json(&saved_str);
    if app1.tabs.len() > 1 {
        assert_eq!(app2.selected_tab, 1);
    }
    assert_eq!(app2.theme, ThemeMode::HighContrast);
}

#[test]
fn test_invalid_json_fallback_to_defaults() {
    let mut app = MathExplorerApp::default();
    let initial_tab = app.selected_tab;
    let invalid_json = "{ invalid_json_content: true, ";
    app.load_state_from_json(invalid_json);
    assert_eq!(app.selected_tab, initial_tab);
    assert!(app.show_info);
    assert_eq!(app.theme, ThemeMode::Dark);
}

fn send_key(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    frame: &mut eframe::Frame,
    key: egui::Key,
    modifiers: egui::Modifiers,
) {
    let raw_input = egui::RawInput {
        events: vec![egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        }],
        ..Default::default()
    };
    let _ = ctx.run(raw_input, |ctx| {
        app.update(ctx, frame);
    });
}

#[test]
fn test_cheatsheet_shortcut_question() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();
    assert!(!app.show_help_menu);
    send_key(
        &ctx,
        &mut app,
        &mut frame,
        egui::Key::Questionmark,
        egui::Modifiers::NONE,
    );
    assert!(app.show_help_menu);
    send_key(
        &ctx,
        &mut app,
        &mut frame,
        egui::Key::Questionmark,
        egui::Modifiers::NONE,
    );
    assert!(!app.show_help_menu);
}

#[test]
fn test_cheatsheet_shortcut_ctrl_slash() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();
    assert!(!app.show_help_menu);
    let modifiers = if cfg!(target_os = "macos") {
        egui::Modifiers::MAC_CMD
    } else {
        egui::Modifiers::CTRL
    };
    send_key(&ctx, &mut app, &mut frame, egui::Key::Slash, modifiers);
    assert!(app.show_help_menu);
}

#[test]
fn test_cheatsheet_shortcut_wants_keyboard_input_ignored() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();
    assert!(!app.show_help_menu);
    let mut dummy_string = String::new();
    let raw_input = egui::RawInput {
        events: vec![egui::Event::Key {
            key: egui::Key::Questionmark,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
        ..Default::default()
    };
    let _ = ctx.run(raw_input, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let re = ui.add(egui::TextEdit::singleline(&mut dummy_string));
            re.request_focus();
        });
        app.update(ctx, &mut frame);
    });
    assert!(!app.show_help_menu);
}

#[test]
fn test_cheatsheet_shortcut_aria_announcements() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();
    send_key(
        &ctx,
        &mut app,
        &mut frame,
        egui::Key::Questionmark,
        egui::Modifiers::NONE,
    );
    assert!(app.show_help_menu);
    let aria_msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert_eq!(aria_msg, Some("Hotkey overlay opened".to_string()));
}

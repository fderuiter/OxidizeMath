#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

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
fn test_app_state_save_and_restore() {
    let mut app1 = MathExplorerApp::default();
    if app1.tabs.len() > 1 {
        app1.selected_tab = 1;
    }
    app1.show_info = false;
    app1.show_warnings = true;

    let json = app1.save_state_to_json();
    assert!(!json.is_empty());

    let mut app2 = MathExplorerApp::default();
    app2.load_state_from_json(&json);

    if app1.tabs.len() > 1 {
        assert_eq!(app2.selected_tab, 1);
    }
    assert!(!app2.show_info);
    assert!(app2.show_warnings);
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
}

#[test]
fn test_invalid_json_fallback_to_defaults() {
    let mut app = MathExplorerApp::default();
    let initial_tab = app.selected_tab;
    let invalid_json = "{ invalid_json_content: true, ";

    app.load_state_from_json(invalid_json);
    assert_eq!(app.selected_tab, initial_tab);
    assert!(app.show_info);
}

#[test]
fn test_cheatsheet_shortcut_question() {
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    assert!(!app.show_help_menu);

    // Press '?'
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
        app.update(ctx, &mut frame);
    });

    assert!(app.show_help_menu);

    // Press '?' again to close
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
        app.update(ctx, &mut frame);
    });

    assert!(!app.show_help_menu);
}

#[test]
fn test_cheatsheet_shortcut_ctrl_slash() {
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    assert!(!app.show_help_menu);

    let modifiers = if cfg!(target_os = "macos") {
        egui::Modifiers::MAC_CMD
    } else {
        egui::Modifiers::CTRL
    };

    let raw_input = egui::RawInput {
        events: vec![egui::Event::Key {
            key: egui::Key::Slash,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        }],
        ..Default::default()
    };

    let _ = ctx.run(raw_input, |ctx| {
        app.update(ctx, &mut frame);
    });

    assert!(app.show_help_menu);
}

#[test]
fn test_cheatsheet_shortcut_wants_keyboard_input_ignored() {
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    assert!(!app.show_help_menu);

    // Simulate focus on a text input so wants_keyboard_input() becomes true
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

    let mut dummy_string = String::new();
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
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

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
        app.update(ctx, &mut frame);
    });

    assert!(app.show_help_menu);

    let aria_msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert_eq!(aria_msg, Some("Hotkey overlay opened".to_string()));
}

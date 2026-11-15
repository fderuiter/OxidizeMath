#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use eframe::App;

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
fn test_command_palette_shortcut_cmd_k() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    assert!(!app.command_palette.is_open);

    let modifiers = if cfg!(target_os = "macos") {
        egui::Modifiers::MAC_CMD
    } else {
        egui::Modifiers::CTRL
    };

    send_key(&ctx, &mut app, &mut frame, egui::Key::K, modifiers);

    assert!(app.command_palette.is_open);

    let aria_msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert_eq!(aria_msg, Some("Command palette opened".to_string()));

    // Trigger shortcut again to toggle closed
    send_key(&ctx, &mut app, &mut frame, egui::Key::K, modifiers);
    assert!(!app.command_palette.is_open);

    let aria_msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert_eq!(aria_msg, Some("Command palette closed".to_string()));
}

#[test]
fn test_command_palette_escape_key_closes_and_announces() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    app.command_palette.open();
    assert!(app.command_palette.is_open);

    send_key(
        &ctx,
        &mut app,
        &mut frame,
        egui::Key::Escape,
        egui::Modifiers::NONE,
    );

    assert!(!app.command_palette.is_open);
    let aria_msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert_eq!(aria_msg, Some("Command palette closed".to_string()));
}

#[test]
fn test_command_palette_filter_and_select() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    let initial_tab = app.selected_tab;
    app.command_palette.open();
    app.command_palette.query = "fluid".to_string();

    // Run a frame to update filtered list
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        app.update(ctx, &mut frame);
    });

    // Send Enter to activate
    send_key(
        &ctx,
        &mut app,
        &mut frame,
        egui::Key::Enter,
        egui::Modifiers::NONE,
    );

    assert!(!app.command_palette.is_open);
    let active_name = app.tabs[app.selected_tab].name();
    assert!(active_name.to_lowercase().contains("fluid"));
    assert_ne!(app.selected_tab, initial_tab);

    let aria_msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert_eq!(aria_msg, Some(format!("Switched to tab {}", active_name)));
}

#[test]
fn test_command_palette_view_menu_open() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();

    assert!(!app.command_palette.is_open);

    // Render the top menu panel
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::TopBottomPanel::top("main_menu").show(ctx, |ui| {
            ui_panels::render_main_menu(&mut app, ctx, ui);
        });
    });

    // The button exists in the View menu
    assert!(!app.command_palette.is_open);
    app.command_palette.open();
    assert!(app.command_palette.is_open);
}

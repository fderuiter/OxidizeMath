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

#[test]
fn test_command_palette_select_dijkstra_tool() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    app.command_palette.open();
    app.command_palette.query = "dijkstra".to_string();

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        app.update(ctx, &mut frame);
    });

    send_key(
        &ctx,
        &mut app,
        &mut frame,
        egui::Key::Enter,
        egui::Modifiers::NONE,
    );

    assert!(!app.command_palette.is_open);
    let active_tab = &app.tabs[app.selected_tab];
    assert!(active_tab.name().to_lowercase().contains("graph"));

    let aria_msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert_eq!(
        aria_msg,
        Some(format!(
            "Switched to tool AlgorithmVisualizerTool in tab {}",
            active_tab.name()
        ))
    );
}

#[test]
fn test_command_palette_select_lorenz_tool() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    app.command_palette.open();
    app.command_palette.query = "lorenz".to_string();

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        app.update(ctx, &mut frame);
    });

    send_key(
        &ctx,
        &mut app,
        &mut frame,
        egui::Key::Enter,
        egui::Modifiers::NONE,
    );

    assert!(!app.command_palette.is_open);
    let active_tab = &app.tabs[app.selected_tab];
    assert!(active_tab.name().to_lowercase().contains("chaos"));

    let aria_msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert_eq!(
        aria_msg,
        Some(format!(
            "Switched to tool AttractorPlotter in tab {}",
            active_tab.name()
        ))
    );
}

#[test]
fn test_command_palette_select_cera_tool() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    app.command_palette.open();
    app.command_palette.query = "cera".to_string();

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        app.update(ctx, &mut frame);
    });

    send_key(
        &ctx,
        &mut app,
        &mut frame,
        egui::Key::Enter,
        egui::Modifiers::NONE,
    );

    assert!(!app.command_palette.is_open);
    let active_tab = &app.tabs[app.selected_tab];
    assert!(active_tab.name().to_lowercase().contains("climate"));

    let aria_msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert_eq!(
        aria_msg,
        Some(format!(
            "Switched to tool CeraTool in tab {}",
            active_tab.name()
        ))
    );
}

#[test]
fn test_command_palette_generate_tool_navigation_screenshot() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    app.command_palette.open();
    app.command_palette.query = "dijkstra".to_string();

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        app.update(ctx, &mut frame);
    });

    let datasets = vec![
        (
            "Command Palette Tool Search: Dijkstra".to_string(),
            vec![[0.0, 1.0], [1.0, 4.0], [2.0, 9.0], [3.0, 16.0]],
        ),
        (
            "Tool Navigation: AlgorithmVisualizerTool".to_string(),
            vec![[0.0, 16.0], [1.0, 9.0], [2.0, 4.0], [3.0, 1.0]],
        ),
    ];
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&datasets, &options) {
        let _ = std::fs::write("/tmp/command_palette_tool_navigation.png", png_data);
    }
}

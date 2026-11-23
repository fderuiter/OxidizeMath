#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use eframe::egui;
use eframe::App;

#[cfg(target_os = "macos")]
use super::macos_menu::edit::{edit_menu_enabled_states, handle_edit_menu_action};
#[cfg(not(target_os = "macos"))]
use super::macos_menu_edit::{edit_menu_enabled_states, handle_edit_menu_action};

fn screen_rect() -> egui::Rect {
    egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0))
}

fn run_frame(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    let mut frame = eframe::Frame::_new_kittest();
    ctx.run(
        egui::RawInput {
            screen_rect: Some(screen_rect()),
            events,
            ..Default::default()
        },
        |c| app.update(c, &mut frame),
    )
}

fn shortcut_mods() -> egui::Modifiers {
    if cfg!(target_os = "macos") {
        egui::Modifiers::MAC_CMD
    } else {
        egui::Modifiers::CTRL
    }
}

fn key_ev(key: egui::Key, modifiers: egui::Modifiers) -> egui::Event {
    egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers,
    }
}

fn redo_key_ev() -> egui::Event {
    let mods = shortcut_mods();
    if cfg!(target_os = "macos") {
        key_ev(egui::Key::Z, mods | egui::Modifiers::SHIFT)
    } else {
        key_ev(egui::Key::Y, mods)
    }
}

fn find_buttons_matching<'a>(
    out: &'a egui::FullOutput,
    needle: &str,
) -> Vec<(egui::accesskit::NodeId, &'a egui::accesskit::Node)> {
    let Some(ak) = out.platform_output.accesskit_update.as_ref() else {
        return Vec::new();
    };
    ak.nodes
        .iter()
        .filter(|(_, node)| {
            node.role() == egui::accesskit::Role::Button
                && node.label().is_some_and(|l| l.contains(needle))
        })
        .map(|(id, node)| (*id, node))
        .collect()
}

fn ak_action(action: egui::accesskit::Action, target: egui::accesskit::NodeId) -> egui::Event {
    egui::Event::AccessKitActionRequest(egui::accesskit::ActionRequest {
        action,
        target,
        data: None,
    })
}

fn touch_ev(phase: egui::TouchPhase, pos: egui::Pos2) -> egui::Event {
    egui::Event::Touch {
        device_id: egui::TouchDeviceId(1),
        id: egui::TouchId(1),
        phase,
        pos,
        force: None,
    }
}

fn click_ak_button(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    out: &egui::FullOutput,
    needle: &str,
) -> egui::FullOutput {
    let (target, _) = *find_buttons_matching(out, needle)
        .first()
        .unwrap_or_else(|| panic!("Missing AccessKit button '{needle}'"));
    run_frame(
        ctx,
        app,
        vec![ak_action(egui::accesskit::Action::Click, target)],
    )
}

#[test]
fn test_all_tabs_render_undo_and_redo_buttons_matching_can_undo_and_can_redo() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();

    #[cfg(all(feature = "pure_math", feature = "physics"))]
    assert_eq!(app.tabs.len(), 23);

    let out0 = run_frame(&ctx, &mut app, Vec::new());
    for tab_idx in 0..app.tabs.len() {
        app.selected_tab = tab_idx;
        app.pane_tabs[0] = tab_idx;
        app.session_manager = WorkspaceSessionManager::default();
        let out_idle = run_frame(&ctx, &mut app, Vec::new());
        let undo_btns = find_buttons_matching(&out_idle, "↩ Undo");
        let redo_btns = find_buttons_matching(&out_idle, "↪ Redo");
        assert_eq!(undo_btns.len(), 1, "Tab {tab_idx} must render 1 Undo");
        assert_eq!(redo_btns.len(), 1, "Tab {tab_idx} must render 1 Redo");
        assert_eq!(!undo_btns[0].1.is_disabled(), app.can_undo());
        assert_eq!(!redo_btns[0].1.is_disabled(), app.can_redo());

        app.split_ratio = 0.65;
        let _ = run_frame(&ctx, &mut app, Vec::new());
        let out_undoable = run_frame(&ctx, &mut app, Vec::new());
        assert!(app.can_undo() && !app.can_redo());
        assert!(!find_buttons_matching(&out_undoable, "↩ Undo")[0]
            .1
            .is_disabled());
        assert!(find_buttons_matching(&out_undoable, "↪ Redo")[0]
            .1
            .is_disabled());

        let _ = click_ak_button(&ctx, &mut app, &out_undoable, "↩ Undo");
        let out_redone = run_frame(&ctx, &mut app, Vec::new());
        assert!(app.can_redo());
        assert!(!find_buttons_matching(&out_redone, "↪ Redo")[0]
            .1
            .is_disabled());
    }
    let _ = out0;
}

#[cfg(feature = "pure_math")]
#[test]
fn test_graph_editor_deduplicates_sidebar_undo_redo_when_workspace_bridge_active() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let graph_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Graph Theory")
        .expect("Graph Theory tab");
    app.select_tab(&ctx, graph_idx);
    assert!(app.tabs[graph_idx].select_tool_by_name("GraphEditorTool"));
    let out = run_frame(&ctx, &mut app, Vec::new());
    assert_eq!(find_buttons_matching(&out, "↩ Undo").len(), 1);
    assert_eq!(find_buttons_matching(&out, "↪ Redo").len(), 1);
}

#[cfg(all(feature = "physics", feature = "ai"))]
fn focus_first_role(ctx: &egui::Context, app: &mut MathExplorerApp, role: egui::accesskit::Role) {
    let out = run_frame(ctx, app, Vec::new());
    let target = out
        .platform_output
        .accesskit_update
        .as_ref()
        .and_then(|ak| {
            ak.nodes.iter().find_map(|(id, n)| {
                let in_left = n.bounds().is_some_and(|b| b.x1 < 600.0);
                (n.role() == role && in_left).then_some(*id)
            })
        })
        .unwrap_or_else(|| panic!("Expected widget with role {role:?} in left pane"));
    let _ = run_frame(
        ctx,
        app,
        vec![ak_action(egui::accesskit::Action::Focus, target)],
    );
}

#[cfg(all(feature = "physics", feature = "ai"))]
fn is_ai_training(app: &MathExplorerApp, pane_idx: usize) -> bool {
    app.pane_tab(pane_idx)
        .and_then(|t| t.framework())
        .and_then(|fw| fw.active_tool.as_ref())
        .and_then(|tool| tool.save_state())
        .is_some_and(|s| s.contains("\"is_training\":true"))
}

#[cfg(all(feature = "physics", feature = "ai"))]
fn has_rendered_text(out: &egui::FullOutput, needle: &str) -> bool {
    fn walk(shapes: &[egui::epaint::Shape], needle: &str) -> bool {
        shapes.iter().any(|s| match s {
            egui::epaint::Shape::Text(ts) => ts.galley.text().contains(needle),
            egui::epaint::Shape::Vec(nested) => walk(nested, needle),
            _ => false,
        })
    }
    let shapes: Vec<_> = out.shapes.iter().map(|cs| cs.shape.clone()).collect();
    walk(&shapes, needle)
}

#[cfg(all(feature = "physics", feature = "ai"))]
fn click_at(ctx: &egui::Context, app: &mut MathExplorerApp, pos: egui::Pos2) {
    let _ = run_frame(
        ctx,
        app,
        vec![
            egui::Event::PointerMoved(pos),
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    let _ = run_frame(
        ctx,
        app,
        vec![egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }],
    );
}

#[cfg(all(feature = "physics", feature = "ai"))]
#[test]
fn test_focused_camera3d_and_pane_allow_ctrl_z_ctrl_y_and_questionmark() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let mods = shortcut_mods();
    let chaos_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Chaos Theory")
        .unwrap();

    app.select_tab(&ctx, chaos_idx);
    assert!(app.tabs[chaos_idx].select_tool_by_name("AttractorPlotter"));
    let _ = run_frame(&ctx, &mut app, Vec::new());
    app.split_ratio = 0.62;
    let _ = run_frame(&ctx, &mut app, Vec::new());

    click_at(&ctx, &mut app, egui::pos2(800.0, 450.0));
    assert!(ctx.wants_keyboard_input());
    let _ = run_frame(&ctx, &mut app, vec![key_ev(egui::Key::Z, mods)]);
    assert!((app.split_ratio - 0.50).abs() < 1e-5);
    let _ = run_frame(&ctx, &mut app, vec![redo_key_ev()]);
    assert!((app.split_ratio - 0.62).abs() < 1e-5);

    let _ = run_frame(
        &ctx,
        &mut app,
        vec![key_ev(egui::Key::Questionmark, egui::Modifiers::NONE)],
    );
    assert!(app.show_help_menu);
    let _ = run_frame(
        &ctx,
        &mut app,
        vec![key_ev(egui::Key::Questionmark, egui::Modifiers::NONE)],
    );
    assert!(!app.show_help_menu);
}

#[cfg(all(feature = "physics", feature = "ai"))]
#[test]
fn test_focused_slider_and_sash_allow_period_and_history_shortcuts() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let mods = shortcut_mods();
    let chaos_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Chaos Theory")
        .unwrap();
    let ai_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Artificial Intelligence")
        .unwrap();

    app.set_layout(ViewportLayout::SplitHorizontal);
    app.set_pane_tab(0, ai_idx);
    app.set_pane_tab(1, chaos_idx);
    assert!(app
        .pane_tab_mut(0)
        .unwrap()
        .select_tool_by_name("TrainingMonitorTool"));
    viewport::focus::focus_pane_interactive(&mut app, &ctx, 0);
    let out0 = run_frame(&ctx, &mut app, Vec::new());
    assert!(has_rendered_text(&out0, "Epoch: 0"));

    focus_first_role(&ctx, &mut app, egui::accesskit::Role::Slider);
    assert!(ctx.wants_keyboard_input());
    let _ = run_frame(
        &ctx,
        &mut app,
        vec![key_ev(egui::Key::Period, egui::Modifiers::NONE)],
    );
    let out1 = run_frame(&ctx, &mut app, Vec::new());
    assert!(has_rendered_text(&out1, "Epoch: 1"));

    app.split_ratio = 0.58;
    let _ = run_frame(&ctx, &mut app, Vec::new());
    let sash_x = 1280.0 * app.split_ratio;
    click_at(&ctx, &mut app, egui::pos2(sash_x, 450.0));
    assert!(ctx.wants_keyboard_input());
    let _ = run_frame(
        &ctx,
        &mut app,
        vec![key_ev(egui::Key::Period, egui::Modifiers::NONE)],
    );
    let out2 = run_frame(&ctx, &mut app, Vec::new());
    assert!(has_rendered_text(&out2, "Epoch: 2"));
    let _ = run_frame(&ctx, &mut app, vec![key_ev(egui::Key::Z, mods)]);
    assert!((app.split_ratio - 0.50).abs() < 1e-5);
}

#[cfg(all(feature = "physics", feature = "ai"))]
#[test]
fn test_playback_transport_allows_space_on_canvas_and_slider_but_reserves_focused_button() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let ai_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Artificial Intelligence")
        .unwrap();
    app.select_tab(&ctx, ai_idx);
    assert!(app.tabs[ai_idx].select_tool_by_name("TrainingMonitorTool"));
    let _ = run_frame(&ctx, &mut app, Vec::new());
    assert!(!is_ai_training(&app, 0));

    let space = || vec![key_ev(egui::Key::Space, egui::Modifiers::NONE)];
    focus_first_role(&ctx, &mut app, egui::accesskit::Role::Slider);
    assert!(ctx.wants_keyboard_input());
    let _ = run_frame(&ctx, &mut app, space());
    assert!(is_ai_training(&app, 0));

    if let Some(cid) = ctx.data(|d| d.get_temp::<egui::Id>(egui::Id::new("LAST_PLOT_ID"))) {
        ctx.memory_mut(|m| m.request_focus(cid));
    }
    let _ = run_frame(&ctx, &mut app, space());
    assert!(!is_ai_training(&app, 0));

    let out = run_frame(&ctx, &mut app, Vec::new());
    let (theory_id, _) = find_buttons_matching(&out, "Theory")[0];
    let _ = run_frame(
        &ctx,
        &mut app,
        vec![ak_action(egui::accesskit::Action::Focus, theory_id)],
    );
    let _ = run_frame(&ctx, &mut app, space());
    assert!(!is_ai_training(&app, 0));
    assert!(app.tabs[ai_idx].framework().unwrap().show_theory_portal);
}

#[test]
fn test_chrome_toggles_and_focused_pane_leave_undo_depth_and_redo_stack_intact() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::SplitHorizontal);
    let _ = run_frame(&ctx, &mut app, Vec::new());

    app.split_ratio = 0.60;
    let _ = run_frame(&ctx, &mut app, Vec::new());
    app.split_ratio = 0.72;
    let _ = run_frame(&ctx, &mut app, Vec::new());

    app.undo();
    for _ in 0..3 {
        let _ = run_frame(&ctx, &mut app, Vec::new());
    }
    let baseline_undo_depth = app.session_manager.undo_stack.len();
    let baseline_redo_depth = app.session_manager.redo_stack.len();
    assert!(baseline_undo_depth >= 1 && baseline_redo_depth == 1);

    let out = run_frame(&ctx, &mut app, Vec::new());
    let _ = click_ak_button(&ctx, &mut app, &out, "Theory");
    let _ = run_frame(&ctx, &mut app, Vec::new());
    assert_eq!(app.session_manager.undo_stack.len(), baseline_undo_depth);
    assert_eq!(app.session_manager.redo_stack.len(), baseline_redo_depth);

    if let Some(fw) = app.pane_tab_mut(0).and_then(|t| t.framework_mut()) {
        fw.show_starter_gallery = true;
    }
    let _ = run_frame(&ctx, &mut app, Vec::new());
    app.show_diagnostics_panel = true;
    app.show_info = false;
    app.show_warnings = false;
    app.theme = ThemeMode::HighContrast;
    app.focus_pane(1);
    let _ = run_frame(&ctx, &mut app, Vec::new());

    assert_eq!(app.session_manager.undo_stack.len(), baseline_undo_depth);
    assert_eq!(app.session_manager.redo_stack.len(), baseline_redo_depth);

    app.redo();
    let _ = run_frame(&ctx, &mut app, Vec::new());
    assert!((app.split_ratio - 0.72).abs() < 1e-5);
    assert_eq!(app.theme, ThemeMode::HighContrast);
    assert_eq!(app.focused_pane, 1);
    assert!(app.show_diagnostics_panel);
    if let Some(fw) = app.pane_tab(0).and_then(|t| t.framework()) {
        assert!(fw.show_theory_portal);
    }
}

#[test]
fn test_touch_mode_and_macos_edit_menu_execute_undo_and_redo_via_toolbar_and_shortcuts() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let mods = shortcut_mods();

    let touch_pos = egui::pos2(300.0, 300.0);
    let _ = run_frame(
        &ctx,
        &mut app,
        vec![touch_ev(egui::TouchPhase::Start, touch_pos)],
    );
    let _ = run_frame(
        &ctx,
        &mut app,
        vec![touch_ev(egui::TouchPhase::End, touch_pos)],
    );
    assert_eq!(edit_menu_enabled_states(&app), (false, false));

    app.split_ratio = 0.68;
    let out = run_frame(&ctx, &mut app, Vec::new());
    assert_eq!(edit_menu_enabled_states(&app), (true, false));

    let (_, undo_node) = find_buttons_matching(&out, "↩ Undo")[0];
    let bounds = undo_node.bounds().expect("Undo button bounds");
    assert!(bounds.width() >= 43.5 && bounds.height() >= 43.5);

    let _ = run_frame(&ctx, &mut app, vec![key_ev(egui::Key::Z, mods)]);
    assert!((app.split_ratio - 0.50).abs() < 1e-5);
    assert_eq!(edit_menu_enabled_states(&app), (false, true));

    let _ = run_frame(&ctx, &mut app, vec![redo_key_ev()]);
    assert!((app.split_ratio - 0.68).abs() < 1e-5);
    assert_eq!(edit_menu_enabled_states(&app), (true, false));

    assert!(handle_edit_menu_action(&mut app, &ctx, "edit_undo"));
    let out_after_menu_undo = run_frame(&ctx, &mut app, Vec::new());
    assert!((app.split_ratio - 0.50).abs() < 1e-5);
    assert_eq!(edit_menu_enabled_states(&app), (false, true));

    let _ = click_ak_button(&ctx, &mut app, &out_after_menu_undo, "↪ Redo");
    let _ = run_frame(&ctx, &mut app, Vec::new());
    assert!((app.split_ratio - 0.68).abs() < 1e-5);

    assert!(handle_edit_menu_action(&mut app, &ctx, "edit_undo"));
    assert!(handle_edit_menu_action(&mut app, &ctx, "edit_redo"));
    assert!((app.split_ratio - 0.68).abs() < 1e-5);
    assert!(!handle_edit_menu_action(&mut app, &ctx, "unknown_item"));
}

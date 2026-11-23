#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::helpers::{
    click_pos, cmd_mods, configure_layout_with_canary_panes, copied_texts, find_ak_nodes,
    find_text_centers, focus_ak_node, has_close_viewport_cmd, has_rendered_text, key_evt, run_app,
    screen_rect,
};
use crate::app::{MathExplorerApp, ViewportLayout};
use eframe::egui::{self, accesskit, Context, Id, Key, Modifiers};
use egui_plot::commands::CommandRegistryData;
use std::collections::HashSet;

fn focus_plot_in_last_pane(ctx: &Context, app: &mut MathExplorerApp) -> Id {
    let out = run_app(ctx, app, Vec::new());
    let plots = find_ak_nodes(&out, accesskit::Role::GraphicsDocument, "Interactive Plot");
    focus_ak_node(ctx, app, plots.last().unwrap().0)
}

fn assert_shortcut_blocked_over_modal(
    ctx: &Context,
    app: &mut MathExplorerApp,
    pane_idx: usize,
    ev: egui::Event,
) {
    let out = run_app(ctx, app, vec![ev]);
    assert!(
        !app.command_palette.is_open,
        "Ctrl+K must be blocked over modal"
    );
    assert!(
        !app.show_help_menu,
        "Ctrl+/ and ? must be blocked over modal"
    );
    assert!(!app.show_diagnostics_panel, "Ctrl+Shift+D must be blocked");
    assert!(
        copied_texts(&out).is_empty(),
        "Ctrl+Shift+L must be blocked"
    );
    assert_eq!(app.viewport_layout, ViewportLayout::SplitHorizontal);
    assert!(
        !has_close_viewport_cmd(&out),
        "Ctrl+Q must not close viewport over modal"
    );
    assert!(
        app.pane_tab(pane_idx)
            .unwrap()
            .framework()
            .unwrap()
            .show_export_modal
    );
}

#[test]
#[cfg(all(feature = "pure_math", feature = "ai", feature = "epidemiology"))]
fn test_nine_global_shortcuts_blocked_over_open_modal_and_preserve_prior_focus() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let pane_idx = configure_layout_with_canary_panes(&mut app, ViewportLayout::SplitHorizontal);
    let prior_focus = focus_plot_in_last_pane(&ctx, &mut app);
    assert_eq!(ctx.memory(|m| m.focused()), Some(prior_focus));

    // Open Framework Export foreground modal in pane_idx
    app.pane_tab_mut(pane_idx)
        .unwrap()
        .framework_mut()
        .unwrap()
        .show_export_modal = true;
    let _ = run_app(&ctx, &mut app, Vec::new());
    let _ = run_app(&ctx, &mut app, Vec::new());

    let m = cmd_mods();
    let sm = m | Modifiers::SHIFT;
    let blocked_shortcuts = [
        key_evt(Key::K, m),
        key_evt(Key::Slash, m),
        key_evt(Key::Questionmark, Modifiers::NONE),
        key_evt(Key::O, m),
        key_evt(Key::S, m),
        key_evt(Key::S, sm),
        key_evt(Key::D, sm),
        key_evt(Key::L, sm),
        key_evt(Key::Q, m),
        key_evt(Key::Num2, sm),
    ];

    for ev in blocked_shortcuts {
        assert_shortcut_blocked_over_modal(&ctx, &mut app, pane_idx, ev);
    }

    // Dismiss modal with Escape and confirm prior_focus was preserved across all blocked shortcuts
    let _ = run_app(&ctx, &mut app, vec![key_evt(Key::Escape, Modifiers::NONE)]);
    assert!(
        !app.pane_tab(pane_idx)
            .unwrap()
            .framework()
            .unwrap()
            .show_export_modal
    );
    assert_eq!(ctx.memory(|m| m.focused()), Some(prior_focus));
}

fn scroll_modal(ctx: &Context, app: &mut MathExplorerApp, delta_y: f32) -> egui::FullOutput {
    let center = screen_rect().center();
    run_app(
        ctx,
        app,
        vec![
            egui::Event::PointerMoved(center),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: egui::vec2(0.0, delta_y),
                modifiers: Modifiers::NONE,
            },
        ],
    )
}

fn assert_all_global_commands_registered_and_rendered(
    ctx: &Context,
    app: &mut MathExplorerApp,
    out_top: &egui::FullOutput,
) {
    let cmds = ctx
        .data(|d| d.get_temp::<CommandRegistryData>(Id::new("CMD_REGISTRY")))
        .unwrap_or_default()
        .commands;
    let names: HashSet<&str> = cmds.iter().map(|c| c.name.as_str()).collect();
    let mut expected = vec![
        "Undo",
        "Redo",
        "Open Workspace Session...",
        "Save Workspace Session As...",
        "Save Workspace Session",
        "Focus Next Pane",
        "Focus Previous Pane",
        "Command Palette",
        "Keyboard Shortcuts",
        "Toggle Issues & Diagnostics",
        "Copy Share Link",
        "Duplicate Focused Pane",
    ];
    if !cfg!(target_os = "macos") {
        expected.push("Redo (Ctrl+Y)");
        expected.push("Quit");
    }
    let reset_top = scroll_modal(ctx, app, 3000.0);
    let scroll_frames: Vec<egui::FullOutput> =
        (0..10).map(|_| scroll_modal(ctx, app, -200.0)).collect();
    for name in &expected {
        assert!(
            names.contains(name),
            "Missing global command '{name}' in CMD_REGISTRY"
        );
        let rendered = has_rendered_text(out_top, name)
            || has_rendered_text(&reset_top, name)
            || scroll_frames.iter().any(|f| has_rendered_text(f, name));
        assert!(rendered, "Keyboard Shortcuts dialog must display '{name}'");
    }
    if !cfg!(target_os = "macos") {
        assert_eq!(expected.len(), 14);
    }
}

#[test]
fn test_keyboard_shortcuts_displays_all_14_global_commands_from_slash_question_and_help_menu() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let _ = run_app(&ctx, &mut app, Vec::new());

    // 1. Open via Ctrl+/ (Cmd+/ on macOS)
    let _ = run_app(&ctx, &mut app, vec![key_evt(Key::Slash, cmd_mods())]);
    let out_slash = run_app(&ctx, &mut app, Vec::new());
    assert!(app.show_help_menu);
    assert_all_global_commands_registered_and_rendered(&ctx, &mut app, &out_slash);
    let _ = run_app(&ctx, &mut app, vec![key_evt(Key::Escape, Modifiers::NONE)]);
    assert!(!app.show_help_menu);

    // 2. Open via `?`
    let _ = run_app(
        &ctx,
        &mut app,
        vec![key_evt(Key::Questionmark, Modifiers::NONE)],
    );
    let out_q = run_app(&ctx, &mut app, Vec::new());
    assert!(app.show_help_menu);
    assert_all_global_commands_registered_and_rendered(&ctx, &mut app, &out_q);
    let _ = run_app(&ctx, &mut app, vec![key_evt(Key::Escape, Modifiers::NONE)]);
    assert!(!app.show_help_menu);

    // 3. Open via Help -> Keyboard Shortcuts menu
    #[cfg(not(target_os = "macos"))]
    {
        let menu_out = run_app(&ctx, &mut app, Vec::new());
        let help_pos = *find_text_centers(&menu_out, "Help")
            .first()
            .expect("Top MenuBar must render Help menu");
        let _ = click_pos(&ctx, &mut app, help_pos);
        let popup_out = run_app(&ctx, &mut app, vec![egui::Event::PointerMoved(help_pos)]);
        let item_pos = *find_text_centers(&popup_out, "Keyboard Shortcuts")
            .first()
            .expect("Help menu must list Keyboard Shortcuts");
        let _ = click_pos(&ctx, &mut app, item_pos);
        let _ = run_app(&ctx, &mut app, Vec::new());
        let out_menu = run_app(&ctx, &mut app, Vec::new());
        assert!(app.show_help_menu);
        assert_all_global_commands_registered_and_rendered(&ctx, &mut app, &out_menu);
    }
}

#[test]
#[cfg(all(feature = "pure_math", feature = "ai", feature = "epidemiology"))]
fn test_command_palette_full_app_close_button_covered_guard_and_selected_pane_focus() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let _ = configure_layout_with_canary_panes(&mut app, ViewportLayout::SplitHorizontal);
    app.focus_pane(1);
    let prior_focus = focus_plot_in_last_pane(&ctx, &mut app);
    assert_eq!(ctx.memory(|m| m.focused()), Some(prior_focus));

    // Open CommandPalette and verify Close button dismisses and restores prior_focus
    app.command_palette.open();
    let _ = run_app(&ctx, &mut app, Vec::new());
    let out_open = run_app(&ctx, &mut app, Vec::new());
    let &(close_node, close_rect) = find_ak_nodes(&out_open, accesskit::Role::Button, "Close")
        .last()
        .expect("Command Palette must expose a keyboard-accessible Close button");
    let _ = close_node;
    let _ = click_pos(&ctx, &mut app, close_rect.center());
    assert!(!app.command_palette.is_open);
    assert_eq!(ctx.memory(|m| m.focused()), Some(prior_focus));

    // Reopen CommandPalette, cover it with Keyboard Shortcuts, verify covered palette ignores Enter/ArrowDown
    app.command_palette.open();
    app.command_palette.query = "fluid".to_string();
    let _ = run_app(&ctx, &mut app, Vec::new());
    app.show_help_menu = true;
    let _ = run_app(&ctx, &mut app, Vec::new());
    let _ = run_app(
        &ctx,
        &mut app,
        vec![
            key_evt(Key::ArrowDown, Modifiers::NONE),
            key_evt(Key::Enter, Modifiers::NONE),
        ],
    );
    assert!(
        app.command_palette.is_open,
        "Covered CommandPalette must ignore Enter"
    );
    assert_eq!(
        app.command_palette.selected_index, 0,
        "Covered CommandPalette must ignore ArrowDown"
    );

    // Close covering help modal, then press Enter on CommandPalette while Pane 1 is targeted
    app.show_help_menu = false;
    app.focus_pane(1);
    let _ = run_app(&ctx, &mut app, Vec::new());
    let _ = run_app(&ctx, &mut app, vec![key_evt(Key::Enter, Modifiers::NONE)]);
    let _ = run_app(&ctx, &mut app, Vec::new());
    assert!(!app.command_palette.is_open);
    assert_eq!(app.focused_pane, 1);
    assert!(app
        .pane_tab(1)
        .unwrap()
        .name()
        .to_lowercase()
        .contains("fluid"));
    let focused_after = ctx
        .memory(|m| m.focused())
        .expect("Selected pane must receive focus");
    let pane1_owns_focus = ctx
        .read_response(focused_after)
        .is_some_and(|r| r.rect.center().x > screen_rect().center().x);
    assert!(pane1_owns_focus, "Focus must land inside selected Pane 1");
}

fn verify_theory_portal_in_tab(tab_idx: usize) {
    let ctx = Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let tab_name = app.tabs[tab_idx].name();
    app.select_tab(&ctx, tab_idx);
    if let Some(fw) = app.tabs[tab_idx].framework_mut() {
        if fw.active_tool.is_none() && !fw.available_tools.is_empty() {
            fw.select_tool(0);
        }
    }

    let out0 = run_app(&ctx, &mut app, Vec::new());
    let toggle_node = find_ak_nodes(&out0, accesskit::Role::Button, "Theory")
        .first()
        .unwrap_or_else(|| panic!("Tab '{tab_name}' must expose 📖 Theory toggle"))
        .0;
    let toggle_id = focus_ak_node(&ctx, &mut app, toggle_node);
    let click_toggle = super::helpers::ak_action_evt(accesskit::Action::Click, toggle_node);

    let _ = run_app(&ctx, &mut app, vec![click_toggle.clone()]);
    let out_open = run_app(&ctx, &mut app, Vec::new());
    assert!(app.tabs[tab_idx].framework().unwrap().show_theory_portal);
    assert!(has_rendered_text(&out_open, "Close"));

    let _ = run_app(&ctx, &mut app, vec![key_evt(Key::Escape, Modifiers::NONE)]);
    let _ = run_app(&ctx, &mut app, Vec::new());
    assert!(!app.tabs[tab_idx].framework().unwrap().show_theory_portal);
    assert_eq!(ctx.memory(|m| m.focused()), Some(toggle_id));

    let _ = run_app(&ctx, &mut app, vec![click_toggle]);
    let out_reopen = run_app(&ctx, &mut app, Vec::new());
    let close_node = find_ak_nodes(&out_reopen, accesskit::Role::Button, "Close")
        .last()
        .unwrap_or_else(|| panic!("Tab '{tab_name}' Theory portal Close button node"))
        .0;
    let click_close = super::helpers::ak_action_evt(accesskit::Action::Click, close_node);
    let _ = run_app(&ctx, &mut app, vec![click_close]);
    let _ = run_app(&ctx, &mut app, Vec::new());
    assert!(!app.tabs[tab_idx].framework().unwrap().show_theory_portal);
    assert_eq!(ctx.memory(|m| m.focused()), Some(toggle_id));
}

#[test]
fn test_theory_context_portal_close_button_escape_and_focus_return_across_all_23_framework_tabs() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let probe_app = MathExplorerApp::default();
    let fw_tab_indices: Vec<usize> = probe_app
        .tabs
        .iter()
        .enumerate()
        .filter_map(|(i, t)| t.framework().map(|_| i))
        .collect();
    assert_eq!(
        fw_tab_indices.len(),
        23,
        "Expected all 23 SimulationFramework tabs to be registered"
    );
    for tab_idx in fw_tab_indices {
        verify_theory_portal_in_tab(tab_idx);
    }
}

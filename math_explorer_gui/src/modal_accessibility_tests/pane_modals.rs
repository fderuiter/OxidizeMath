#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::helpers::{
    ak_action_evt, click_pos, cmd_mods, configure_layout_with_canary_panes, copied_texts,
    find_ak_node_in_pane, find_ak_nodes, find_text_centers, focus_ak_node, has_rendered_text,
    key_evt, run_app, secondary_click_pos, snapshot_all_pane_states, target_pane_rect,
};
use crate::app::{MathExplorerApp, ViewportLayout};
use eframe::egui::{self, accesskit, Context, Event, Id, Key, Modifiers, Rect};

fn locate_target_pane_plot(
    ctx: &Context,
    app: &mut MathExplorerApp,
    pane_idx: usize,
) -> (Id, Rect) {
    let out = run_app(ctx, app, Vec::new());
    let pane_rect = target_pane_rect(ctx, pane_idx);
    let plot_node = find_ak_node_in_pane(
        ctx,
        &out,
        pane_rect,
        accesskit::Role::GraphicsDocument,
        "Interactive Plot",
    );
    let plot_id = focus_ak_node(ctx, app, plot_node);
    let plot_rect = ctx
        .read_response(plot_id)
        .expect("Focused plot widget must have a stored Response")
        .rect;
    (plot_id, plot_rect)
}

fn open_plot_button_modal(
    ctx: &Context,
    app: &mut MathExplorerApp,
    pane_idx: usize,
    label: &str,
) -> Id {
    let out = run_app(ctx, app, Vec::new());
    let pane_rect = target_pane_rect(ctx, pane_idx);
    let btn_node = find_ak_node_in_pane(ctx, &out, pane_rect, accesskit::Role::Button, label);
    let btn_id = focus_ak_node(ctx, app, btn_node);
    let _ = run_app(
        ctx,
        app,
        vec![ak_action_evt(accesskit::Action::Click, btn_node)],
    );
    let _ = run_app(ctx, app, Vec::new());
    btn_id
}

fn focus_modal_close_button(ctx: &Context, app: &mut MathExplorerApp) {
    let out = run_app(ctx, app, Vec::new());
    let &(close_node, _) = find_ak_nodes(&out, accesskit::Role::Button, "Close")
        .last()
        .expect("Foreground modal must expose Close button in AccessKit");
    let _ = focus_ak_node(ctx, app, close_node);
}

fn verify_data_table_modal(ctx: &Context, app: &mut MathExplorerApp, pane_idx: usize) {
    let prior = open_plot_button_modal(ctx, app, pane_idx, "Table View");
    let out_init = run_app(ctx, app, Vec::new());
    assert!(has_rendered_text(&out_init, "Row 1/"));

    let out_down = run_app(ctx, app, vec![key_evt(Key::ArrowDown, Modifiers::NONE)]);
    assert!(has_rendered_text(&out_down, "Row 2/"));
    let out_up = run_app(ctx, app, vec![key_evt(Key::ArrowUp, Modifiers::NONE)]);
    assert!(has_rendered_text(&out_up, "Row 1/"));

    let _ = run_app(
        ctx,
        app,
        vec![
            Event::Copy,
            Event::Text("bg_leak_probe".into()),
            Event::Ime(egui::ImeEvent::Enabled),
            Event::Ime(egui::ImeEvent::Commit("!".into())),
        ],
    );

    focus_modal_close_button(ctx, app);
    let _ = run_app(ctx, app, vec![key_evt(Key::Space, Modifiers::NONE)]);
    assert_eq!(ctx.memory(|m| m.focused()), Some(prior));
    let out_after_space = run_app(ctx, app, Vec::new());
    assert!(!has_rendered_text(&out_after_space, "Row 1/"));

    let prior2 = open_plot_button_modal(ctx, app, pane_idx, "Table View");
    focus_modal_close_button(ctx, app);
    let _ = run_app(ctx, app, vec![key_evt(Key::Enter, Modifiers::NONE)]);
    assert_eq!(ctx.memory(|m| m.focused()), Some(prior2));
    let out_after_enter = run_app(ctx, app, Vec::new());
    assert!(!has_rendered_text(&out_after_enter, "Row 1/"));

    let prior3 = open_plot_button_modal(ctx, app, pane_idx, "Table View");
    let _ = run_app(ctx, app, vec![key_evt(Key::Escape, Modifiers::NONE)]);
    assert_eq!(ctx.memory(|m| m.focused()), Some(prior3));
    let out_after_esc = run_app(ctx, app, Vec::new());
    assert!(!has_rendered_text(&out_after_esc, "Row 1/"));
}

fn verify_plot_export_modal(ctx: &Context, app: &mut MathExplorerApp, pane_idx: usize) {
    let prior = open_plot_button_modal(ctx, app, pane_idx, "Save PNG");
    let out0 = run_app(ctx, app, Vec::new());
    assert!(has_rendered_text(&out0, "1920 x 1080"));

    // Tab -> "2x (4K)" -> Space -> 3840 x 2160; Tab -> "4x" -> Enter -> 7680 x 4320
    let _ = run_app(ctx, app, vec![key_evt(Key::Tab, Modifiers::NONE)]);
    let out_2x = run_app(ctx, app, vec![key_evt(Key::Space, Modifiers::NONE)]);
    assert!(has_rendered_text(&out_2x, "3840 x 2160"));

    let _ = run_app(ctx, app, vec![key_evt(Key::Tab, Modifiers::NONE)]);
    let out_4x = run_app(ctx, app, vec![key_evt(Key::Enter, Modifiers::NONE)]);
    assert!(has_rendered_text(&out_4x, "Capped at max dimension"));

    // Tab -> Aspect Ratio ComboBox -> Tab -> Line Thickness Slider (ArrowUp)
    let _ = run_app(ctx, app, vec![key_evt(Key::Tab, Modifiers::NONE)]);
    let _ = run_app(ctx, app, vec![key_evt(Key::Tab, Modifiers::NONE)]);
    let _ = run_app(
        ctx,
        app,
        vec![
            key_evt(Key::ArrowUp, Modifiers::NONE),
            Event::Text("no_leak".into()),
            Event::Ime(egui::ImeEvent::Enabled),
            Event::Ime(egui::ImeEvent::Commit("!".into())),
            Event::Copy,
        ],
    );
    let _ = run_app(ctx, app, vec![key_evt(Key::Escape, Modifiers::NONE)]);
    assert_eq!(ctx.memory(|m| m.focused()), Some(prior));
    let out_esc = run_app(ctx, app, Vec::new());
    assert!(!has_rendered_text(&out_esc, "Target Canvas Size:"));
}

fn is_ak_checkbox_checked(out: &egui::FullOutput, label_substr: &str) -> bool {
    out.platform_output
        .accesskit_update
        .as_ref()
        .is_some_and(|ak| {
            ak.nodes.iter().any(|(_, n)| {
                n.role() == accesskit::Role::CheckBox
                    && n.label().is_some_and(|l| l.contains(label_substr))
                    && n.toggled() == Some(accesskit::Toggled::True)
            })
        })
}

fn verify_plot_settings_modal(ctx: &Context, app: &mut MathExplorerApp, pane_idx: usize) {
    let prior = open_plot_button_modal(ctx, app, pane_idx, "Settings");

    // First focusable control is "Logarithmic X-Axis" checkbox
    let out_on = run_app(ctx, app, vec![key_evt(Key::Space, Modifiers::NONE)]);
    assert!(is_ak_checkbox_checked(&out_on, "Logarithmic X-Axis"));

    let out_off = run_app(ctx, app, vec![key_evt(Key::Enter, Modifiers::NONE)]);
    assert!(!is_ak_checkbox_checked(&out_off, "Logarithmic X-Axis"));

    // Tab -> Log Y -> Tab -> Min Line Spacing slider (ArrowUp)
    let _ = run_app(ctx, app, vec![key_evt(Key::Tab, Modifiers::NONE)]);
    let _ = run_app(ctx, app, vec![key_evt(Key::Tab, Modifiers::NONE)]);
    let _ = run_app(
        ctx,
        app,
        vec![
            key_evt(Key::ArrowUp, Modifiers::NONE),
            Event::Text("settings_probe".into()),
            Event::Ime(egui::ImeEvent::Enabled),
            Event::Ime(egui::ImeEvent::Commit("!".into())),
            Event::Copy,
        ],
    );
    let _ = run_app(ctx, app, vec![key_evt(Key::Escape, Modifiers::NONE)]);
    assert_eq!(ctx.memory(|m| m.focused()), Some(prior));
    let out_esc = run_app(ctx, app, Vec::new());
    assert!(!has_rendered_text(&out_esc, "Grid & Axis Settings"));
}

fn open_add_callout_via_context_menu(
    ctx: &Context,
    app: &mut MathExplorerApp,
    plot_id: Id,
    plot_rect: Rect,
) {
    ctx.memory_mut(|m| m.request_focus(plot_id));
    let _ = run_app(ctx, app, Vec::new());
    let center = ctx
        .read_response(plot_id)
        .map_or(plot_rect, |r| r.rect)
        .center();
    let _ = secondary_click_pos(ctx, app, center);
    let menu_out = run_app(ctx, app, vec![Event::PointerMoved(center)]);
    let add_pos = *find_text_centers(&menu_out, "Add Callout")
        .last()
        .expect("Plot context menu must expose Add Callout");
    let _ = click_pos(ctx, app, add_pos);
    let _ = run_app(ctx, app, Vec::new());
}

fn verify_add_callout_modal(
    ctx: &Context,
    app: &mut MathExplorerApp,
    plot_id: Id,
    plot_rect: Rect,
) {
    open_add_callout_via_context_menu(ctx, app, plot_id, plot_rect);

    let _ = run_app(
        ctx,
        app,
        vec![
            Event::Text("Peak".into()),
            Event::Ime(egui::ImeEvent::Enabled),
            Event::Ime(egui::ImeEvent::Commit("!".into())),
            Event::Ime(egui::ImeEvent::Disabled),
            Event::Text(" ".into()),
            key_evt(Key::Space, Modifiers::NONE),
            Event::Text("A".into()),
            key_evt(Key::ArrowLeft, Modifiers::NONE),
            key_evt(Key::ArrowUp, Modifiers::NONE),
            key_evt(Key::ArrowDown, Modifiers::NONE),
        ],
    );
    let copy_out = run_app(ctx, app, vec![key_evt(Key::A, cmd_mods()), Event::Copy]);
    assert!(copied_texts(&copy_out).iter().any(|c| c == "Peak! A"));

    let _ = run_app(ctx, app, vec![key_evt(Key::Enter, Modifiers::NONE)]);
    let out_after = run_app(ctx, app, Vec::new());
    assert!(!has_rendered_text(&out_after, "New Callout Annotation"));
    assert!(has_rendered_text(&out_after, "Peak! A"));

    // Reopen Add Callout and verify Escape cancels and restores prior focus
    open_add_callout_via_context_menu(ctx, app, plot_id, plot_rect);
    let _ = run_app(ctx, app, vec![key_evt(Key::Escape, Modifiers::NONE)]);
    let out_esc = run_app(ctx, app, Vec::new());
    assert_eq!(ctx.memory(|m| m.focused()), Some(plot_id));
    assert!(!has_rendered_text(&out_esc, "New Callout Annotation"));
}

fn open_edit_callout_by_badge_click(
    ctx: &Context,
    app: &mut MathExplorerApp,
    plot_id: Id,
    badge_label: &str,
) {
    ctx.memory_mut(|m| m.request_focus(plot_id));
    let out = run_app(ctx, app, Vec::new());
    let badge_pos = *find_text_centers(&out, badge_label)
        .last()
        .unwrap_or_else(|| panic!("Expected rendered callout badge '{badge_label}'"));
    let _ = click_pos(ctx, app, badge_pos);
    let _ = run_app(ctx, app, Vec::new());
}

fn verify_edit_callout_modal(ctx: &Context, app: &mut MathExplorerApp, plot_id: Id) {
    open_edit_callout_by_badge_click(ctx, app, plot_id, "Peak! A");

    let _ = run_app(
        ctx,
        app,
        vec![
            key_evt(Key::A, cmd_mods()),
            Event::Text("Edit".into()),
            Event::Ime(egui::ImeEvent::Enabled),
            Event::Ime(egui::ImeEvent::Commit("!".into())),
            Event::Ime(egui::ImeEvent::Disabled),
            Event::Text(" ".into()),
            key_evt(Key::Space, Modifiers::NONE),
            Event::Text("B".into()),
            key_evt(Key::ArrowUp, Modifiers::NONE),
            key_evt(Key::ArrowDown, Modifiers::NONE),
        ],
    );
    let copy_out = run_app(ctx, app, vec![key_evt(Key::A, cmd_mods()), Event::Copy]);
    assert!(copied_texts(&copy_out).iter().any(|c| c == "Edit! B"));

    // Tab: TextEdit -> dx DragValue -> dy DragValue -> Update button, then Enter commits
    for _ in 0..3 {
        let _ = run_app(ctx, app, vec![key_evt(Key::Tab, Modifiers::NONE)]);
    }
    let _ = run_app(ctx, app, vec![key_evt(Key::Enter, Modifiers::NONE)]);
    assert_eq!(ctx.memory(|m| m.focused()), Some(plot_id));
    let out_upd = run_app(ctx, app, Vec::new());
    assert!(!has_rendered_text(&out_upd, "Edit Callout Annotation"));
    assert!(has_rendered_text(&out_upd, "Edit! B"));

    open_edit_callout_by_badge_click(ctx, app, plot_id, "Edit! B");
    let _ = run_app(ctx, app, vec![key_evt(Key::Escape, Modifiers::NONE)]);
    assert_eq!(ctx.memory(|m| m.focused()), Some(plot_id));
    let out_esc = run_app(ctx, app, Vec::new());
    assert!(!has_rendered_text(&out_esc, "Edit Callout Annotation"));
}

fn pane_export_options(app: &MathExplorerApp, pane_idx: usize) -> egui_plot::export::ExportOptions {
    app.pane_tab(pane_idx)
        .unwrap()
        .framework()
        .unwrap()
        .export_options
        .clone()
}

fn verify_framework_export_modal(
    ctx: &Context,
    app: &mut MathExplorerApp,
    pane_idx: usize,
    prior: Id,
) {
    ctx.memory_mut(|m| m.request_focus(prior));
    let _ = run_app(ctx, app, Vec::new());

    app.pane_tab_mut(pane_idx)
        .unwrap()
        .framework_mut()
        .unwrap()
        .show_export_modal = true;
    let _ = run_app(ctx, app, Vec::new());
    let _ = run_app(ctx, app, Vec::new());

    let _ = run_app(ctx, app, vec![key_evt(Key::Tab, Modifiers::NONE)]);
    let _ = run_app(ctx, app, vec![key_evt(Key::Space, Modifiers::NONE)]);
    assert!((pane_export_options(app, pane_idx).scale - 2.0).abs() < 1e-5);

    let _ = run_app(ctx, app, vec![key_evt(Key::Tab, Modifiers::NONE)]);
    let _ = run_app(ctx, app, vec![key_evt(Key::Enter, Modifiers::NONE)]);
    assert!((pane_export_options(app, pane_idx).scale - 4.0).abs() < 1e-5);

    let _ = run_app(ctx, app, vec![key_evt(Key::Tab, Modifiers::NONE)]);
    let _ = run_app(ctx, app, vec![key_evt(Key::Tab, Modifiers::NONE)]);
    let prev_thick = pane_export_options(app, pane_idx).line_thickness;
    let _ = run_app(
        ctx,
        app,
        vec![
            key_evt(Key::ArrowRight, Modifiers::NONE),
            key_evt(Key::ArrowUp, Modifiers::NONE),
            Event::Text("fw_export_probe".into()),
            Event::Ime(egui::ImeEvent::Enabled),
            Event::Ime(egui::ImeEvent::Commit("!".into())),
            Event::Copy,
        ],
    );
    assert!(pane_export_options(app, pane_idx).line_thickness > prev_thick);

    let _ = run_app(ctx, app, vec![key_evt(Key::Escape, Modifiers::NONE)]);
    assert!(
        !app.pane_tab(pane_idx)
            .unwrap()
            .framework()
            .unwrap()
            .show_export_modal
    );
    assert_eq!(ctx.memory(|m| m.focused()), Some(prior));
}

#[test]
#[cfg(all(feature = "pure_math", feature = "ai", feature = "epidemiology"))]
fn test_all_six_pane_modals_across_all_four_viewport_layouts() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    for layout in [
        ViewportLayout::Single,
        ViewportLayout::SplitHorizontal,
        ViewportLayout::SplitVertical,
        ViewportLayout::QuadGrid,
    ] {
        let ctx = Context::default();
        ctx.enable_accesskit();
        let mut app = MathExplorerApp::default();
        let modal_pane = configure_layout_with_canary_panes(&mut app, layout);
        let (plot_id, plot_rect) = locate_target_pane_plot(&ctx, &mut app, modal_pane);
        let baseline_states = snapshot_all_pane_states(&app, Some(modal_pane));

        verify_data_table_modal(&ctx, &mut app, modal_pane);
        verify_plot_export_modal(&ctx, &mut app, modal_pane);
        verify_plot_settings_modal(&ctx, &mut app, modal_pane);
        verify_add_callout_modal(&ctx, &mut app, plot_id, plot_rect);
        verify_edit_callout_modal(&ctx, &mut app, plot_id);
        verify_framework_export_modal(&ctx, &mut app, modal_pane, plot_id);

        assert_eq!(
            snapshot_all_pane_states(&app, Some(modal_pane)),
            baseline_states,
            "Background controls in active and inactive panes must receive zero events in {layout:?}"
        );
        let modal_pane_saved = app
            .pane_tab(modal_pane)
            .and_then(|t| t.save_state())
            .unwrap_or_default();
        assert!(modal_pane_saved.contains("\"label\":\"Edit! B\""));
    }
}

#[test]
#[cfg(all(feature = "pure_math", feature = "ai", feature = "epidemiology"))]
fn test_covered_data_table_ignores_arrows_and_copy_in_full_app() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let modal_pane = configure_layout_with_canary_panes(&mut app, ViewportLayout::SplitHorizontal);
    let _ = locate_target_pane_plot(&ctx, &mut app, modal_pane);

    let _ = open_plot_button_modal(&ctx, &mut app, modal_pane, "Table View");
    let out_row2 = run_app(
        &ctx,
        &mut app,
        vec![key_evt(Key::ArrowDown, Modifiers::NONE)],
    );
    assert!(has_rendered_text(&out_row2, "Row 2/"));

    // Cover Data Table with CommandPalette and verify Data Table ignores ArrowDown and Copy
    app.command_palette.open();
    let _ = run_app(&ctx, &mut app, Vec::new());
    let out_cov = run_app(
        &ctx,
        &mut app,
        vec![key_evt(Key::ArrowDown, Modifiers::NONE), Event::Copy],
    );
    assert!(has_rendered_text(&out_cov, "Row 2/"));
    assert!(copied_texts(&out_cov).is_empty());

    // Escape closes CommandPalette first while keeping Data Table open on Row 2
    let out_after = run_app(&ctx, &mut app, vec![key_evt(Key::Escape, Modifiers::NONE)]);
    assert!(!app.command_palette.is_open);
    assert!(has_rendered_text(&out_after, "Row 2/"));
}

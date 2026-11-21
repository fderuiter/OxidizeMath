use super::*;
use crate::app::SplitLayoutMode;

#[test]
fn test_handle_view_layout_event_updates_viewport_layout_and_swaps_panes() {
    let mut app = MathExplorerApp::default();

    for (menu_id, expected_layout, expected_split_mode) in [
        (
            "layout_split_h",
            ViewportLayout::SplitHorizontal,
            SplitLayoutMode::Horizontal,
        ),
        (
            "layout_split_v",
            ViewportLayout::SplitVertical,
            SplitLayoutMode::Vertical,
        ),
        (
            "layout_quad",
            ViewportLayout::QuadGrid,
            SplitLayoutMode::Single,
        ),
        (
            "layout_single",
            ViewportLayout::Single,
            SplitLayoutMode::Single,
        ),
    ] {
        assert!(handle_view_layout_event(&mut app, menu_id));
        assert_eq!(app.viewport_layout, expected_layout);
        assert_eq!(app.split_layout_mode, expected_split_mode);
    }

    if app.tabs.len() > 1 {
        app.selected_tab = 0;
        app.secondary_tab = 1;
        app.pane_tabs[0] = 0;
        app.pane_tabs[1] = 1;
        assert!(handle_view_layout_event(&mut app, "layout_swap"));
        assert_eq!(app.selected_tab, 1);
        assert_eq!(app.secondary_tab, 0);
        assert_eq!(app.pane_tabs[0], 1);
        assert_eq!(app.pane_tabs[1], 0);
    } else {
        assert!(handle_view_layout_event(&mut app, "layout_swap"));
    }
}

#[test]
fn test_handle_view_layout_event_rejects_removed_view_split_ids() {
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::SplitHorizontal);
    let initial_primary = app.selected_tab;
    let initial_secondary = app.secondary_tab;

    for legacy_id in [
        "view_split_single",
        "view_split_horizontal",
        "view_split_vertical",
        "view_split_swap",
        "unknown_layout_id",
    ] {
        assert!(
            !handle_view_layout_event(&mut app, legacy_id),
            "Removed or unknown menu ID '{legacy_id}' must return false"
        );
        assert_eq!(app.viewport_layout, ViewportLayout::SplitHorizontal);
        assert_eq!(app.selected_tab, initial_primary);
        assert_eq!(app.secondary_tab, initial_secondary);
    }
}

#[test]
fn test_new_succeeds_on_worker_thread_without_appkit_main_thread_panic() {
    let handle = std::thread::spawn(|| {
        let _guard = crate::reflective_ui::tests::lock_test_mutex();
        let ctx = egui::Context::default();
        let cc = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut app = MathExplorerApp::new(&cc);
        let mut frame = eframe::Frame::_new_kittest();
        let _ = ctx.run(egui::RawInput::default(), |ui_ctx| {
            eframe::App::update(&mut app, ui_ctx, &mut frame);
        });
        app.theme
    });
    assert_eq!(
        handle.join().expect("worker thread must not panic"),
        ThemeMode::Dark
    );
}

#[test]
fn test_handle_quad_pair_swap_rotation_and_focus_pane_events() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::QuadGrid);
    if !app.tabs.is_empty() {
        for p in 0..4 {
            app.set_pane_tab(p, p % app.tabs.len());
        }
    }
    let baseline = app.pane_tabs.clone();
    assert!(handle_view_layout_event(&mut app, "layout_swap_0_2"));
    assert_eq!(app.pane_tabs[0], baseline[2]);
    assert_eq!(app.pane_tabs[2], baseline[0]);
    assert!(handle_view_layout_event(&mut app, "layout_swap_0_2"));
    assert_eq!(app.pane_tabs, baseline);

    assert!(handle_view_layout_event(&mut app, "layout_rotate_cw"));
    assert!(handle_view_layout_event(&mut app, "layout_rotate_ccw"));
    assert_eq!(app.pane_tabs, baseline);

    app.focus_pane(0);
    assert!(handle_focus_pane_event(&mut app, &ctx, "focus_pane_next"));
    assert_eq!(app.focused_pane, 1);
    assert!(handle_focus_pane_event(&mut app, &ctx, "focus_pane_prev"));
    assert_eq!(app.focused_pane, 0);
    assert!(handle_focus_pane_event(&mut app, &ctx, "focus_pane_3"));
    assert_eq!(app.focused_pane, 3);
    assert!(!handle_focus_pane_event(
        &mut app,
        &ctx,
        "focus_pane_unknown"
    ));
    setup_macos_menu(&app);
}

#[test]
fn test_setup_macos_menu_and_share_link_event_safe_on_worker_threads() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let _ = crate::url_state::take_last_copied_share_url();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();

    setup_macos_menu(&app);
    assert!(handle_file_event(&mut app, &ctx, "file_share_link"));
    let copied =
        crate::url_state::take_last_copied_share_url().expect("macOS menu copied share link");
    assert!(copied.starts_with("https://fderuiter.github.io/OxidizeMath/#tab="));
}

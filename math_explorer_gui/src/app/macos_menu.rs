#![cfg(target_os = "macos")]
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::macos_menu_view::{build_view_menu, handle_focus_pane_event, handle_view_layout_event};
use super::{MathExplorerApp, ThemeMode};
use eframe::egui;
use muda::{Menu, MenuId, MenuItem, PredefinedMenuItem, Submenu};

fn build_file_menu() -> Submenu {
    let file_menu = Submenu::new("File", true);
    let open_item = MenuItem::with_id(
        MenuId::new("file_open"),
        "Open Workspace Session...",
        true,
        None,
    );
    let save_item = MenuItem::with_id(
        MenuId::new("file_save"),
        "Save Workspace Session",
        true,
        None,
    );
    let save_as_item = MenuItem::with_id(
        MenuId::new("file_save_as"),
        "Save Workspace Session As...",
        true,
        None,
    );
    let share_link_item = MenuItem::with_id(
        MenuId::new("file_share_link"),
        "Copy Share Link",
        true,
        None,
    );
    file_menu.append(&open_item).unwrap();
    file_menu.append(&save_item).unwrap();
    file_menu.append(&save_as_item).unwrap();
    file_menu.append(&PredefinedMenuItem::separator()).unwrap();
    file_menu.append(&share_link_item).unwrap();
    file_menu
}

fn build_edit_menu() -> Submenu {
    let edit_menu = Submenu::new("Edit", true);
    let undo_item = MenuItem::with_id(MenuId::new("edit_undo"), "Undo", true, None);
    let redo_item = MenuItem::with_id(MenuId::new("edit_redo"), "Redo", true, None);
    edit_menu.append(&undo_item).unwrap();
    edit_menu.append(&redo_item).unwrap();
    edit_menu
}

fn build_appearance_menu() -> Submenu {
    let appearance_menu = Submenu::new("Appearance", true);
    let dark_item = MenuItem::with_id(
        MenuId::new("theme_dark"),
        ThemeMode::Dark.name(),
        true,
        None,
    );
    let light_item = MenuItem::with_id(
        MenuId::new("theme_light"),
        ThemeMode::Light.name(),
        true,
        None,
    );
    let hc_item = MenuItem::with_id(
        MenuId::new("theme_high_contrast"),
        ThemeMode::HighContrast.name(),
        true,
        None,
    );
    appearance_menu.append(&dark_item).unwrap();
    appearance_menu.append(&light_item).unwrap();
    appearance_menu.append(&hc_item).unwrap();
    appearance_menu
}

fn build_help_menu() -> Submenu {
    let help_menu = Submenu::new("Help", true);
    let shortcuts_item = MenuItem::with_id(
        MenuId::new("help_commands"),
        "Keyboard Shortcuts",
        true,
        None,
    );
    help_menu.append(&shortcuts_item).unwrap();
    help_menu
}

fn can_init_nsapp_menu() -> bool {
    !cfg!(test) && std::thread::current().name() == Some("main")
}

pub(super) fn setup_macos_menu(app: &MathExplorerApp) {
    if !can_init_nsapp_menu() {
        return;
    }
    let menu = Menu::new();

    let app_menu = Submenu::new("Math Explorer", true);
    app_menu.append(&PredefinedMenuItem::quit(None)).unwrap();
    menu.append(&app_menu).unwrap();

    menu.append(&build_file_menu()).unwrap();
    menu.append(&build_edit_menu()).unwrap();
    menu.append(&build_view_menu(app)).unwrap();
    menu.append(&build_appearance_menu()).unwrap();
    menu.append(&build_help_menu()).unwrap();

    menu.init_for_nsapp();
}

fn handle_file_event(app: &mut MathExplorerApp, ctx: &egui::Context, id: &str) -> bool {
    match id {
        "file_open" => {
            app.session_manager.trigger_open();
            true
        }
        "file_save" => {
            app.clear_active_pane_caches();
            let state = app.get_app_state();
            let env_json = app.session_manager.create_envelope(state);
            app.session_manager.trigger_save(env_json);
            true
        }
        "file_save_as" => {
            app.clear_active_pane_caches();
            let state = app.get_app_state();
            let env_json = app.session_manager.create_envelope(state);
            app.session_manager.trigger_save_as(env_json);
            true
        }
        "file_share_link" => {
            app.copy_current_share_link(ctx);
            true
        }
        _ => false,
    }
}

fn handle_palette_theme_event(app: &mut MathExplorerApp, ctx: &egui::Context, id: &str) -> bool {
    match id {
        "cmd_palette" => {
            app.command_palette.toggle();
            let status_msg = if app.command_palette.is_open {
                "Command palette opened"
            } else {
                "Command palette closed"
            };
            crate::accessibility::announce_live_status(ctx, status_msg);
        }
        "help_commands" => app.show_help_menu = !app.show_help_menu,
        "theme_dark" => {
            app.theme = ThemeMode::Dark;
            app.theme.apply_to_context(ctx);
        }
        "theme_light" => {
            app.theme = ThemeMode::Light;
            app.theme.apply_to_context(ctx);
        }
        "theme_high_contrast" => {
            app.theme = ThemeMode::HighContrast;
            app.theme.apply_to_context(ctx);
        }
        "view_toggle_diagnostics" => {
            app.show_diagnostics_panel = !app.show_diagnostics_panel;
        }
        _ => return false,
    }
    true
}

pub(super) fn handle_macos_menu_events(app: &mut MathExplorerApp, ctx: &egui::Context) {
    if let Ok(event) = muda::MenuEvent::receiver().try_recv() {
        let id = event.id.0.as_str();
        if handle_file_event(app, ctx, id)
            || handle_view_layout_event(app, ctx, id)
            || handle_focus_pane_event(app, ctx, id)
            || handle_palette_theme_event(app, ctx, id)
        {
            return;
        }
        match id {
            "edit_undo" => app.undo(),
            "edit_redo" => app.redo(),
            _ => {
                if let Some(idx) = id
                    .strip_prefix("tab_")
                    .and_then(|s| s.parse::<usize>().ok())
                {
                    super::domain_bar::select_primary_tab(app, ctx, idx);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{SplitLayoutMode, ViewportLayout};

    #[test]
    fn test_handle_view_layout_event_updates_viewport_layout_and_swaps_panes() {
        let ctx = egui::Context::default();
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
            assert!(handle_view_layout_event(&mut app, &ctx, menu_id));
            assert_eq!(app.viewport_layout, expected_layout);
            assert_eq!(app.split_layout_mode, expected_split_mode);
        }

        if app.tabs.len() > 1 {
            app.selected_tab = 0;
            app.secondary_tab = 1;
            app.pane_tabs[0] = 0;
            app.pane_tabs[1] = 1;
            assert!(handle_view_layout_event(&mut app, &ctx, "layout_swap"));
            assert_eq!(app.selected_tab, 1);
            assert_eq!(app.secondary_tab, 0);
            assert_eq!(app.pane_tabs[0], 1);
            assert_eq!(app.pane_tabs[1], 0);
        } else {
            assert!(handle_view_layout_event(&mut app, &ctx, "layout_swap"));
        }
    }

    #[test]
    fn test_handle_duplicate_and_comparison_sync_menu_events() {
        let _guard = crate::reflective_ui::tests::lock_test_mutex();
        let ctx = egui::Context::default();
        let mut app = MathExplorerApp::default();
        assert!(handle_view_layout_event(
            &mut app,
            &ctx,
            "layout_duplicate_focused"
        ));
        assert_eq!(app.viewport_layout, ViewportLayout::SplitHorizontal);
        assert_eq!(app.focused_pane, 1);

        assert!(handle_view_layout_event(
            &mut app,
            &ctx,
            "compare_link_axes"
        ));
        assert!(app.compare_sync.link_axes);
        assert!(handle_view_layout_event(
            &mut app,
            &ctx,
            "compare_link_cursors"
        ));
        assert!(app.compare_sync.link_cursors);
        assert!(handle_view_layout_event(
            &mut app,
            &ctx,
            "compare_sync_playback"
        ));
        assert!(app.compare_sync.sync_playback);
    }

    #[test]
    fn test_handle_view_layout_event_rejects_removed_view_split_ids() {
        let ctx = egui::Context::default();
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
                !handle_view_layout_event(&mut app, &ctx, legacy_id),
                "Removed or unknown menu ID '{legacy_id}' must return false"
            );
            assert_eq!(app.viewport_layout, ViewportLayout::SplitHorizontal);
            assert_eq!(app.selected_tab, initial_primary);
            assert_eq!(app.secondary_tab, initial_secondary);
        }
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
        assert!(handle_view_layout_event(&mut app, &ctx, "layout_swap_0_2"));
        assert_eq!(app.pane_tabs[0], baseline[2]);
        assert_eq!(app.pane_tabs[2], baseline[0]);
        assert!(handle_view_layout_event(&mut app, &ctx, "layout_swap_0_2"));
        assert_eq!(app.pane_tabs, baseline);

        assert!(handle_view_layout_event(&mut app, &ctx, "layout_rotate_cw"));
        assert!(handle_view_layout_event(
            &mut app,
            &ctx,
            "layout_rotate_ccw"
        ));
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
}

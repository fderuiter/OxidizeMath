#![cfg(target_os = "macos")]
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::domain_bar::AcademicDomain;
use super::{MathExplorerApp, ThemeMode, ViewportLayout};
use eframe::egui;
use muda::{CheckMenuItem, Menu, MenuId, MenuItem, PredefinedMenuItem, Submenu};

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
    file_menu.append(&open_item).unwrap();
    file_menu.append(&save_item).unwrap();
    file_menu.append(&save_as_item).unwrap();
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
fn build_layout_submenu() -> Submenu {
    let layout_menu = Submenu::new("Viewport Layout", true);
    layout_menu
        .append(&MenuItem::with_id(
            MenuId::new("layout_single"),
            ViewportLayout::Single.name(),
            true,
            None,
        ))
        .unwrap();
    layout_menu
        .append(&MenuItem::with_id(
            MenuId::new("layout_split_h"),
            ViewportLayout::SplitHorizontal.name(),
            true,
            None,
        ))
        .unwrap();
    layout_menu
        .append(&MenuItem::with_id(
            MenuId::new("layout_split_v"),
            ViewportLayout::SplitVertical.name(),
            true,
            None,
        ))
        .unwrap();
    layout_menu
        .append(&MenuItem::with_id(
            MenuId::new("layout_quad"),
            ViewportLayout::QuadGrid.name(),
            true,
            None,
        ))
        .unwrap();
    layout_menu
        .append(&PredefinedMenuItem::separator())
        .unwrap();
    layout_menu
        .append(&MenuItem::with_id(
            MenuId::new("layout_swap"),
            "Swap Panes",
            true,
            None,
        ))
        .unwrap();
    layout_menu
}

fn build_focus_pane_submenu() -> Submenu {
    let focus_menu = Submenu::new("Focus Pane", true);
    focus_menu
        .append(&MenuItem::with_id(
            MenuId::new("focus_pane_next"),
            "Next Pane",
            true,
            None,
        ))
        .unwrap();
    focus_menu
        .append(&MenuItem::with_id(
            MenuId::new("focus_pane_prev"),
            "Previous Pane",
            true,
            None,
        ))
        .unwrap();
    focus_menu.append(&PredefinedMenuItem::separator()).unwrap();
    for (idx, label) in [
        (0, "Pane 1 (Primary / Top-Left)"),
        (1, "Pane 2 (Secondary / Top-Right)"),
        (2, "Pane 3 (Bottom-Left)"),
        (3, "Pane 4 (Bottom-Right)"),
    ] {
        focus_menu
            .append(&MenuItem::with_id(
                MenuId::new(format!("focus_pane_{idx}")),
                label,
                true,
                None,
            ))
            .unwrap();
    }
    focus_menu
}

fn build_primary_module_submenu(app: &MathExplorerApp) -> Submenu {
    let primary_menu = Submenu::new("Primary Module", true);
    for &domain in AcademicDomain::ALL {
        let domain_menu = Submenu::new(domain.name(), true);
        let mut has_items = false;
        for (i, tab) in app.tabs.iter().enumerate() {
            if AcademicDomain::from_str(tab.academic_domain()) == Some(domain) {
                let item =
                    MenuItem::with_id(MenuId::new(format!("tab_{}", i)), tab.name(), true, None);
                domain_menu.append(&item).unwrap();
                has_items = true;
            }
        }
        if has_items {
            primary_menu.append(&domain_menu).unwrap();
        }
    }
    primary_menu
}

fn build_view_menu(app: &MathExplorerApp) -> Submenu {
    let view_menu = Submenu::new("View", true);
    let palette_item =
        MenuItem::with_id(MenuId::new("cmd_palette"), "Command Palette...", true, None);
    view_menu.append(&palette_item).unwrap();
    view_menu.append(&PredefinedMenuItem::separator()).unwrap();

    view_menu.append(&build_layout_submenu()).unwrap();
    view_menu.append(&build_focus_pane_submenu()).unwrap();
    view_menu.append(&PredefinedMenuItem::separator()).unwrap();

    view_menu
        .append(&build_primary_module_submenu(app))
        .unwrap();
    view_menu.append(&PredefinedMenuItem::separator()).unwrap();

    let diag_item = CheckMenuItem::with_id(
        MenuId::new("view_toggle_diagnostics"),
        "Issues & Diagnostics Panel",
        true,
        app.show_diagnostics_panel,
        None,
    );
    view_menu.append(&diag_item).unwrap();
    view_menu
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

pub(super) fn setup_macos_menu(app: &MathExplorerApp) {
    let menu = Menu::new();

    // App Menu
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

fn handle_file_event(app: &mut MathExplorerApp, id: &str) -> bool {
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
        _ => false,
    }
}

fn handle_view_layout_event(app: &mut MathExplorerApp, id: &str) -> bool {
    match id {
        "layout_single" => app.set_layout(ViewportLayout::Single),
        "layout_split_h" => app.set_layout(ViewportLayout::SplitHorizontal),
        "layout_split_v" => app.set_layout(ViewportLayout::SplitVertical),
        "layout_quad" => app.set_layout(ViewportLayout::QuadGrid),
        "layout_swap" => app.swap_panes(),
        _ => return false,
    }
    true
}

fn handle_focus_pane_event(app: &mut MathExplorerApp, ctx: &egui::Context, id: &str) -> bool {
    match id {
        "focus_pane_next" => super::viewport::focus::cycle_focused_pane(app, ctx, false),
        "focus_pane_prev" => super::viewport::focus::cycle_focused_pane(app, ctx, true),
        "focus_pane_0" => super::viewport::focus::focus_pane_interactive(app, ctx, 0),
        "focus_pane_1" => super::viewport::focus::focus_pane_interactive(app, ctx, 1),
        "focus_pane_2" => super::viewport::focus::focus_pane_interactive(app, ctx, 2),
        "focus_pane_3" => super::viewport::focus::focus_pane_interactive(app, ctx, 3),
        _ => return false,
    }
    true
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

fn handle_edit_event(app: &mut MathExplorerApp, id: &str) -> bool {
    match id {
        "edit_undo" => {
            if app.can_undo() {
                app.undo();
            }
            true
        }
        "edit_redo" => {
            if app.can_redo() {
                app.redo();
            }
            true
        }
        _ => false,
    }
}

pub(super) fn handle_macos_menu_events(app: &mut MathExplorerApp, ctx: &egui::Context) {
    if let Ok(event) = muda::MenuEvent::receiver().try_recv() {
        let id = event.id.0.as_str();
        if handle_file_event(app, id)
            || handle_edit_event(app, id)
            || handle_view_layout_event(app, id)
            || handle_focus_pane_event(app, ctx, id)
            || handle_palette_theme_event(app, ctx, id)
        {
            return;
        }
        if let Some(rest) = id.strip_prefix("tab_") {
            if let Ok(idx) = rest.parse::<usize>() {
                super::domain_bar::select_primary_tab(app, ctx, idx);
            }
        }
    }
}

#[cfg(test)]
mod tests {
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
    fn test_handle_edit_event_respects_undo_redo_availability() {
        let mut app = MathExplorerApp::default();
        assert!(!app.can_undo());
        assert!(!app.can_redo());
        assert!(handle_edit_event(&mut app, "edit_undo"));
        assert!(handle_edit_event(&mut app, "edit_redo"));
        assert!(!handle_edit_event(&mut app, "unknown_edit_id"));
    }
}

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
fn build_swap_pair_submenu() -> Submenu {
    let swap_pair_menu = Submenu::new("Swap Pane Pair", true);
    let labels = ViewportLayout::QuadGrid.pane_labels();
    for (a, b) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        let item = MenuItem::with_id(
            MenuId::new(format!("layout_swap_{a}_{b}")),
            format!("{} <-> {}", labels[a], labels[b]),
            true,
            None,
        );
        swap_pair_menu.append(&item).unwrap();
    }
    swap_pair_menu
}

fn build_layout_submenu() -> Submenu {
    let layout_menu = Submenu::new("Viewport Layout", true);
    for (id, layout) in [
        ("layout_single", ViewportLayout::Single),
        ("layout_split_h", ViewportLayout::SplitHorizontal),
        ("layout_split_v", ViewportLayout::SplitVertical),
        ("layout_quad", ViewportLayout::QuadGrid),
    ] {
        let item = MenuItem::with_id(MenuId::new(id), layout.name(), true, None);
        layout_menu.append(&item).unwrap();
    }
    layout_menu
        .append(&PredefinedMenuItem::separator())
        .unwrap();
    let swap_item = MenuItem::with_id(MenuId::new("layout_swap"), "Swap Panes", true, None);
    layout_menu.append(&swap_item).unwrap();
    layout_menu.append(&build_swap_pair_submenu()).unwrap();
    for (id, label) in [
        ("layout_rotate_cw", "Rotate Panes Clockwise"),
        ("layout_rotate_ccw", "Rotate Panes Counter-Clockwise"),
    ] {
        let item = MenuItem::with_id(MenuId::new(id), label, true, None);
        layout_menu.append(&item).unwrap();
    }
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

fn can_init_nsapp_menu() -> bool {
    !cfg!(test) && std::thread::current().name() == Some("main")
}

pub(super) fn setup_macos_menu(app: &MathExplorerApp) {
    if !can_init_nsapp_menu() {
        return;
    }
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

fn handle_view_layout_event(app: &mut MathExplorerApp, id: &str) -> bool {
    match id {
        "layout_single" => app.set_layout(ViewportLayout::Single),
        "layout_split_h" => app.set_layout(ViewportLayout::SplitHorizontal),
        "layout_split_v" => app.set_layout(ViewportLayout::SplitVertical),
        "layout_quad" => app.set_layout(ViewportLayout::QuadGrid),
        "layout_swap" | "layout_swap_0_1" => app.swap_pane_pair(0, 1),
        "layout_swap_0_2" => app.swap_pane_pair(0, 2),
        "layout_swap_0_3" => app.swap_pane_pair(0, 3),
        "layout_swap_1_2" => app.swap_pane_pair(1, 2),
        "layout_swap_1_3" => app.swap_pane_pair(1, 3),
        "layout_swap_2_3" => app.swap_pane_pair(2, 3),
        "layout_rotate_cw" => app.rotate_panes_clockwise(),
        "layout_rotate_ccw" => app.rotate_panes_counter_clockwise(),
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

pub(super) fn handle_macos_menu_events(app: &mut MathExplorerApp, ctx: &egui::Context) {
    if let Ok(event) = muda::MenuEvent::receiver().try_recv() {
        let id = event.id.0.as_str();
        if handle_file_event(app, ctx, id)
            || handle_view_layout_event(app, id)
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
#[path = "macos_menu_tests.rs"]
mod tests;

#![cfg(target_os = "macos")]
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{MathExplorerApp, SplitLayoutMode, ThemeMode, ViewportLayout};
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
fn build_split_submenu() -> Submenu {
    let split_submenu = Submenu::new("Split Layout", true);
    let split_single = MenuItem::with_id(
        MenuId::new("view_split_single"),
        SplitLayoutMode::Single.label(),
        true,
        None,
    );
    let split_horiz = MenuItem::with_id(
        MenuId::new("view_split_horizontal"),
        SplitLayoutMode::Horizontal.label(),
        true,
        None,
    );
    let split_vert = MenuItem::with_id(
        MenuId::new("view_split_vertical"),
        SplitLayoutMode::Vertical.label(),
        true,
        None,
    );
    let split_swap = MenuItem::with_id(MenuId::new("view_split_swap"), "Swap Panes", true, None);
    split_submenu.append(&split_single).unwrap();
    split_submenu.append(&split_horiz).unwrap();
    split_submenu.append(&split_vert).unwrap();
    split_submenu.append(&split_swap).unwrap();
    split_submenu
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
}

fn build_view_menu(app: &MathExplorerApp) -> Submenu {
    let view_menu = Submenu::new("View", true);
    let palette_item =
        MenuItem::with_id(MenuId::new("cmd_palette"), "Command Palette...", true, None);
    view_menu.append(&palette_item).unwrap();
    view_menu.append(&PredefinedMenuItem::separator()).unwrap();

    view_menu.append(&build_split_submenu()).unwrap();
    view_menu.append(&PredefinedMenuItem::separator()).unwrap();

    view_menu.append(&build_layout_submenu()).unwrap();
    view_menu.append(&PredefinedMenuItem::separator()).unwrap();

    for (i, name) in app.tab_names().into_iter().enumerate() {
        let item = MenuItem::with_id(MenuId::new(format!("tab_{}", i)), name, true, None);
        view_menu.append(&item).unwrap();
    }
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

    menu.init_for_nsapp();
}

fn handle_file_event(app: &mut MathExplorerApp, id: &str) -> bool {
    match id {
        "file_open" => {
            app.session_manager.trigger_open();
            true
        }
        "file_save" => {
            let state = app.get_app_state();
            let env_json = app.session_manager.create_envelope(state);
            app.session_manager.trigger_save(env_json);
            true
        }
        "file_save_as" => {
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
        "view_split_single" => app.split_layout_mode = SplitLayoutMode::Single,
        "view_split_horizontal" => app.split_layout_mode = SplitLayoutMode::Horizontal,
        "view_split_vertical" => app.split_layout_mode = SplitLayoutMode::Vertical,
        "view_split_swap" => app.swap_panes(),
        "layout_single" => app.viewport_layout = ViewportLayout::Single,
        "layout_split_h" => app.viewport_layout = ViewportLayout::SplitHorizontal,
        "layout_split_v" => app.viewport_layout = ViewportLayout::SplitVertical,
        "layout_quad" => app.viewport_layout = ViewportLayout::QuadGrid,
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
            ctx.data_mut(|d| {
                d.insert_temp(egui::Id::new("aria_live_message"), status_msg.to_string());
            });
        }
        "help_commands" => app.show_help_menu = !app.show_help_menu,
        "theme_dark" => {
            app.theme = ThemeMode::Dark;
            ctx.set_visuals(app.theme.visuals());
        }
        "theme_light" => {
            app.theme = ThemeMode::Light;
            ctx.set_visuals(app.theme.visuals());
        }
        "theme_high_contrast" => {
            app.theme = ThemeMode::HighContrast;
            ctx.set_visuals(app.theme.visuals());
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
        if handle_file_event(app, id)
            || handle_view_layout_event(app, id)
            || handle_palette_theme_event(app, ctx, id)
        {
            return;
        }
        match id {
            "edit_undo" => app.undo(),
            "edit_redo" => app.redo(),
            other if other.starts_with("tab_") => {
                if let Ok(idx) = other[4..].parse::<usize>() {
                    app.selected_tab = idx;
                }
            }
            _ => {}
        }
    }
}

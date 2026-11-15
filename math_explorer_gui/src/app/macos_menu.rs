#![cfg(target_os = "macos")]

use super::MathExplorerApp;
use muda::{CheckMenuItem, Menu, MenuId, MenuItem, PredefinedMenuItem, Submenu};

pub(super) fn setup_macos_menu(app: &MathExplorerApp) {
    let menu = Menu::new();

    // App Menu
    let app_menu = Submenu::new("Math Explorer", true);
    app_menu.append(&PredefinedMenuItem::quit(None)).unwrap();
    menu.append(&app_menu).unwrap();

    // File Menu
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
    menu.append(&file_menu).unwrap();

    // Edit Menu
    let edit_menu = Submenu::new("Edit", true);
    let undo_item = MenuItem::with_id(MenuId::new("edit_undo"), "Undo", true, None);
    let redo_item = MenuItem::with_id(MenuId::new("edit_redo"), "Redo", true, None);
    edit_menu.append(&undo_item).unwrap();
    edit_menu.append(&redo_item).unwrap();
    menu.append(&edit_menu).unwrap();

    // View Menu
    let view_menu = Submenu::new("View", true);
    let palette_item =
        MenuItem::with_id(MenuId::new("cmd_palette"), "Command Palette...", true, None);
    view_menu.append(&palette_item).unwrap();
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
    menu.append(&view_menu).unwrap();

    // Appearance Menu
    let appearance_menu = Submenu::new("Appearance", true);
    let dark_item = MenuItem::with_id(MenuId::new("theme_dark"), "Dark", true, None);
    let light_item = MenuItem::with_id(MenuId::new("theme_light"), "Light", true, None);
    let hc_item = MenuItem::with_id(
        MenuId::new("theme_high_contrast"),
        "High Contrast",
        true,
        None,
    );
    appearance_menu.append(&dark_item).unwrap();
    appearance_menu.append(&light_item).unwrap();
    appearance_menu.append(&hc_item).unwrap();
    menu.append(&appearance_menu).unwrap();

    menu.init_for_nsapp();
}

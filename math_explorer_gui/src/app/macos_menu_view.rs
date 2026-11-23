#![cfg(target_os = "macos")]
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::domain_bar::AcademicDomain;
use super::{MathExplorerApp, ViewportLayout};
use eframe::egui;
use muda::{CheckMenuItem, MenuId, MenuItem, PredefinedMenuItem, Submenu};

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

fn build_layout_submenu(app: &MathExplorerApp) -> Submenu {
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
    let duplicate_item = MenuItem::with_id(
        MenuId::new("layout_duplicate_focused"),
        "Duplicate Focused Pane",
        true,
        None,
    );
    layout_menu.append(&duplicate_item).unwrap();
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
        .append(&PredefinedMenuItem::separator())
        .unwrap();
    append_compare_sync_items(&layout_menu, app);
    layout_menu
}

fn append_compare_sync_items(layout_menu: &Submenu, app: &MathExplorerApp) {
    let sync = app.compare_sync;
    for (id, label, checked) in [
        ("compare_link_axes", "Link Plot Axes", sync.link_axes),
        (
            "compare_link_cursors",
            "Link Plot Cursors",
            sync.link_cursors,
        ),
        ("compare_sync_playback", "Sync Playback", sync.sync_playback),
    ] {
        let item = CheckMenuItem::with_id(MenuId::new(id), label, true, checked, None);
        layout_menu.append(&item).unwrap();
    }
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
                    MenuItem::with_id(MenuId::new(format!("tab_{i}")), tab.name(), true, None);
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

pub(super) fn build_view_menu(app: &MathExplorerApp) -> Submenu {
    let view_menu = Submenu::new("View", true);
    let catalog_item = MenuItem::with_id(
        MenuId::new("view_explore_catalog"),
        "Explore Catalog...",
        true,
        None,
    );
    view_menu.append(&catalog_item).unwrap();
    let palette_item =
        MenuItem::with_id(MenuId::new("cmd_palette"), "Command Palette...", true, None);
    view_menu.append(&palette_item).unwrap();
    view_menu.append(&PredefinedMenuItem::separator()).unwrap();

    view_menu.append(&build_layout_submenu(app)).unwrap();
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

pub(super) fn handle_view_layout_event(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    id: &str,
) -> bool {
    match id {
        "layout_single" => app.set_layout(ViewportLayout::Single),
        "layout_split_h" => app.set_layout(ViewportLayout::SplitHorizontal),
        "layout_split_v" => app.set_layout(ViewportLayout::SplitVertical),
        "layout_quad" => app.set_layout(ViewportLayout::QuadGrid),
        "layout_duplicate_focused" => {
            let _ = app.duplicate_focused_pane(Some(ctx));
        }
        "layout_swap" | "layout_swap_0_1" => app.swap_pane_pair(0, 1),
        "layout_swap_0_2" => app.swap_pane_pair(0, 2),
        "layout_swap_0_3" => app.swap_pane_pair(0, 3),
        "layout_swap_1_2" => app.swap_pane_pair(1, 2),
        "layout_swap_1_3" => app.swap_pane_pair(1, 3),
        "layout_swap_2_3" => app.swap_pane_pair(2, 3),
        "layout_rotate_cw" => app.rotate_panes_clockwise(),
        "layout_rotate_ccw" => app.rotate_panes_counter_clockwise(),
        "compare_link_axes" => app.set_link_plot_axes(Some(ctx), !app.compare_sync.link_axes),
        "compare_link_cursors" => {
            app.set_link_plot_cursors(Some(ctx), !app.compare_sync.link_cursors);
        }
        "compare_sync_playback" => {
            app.set_sync_playback(Some(ctx), !app.compare_sync.sync_playback);
        }
        _ => return false,
    }
    true
}

pub(super) fn handle_focus_pane_event(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    id: &str,
) -> bool {
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

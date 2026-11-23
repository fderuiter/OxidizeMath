#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::app::viewport::compare_sync::render_comparison_sync_checkboxes;
use crate::app::{MathExplorerApp, ViewportLayout};
use eframe::egui;

/// Renders the `Viewport Layout` flyout submenu including pane cloning, swap, rotation, and comparison sync toggles.
pub(super) fn render_viewport_layout_submenu(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    ui: &mut egui::Ui,
) {
    let dup_shortcut = if cfg!(target_os = "macos") {
        "Cmd+Shift+2"
    } else {
        "Ctrl+Shift+2"
    };
    ui.menu_button("Viewport Layout", |ui| {
        for layout in [
            ViewportLayout::Single,
            ViewportLayout::SplitHorizontal,
            ViewportLayout::SplitVertical,
            ViewportLayout::QuadGrid,
        ] {
            let mut current = app.viewport_layout;
            if ui
                .radio_value(&mut current, layout, layout.name())
                .clicked()
            {
                app.set_layout(layout);
                ui.close();
            }
        }

        ui.separator();
        if ui
            .add(egui::Button::new("⧉ Duplicate Focused Pane").shortcut_text(dup_shortcut))
            .clicked()
        {
            let _ = app.duplicate_focused_pane(Some(ctx));
            ui.close();
            return;
        }

        if app.viewport_layout != ViewportLayout::Single {
            if ui.button("⇄ Swap Panes").clicked() {
                app.swap_panes();
                ui.close();
                return;
            }
            if app.viewport_layout == ViewportLayout::QuadGrid {
                render_quad_swap_and_rotate_items(app, ctx, ui);
            }
            ui.separator();
            render_comparison_sync_checkboxes(app, ctx, ui);
        }
    });
}

fn render_quad_swap_and_rotate_items(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    ui: &mut egui::Ui,
) {
    let labels = ViewportLayout::QuadGrid.pane_labels();
    let src_pane = app.active_focused_pane();
    ui.menu_button("⧉ Duplicate Pane To", |clone_ui| {
        for (dst_idx, &dst_lbl) in labels.iter().enumerate() {
            if dst_idx != src_pane && clone_ui.button(format!("⧉ {dst_lbl}")).clicked() {
                let _ = app.clone_pane_interactive(ctx, src_pane, dst_idx);
                clone_ui.close();
                return;
            }
        }
    });
    ui.menu_button("⇄ Swap Pane Pair", |pair_ui| {
        for (a, b) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
            let text = format!("{} ⇄ {}", labels[a], labels[b]);
            if pair_ui.button(text).clicked() {
                app.swap_pane_pair(a, b);
                pair_ui.close();
                return;
            }
        }
    });
    if ui.button("↻ Rotate Panes Clockwise").clicked() {
        app.rotate_panes_clockwise();
        ui.close();
    } else if ui.button("↺ Rotate Panes Counter-Clockwise").clicked() {
        app.rotate_panes_counter_clockwise();
        ui.close();
    }
}

/// Renders the `Focus Pane` submenu (`F6` / `Shift+F6` and direct pane targets).
pub(super) fn render_focus_pane_submenu(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    ui: &mut egui::Ui,
) {
    ui.menu_button("Focus Pane", |ui| {
        if ui
            .add(egui::Button::new("Next Pane").shortcut_text("F6"))
            .clicked()
        {
            crate::app::viewport::focus::cycle_focused_pane(app, ctx, false);
            ui.close();
            return;
        }
        if ui
            .add(egui::Button::new("Previous Pane").shortcut_text("Shift+F6"))
            .clicked()
        {
            crate::app::viewport::focus::cycle_focused_pane(app, ctx, true);
            ui.close();
            return;
        }
        ui.separator();

        let pane_count = app.viewport_layout.pane_count();
        let pane_labels = app.viewport_layout.pane_labels();
        let active_pane = app.active_focused_pane();
        for pane_idx in 0..pane_count {
            let label = pane_labels
                .get(pane_idx)
                .copied()
                .unwrap_or("Main Viewport");
            let tab_idx = if pane_idx == 0 {
                app.active_primary_tab_index()
            } else {
                app.pane_tabs.get(pane_idx).copied().unwrap_or(0)
            };
            let tab_name = app.tabs.get(tab_idx).map_or("No module", |t| t.name());
            let entry_text = format!("{label} — {tab_name}");
            if ui
                .selectable_label(pane_idx == active_pane, entry_text)
                .clicked()
            {
                crate::app::viewport::focus::focus_pane_interactive(app, ctx, pane_idx);
                ui.close();
                return;
            }
        }
    });
}

/// Renders the `Pane Assignments` submenu for multi-pane layouts.
pub(super) fn render_pane_assignments_submenu(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    ui: &mut egui::Ui,
) {
    ui.menu_button("Pane Assignments", |ui| {
        let pane_count = app.viewport_layout.pane_count();
        let pane_prefixes = app.viewport_layout.pane_label_prefixes();
        let mut pending_assignment = None;

        for pane_idx in 0..pane_count {
            let prefix = pane_prefixes.get(pane_idx).copied().unwrap_or("Pane:");
            let assigned = app.pane_tabs.get(pane_idx).copied().unwrap_or(0);
            let assigned_name = app.tabs.get(assigned).map_or("Unknown", |t| t.name());

            ui.push_id(("pane_assign", pane_idx), |ui| {
                ui.menu_button((prefix, assigned_name), |pane_ui| {
                    if let Some(tab_idx) = crate::app::domain_bar::render_domain_module_submenus(
                        pane_ui, &app.tabs, assigned,
                    ) {
                        pending_assignment = Some((pane_idx, tab_idx));
                    }
                });
            });
        }

        if let Some((pane_idx, tab_idx)) = pending_assignment {
            app.select_pane_tab(ctx, pane_idx, tab_idx);
            ui.close();
        }
    });
}

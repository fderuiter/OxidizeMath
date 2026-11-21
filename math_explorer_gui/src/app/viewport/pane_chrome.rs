#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{focus, CURRENT_PANE_RECT_KEY};
use crate::app::MathExplorerApp;
use eframe::egui;

pub(super) fn build_pane_region_ui(
    parent_ui: &mut egui::Ui,
    pane_idx: usize,
    pane_rect: egui::Rect,
    label: &str,
) -> egui::Ui {
    let mut pane_ui = parent_ui.new_child(
        egui::UiBuilder::new()
            .id_salt(("viewport_pane_region", pane_idx))
            .max_rect(pane_rect)
            .layout(egui::Layout::top_down(egui::Align::Min)),
    );
    pane_ui.expand_to_include_rect(pane_rect);
    pane_ui.set_clip_rect(pane_rect);
    let region_id = pane_ui.unique_id();
    pane_ui.ctx().accesskit_node_builder(region_id, |builder| {
        builder.set_role(egui::accesskit::Role::Region);
        builder.set_label(label);
        builder.set_bounds(egui::accesskit::Rect {
            x0: f64::from(pane_rect.min.x),
            y0: f64::from(pane_rect.min.y),
            x1: f64::from(pane_rect.max.x),
            y1: f64::from(pane_rect.max.y),
        });
    });
    pane_ui
}

fn render_pane_header_layout_actions(
    app: &mut MathExplorerApp,
    header_ui: &mut egui::Ui,
    pane_idx: usize,
    header_width: f32,
) {
    let pane_count = app.viewport_layout.pane_count();
    if pane_count == 2 && pane_idx == 0 {
        if header_ui.small_button("⇄ Swap").clicked() {
            app.swap_panes();
        }
        header_ui.separator();
        return;
    }
    if pane_count != 4 {
        return;
    }
    let labels = app.viewport_layout.pane_labels();
    header_ui.push_id(("pane_quad_ops", pane_idx), |ops_ui| {
        ops_ui.menu_button("⇄", |menu_ui| {
            for (other_idx, &other_lbl) in labels.iter().enumerate() {
                if other_idx != pane_idx
                    && menu_ui.button(format!("⇄ Swap with {other_lbl}")).clicked()
                {
                    app.swap_pane_pair(pane_idx, other_idx);
                    menu_ui.close();
                    return;
                }
            }
            menu_ui.separator();
            if menu_ui.button("↻ Rotate Panes Clockwise").clicked() {
                app.rotate_panes_clockwise();
                menu_ui.close();
            } else if menu_ui.button("↺ Rotate Panes Counter-Clockwise").clicked() {
                app.rotate_panes_counter_clockwise();
                menu_ui.close();
            }
        });
        if pane_idx == 0 && header_width >= 320.0 {
            if ops_ui
                .small_button("↻")
                .on_hover_text("Rotate panes clockwise")
                .clicked()
            {
                app.rotate_panes_clockwise();
            }
            if ops_ui
                .small_button("↺")
                .on_hover_text("Rotate panes counter-clockwise")
                .clicked()
            {
                app.rotate_panes_counter_clockwise();
            }
        }
    });
    header_ui.separator();
}

pub(super) fn render_pane_header(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    pane_ui: &mut egui::Ui,
    pane_idx: usize,
    header_rect: egui::Rect,
    label: &str,
    suppress_focus_hint: bool,
) {
    let mut header_ui = pane_ui.new_child(
        egui::UiBuilder::new()
            .id_salt(("pane_header", pane_idx))
            .max_rect(header_rect)
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
    );
    header_ui.expand_to_include_rect(header_rect);
    header_ui.set_clip_rect(header_rect);
    header_ui.painter().rect_filled(
        header_rect,
        0.0,
        header_ui.visuals().widgets.noninteractive.bg_fill,
    );

    let assigned_tab_idx = app.pane_tabs.get(pane_idx).copied().unwrap_or(0);
    let current_tab_name = app
        .tabs
        .get(assigned_tab_idx)
        .map_or("Select Tab", |t| t.name());

    header_ui.add_space(8.0);
    header_ui.strong(label);
    header_ui.separator();
    render_pane_header_layout_actions(app, &mut header_ui, pane_idx, header_rect.width());

    let mut tab_to_switch = None;
    header_ui.push_id(("pane_header_selector", pane_idx), |selector_ui| {
        let menu_resp = selector_ui.menu_button((current_tab_name, "▾"), |menu_ui| {
            tab_to_switch = crate::app::domain_bar::render_domain_module_submenus(
                menu_ui,
                &app.tabs,
                assigned_tab_idx,
            );
        });
        if suppress_focus_hint {
            menu_resp
                .response
                .on_hover_text(format!("Select simulation module for {label}"));
        } else {
            crate::app::domain_bar::attach_hover_hint(menu_resp.response, || {
                format!("Select simulation module for {label}")
            });
        }
    });

    if let Some(new_idx) = tab_to_switch {
        app.select_pane_tab(ctx, pane_idx, new_idx);
    }
}

pub(super) fn render_pane_content(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    frame: &mut eframe::Frame,
    pane_ui: &mut egui::Ui,
    pane_idx: usize,
    content_rect: egui::Rect,
) {
    let mut content_ui = pane_ui.new_child(
        egui::UiBuilder::new()
            .id_salt(("pane_content", pane_idx))
            .max_rect(content_rect)
            .layout(egui::Layout::top_down(egui::Align::Min)),
    );
    content_ui.expand_to_include_rect(content_rect);
    content_ui.set_clip_rect(content_rect);

    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::new(focus::CURRENT_RENDERING_PANE_KEY), pane_idx);
        d.insert_temp(egui::Id::new(CURRENT_PANE_RECT_KEY), content_rect);
    });
    crate::reflective_ui::slider::begin_pane_sweep_scope(ctx, pane_idx);
    egui_plot::clear_last_plot_datasets();
    focus::with_pane_scoped_input(ctx, pane_idx, || {
        if let Some(tab) = app.pane_tab_mut(pane_idx) {
            tab.show_inside(&mut content_ui, frame);
        } else {
            content_ui.centered_and_justified(|ui| {
                ui.label("No module selected.");
            });
        }
    });
    egui_plot::clear_last_plot_datasets();
    crate::reflective_ui::slider::merge_pane_sweep_flags(ctx, pane_idx);
    ctx.data_mut(|d| {
        d.remove::<usize>(egui::Id::new(focus::CURRENT_RENDERING_PANE_KEY));
        d.remove::<egui::Rect>(egui::Id::new(CURRENT_PANE_RECT_KEY));
    });
}

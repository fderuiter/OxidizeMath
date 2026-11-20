pub(crate) mod focus;
pub(crate) mod sash;
#[cfg(test)]
mod sash_tests;

use crate::app::MathExplorerApp;
use eframe::egui;
use serde::{Deserialize, Serialize};

/// Supported viewport layout presets.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewportLayout {
    /// Single viewport showing one active tab.
    #[default]
    Single,
    /// Side-by-side view with left and right panes.
    SplitHorizontal,
    /// Stacked view with top and bottom panes.
    SplitVertical,
    /// Four-quadrant grid view (2x2).
    QuadGrid,
}

impl ViewportLayout {
    /// Returns the user-facing preset name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Single => "Single",
            Self::SplitHorizontal => "Split Horizontal",
            Self::SplitVertical => "Split Vertical",
            Self::QuadGrid => "2x2 Quad Grid",
        }
    }

    /// Returns the number of active panes for this layout preset.
    pub fn pane_count(&self) -> usize {
        match self {
            Self::Single => 1,
            Self::SplitHorizontal | Self::SplitVertical => 2,
            Self::QuadGrid => 4,
        }
    }

    /// Returns descriptive labels for each pane in this layout.
    pub fn pane_labels(&self) -> &'static [&'static str] {
        match self {
            Self::Single => &["Main Viewport"],
            Self::SplitHorizontal => &["Left Pane", "Right Pane"],
            Self::SplitVertical => &["Top Pane", "Bottom Pane"],
            Self::QuadGrid => &[
                "Top-Left Pane",
                "Top-Right Pane",
                "Bottom-Left Pane",
                "Bottom-Right Pane",
            ],
        }
    }

    /// Returns descriptive pane label prefixes with trailing colons for zero-allocation menu buttons.
    #[cfg(any(not(target_os = "macos"), test))]
    pub fn pane_label_prefixes(&self) -> &'static [&'static str] {
        match self {
            Self::Single => &["Main Viewport:"],
            Self::SplitHorizontal => &["Left Pane:", "Right Pane:"],
            Self::SplitVertical => &["Top Pane:", "Bottom Pane:"],
            Self::QuadGrid => &[
                "Top-Left Pane:",
                "Top-Right Pane:",
                "Bottom-Left Pane:",
                "Bottom-Right Pane:",
            ],
        }
    }
}

/// Computes pane bounding rectangles within the available container area using default equal split.
pub fn calculate_pane_rects(layout: ViewportLayout, total_rect: egui::Rect) -> Vec<egui::Rect> {
    match layout {
        ViewportLayout::Single => vec![total_rect],
        ViewportLayout::SplitHorizontal => {
            let half_w = (total_rect.width() * 0.5).floor();
            let rect_left =
                egui::Rect::from_min_size(total_rect.min, egui::vec2(half_w, total_rect.height()));
            let rect_right = egui::Rect::from_min_max(
                egui::pos2(total_rect.min.x + half_w, total_rect.min.y),
                total_rect.max,
            );
            vec![rect_left, rect_right]
        }
        ViewportLayout::SplitVertical => {
            let half_h = (total_rect.height() * 0.5).floor();
            let rect_top =
                egui::Rect::from_min_size(total_rect.min, egui::vec2(total_rect.width(), half_h));
            let rect_bottom = egui::Rect::from_min_max(
                egui::pos2(total_rect.min.x, total_rect.min.y + half_h),
                total_rect.max,
            );
            vec![rect_top, rect_bottom]
        }
        ViewportLayout::QuadGrid => {
            let half_w = (total_rect.width() * 0.5).floor();
            let half_h = (total_rect.height() * 0.5).floor();
            let mid_x = total_rect.min.x + half_w;
            let mid_y = total_rect.min.y + half_h;

            let top_left = egui::Rect::from_min_max(total_rect.min, egui::pos2(mid_x, mid_y));
            let top_right = egui::Rect::from_min_max(
                egui::pos2(mid_x, total_rect.min.y),
                egui::pos2(total_rect.max.x, mid_y),
            );
            let bottom_left = egui::Rect::from_min_max(
                egui::pos2(total_rect.min.x, mid_y),
                egui::pos2(mid_x, total_rect.max.y),
            );
            let bottom_right = egui::Rect::from_min_max(egui::pos2(mid_x, mid_y), total_rect.max);

            vec![top_left, top_right, bottom_left, bottom_right]
        }
    }
}

/// Computes pane bounding rectangles and optional divider bar rectangle respecting split ratio limits.
pub fn calculate_pane_rects_with_ratio(
    layout: ViewportLayout,
    total_rect: egui::Rect,
    split_ratio: f32,
) -> (Vec<egui::Rect>, Option<egui::Rect>) {
    let clamped = split_ratio.clamp(0.20, 0.80);
    const THICKNESS: f32 = 6.0;

    match layout {
        ViewportLayout::Single => (vec![total_rect], None),
        ViewportLayout::SplitHorizontal => {
            let div_x = total_rect.min.x + total_rect.width() * clamped;
            let rect_left = egui::Rect::from_min_max(
                total_rect.min,
                egui::pos2(div_x - THICKNESS * 0.5, total_rect.max.y),
            );
            let divider = egui::Rect::from_min_max(
                egui::pos2(div_x - THICKNESS * 0.5, total_rect.min.y),
                egui::pos2(div_x + THICKNESS * 0.5, total_rect.max.y),
            );
            let rect_right = egui::Rect::from_min_max(
                egui::pos2(div_x + THICKNESS * 0.5, total_rect.min.y),
                total_rect.max,
            );
            (vec![rect_left, rect_right], Some(divider))
        }
        ViewportLayout::SplitVertical => {
            let div_y = total_rect.min.y + total_rect.height() * clamped;
            let rect_top = egui::Rect::from_min_max(
                total_rect.min,
                egui::pos2(total_rect.max.x, div_y - THICKNESS * 0.5),
            );
            let divider = egui::Rect::from_min_max(
                egui::pos2(total_rect.min.x, div_y - THICKNESS * 0.5),
                egui::pos2(total_rect.max.x, div_y + THICKNESS * 0.5),
            );
            let rect_bottom = egui::Rect::from_min_max(
                egui::pos2(total_rect.min.x, div_y + THICKNESS * 0.5),
                total_rect.max,
            );
            (vec![rect_top, rect_bottom], Some(divider))
        }
        ViewportLayout::QuadGrid => (
            calculate_pane_rects(ViewportLayout::QuadGrid, total_rect),
            None,
        ),
    }
}

fn build_pane_region_ui(
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

fn render_pane_header(
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
        .map(|t| t.name())
        .unwrap_or("Select Tab");

    header_ui.add_space(8.0);
    header_ui.strong(label);
    header_ui.separator();

    if app.viewport_layout.pane_count() == 2 && pane_idx == 0 {
        if header_ui.small_button("⇄ Swap").clicked() {
            app.swap_panes();
        }
        header_ui.separator();
    }

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
                .on_hover_text(format!("Select simulation module for {}", label));
        } else {
            crate::app::domain_bar::attach_hover_hint(menu_resp.response, || {
                format!("Select simulation module for {}", label)
            });
        }
    });

    if let Some(new_idx) = tab_to_switch {
        app.select_pane_tab(ctx, pane_idx, new_idx);
    }
}

fn render_pane_content(
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
    });
    egui_plot::clear_last_plot_datasets();
    if let Some(tab) = app.pane_tab_mut(pane_idx) {
        tab.show_inside(&mut content_ui, frame);
    } else {
        content_ui.centered_and_justified(|ui| {
            ui.label("No module selected.");
        });
    }
    egui_plot::clear_last_plot_datasets();
    ctx.data_mut(|d| {
        d.remove::<usize>(egui::Id::new(focus::CURRENT_RENDERING_PANE_KEY));
    });
}

/// Renders `ViewportLayout::Single` inside the root central panel with a `Main Viewport` `Role::Region`.
pub(super) fn render_single_pane_viewport(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    frame: &mut eframe::Frame,
) {
    egui::CentralPanel::default()
        .frame(egui::Frame::NONE)
        .show(ctx, |ui| {
            let total_rect = ui.available_rect_before_wrap();
            let pending_focus = ctx
                .data_mut(|d| d.remove_temp::<usize>(egui::Id::new(focus::PENDING_PANE_FOCUS_KEY)));
            let shell_focused = focus::sync_focused_pane_from_widget_focus(
                app,
                ctx,
                &[total_rect],
                false,
                pending_focus.is_some(),
            );
            ctx.data_mut(|d| {
                d.insert_temp(egui::Id::new(focus::ACTIVE_FOCUSED_PANE_KEY), 0usize);
                d.insert_temp(
                    egui::Id::new(focus::SHELL_MODAL_OR_SASH_FOCUSED_KEY),
                    shell_focused,
                );
            });

            let mut pane_ui = build_pane_region_ui(ui, 0, total_rect, "Main Viewport");
            let should_focus = pending_focus == Some(0);
            focus::begin_pending_pane_focus(ctx, should_focus);
            render_pane_content(app, ctx, frame, &mut pane_ui, 0, total_rect);
            focus::finish_pending_pane_focus(ctx, should_focus);
        });
}

fn render_multi_pane_regions(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    frame: &mut eframe::Frame,
    ui: &mut egui::Ui,
    pane_rects: &[egui::Rect],
    pending_focus: Option<usize>,
) {
    let pane_count = app.viewport_layout.pane_count();
    let pane_labels = app.viewport_layout.pane_labels();
    const HEADER_HEIGHT: f32 = 28.0;
    for (pane_idx, &pane_rect) in pane_rects.iter().take(pane_count).enumerate() {
        let label = pane_labels.get(pane_idx).copied().unwrap_or("Pane");
        let header_rect =
            egui::Rect::from_min_size(pane_rect.min, egui::vec2(pane_rect.width(), HEADER_HEIGHT));
        let content_rect = egui::Rect::from_min_max(
            egui::pos2(pane_rect.min.x, pane_rect.min.y + HEADER_HEIGHT),
            pane_rect.max,
        );

        let mut pane_ui = build_pane_region_ui(ui, pane_idx, pane_rect, label);
        let should_focus = pending_focus == Some(pane_idx);
        focus::begin_pending_pane_focus(ctx, should_focus);
        render_pane_header(
            app,
            ctx,
            &mut pane_ui,
            pane_idx,
            header_rect,
            label,
            should_focus,
        );
        render_pane_content(app, ctx, frame, &mut pane_ui, pane_idx, content_rect);
        focus::finish_pending_pane_focus(ctx, should_focus);
    }
}

/// Renders the multi-pane viewport manager layout in the central panel on the root `egui::Context`.
pub fn render_multi_pane_manager(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    frame: &mut eframe::Frame,
) {
    let pane_count = app.viewport_layout.pane_count();

    egui::CentralPanel::default().show(ctx, |ui| {
        let total_rect = ui.available_rect_before_wrap();
        if total_rect.width() < 20.0 || total_rect.height() < 20.0 {
            return;
        }

        let (mut pane_rects, mut maybe_divider) =
            calculate_pane_rects_with_ratio(app.viewport_layout, total_rect, app.split_ratio);

        let sash_state = maybe_divider.map(|divider_rect| {
            let prev_ratio = app.split_ratio;
            let state = sash::interact_split_sash(app, ctx, ui, total_rect, divider_rect);
            if (app.split_ratio - prev_ratio).abs() > 1e-6 {
                let updated = calculate_pane_rects_with_ratio(
                    app.viewport_layout,
                    total_rect,
                    app.split_ratio,
                );
                pane_rects = updated.0;
                maybe_divider = updated.1;
            }
            state
        });

        let sash_focused = sash_state.as_ref().is_some_and(|s| s.has_focus);
        let pending_focus = focus::resolve_frame_pane_focus(app, ctx, &pane_rects, sash_focused);

        render_multi_pane_regions(app, ctx, frame, ui, &pane_rects, pending_focus);

        let _ = focus::sync_focused_pane_from_widget_focus(
            app,
            ctx,
            &pane_rects,
            sash_focused,
            pending_focus.is_some(),
        );
        let final_focused_idx = app.active_focused_pane();
        for (pane_idx, &pane_rect) in pane_rects.iter().take(pane_count).enumerate() {
            focus::paint_pane_border(
                ui.painter(),
                pane_rect,
                pane_idx == final_focused_idx,
                ui.visuals(),
            );
        }

        if let (Some(divider_rect), Some(state)) = (maybe_divider, sash_state) {
            sash::paint_split_sash(
                ui,
                total_rect,
                divider_rect,
                app.viewport_layout,
                app.is_dragging_divider,
                &state,
            );
        }
    });
}

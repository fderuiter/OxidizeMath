mod pane_sync;

use crate::app::MathExplorerApp;
use eframe::egui;
use pane_sync::{merge_pane_commands, remap_shape_textures, sync_pane_textures};
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

/// Filters raw input events to isolate interaction within a specific pane rectangle.
pub fn filter_raw_input_for_pane(
    raw_input: &egui::RawInput,
    pane_rect: egui::Rect,
    pointer_pos: Option<egui::Pos2>,
) -> egui::RawInput {
    let mut pane_input = raw_input.clone();
    pane_input.screen_rect = Some(pane_rect);

    let is_pointer_inside = pointer_pos.is_some_and(|pos| pane_rect.contains(pos));

    pane_input.events.retain(|event| match event {
        egui::Event::PointerMoved(pos) => pane_rect.contains(*pos),
        egui::Event::PointerButton { pos, .. } => pane_rect.contains(*pos),
        egui::Event::PointerGone => true,
        egui::Event::MouseWheel { .. } => is_pointer_inside,
        egui::Event::Key { .. } | egui::Event::Text(_) => is_pointer_inside,
        _ => true,
    });

    pane_input
}

fn render_pane_header(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    ui: &mut egui::Ui,
    pane_idx: usize,
    header_rect: egui::Rect,
    label: &str,
) {
    let mut header_ui = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(header_rect)
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
    );
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
        crate::app::domain_bar::attach_hover_hint(menu_resp.response, || {
            format!("Select simulation module for {}", label)
        });
    });

    if let Some(new_idx) = tab_to_switch {
        app.select_pane_tab(ctx, pane_idx, new_idx);
    }
}

fn render_pane_content(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    frame: &mut eframe::Frame,
    ui: &mut egui::Ui,
    pane_idx: usize,
    content_rect: egui::Rect,
    pane_input: egui::RawInput,
) {
    let current_assigned = app.pane_tabs.get(pane_idx).copied().unwrap_or(0);
    if let Some(tab) = app.tabs.get_mut(current_assigned) {
        let pane_ctx = &app.pane_contexts[pane_idx];
        pane_ctx.set_style((*ctx.style()).clone());
        pane_ctx.data_mut(|d| {
            d.insert_temp(
                egui::Id::new("CMD_REGISTRY"),
                egui_plot::commands::CommandRegistryData::default(),
            );
        });
        let output = pane_ctx.run(pane_input, |sub_ctx| {
            tab.show(sub_ctx, frame);
        });
        merge_pane_commands(ctx, pane_ctx);

        sync_pane_textures(
            ctx,
            pane_idx,
            &mut app.pane_textures[pane_idx],
            output.textures_delta,
        );

        let should_repaint = output
            .viewport_output
            .values()
            .any(|v| v.repaint_delay < std::time::Duration::from_millis(500))
            || pane_ctx.has_requested_repaint();
        if should_repaint {
            ctx.request_repaint();
        }

        let tex_map = &app.pane_textures[pane_idx];
        for mut clipped_shape in output.shapes {
            if !tex_map.is_empty() {
                remap_shape_textures(&mut clipped_shape.shape, tex_map);
            }
            let clip = clipped_shape.clip_rect.intersect(content_rect);
            ui.painter().with_clip_rect(clip).add(clipped_shape.shape);
        }
    }
}

/// Renders the multi-pane viewport manager layout in the central panel.
#[allow(clippy::too_many_lines)]
pub fn render_multi_pane_manager(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    frame: &mut eframe::Frame,
) {
    let pane_count = app.viewport_layout.pane_count();
    let pane_labels = app.viewport_layout.pane_labels();

    egui::CentralPanel::default().show(ctx, |ui| {
        let total_rect = ui.available_rect_before_wrap();
        if total_rect.width() < 20.0 || total_rect.height() < 20.0 {
            return;
        }

        let (pane_rects, maybe_divider) =
            calculate_pane_rects_with_ratio(app.viewport_layout, total_rect, app.split_ratio);

        if let Some(divider_rect) = maybe_divider {
            let hover_pos = ctx.input(|i| i.pointer.hover_pos());
            let is_hovered = hover_pos.is_some_and(|pos| divider_rect.contains(pos));
            let is_down = ctx.input(|i| i.pointer.primary_down());

            if is_hovered && ctx.input(|i| i.pointer.any_pressed()) {
                app.is_dragging_divider = true;
            }
            if !is_down {
                app.is_dragging_divider = false;
            }
            app.split_view_state.is_dragging_divider = app.is_dragging_divider;

            if app.is_dragging_divider {
                let delta = ctx.input(|i| i.pointer.delta());
                let ratio_delta = match app.viewport_layout {
                    ViewportLayout::SplitHorizontal if total_rect.width() > 0.0 => {
                        delta.x / total_rect.width()
                    }
                    ViewportLayout::SplitVertical if total_rect.height() > 0.0 => {
                        delta.y / total_rect.height()
                    }
                    _ => 0.0,
                };
                app.split_ratio = (app.split_ratio + ratio_delta).clamp(0.20, 0.80);
            }

            if is_hovered || app.is_dragging_divider {
                ctx.set_cursor_icon(match app.viewport_layout {
                    ViewportLayout::SplitHorizontal => egui::CursorIcon::ResizeColumn,
                    ViewportLayout::SplitVertical => egui::CursorIcon::ResizeRow,
                    _ => egui::CursorIcon::Default,
                });
            }

            let divider_color = if app.is_dragging_divider {
                egui::Color32::from_rgb(90, 150, 255)
            } else if is_hovered {
                egui::Color32::from_rgb(140, 140, 150)
            } else {
                egui::Color32::from_gray(50)
            };
            ui.painter().rect_filled(divider_rect, 0.0, divider_color);

            let center = divider_rect.center();
            let handle_color = if app.is_dragging_divider || is_hovered {
                egui::Color32::WHITE
            } else {
                egui::Color32::from_gray(130)
            };
            let (p1, p2) = if app.viewport_layout == ViewportLayout::SplitHorizontal {
                (
                    center - egui::vec2(0.0, 16.0),
                    center + egui::vec2(0.0, 16.0),
                )
            } else {
                (
                    center - egui::vec2(16.0, 0.0),
                    center + egui::vec2(16.0, 0.0),
                )
            };
            ui.painter()
                .line_segment([p1, p2], egui::Stroke::new(2.0_f32, handle_color));
        }

        let raw_input = ctx.input(|i| i.raw.clone());
        let pointer_pos = if app.is_dragging_divider {
            None
        } else {
            ctx.input(|i| i.pointer.latest_pos())
        };

        const HEADER_HEIGHT: f32 = 28.0;

        for pane_idx in 0..pane_count {
            if pane_idx >= pane_rects.len() {
                break;
            }
            let pane_rect = pane_rects[pane_idx];
            let label = pane_labels.get(pane_idx).copied().unwrap_or("Pane");

            let header_rect = egui::Rect::from_min_size(
                pane_rect.min,
                egui::vec2(pane_rect.width(), HEADER_HEIGHT),
            );
            let content_rect = egui::Rect::from_min_max(
                egui::pos2(pane_rect.min.x, pane_rect.min.y + HEADER_HEIGHT),
                pane_rect.max,
            );

            ui.painter().rect_stroke(
                pane_rect,
                0.0,
                egui::Stroke::new(1.0_f32, ui.visuals().widgets.noninteractive.bg_stroke.color),
                egui::StrokeKind::Inside,
            );

            render_pane_header(app, ctx, ui, pane_idx, header_rect, label);

            let pane_input = filter_raw_input_for_pane(&raw_input, content_rect, pointer_pos);
            render_pane_content(app, ctx, frame, ui, pane_idx, content_rect, pane_input);
        }
    });
}

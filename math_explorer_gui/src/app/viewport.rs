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
}

/// Computes pane bounding rectangles within the available container area.
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

    let mut tab_to_switch = None;
    egui::ComboBox::from_id_salt(format!("header_combo_{}", pane_idx))
        .selected_text(current_tab_name)
        .show_ui(&mut header_ui, |cb_ui| {
            let tab_names = app.tab_names();
            for (i, name) in tab_names.into_iter().enumerate() {
                if cb_ui
                    .selectable_label(assigned_tab_idx == i, name)
                    .clicked()
                {
                    tab_to_switch = Some(i);
                }
            }
        });

    if let Some(new_idx) = tab_to_switch {
        app.set_pane_tab(pane_idx, new_idx);
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
        let output = pane_ctx.run(pane_input, |sub_ctx| {
            tab.show(sub_ctx, frame);
        });

        let should_repaint = output
            .viewport_output
            .values()
            .any(|v| v.repaint_delay < std::time::Duration::from_millis(500));
        if should_repaint {
            ctx.request_repaint();
        }

        for clipped_shape in output.shapes {
            let clip = clipped_shape.clip_rect.intersect(content_rect);
            ui.painter().with_clip_rect(clip).add(clipped_shape.shape);
        }
    }
}

/// Renders the multi-pane viewport manager layout in the central panel.
pub fn render_multi_pane_manager(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    frame: &mut eframe::Frame,
) {
    let pane_count = app.viewport_layout.pane_count();
    let pane_labels = app.viewport_layout.pane_labels();

    egui::CentralPanel::default().show(ctx, |ui| {
        let total_rect = ui.available_rect_before_wrap();
        let pane_rects = calculate_pane_rects(app.viewport_layout, total_rect);
        let raw_input = ctx.input(|i| i.raw.clone());
        let pointer_pos = ctx.input(|i| i.pointer.latest_pos());

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

            render_pane_header(app, ui, pane_idx, header_rect, label);

            let pane_input = filter_raw_input_for_pane(&raw_input, content_rect, pointer_pos);
            render_pane_content(app, ctx, frame, ui, pane_idx, content_rect, pane_input);
        }
    });
}

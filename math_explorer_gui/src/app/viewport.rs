pub(crate) mod compare_sync;
pub(crate) mod focus;
mod pane_chrome;
pub(crate) mod quad_sash;
pub(crate) mod sash;
#[cfg(test)]
mod sash_tests;

use crate::app::MathExplorerApp;
use eframe::egui;
use pane_chrome::{build_pane_region_ui, render_pane_content, render_pane_header};
use serde::{Deserialize, Serialize};

/// Temp context key storing the `egui::Rect` of the currently rendering viewport pane.
pub(crate) const CURRENT_PANE_RECT_KEY: &str = "CURRENT_PANE_RECT";

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
#[cfg(all(test, feature = "epidemiology", feature = "pure_math"))]
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

            vec![
                egui::Rect::from_min_max(total_rect.min, egui::pos2(mid_x, mid_y)),
                egui::Rect::from_min_max(
                    egui::pos2(mid_x, total_rect.min.y),
                    egui::pos2(total_rect.max.x, mid_y),
                ),
                egui::Rect::from_min_max(
                    egui::pos2(total_rect.min.x, mid_y),
                    egui::pos2(mid_x, total_rect.max.y),
                ),
                egui::Rect::from_min_max(egui::pos2(mid_x, mid_y), total_rect.max),
            ]
        }
    }
}

/// Computes the four `QuadGrid` pane rectangles and `(vertical_divider, horizontal_divider)`.
pub fn calculate_quad_pane_rects_with_ratios(
    total_rect: egui::Rect,
    quad_ratios: (f32, f32),
) -> (Vec<egui::Rect>, egui::Rect, egui::Rect) {
    let rx = quad_ratios.0.clamp(0.20, 0.80);
    let ry = quad_ratios.1.clamp(0.20, 0.80);
    const HALF_T: f32 = 3.0;
    let div_x = total_rect.min.x + total_rect.width() * rx;
    let div_y = total_rect.min.y + total_rect.height() * ry;

    let top_left =
        egui::Rect::from_min_max(total_rect.min, egui::pos2(div_x - HALF_T, div_y - HALF_T));
    let top_right = egui::Rect::from_min_max(
        egui::pos2(div_x + HALF_T, total_rect.min.y),
        egui::pos2(total_rect.max.x, div_y - HALF_T),
    );
    let bottom_left = egui::Rect::from_min_max(
        egui::pos2(total_rect.min.x, div_y + HALF_T),
        egui::pos2(div_x - HALF_T, total_rect.max.y),
    );
    let bottom_right =
        egui::Rect::from_min_max(egui::pos2(div_x + HALF_T, div_y + HALF_T), total_rect.max);
    let vert_div = egui::Rect::from_min_max(
        egui::pos2(div_x - HALF_T, total_rect.min.y),
        egui::pos2(div_x + HALF_T, total_rect.max.y),
    );
    let horiz_div = egui::Rect::from_min_max(
        egui::pos2(total_rect.min.x, div_y - HALF_T),
        egui::pos2(total_rect.max.x, div_y + HALF_T),
    );
    (
        vec![top_left, top_right, bottom_left, bottom_right],
        vert_div,
        horiz_div,
    )
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
            calculate_quad_pane_rects_with_ratios(total_rect, (clamped, clamped)).0,
            None,
        ),
    }
}

/// Renders `ViewportLayout::Single` inside the root central panel with a `Main Viewport` `Role::Region`.
pub(super) fn render_single_pane_viewport(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    frame: &mut eframe::Frame,
) {
    app.flush_stale_pane_focus(ctx);
    crate::starter_gallery::begin_root_starter_gallery_scope(ctx);
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
                d.remove::<Vec<egui::Rect>>(egui::Id::new("LAST_PANE_RECTS"));
                d.insert_temp(egui::Id::new(focus::ACTIVE_FOCUSED_PANE_KEY), 0usize);
                d.insert_temp(
                    egui::Id::new(focus::SHELL_MODAL_OR_SASH_FOCUSED_KEY),
                    shell_focused,
                );
            });
            compare_sync::clear_single_pane_sync_context(ctx);

            let mut pane_ui = build_pane_region_ui(ui, 0, total_rect, "Main Viewport");
            let should_focus = pending_focus == Some(0);
            focus::begin_pending_pane_focus(ctx, should_focus);
            render_pane_content(app, ctx, frame, &mut pane_ui, 0, total_rect);
            focus::finish_pending_pane_focus(ctx, should_focus);
            focus::record_resolved_pane_focus(ctx, 0);
        });
    crate::starter_gallery::flush_deferred_starter_galleries(app, ctx);
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

struct FrameSashOutcome {
    pane_rects: Vec<egui::Rect>,
    dual_sash: Option<(egui::Rect, sash::SashFrameState)>,
    quad_sashes: Option<(egui::Rect, egui::Rect, quad_sash::QuadSashFrameState)>,
    sash_focused: bool,
}

fn resolve_sashes_and_rects(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    ui: &mut egui::Ui,
    total_rect: egui::Rect,
) -> FrameSashOutcome {
    if app.viewport_layout == ViewportLayout::QuadGrid {
        let prev = app.quad_split_ratios;
        let (mut rects, mut vd, mut hd) = calculate_quad_pane_rects_with_ratios(total_rect, prev);
        let qs = quad_sash::interact_quad_sashes(app, ctx, ui, total_rect, vd, hd);
        if (app.quad_split_ratios.0 - prev.0).abs() > 1e-6
            || (app.quad_split_ratios.1 - prev.1).abs() > 1e-6
        {
            (rects, vd, hd) =
                calculate_quad_pane_rects_with_ratios(total_rect, app.quad_split_ratios);
        }
        return FrameSashOutcome {
            pane_rects: rects,
            dual_sash: None,
            sash_focused: qs.any_focused(),
            quad_sashes: Some((vd, hd, qs)),
        };
    }

    let (mut rects, mut md) =
        calculate_pane_rects_with_ratio(app.viewport_layout, total_rect, app.split_ratio);
    let dual = md.and_then(|div| {
        let prev = app.split_ratio;
        let st = sash::interact_split_sash(app, ctx, ui, total_rect, div);
        if (app.split_ratio - prev).abs() > 1e-6 {
            (rects, md) =
                calculate_pane_rects_with_ratio(app.viewport_layout, total_rect, app.split_ratio);
        }
        md.map(|d| (d, st))
    });
    let sash_focused = dual.as_ref().is_some_and(|(_, s)| s.has_focus);
    FrameSashOutcome {
        pane_rects: rects,
        dual_sash: dual,
        quad_sashes: None,
        sash_focused,
    }
}

/// Renders the multi-pane viewport manager layout in the central panel on the root `egui::Context`.
pub fn render_multi_pane_manager(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    frame: &mut eframe::Frame,
) {
    app.flush_stale_pane_focus(ctx);
    crate::starter_gallery::begin_root_starter_gallery_scope(ctx);
    let pane_count = app.viewport_layout.pane_count();

    egui::CentralPanel::default().show(ctx, |ui| {
        let total_rect = ui.available_rect_before_wrap();
        if total_rect.width() < 20.0 || total_rect.height() < 20.0 {
            return;
        }

        let outcome = resolve_sashes_and_rects(app, ctx, ui, total_rect);
        ctx.data_mut(|d| {
            d.insert_temp(egui::Id::new("LAST_PANE_RECTS"), outcome.pane_rects.clone())
        });
        let pending_focus =
            focus::resolve_frame_pane_focus(app, ctx, &outcome.pane_rects, outcome.sash_focused);
        compare_sync::publish_multi_pane_sync_context(app, ctx);

        render_multi_pane_regions(app, ctx, frame, ui, &outcome.pane_rects, pending_focus);

        let _ = focus::sync_focused_pane_from_widget_focus(
            app,
            ctx,
            &outcome.pane_rects,
            outcome.sash_focused,
            pending_focus.is_some(),
        );
        let final_focused_idx = app.active_focused_pane();
        focus::record_resolved_pane_focus(ctx, final_focused_idx);
        for (pane_idx, &pane_rect) in outcome.pane_rects.iter().take(pane_count).enumerate() {
            focus::paint_pane_border(
                ui.painter(),
                pane_rect,
                pane_idx == final_focused_idx,
                ui.visuals(),
            );
        }

        if let Some((div, st)) = outcome.dual_sash {
            sash::paint_split_sash(
                ui,
                total_rect,
                div,
                app.viewport_layout,
                app.is_dragging_divider,
                &st,
            );
        }
        if let Some((vd, hd, qs)) = outcome.quad_sashes {
            quad_sash::paint_quad_sashes(ui, total_rect, vd, hd, app.dragging_quad_axes, &qs);
        }
    });
    crate::starter_gallery::flush_deferred_starter_galleries(app, ctx);
}

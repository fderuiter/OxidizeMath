#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::sash::{
    apply_accesskit_actions, apply_keyboard_adjustments, focus_outline_color,
    quad_horizontal_sash_id, quad_vertical_sash_id, register_axis_splitter_accessibility,
    SashFrameState, MAX_SPLIT_RATIO, MIN_SPLIT_RATIO,
};
use crate::app::MathExplorerApp;
use eframe::egui;

/// Combined per-frame interaction outcome for the 2D `QuadGrid` vertical and horizontal sashes.
#[derive(Debug, Clone, Copy)]
pub(super) struct QuadSashFrameState {
    pub vertical: SashFrameState,
    pub horizontal: SashFrameState,
}

impl QuadSashFrameState {
    pub fn any_focused(&self) -> bool {
        self.vertical.has_focus || self.horizontal.has_focus
    }
}

fn announce_quad_axis_ratio(ctx: &egui::Context, axis_label: &str, ratio: f32) {
    let clamped = ratio.clamp(MIN_SPLIT_RATIO, MAX_SPLIT_RATIO);
    let pct = (clamped * 100.0).round() as i32;
    crate::accessibility::announce_live_status(ctx, format!("{axis_label} {pct}%"));
}

struct QuadAxisSpec {
    label: &'static str,
    announce_prefix: &'static str,
    orientation: egui::accesskit::Orientation,
    modal_open: bool,
    drag_changed: bool,
}

fn begin_or_end_quad_drag(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    vert_resp: &egui::Response,
    horiz_resp: &egui::Response,
    vert_rect: egui::Rect,
    horiz_rect: egui::Rect,
    modal_open: bool,
) -> ((bool, bool), (bool, bool)) {
    let hover_pos = (!modal_open)
        .then(|| ctx.input(|i| i.pointer.hover_pos()))
        .flatten();
    let intersection = vert_rect.intersect(horiz_rect).expand(4.0);
    let in_intersection = hover_pos.is_some_and(|p| intersection.contains(p));
    let vert_hovered = !modal_open
        && (in_intersection
            || vert_resp.hovered()
            || hover_pos.is_some_and(|p| vert_rect.contains(p)));
    let horiz_hovered = !modal_open
        && (in_intersection
            || horiz_resp.hovered()
            || hover_pos.is_some_and(|p| horiz_rect.contains(p)));

    let was_dragging = app.dragging_quad_axes;
    let start_key = vert_resp.id.with("quad_drag_start");
    if (vert_hovered || horiz_hovered) && ctx.input(|i| i.pointer.any_pressed()) {
        if app.dragging_quad_axes == (false, false) {
            ctx.data_mut(|d| d.insert_temp(start_key, app.quad_split_ratios));
        }
        app.dragging_quad_axes = if in_intersection || (vert_hovered && horiz_hovered) {
            vert_resp.request_focus();
            (true, true)
        } else if vert_hovered {
            vert_resp.request_focus();
            (true, false)
        } else {
            horiz_resp.request_focus();
            (false, true)
        };
    }
    if !ctx.input(|i| i.pointer.primary_down()) {
        app.dragging_quad_axes = (false, false);
    }
    ((vert_hovered, horiz_hovered), was_dragging)
}

fn apply_quad_drag_delta_and_cursor(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    start_key: egui::Id,
    total_rect: egui::Rect,
    hovered: (bool, bool),
    was_dragging: (bool, bool),
) -> (bool, bool) {
    let mut changed = (false, false);
    let delta = ctx.input(|i| i.pointer.delta());
    if app.dragging_quad_axes.0 && total_rect.width() > 0.0 {
        let prev = app.quad_split_ratios.0;
        app.quad_split_ratios.0 =
            (prev + delta.x / total_rect.width()).clamp(MIN_SPLIT_RATIO, MAX_SPLIT_RATIO);
        changed.0 = (app.quad_split_ratios.0 - prev).abs() > 1e-6;
    }
    if app.dragging_quad_axes.1 && total_rect.height() > 0.0 {
        let prev = app.quad_split_ratios.1;
        app.quad_split_ratios.1 =
            (prev + delta.y / total_rect.height()).clamp(MIN_SPLIT_RATIO, MAX_SPLIT_RATIO);
        changed.1 = (app.quad_split_ratios.1 - prev).abs() > 1e-6;
    }

    if app.dragging_quad_axes == (false, false) && was_dragging != (false, false) {
        let start = ctx.data_mut(|d| {
            let val = d.get_temp::<(f32, f32)>(start_key);
            d.remove::<(f32, f32)>(start_key);
            val
        });
        if let Some((sx, sy)) = start {
            if was_dragging.0 && (app.quad_split_ratios.0 - sx).abs() > 1e-4 {
                announce_quad_axis_ratio(ctx, "Quad vertical split ratio", app.quad_split_ratios.0);
            }
            if was_dragging.1 && (app.quad_split_ratios.1 - sy).abs() > 1e-4 {
                announce_quad_axis_ratio(
                    ctx,
                    "Quad horizontal split ratio",
                    app.quad_split_ratios.1,
                );
            }
        }
    }

    let active = app.dragging_quad_axes;
    if active == (true, true) || (hovered.0 && hovered.1) {
        ctx.set_cursor_icon(egui::CursorIcon::Move);
    } else if active.0 || hovered.0 {
        ctx.set_cursor_icon(egui::CursorIcon::ResizeColumn);
    } else if active.1 || hovered.1 {
        ctx.set_cursor_icon(egui::CursorIcon::ResizeRow);
    }
    changed
}

fn process_axis_keyboard_and_a11y(
    ctx: &egui::Context,
    ui: &egui::Ui,
    sash_resp: &mut egui::Response,
    ratio: &mut f32,
    spec: QuadAxisSpec,
) {
    let mut changed = spec.drag_changed;
    let sash_id = sash_resp.id;
    if sash_resp.has_focus() {
        ctx.memory_mut(|m| {
            m.set_focus_lock_filter(
                sash_id,
                egui::EventFilter {
                    tab: false,
                    horizontal_arrows: true,
                    vertical_arrows: true,
                    escape: false,
                },
            );
        });
        if !spec.modal_open {
            let (kb_changed, _consumed, arrow_pressed) = apply_keyboard_adjustments(ctx, ratio);
            if arrow_pressed {
                ctx.memory_mut(|m| m.move_focus(egui::FocusDirection::None));
            }
            if kb_changed {
                changed = true;
                announce_quad_axis_ratio(ctx, spec.announce_prefix, *ratio);
            }
        }
    }

    if apply_accesskit_actions(ctx, sash_id, ratio) {
        changed = true;
        announce_quad_axis_ratio(ctx, spec.announce_prefix, *ratio);
    }

    if changed {
        sash_resp.mark_changed();
        ctx.request_repaint();
    }

    register_axis_splitter_accessibility(ui, sash_resp, spec.label, *ratio, spec.orientation);
}

/// Allocates and processes pointer, keyboard, and AccessKit interactions for both `QuadGrid` sashes.
pub(super) fn interact_quad_sashes(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    ui: &mut egui::Ui,
    total_rect: egui::Rect,
    vert_rect: egui::Rect,
    horiz_rect: egui::Rect,
) -> QuadSashFrameState {
    let sense = egui::Sense::click_and_drag();
    let mut vert_resp = ui.interact(vert_rect, quad_vertical_sash_id(), sense);
    let mut horiz_resp = ui.interact(horiz_rect, quad_horizontal_sash_id(), sense);
    let modal_open = app.command_palette.is_open
        || app.show_help_menu
        || egui::Popup::is_any_open(ctx)
        || ctx.memory(|m| m.top_modal_layer().is_some());

    let (hovered, was_dragging) = begin_or_end_quad_drag(
        app,
        ctx,
        &vert_resp,
        &horiz_resp,
        vert_rect,
        horiz_rect,
        modal_open,
    );
    let start_key = vert_resp.id.with("quad_drag_start");
    let (drag_x, drag_y) =
        apply_quad_drag_delta_and_cursor(app, ctx, start_key, total_rect, hovered, was_dragging);

    let v_spec = QuadAxisSpec {
        label: "Quad grid vertical split divider",
        announce_prefix: "Quad vertical split ratio",
        orientation: egui::accesskit::Orientation::Vertical,
        modal_open,
        drag_changed: drag_x,
    };
    process_axis_keyboard_and_a11y(
        ctx,
        ui,
        &mut vert_resp,
        &mut app.quad_split_ratios.0,
        v_spec,
    );

    let h_spec = QuadAxisSpec {
        label: "Quad grid horizontal split divider",
        announce_prefix: "Quad horizontal split ratio",
        orientation: egui::accesskit::Orientation::Horizontal,
        modal_open,
        drag_changed: drag_y,
    };
    process_axis_keyboard_and_a11y(
        ctx,
        ui,
        &mut horiz_resp,
        &mut app.quad_split_ratios.1,
        h_spec,
    );

    QuadSashFrameState {
        vertical: SashFrameState {
            is_hovered: hovered.0,
            has_focus: vert_resp.has_focus(),
        },
        horizontal: SashFrameState {
            is_hovered: hovered.1,
            has_focus: horiz_resp.has_focus(),
        },
    }
}

fn paint_single_quad_bar(
    ui: &mut egui::Ui,
    total_rect: egui::Rect,
    bar_rect: egui::Rect,
    is_vertical_bar: bool,
    is_dragging: bool,
    state: &SashFrameState,
) {
    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    let fill = if is_dragging {
        palette.accent
    } else if state.is_hovered || state.has_focus {
        palette.graph_edge
    } else {
        palette.surface_fill
    };
    ui.painter().rect_filled(bar_rect, 0.0, fill);

    let center = bar_rect.center();
    let handle_color = if is_dragging || state.is_hovered || state.has_focus {
        crate::app::contrasting_text_color(fill)
    } else {
        palette.muted_text
    };
    let (p1, p2) = if is_vertical_bar {
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

    if state.has_focus {
        let focus_rect = bar_rect.expand(1.5).intersect(total_rect);
        ui.painter().rect_stroke(
            focus_rect,
            1.0,
            egui::Stroke::new(2.0_f32, focus_outline_color(ui.visuals())),
            egui::StrokeKind::Inside,
        );
    }
}

/// Paints both `QuadGrid` split sashes, center grip handles, and high-contrast focus outlines.
pub(super) fn paint_quad_sashes(
    ui: &mut egui::Ui,
    total_rect: egui::Rect,
    vert_rect: egui::Rect,
    horiz_rect: egui::Rect,
    dragging_axes: (bool, bool),
    state: &QuadSashFrameState,
) {
    paint_single_quad_bar(
        ui,
        total_rect,
        vert_rect,
        true,
        dragging_axes.0,
        &state.vertical,
    );
    paint_single_quad_bar(
        ui,
        total_rect,
        horiz_rect,
        false,
        dragging_axes.1,
        &state.horizontal,
    );
    let hub = vert_rect.intersect(horiz_rect);
    if hub.is_positive() {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let hub_color = if dragging_axes != (false, false) {
            palette.accent
        } else {
            palette.graph_edge
        };
        ui.painter().circle_filled(hub.center(), 3.0, hub_color);
    }
}

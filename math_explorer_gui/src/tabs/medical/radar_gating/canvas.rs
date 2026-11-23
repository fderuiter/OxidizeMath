//! Interactive AccessKit-enabled 2D clinical room canvas for ULA steering and scatterer dragging.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::spectra::SpectralSnapshot;
use super::state::{RadarGatingState, ScattererPoint};
use crate::accessibility::{
    lock_canvas_arrow_keys, paint_canvas_focus_ring, register_canvas_accessibility,
};
use crate::app::theme::SemanticPalette;
use eframe::egui::{
    self, accesskit::Role, Key, Pos2, Rect, Sense, Stroke, Vec2,
};

const ROOM_HALF_WIDTH_M: f32 = 1.25;
const ROOM_MIN_DEPTH_M: f32 = 0.0;
const ROOM_MAX_DEPTH_M: f32 = 1.80;

/// Outcome of rendering and interacting with the 2D clinical radar room canvas.
pub struct CanvasInteraction {
    /// Canvas widget ID when it currently holds keyboard focus.
    pub focused_id: Option<egui::Id>,
    /// True if scatterer positions or steering angle changed this frame.
    pub state_changed: bool,
}

/// Renders the focusable 2D clinical radar room, ULA steering sector, and draggable scatterers.
pub fn show_room_canvas(
    ui: &mut egui::Ui,
    state: &mut RadarGatingState,
    spectra: &SpectralSnapshot,
    selected_target: &mut usize,
    dragging_target: &mut Option<usize>,
) -> CanvasInteraction {
    let desired = Vec2::new(ui.available_width().max(280.0), 235.0);
    let (response, painter) = ui.allocate_painter(
        desired,
        Sense::click_and_drag() | Sense::focusable_noninteractive(),
    );
    let rect = response.rect;

    let mut changed = handle_pointer_interaction(
        &response,
        rect,
        state,
        selected_target,
        dragging_target,
    );
    if response.has_focus() {
        lock_canvas_arrow_keys(ui, &response);
        if handle_keyboard_navigation(ui, state, selected_target) {
            changed = true;
        }
    }

    let palette = SemanticPalette::from_visuals(ui.visuals());
    let bg_fill = ui.visuals().extreme_bg_color;
    let text_fg = ui.visuals().text_color();
    paint_room_backdrop(&painter, rect, spectra.rayleigh_limit_m, bg_fill, &palette);
    paint_steering_cone(
        &painter,
        rect,
        state.steering_deg,
        spectra.angle_resolution_deg,
        &palette,
    );
    paint_scatterer_markers(&painter, rect, state, *selected_target, &palette);
    paint_canvas_legend(&painter, rect, state, spectra, text_fg, &palette);

    let sep_cm = state.target_separation_m() * 100.0;
    let limit_cm = spectra.rayleigh_limit_m * 100.0;
    let a11y_label = format!(
        "60 GHz MIMO Radar Room Visualization: primary chest target at {:.3} m, secondary scatterer at {:.3} m, separation {:.2} cm vs Rayleigh limit {:.2} cm, steering {:.1} deg. Arrow keys nudge active target, 1 and 2 switch targets.",
        state.primary_target.range_m(),
        state.secondary_target.range_m(),
        sep_cm,
        limit_cm,
        state.steering_deg
    );
    register_canvas_accessibility(ui, &response, Role::GraphicsDocument, &a11y_label, None);
    paint_canvas_focus_ring(ui, &response);

    CanvasInteraction {
        focused_id: response.has_focus().then_some(response.id),
        state_changed: changed,
    }
}

fn world_to_screen(rect: Rect, x_m: f64, y_m: f64) -> Pos2 {
    let pad = 16.0;
    let inner = rect.shrink(pad);
    let u = ((x_m as f32 + ROOM_HALF_WIDTH_M) / (2.0 * ROOM_HALF_WIDTH_M)).clamp(0.0, 1.0);
    let v = ((y_m as f32 - ROOM_MIN_DEPTH_M) / (ROOM_MAX_DEPTH_M - ROOM_MIN_DEPTH_M)).clamp(0.0, 1.0);
    Pos2::new(
        inner.left() + u * inner.width(),
        inner.bottom() - v * inner.height(),
    )
}

fn screen_to_world(rect: Rect, pos: Pos2) -> (f64, f64) {
    let inner = rect.shrink(16.0);
    let u = ((pos.x - inner.left()) / inner.width().max(1.0)).clamp(0.0, 1.0);
    let v = ((inner.bottom() - pos.y) / inner.height().max(1.0)).clamp(0.0, 1.0);
    let x_m = (-ROOM_HALF_WIDTH_M + u * 2.0 * ROOM_HALF_WIDTH_M) as f64;
    let y_m = (ROOM_MIN_DEPTH_M + v * (ROOM_MAX_DEPTH_M - ROOM_MIN_DEPTH_M)) as f64;
    (x_m.clamp(-1.20, 1.20), y_m.clamp(0.22, 1.75))
}

fn handle_pointer_interaction(
    response: &egui::Response,
    rect: Rect,
    state: &mut RadarGatingState,
    selected_target: &mut usize,
    dragging_target: &mut Option<usize>,
) -> bool {
    let mut changed = false;
    if response.drag_started() || response.clicked() {
        response.request_focus();
        if let Some(pos) = response.interact_pointer_pos() {
            let p1 = world_to_screen(rect, state.primary_target.x_m, state.primary_target.y_m);
            let p2 = world_to_screen(rect, state.secondary_target.x_m, state.secondary_target.y_m);
            let d1 = pos.distance(p1);
            let d2 = pos.distance(p2);
            if d1 <= 26.0 || d2 <= 26.0 {
                let chosen = if d1 <= d2 { 0 } else { 1 };
                *selected_target = chosen;
                *dragging_target = Some(chosen);
            } else if response.clicked() {
                let (wx, wy) = screen_to_world(rect, pos);
                state.steering_deg = wx.atan2(wy).to_degrees().clamp(-60.0, 60.0);
                changed = true;
            }
        }
    }
    if response.dragged() {
        if let (Some(target_idx), Some(pos)) = (*dragging_target, response.interact_pointer_pos()) {
            let (wx, wy) = screen_to_world(rect, pos);
            let target = target_mut(state, target_idx);
            target.x_m = wx;
            target.y_m = wy;
            changed = true;
        }
    }
    if response.drag_stopped() {
        *dragging_target = None;
    }
    changed
}

fn handle_keyboard_navigation(
    ui: &egui::Ui,
    state: &mut RadarGatingState,
    selected_target: &mut usize,
) -> bool {
    let step = ui.input(|i| if i.modifiers.shift { 0.002 } else { 0.005 });
    let mut dx = 0.0;
    let mut dy = 0.0;
    let mut d_steer = 0.0;
    ui.input(|i| {
        if i.key_pressed(Key::Num1) {
            *selected_target = 0;
        }
        if i.key_pressed(Key::Num2) {
            *selected_target = 1;
        }
        if i.key_pressed(Key::ArrowLeft) {
            dx -= step;
        }
        if i.key_pressed(Key::ArrowRight) {
            dx += step;
        }
        if i.key_pressed(Key::ArrowUp) {
            dy += step;
        }
        if i.key_pressed(Key::ArrowDown) {
            dy -= step;
        }
        if i.key_pressed(Key::OpenBracket) {
            d_steer -= 2.0;
        }
        if i.key_pressed(Key::CloseBracket) {
            d_steer += 2.0;
        }
    });
    if dx == 0.0 && dy == 0.0 && d_steer == 0.0 {
        return false;
    }
    if dx != 0.0 || dy != 0.0 {
        let tgt = target_mut(state, *selected_target);
        tgt.x_m = (tgt.x_m + dx).clamp(-1.20, 1.20);
        tgt.y_m = (tgt.y_m + dy).clamp(0.22, 1.75);
    }
    if d_steer != 0.0 {
        state.steering_deg = (state.steering_deg + d_steer).clamp(-60.0, 60.0);
    }
    true
}

fn target_mut(state: &mut RadarGatingState, idx: usize) -> &mut ScattererPoint {
    if idx == 0 {
        &mut state.primary_target
    } else {
        &mut state.secondary_target
    }
}

fn paint_room_backdrop(
    painter: &egui::Painter,
    rect: Rect,
    rayleigh_limit_m: f64,
    bg_fill: egui::Color32,
    palette: &SemanticPalette,
) {
    painter.rect_filled(rect, 6.0, bg_fill);
    painter.rect_stroke(
        rect,
        6.0,
        Stroke::new(1.0_f32, palette.graph_edge),
        egui::StrokeKind::Inside,
    );
    let origin = world_to_screen(rect, 0.0, 0.0);
    let grid_ring = palette.graph_edge.gamma_multiply(0.35);
    for ring_idx in 1..=4 {
        let r_m = ring_idx as f64 * 0.40;
        let top_pt = world_to_screen(rect, 0.0, r_m);
        let radius_px = (origin.y - top_pt.y).abs();
        painter.circle_stroke(origin, radius_px, Stroke::new(1.0_f32, grid_ring));
    }
    let res_px = (origin.y - world_to_screen(rect, 0.0, rayleigh_limit_m).y).abs();
    if res_px >= 2.0 {
        let ref_center = world_to_screen(rect, 0.0, 0.70);
        painter.line_segment(
            [
                Pos2::new(rect.right() - 78.0, ref_center.y - 0.5 * res_px),
                Pos2::new(rect.right() - 78.0, ref_center.y + 0.5 * res_px),
            ],
            Stroke::new(2.5_f32, palette.warning),
        );
    }
    painter.rect_filled(
        Rect::from_center_size(Pos2::new(origin.x, origin.y - 3.0), Vec2::new(42.0, 7.0)),
        2.0,
        palette.info,
    );
}

fn paint_steering_cone(
    painter: &egui::Painter,
    rect: Rect,
    steering_deg: f64,
    angle_res_deg: f64,
    palette: &SemanticPalette,
) {
    let origin = world_to_screen(rect, 0.0, 0.0);
    let half_bw = (0.5 * angle_res_deg).clamp(2.5, 45.0);
    let left_rad = (steering_deg - half_bw).to_radians();
    let right_rad = (steering_deg + half_bw).to_radians();
    let center_rad = steering_deg.to_radians();
    let reach = 1.68;

    let p_left = world_to_screen(rect, reach * left_rad.sin(), reach * left_rad.cos());
    let p_right = world_to_screen(rect, reach * right_rad.sin(), reach * right_rad.cos());
    let p_center = world_to_screen(rect, reach * center_rad.sin(), reach * center_rad.cos());

    painter.add(egui::Shape::convex_polygon(
        vec![origin, p_left, p_right],
        palette.info.gamma_multiply(0.16),
        Stroke::new(1.0_f32, palette.info.gamma_multiply(0.55)),
    ));
    painter.line_segment(
        [origin, p_center],
        Stroke::new(1.8_f32, palette.info),
    );
}

fn paint_scatterer_markers(
    painter: &egui::Painter,
    rect: Rect,
    state: &RadarGatingState,
    selected_target: usize,
    palette: &SemanticPalette,
) {
    let p1 = world_to_screen(rect, state.primary_target.x_m, state.primary_target.y_m);
    let p2 = world_to_screen(rect, state.secondary_target.x_m, state.secondary_target.y_m);
    painter.line_segment([p1, p2], Stroke::new(1.2_f32, palette.accent));

    painter.circle_filled(p1, 8.5, palette.success);
    painter.circle_filled(p2, 7.5, palette.series_color(3));
    if selected_target == 0 {
        painter.circle_stroke(p1, 12.5, Stroke::new(2.0_f32, palette.node_stroke));
    } else {
        painter.circle_stroke(p2, 11.5, Stroke::new(2.0_f32, palette.node_stroke));
    }
}

fn paint_canvas_legend(
    painter: &egui::Painter,
    rect: Rect,
    state: &RadarGatingState,
    spectra: &SpectralSnapshot,
    text_fg: egui::Color32,
    palette: &SemanticPalette,
) {
    let sep_cm = state.target_separation_m() * 100.0;
    let rayleigh_cm = spectra.rayleigh_limit_m * 100.0;
    let regime = if sep_cm < rayleigh_cm {
        "SUB-RAYLEIGH (< c/2B)"
    } else {
        "RAYLEIGH RESOLVED"
    };
    let summary = format!(
        "ΔR = {:.2} cm vs c/2B = {:.2} cm [{}] | Beam θ = {:+.1}° (Δθ = {:.1}°)",
        sep_cm, rayleigh_cm, regime, state.steering_deg, spectra.angle_resolution_deg
    );
    painter.text(
        Pos2::new(rect.left() + 12.0, rect.top() + 10.0),
        egui::Align2::LEFT_TOP,
        summary,
        egui::FontId::proportional(12.5),
        text_fg,
    );
    painter.text(
        Pos2::new(rect.left() + 12.0, rect.bottom() - 20.0),
        egui::Align2::LEFT_TOP,
        "Drag Chest / Secondary targets, click room to steer ULA, or use 1/2 + Arrows + [/]",
        egui::FontId::proportional(11.0),
        palette.muted_text,
    );
}

// [cite:mmwave_radiotherapy_setup]

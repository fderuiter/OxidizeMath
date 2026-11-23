//! Focusable AccessKit 2D phase-portrait and click-to-kick perturbation canvas.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::pipeline::ClosedLoopPipeline;
use super::state::{ClosedLoopState, PlantKind};
use crate::accessibility::{
    lock_canvas_arrow_keys, paint_canvas_focus_ring, register_canvas_accessibility,
};
use crate::app::theme::SemanticPalette;
use eframe::egui::{self, accesskit::Role, Color32, Key, Pos2, Rect, Sense, Stroke, Vec2};

/// Outcome of rendering and interacting with the 2D phase-portrait perturbation pad.
pub struct PhaseCanvasInteraction {
    /// Canvas widget ID when it holds keyboard focus.
    pub focused_id: Option<egui::Id>,
    /// Normalized `[dx, dy]` perturbation kick requested via pointer click or keyboard.
    pub kick_request: Option<[f64; 2]>,
}

/// Renders the interactive 2D phase portrait, EKF `1-sigma` ellipse, and click-to-kick pad.
pub fn show_phase_kick_canvas(
    ui: &mut egui::Ui,
    state: &ClosedLoopState,
    pipeline: &ClosedLoopPipeline,
    height: f32,
) -> PhaseCanvasInteraction {
    let desired = Vec2::new(ui.available_width().max(260.0), height.max(150.0));
    let (response, painter) = ui.allocate_painter(
        desired,
        Sense::click_and_drag() | Sense::focusable_noninteractive(),
    );
    let rect = response.rect;
    let bounds = phase_bounds(state);

    let mut kick_request = handle_pointer_kick(&response, rect, &bounds, pipeline);
    if response.has_focus() {
        lock_canvas_arrow_keys(ui, &response);
        if let Some(kb_kick) = handle_keyboard_kick(ui) {
            kick_request = Some(kb_kick);
        }
    }

    let palette = SemanticPalette::from_visuals(ui.visuals());
    paint_phase_backdrop(&painter, rect, state, &bounds, ui.visuals().extreme_bg_color, &palette);
    paint_phase_trajectories(&painter, rect, &bounds, pipeline, &palette);
    paint_ekf_uncertainty_and_markers(&painter, rect, &bounds, pipeline, &palette);
    paint_overlay_caption(&painter, rect, state, pipeline, ui.visuals().text_color(), &palette);

    let out = pipeline.last_output;
    let a11y_label = format!(
        "Closed-Loop Phase & Perturbation Visualization: {} with {}, controlled state {:.2}, EKF estimate {:.2} +/- {:.2}, latest innovation {:+.3}. Click canvas or press arrow keys to inject state kick.",
        state.plant.label(),
        state.controller.label(),
        out.controlled_primary,
        out.ekf_primary,
        out.ekf_primary_std,
        out.innovation,
    );
    let a11y_val = format!(
        "u={:+.3}, innovation={:+.3}",
        out.actuation, out.innovation
    );
    register_canvas_accessibility(
        ui,
        &response,
        Role::GraphicsDocument,
        &a11y_label,
        Some(&a11y_val),
    );
    paint_canvas_focus_ring(ui, &response);

    PhaseCanvasInteraction {
        focused_id: response.has_focus().then_some(response.id),
        kick_request,
    }
}

#[derive(Clone, Copy)]
struct PhaseBounds {
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
}

fn phase_bounds(state: &ClosedLoopState) -> PhaseBounds {
    match state.plant {
        PlantKind::LorenzChaos => PhaseBounds {
            x_min: -24.0,
            x_max: 24.0,
            y_min: -30.0,
            y_max: 30.0,
        },
        PlantKind::SeirEpidemic => PhaseBounds {
            x_min: 0.0,
            x_max: 420.0,
            y_min: 0.0,
            y_max: 340.0,
        },
        PlantKind::Pharmacokinetics => PhaseBounds {
            x_min: 0.0,
            x_max: (state.pk_mtc * 1.85).max(5.5),
            y_min: 0.4,
            y_max: 4.2,
        },
    }
}

fn phase_to_screen(rect: Rect, b: &PhaseBounds, x: f64, y: f64) -> Pos2 {
    let inner = rect.shrink2(Vec2::new(16.0, 22.0));
    let u = ((x - b.x_min) / (b.x_max - b.x_min).max(1.0e-6)).clamp(0.0, 1.0) as f32;
    let v = ((y - b.y_min) / (b.y_max - b.y_min).max(1.0e-6)).clamp(0.0, 1.0) as f32;
    Pos2::new(
        inner.left() + u * inner.width(),
        inner.bottom() - v * inner.height(),
    )
}

fn handle_pointer_kick(
    response: &egui::Response,
    rect: Rect,
    bounds: &PhaseBounds,
    pipeline: &ClosedLoopPipeline,
) -> Option<[f64; 2]> {
    if !(response.clicked() || response.drag_started()) {
        return None;
    }
    response.request_focus();
    let pos = response.interact_pointer_pos()?;
    let cur = phase_to_screen(
        rect,
        bounds,
        pipeline.last_output.controlled_primary,
        pipeline.last_output.true_latent,
    );
    let half_w = (0.35 * rect.width()).max(24.0);
    let half_h = (0.35 * rect.height()).max(24.0);
    let dx = ((pos.x - cur.x) / half_w).clamp(-1.0, 1.0) as f64;
    let dy = ((cur.y - pos.y) / half_h).clamp(-1.0, 1.0) as f64;
    Some([dx, dy])
}

fn handle_keyboard_kick(ui: &egui::Ui) -> Option<[f64; 2]> {
    let mut dx: f64 = 0.0;
    let mut dy: f64 = 0.0;
    let mut pulse = false;
    ui.input(|i| {
        if i.key_pressed(Key::ArrowRight) {
            dx += 0.65;
        }
        if i.key_pressed(Key::ArrowLeft) {
            dx -= 0.65;
        }
        if i.key_pressed(Key::ArrowUp) {
            dy += 0.65;
        }
        if i.key_pressed(Key::ArrowDown) {
            dy -= 0.65;
        }
        if i.key_pressed(Key::Space) || i.key_pressed(Key::Enter) {
            pulse = true;
        }
    });
    if pulse && dx == 0.0 && dy == 0.0 {
        Some([0.75, 0.40])
    } else if dx != 0.0 || dy != 0.0 {
        Some([dx, dy])
    } else {
        None
    }
}

fn paint_phase_backdrop(
    painter: &egui::Painter,
    rect: Rect,
    state: &ClosedLoopState,
    bounds: &PhaseBounds,
    bg_fill: Color32,
    palette: &SemanticPalette,
) {
    painter.rect_filled(rect, 6.0, bg_fill);
    painter.rect_stroke(
        rect,
        6.0,
        Stroke::new(1.0_f32, palette.graph_edge),
        egui::StrokeKind::Inside,
    );

    let grid_stroke = Stroke::new(1.0_f32, palette.graph_edge.gamma_multiply(0.28));
    for k in 1..4 {
        let frac = k as f64 * 0.25;
        let gx = bounds.x_min + frac * (bounds.x_max - bounds.x_min);
        let gy = bounds.y_min + frac * (bounds.y_max - bounds.y_min);
        painter.line_segment(
            [
                phase_to_screen(rect, bounds, gx, bounds.y_min),
                phase_to_screen(rect, bounds, gx, bounds.y_max),
            ],
            grid_stroke,
        );
        painter.line_segment(
            [
                phase_to_screen(rect, bounds, bounds.x_min, gy),
                phase_to_screen(rect, bounds, bounds.x_max, gy),
            ],
            grid_stroke,
        );
    }

    if state.plant == PlantKind::Pharmacokinetics {
        let p_min = phase_to_screen(rect, bounds, state.pk_mec, bounds.y_max);
        let p_max = phase_to_screen(rect, bounds, state.pk_mtc, bounds.y_min);
        let band_rect = Rect::from_min_max(p_min, p_max);
        painter.rect_filled(band_rect, 0.0, palette.success.gamma_multiply(0.14));
        painter.line_segment(
            [Pos2::new(p_min.x, band_rect.top()), Pos2::new(p_min.x, band_rect.bottom())],
            Stroke::new(1.4_f32, palette.success),
        );
        painter.line_segment(
            [Pos2::new(p_max.x, band_rect.top()), Pos2::new(p_max.x, band_rect.bottom())],
            Stroke::new(1.4_f32, palette.error),
        );
    }
}

fn paint_phase_trajectories(
    painter: &egui::Painter,
    rect: Rect,
    bounds: &PhaseBounds,
    pipeline: &ClosedLoopPipeline,
    palette: &SemanticPalette,
) {
    if pipeline.phase_ctrl_trace.len() == pipeline.phase_open_trace.len() {
        // Draw Viswanath UPO reference curve when Lorenz points lie in bounds.
        if bounds.x_min < -10.0 {
            let mut upo_pts: Vec<Pos2> = pipeline
                .live_lorenz
                .upo_points
                .iter()
                .step_by(2)
                .map(|p| phase_to_screen(rect, bounds, p[0], p[1]))
                .collect();
            if let Some(&first) = upo_pts.first() {
                upo_pts.push(first);
            }
            painter.add(egui::Shape::line(
                upo_pts,
                Stroke::new(1.4_f32, palette.warning.gamma_multiply(0.65)),
            ));
        }
    }

    draw_polyline(
        painter,
        rect,
        bounds,
        &pipeline.phase_open_trace,
        Stroke::new(1.4_f32, palette.error.gamma_multiply(0.60)),
    );
    draw_polyline(
        painter,
        rect,
        bounds,
        &pipeline.phase_ctrl_trace,
        Stroke::new(2.1_f32, palette.info),
    );
    draw_polyline(
        painter,
        rect,
        bounds,
        &pipeline.phase_ekf_trace,
        Stroke::new(1.8_f32, palette.success),
    );
}

fn draw_polyline(
    painter: &egui::Painter,
    rect: Rect,
    bounds: &PhaseBounds,
    pts: &[[f64; 2]],
    stroke: Stroke,
) {
    if pts.len() < 2 {
        return;
    }
    let screen_pts: Vec<Pos2> = pts
        .iter()
        .map(|&[x, y]| phase_to_screen(rect, bounds, x, y))
        .collect();
    painter.add(egui::Shape::line(screen_pts, stroke));
}

fn paint_ekf_uncertainty_and_markers(
    painter: &egui::Painter,
    rect: Rect,
    bounds: &PhaseBounds,
    pipeline: &ClosedLoopPipeline,
    palette: &SemanticPalette,
) {
    let out = pipeline.last_output;
    let ekf_pos = phase_to_screen(rect, bounds, out.ekf_primary, out.ekf_latent);
    let rx_pos = phase_to_screen(
        rect,
        bounds,
        out.ekf_primary + out.ekf_primary_std,
        out.ekf_latent,
    );
    let ry_pos = phase_to_screen(
        rect,
        bounds,
        out.ekf_primary,
        out.ekf_latent + out.ekf_latent_std,
    );
    let rx = (rx_pos.x - ekf_pos.x).abs().clamp(4.0, 48.0);
    let ry = (ekf_pos.y - ry_pos.y).abs().clamp(4.0, 48.0);

    let mut ellipse_pts = Vec::with_capacity(25);
    for k in 0..=24 {
        let ang = (k as f32 / 24.0) * std::f32::consts::TAU;
        ellipse_pts.push(Pos2::new(
            ekf_pos.x + rx * ang.cos(),
            ekf_pos.y + ry * ang.sin(),
        ));
    }
    painter.add(egui::Shape::convex_polygon(
        ellipse_pts,
        palette.success.gamma_multiply(0.18),
        Stroke::new(1.2_f32, palette.success),
    ));

    let open_pos = pipeline
        .phase_open_trace
        .last()
        .map(|&[x, y]| phase_to_screen(rect, bounds, x, y))
        .unwrap_or(ekf_pos);
    let ctrl_pos = phase_to_screen(rect, bounds, out.controlled_primary, out.true_latent);

    painter.circle_filled(open_pos, 4.5, palette.error);
    painter.circle_filled(ctrl_pos, 5.5, palette.info);
    painter.circle_stroke(ekf_pos, 6.5, Stroke::new(2.0_f32, palette.success));

    if let Some([kx, ky]) = pipeline.last_kick_vec {
        let tail = Pos2::new(ctrl_pos.x - kx as f32 * 26.0, ctrl_pos.y + ky as f32 * 26.0);
        painter.arrow(tail, ctrl_pos - tail, Stroke::new(2.0_f32, palette.warning));
    }
}

fn paint_overlay_caption(
    painter: &egui::Painter,
    rect: Rect,
    state: &ClosedLoopState,
    pipeline: &ClosedLoopPipeline,
    text_fg: Color32,
    palette: &SemanticPalette,
) {
    let out = pipeline.last_output;
    let headline = format!(
        "2D Phase Space & Click-to-Kick Pad — {} vs {} | EKF ν = {:+.3} | u = {:+.3}",
        state.plant.primary_y_label(),
        state.plant.latent_label(),
        out.innovation,
        out.actuation
    );
    painter.text(
        Pos2::new(rect.left() + 10.0, rect.top() + 5.0),
        egui::Align2::LEFT_TOP,
        headline,
        egui::FontId::proportional(12.0),
        text_fg,
    );
    painter.text(
        Pos2::new(rect.left() + 10.0, rect.bottom() - 17.0),
        egui::Align2::LEFT_TOP,
        "Click anywhere in phase space (or focus + Arrow/Space keys) to inject a state disturbance kick",
        egui::FontId::proportional(10.8),
        palette.muted_text,
    );
}

// [cite:grpo]

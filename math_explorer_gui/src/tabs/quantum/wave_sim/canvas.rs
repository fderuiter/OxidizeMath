#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
// [cite:quantum_mechanics]

use super::presets::{GRID_X_MAX, GRID_X_MIN, POTENTIAL_DISPLAY_SCALE};
use super::runner::WaveData;
use super::super::wave_sim_sculpt::consume_reticle_key_deltas;
use super::WaveSimulatorTool;
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility,
};
use crate::async_sim::ui::render_empty_plot_guidance;
use crate::framework::{BrushMode, BrushStroke};
use eframe::egui;
use egui_plot::{
    Arrows, Legend, Line, LineStyle, MarkerShape, Plot, PlotPoint, PlotPoints, Points, VLine,
};
use math_explorer::physics::quantum::{MAX_SCULPT_POTENTIAL, MIN_SCULPT_POTENTIAL};

pub(crate) const EMITTER_HANDLE_Y: f64 = 0.20;
pub(crate) const EMITTER_SIGMA_Y: f64 = 0.16;
pub(crate) const K0_ARROW_SCALE: f64 = 0.12;
const HANDLE_HIT_RADIUS_PX: f32 = 14.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ActiveCanvasDrag {
    None,
    Sculpt {
        last_idx: usize,
        last_v: f64,
        erase: bool,
    },
    EmitterX0,
    EmitterK0,
    EmitterSigma,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct QuantumCanvasState {
    pub reticle_idx: usize,
    pub drag: ActiveCanvasDrag,
}

impl Default for QuantumCanvasState {
    fn default() -> Self {
        Self {
            reticle_idx: 100,
            drag: ActiveCanvasDrag::None,
        }
    }
}

pub(super) fn x_to_grid_index(x: f64, n_points: usize) -> usize {
    if n_points <= 1 {
        return 0;
    }
    let ratio = ((x - GRID_X_MIN) / (GRID_X_MAX - GRID_X_MIN)).clamp(0.0, 1.0);
    (ratio * (n_points - 1) as f64).round() as usize
}

fn marker_point(
    name: &str,
    xy: [f64; 2],
    r: f32,
    shape: MarkerShape,
    c: egui::Color32,
) -> Points<'_> {
    Points::new(name, vec![xy]).radius(r).shape(shape).color(c)
}

fn draw_hud_badges(ui: &mut egui::Ui, frame: &WaveData) {
    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    ui.horizontal_wrapped(|ui| {
        ui.label(format!("Time: {:.3}", frame.time));
        ui.separator();
        ui.colored_label(palette.series_color(0), "━ |ψ(x, t)|²");
        ui.colored_label(
            palette.series_color(1),
            "━ V(x) (drag sculpt, right-drag erase)",
        );
        ui.colored_label(
            palette.series_color(2),
            format!("┅ E0 = {:.2} (scaled)", frame.kinetic_energy),
        );
        ui.separator();
        ui.colored_label(
            palette.success,
            format!("T(t): {:.1}%", frame.transmission * 100.0),
        );
        ui.colored_label(
            palette.info,
            format!("R(t): {:.1}%", frame.reflection * 100.0),
        );
    });
}

fn draw_wave_curves(
    plot_ui: &mut egui_plot::PlotUi<'_>,
    frame: &WaveData,
    palette: &crate::app::SemanticPalette,
) {
    let len = frame
        .x_axis
        .len()
        .min(frame.prob_density.len())
        .min(frame.potential.len());
    let mut psi_points = Vec::with_capacity(len);
    let mut v_points = Vec::with_capacity(len);
    for i in 0..len {
        let x = frame.x_axis[i];
        psi_points.push([x, frame.prob_density[i]]);
        v_points.push([x, frame.potential[i] * POTENTIAL_DISPLAY_SCALE]);
    }
    let x_start = frame.x_axis.first().copied().unwrap_or(GRID_X_MIN);
    let x_end = frame.x_axis.last().copied().unwrap_or(GRID_X_MAX);
    let e0_scaled = frame.kinetic_energy * POTENTIAL_DISPLAY_SCALE;

    plot_ui.line(
        Line::new("Probability |ψ(x, t)|²", PlotPoints::new(psi_points))
            .color(palette.series_color(0))
            .width(2.0_f32),
    );
    plot_ui.line(
        Line::new("Potential V(x) (scaled)", PlotPoints::new(v_points))
            .color(palette.series_color(1))
            .width(1.75_f32),
    );
    plot_ui.line(
        Line::new(
            format!("Kinetic Energy E0 = {:.2} (scaled)", frame.kinetic_energy),
            PlotPoints::new(vec![[x_start, e0_scaled], [x_end, e0_scaled]]),
        )
        .style(LineStyle::Dashed { length: 6.0_f32 })
        .color(palette.series_color(2)),
    );
}

fn draw_emitter_and_reticle_overlays(
    plot_ui: &mut egui_plot::PlotUi<'_>,
    frame: &WaveData,
    emitter: (f64, f64, f64),
    reticle_idx: usize,
    has_focus: bool,
    palette: &crate::app::SemanticPalette,
) {
    let (x0, k0, sigma) = emitter;
    let k0_tip_x = x0 + k0 * K0_ARROW_SCALE;
    let span = vec![[x0 - sigma, EMITTER_SIGMA_Y], [x0 + sigma, EMITTER_SIGMA_Y]];
    plot_ui.line(
        Line::new("Packet Width Envelope", PlotPoints::new(span))
            .style(LineStyle::Dashed { length: 4.0_f32 })
            .color(palette.accent),
    );
    plot_ui.arrows(
        Arrows::new(
            "Momentum Vector k₀",
            vec![[x0, EMITTER_HANDLE_Y]],
            vec![[k0_tip_x, EMITTER_HANDLE_Y]],
        )
        .color(palette.warning),
    );
    plot_ui.points(marker_point(
        "Emitter x₀",
        [x0, EMITTER_HANDLE_Y],
        6.0,
        MarkerShape::Diamond,
        palette.accent,
    ));
    plot_ui.points(marker_point(
        "Momentum Tip k₀",
        [k0_tip_x, EMITTER_HANDLE_Y],
        5.0,
        MarkerShape::Circle,
        palette.warning,
    ));
    plot_ui.points(marker_point(
        "Width Handle σ",
        [x0 + sigma, EMITTER_SIGMA_Y],
        5.0,
        MarkerShape::Square,
        palette.success,
    ));
    if has_focus {
        if let (Some(&rx), Some(&rv)) = (
            frame.x_axis.get(reticle_idx),
            frame.potential.get(reticle_idx),
        ) {
            plot_ui.vline(
                VLine::new("Sculpt Reticle", rx)
                    .style(LineStyle::Dotted { spacing: 4.0_f32 })
                    .color(palette.accent),
            );
            plot_ui.points(marker_point(
                "Reticle Cursor",
                [rx, rv * POTENTIAL_DISPLAY_SCALE],
                6.5,
                MarkerShape::Cross,
                palette.accent,
            ));
        }
    }
}

fn pick_pointer_drag_target(
    plot_ui: &egui_plot::PlotUi<'_>,
    pointer_screen: egui::Pos2,
    emitter: (f64, f64, f64),
    n_points: usize,
) -> ActiveCanvasDrag {
    let (x0, k0, sigma) = emitter;
    let candidates = [
        (
            ActiveCanvasDrag::EmitterK0,
            PlotPoint::new(x0 + k0 * K0_ARROW_SCALE, EMITTER_HANDLE_Y),
        ),
        (
            ActiveCanvasDrag::EmitterSigma,
            PlotPoint::new(x0 + sigma, EMITTER_SIGMA_Y),
        ),
        (
            ActiveCanvasDrag::EmitterX0,
            PlotPoint::new(x0, EMITTER_HANDLE_Y),
        ),
    ];
    if let Some((target, _)) = candidates
        .into_iter()
        .map(|(t, pt)| (t, plot_ui.screen_from_plot(pt).distance(pointer_screen)))
        .filter(|(_, dist)| *dist <= HANDLE_HIT_RADIUS_PX)
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    {
        return target;
    }
    let plot_pt = plot_ui.plot_from_screen(pointer_screen);
    let idx = x_to_grid_index(plot_pt.x, n_points);
    let v = (plot_pt.y / POTENTIAL_DISPLAY_SCALE).clamp(MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL);
    ActiveCanvasDrag::Sculpt {
        last_idx: idx,
        last_v: v,
        erase: false,
    }
}

pub(super) fn apply_emitter_drag(
    tool: &mut WaveSimulatorTool,
    drag: ActiveCanvasDrag,
    plot_x: f64,
    ctx: Option<&egui::Context>,
) -> bool {
    let mut x0 = tool.x0;
    let mut k0 = tool.k0;
    let mut sigma = tool.sigma;
    match drag {
        ActiveCanvasDrag::EmitterX0 => x0 = plot_x,
        ActiveCanvasDrag::EmitterK0 => k0 = (plot_x - x0) / K0_ARROW_SCALE,
        ActiveCanvasDrag::EmitterSigma => sigma = (plot_x - x0).abs(),
        _ => return false,
    }
    tool.update_wavepacket_live(x0, k0, sigma, ctx);
    true
}

fn dispatch_sculpt_drag(
    egui_ctx: &egui::Context,
    tool: &mut WaveSimulatorTool,
    frame: &WaveData,
    plot_pt: PlotPoint,
    prev: (usize, f64, bool),
) -> (usize, f64) {
    let (last_idx, last_v, erase) = prev;
    let idx = x_to_grid_index(plot_pt.x, frame.x_axis.len());
    let next_v = if erase {
        0.0
    } else {
        (plot_pt.y / POTENTIAL_DISPLAY_SCALE).clamp(MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL)
    };
    let mode = tool.brush_kit.mode;
    if !erase && matches!(mode, BrushMode::PotentialWell | BrushMode::PotentialStep | BrushMode::Erase) {
        let x0 = frame.x_axis.get(last_idx).copied().unwrap_or(plot_pt.x);
        let stroke = BrushStroke {
            start: [x0, plot_pt.y],
            end: [plot_pt.x, plot_pt.y],
            radius: tool.brush_kit.radius,
            mode,
            drag_vector: [plot_pt.x - x0, 0.0],
        };
        tool.apply_ui_brush_stroke(stroke, Some(egui_ctx));
    } else {
        tool.apply_sculpt_stroke(&frame.potential, last_idx, last_v, idx, next_v);
        egui_ctx.request_repaint();
    }
    (idx, next_v)
}

fn handle_pointer_interactions(
    plot_ui: &egui_plot::PlotUi<'_>,
    egui_ctx: &egui::Context,
    tool: &mut WaveSimulatorTool,
    frame: &WaveData,
    state: &mut QuantumCanvasState,
) {
    let (pressed, p_down, s_down, pos_opt) = egui_ctx.input(|i| {
        (
            i.pointer.primary_pressed(),
            i.pointer.primary_down(),
            i.pointer.secondary_down(),
            i.pointer.interact_pos(),
        )
    });
    if !p_down && !s_down {
        state.drag = ActiveCanvasDrag::None;
        return;
    }
    let Some(screen_pos) = pos_opt else { return };
    let plot_pt = plot_ui.plot_from_screen(screen_pos);
    if matches!(state.drag, ActiveCanvasDrag::None) && plot_ui.response().rect.contains(screen_pos)
    {
        if s_down {
            plot_ui.response().request_focus();
            let idx = x_to_grid_index(plot_pt.x, frame.x_axis.len());
            state.reticle_idx = idx;
            state.drag = ActiveCanvasDrag::Sculpt { last_idx: idx, last_v: 0.0, erase: true };
        } else if pressed {
            plot_ui.response().request_focus();
            let emitter = (tool.x0, tool.k0, tool.sigma);
            state.drag = pick_pointer_drag_target(plot_ui, screen_pos, emitter, frame.x_axis.len());
        }
    }
    match state.drag {
        ActiveCanvasDrag::Sculpt { last_idx, last_v, erase } => {
            let (idx, next_v) =
                dispatch_sculpt_drag(egui_ctx, tool, frame, plot_pt, (last_idx, last_v, erase));
            state.reticle_idx = idx;
            state.drag = ActiveCanvasDrag::Sculpt { last_idx: idx, last_v: next_v, erase };
        }
        ActiveCanvasDrag::EmitterX0 | ActiveCanvasDrag::EmitterK0 | ActiveCanvasDrag::EmitterSigma => {
            let _ = apply_emitter_drag(tool, state.drag, plot_pt.x, Some(egui_ctx));
        }
        ActiveCanvasDrag::None => {}
    }
}

fn handle_reticle_keyboard(
    egui_ctx: &egui::Context,
    tool: &mut WaveSimulatorTool,
    frame: &WaveData,
    state: &mut QuantumCanvasState,
) -> Option<String> {
    let n = frame.x_axis.len();
    if n == 0 {
        return None;
    }
    let (dx_idx, dv_delta, zero_cell) = consume_reticle_key_deltas(egui_ctx);
    if dx_idx == 0 && dv_delta == 0.0 && !zero_cell {
        return None;
    }
    state.reticle_idx =
        (state.reticle_idx.min(n - 1) as isize + dx_idx).clamp(0, (n - 1) as isize) as usize;
    let cur_v = tool
        .custom_potential
        .as_ref()
        .and_then(|v| v.get(state.reticle_idx).copied())
        .unwrap_or_else(|| frame.potential[state.reticle_idx]);
    let next_v = if zero_cell {
        0.0
    } else if dv_delta != 0.0 {
        (cur_v + dv_delta).clamp(MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL)
    } else {
        cur_v
    };
    if dv_delta != 0.0 || zero_cell {
        let idx = state.reticle_idx;
        tool.apply_sculpt_stroke(&frame.potential, idx, next_v, idx, next_v);
        egui_ctx.request_repaint();
    }
    Some(format!(
        "Reticle x = {:.2}, V(x) = {next_v:.1}, P_T = {:.1}%, P_R = {:.1}%",
        frame.x_axis[state.reticle_idx],
        frame.p_transmitted * 100.0,
        frame.p_reflected * 100.0
    ))
}

fn attach_canvas_accessibility(
    ui: &egui::Ui,
    response: &egui::Response,
    frame: &WaveData,
    reticle_idx: usize,
    pending_announcement: Option<String>,
) {
    let rx = frame.x_axis.get(reticle_idx).copied().unwrap_or(0.0);
    let rv = frame.potential.get(reticle_idx).copied().unwrap_or(0.0);
    let summary = format!(
        "Reticle x = {rx:.2}, V(x) = {rv:.1}, P_T = {:.1}%, P_R = {:.1}%",
        frame.p_transmitted * 100.0,
        frame.p_reflected * 100.0
    );
    ui.ctx()
        .data_mut(|d| d.insert_temp(egui::Id::new("quantum_plot_response_id"), response.id));
    lock_canvas_arrow_keys(ui, response);
    paint_canvas_focus_ring(ui, response);
    register_canvas_accessibility(
        ui,
        response,
        egui::accesskit::Role::GraphicsDocument,
        "Quantum Potential Sculpting Canvas",
        Some(&summary),
    );
    if let Some(msg) = pending_announcement {
        announce_live_status(ui.ctx(), msg);
    } else if response.gained_focus() {
        announce_live_status(
            ui.ctx(),
            format!("Quantum Potential Sculpting Canvas — {summary}."),
        );
    }
}

pub(super) fn render_wave_central_panel(tool: &mut WaveSimulatorTool, parent_ui: &mut egui::Ui) {
    egui::CentralPanel::default().show_inside(parent_ui, |ui| {
        let Some(frame) = tool.latest_wave_data().cloned() else {
            render_empty_plot_guidance(
                ui,
                "Wavepacket ready — click ▶ Play in the side panel to evolve ψ(x, t).",
            );
            return;
        };
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        draw_hud_badges(ui, &frame);
        let state_id = ui.make_persistent_id("quantum_plot_canvas_state");
        let mut state = ui
            .ctx()
            .data_mut(|d| d.get_temp::<QuantumCanvasState>(state_id))
            .unwrap_or_default();
        let egui_ctx = ui.ctx().clone();
        let allow_pan = !tool.brush_kit.is_sculpting();
        let mut pending_announcement = None;

        let plot_resp = Plot::new("quantum_plot")
            .legend(Legend::default())
            .x_axis_label("Position (x)")
            .y_axis_label("Probability Density |ψ|² / Scaled Energy")
            .view_aspect(2.0)
            .allow_drag(allow_pan)
            .allow_zoom(allow_pan)
            .allow_boxed_zoom(false)
            .show(ui, |plot_ui| {
                let focused = plot_ui.response().has_focus();
                if focused {
                    pending_announcement =
                        handle_reticle_keyboard(&egui_ctx, tool, &frame, &mut state);
                }
                if !allow_pan {
                    handle_pointer_interactions(plot_ui, &egui_ctx, tool, &frame, &mut state);
                } else {
                    state.drag = ActiveCanvasDrag::None;
                }
                draw_wave_curves(plot_ui, &frame, &palette);
                draw_emitter_and_reticle_overlays(
                    plot_ui,
                    &frame,
                    (tool.x0, tool.k0, tool.sigma),
                    state.reticle_idx,
                    focused,
                    &palette,
                );
            });
        ui.ctx().data_mut(|d| d.insert_temp(state_id, state));
        attach_canvas_accessibility(
            ui,
            &plot_resp.response,
            &frame,
            state.reticle_idx,
            pending_announcement,
        );
    });
}

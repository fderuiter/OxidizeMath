#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
// [cite:quantum_mechanics]

use super::{
    CUSTOM_POTENTIAL_KEY, DEFAULT_K0, DEFAULT_SIGMA, DEFAULT_X0, POTENTIAL_DISPLAY_SCALE, WaveData,
    X_MAX, X_MIN, clamp_wavepacket_params,
};
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility,
};
use crate::async_sim::unified::UnifiedCanvasContext;
use crate::async_sim::{SimCommand, StateSnapshot};
use crate::url_state::quantize_url_centi;
use eframe::egui;
use egui_plot::{
    Arrows, Legend, Line, LineStyle, MarkerShape, Plot, PlotPoint, PlotPoints, Points, VLine,
};
use math_explorer::physics::quantum::{
    MAX_SCULPT_POTENTIAL, MIN_SCULPT_POTENTIAL, apply_bresenham_potential_stroke,
};
use nalgebra::DVector;

pub(crate) const EMITTER_HANDLE_Y: f64 = 0.20;
pub(crate) const EMITTER_SIGMA_Y: f64 = 0.16;
pub(crate) const K0_ARROW_SCALE: f64 = 0.12;
const HANDLE_HIT_RADIUS_PX: f32 = 14.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ActiveCanvasDrag {
    None,
    Sculpt { last_idx: usize, last_v: f64, erase: bool },
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
        Self { reticle_idx: 100, drag: ActiveCanvasDrag::None }
    }
}

fn extract_wave_frame(snapshot_opt: Option<&StateSnapshot>) -> Option<WaveData> {
    snapshot_opt?.structured_data.as_ref()?.downcast_ref::<WaveData>().cloned()
}

pub(super) fn x_to_grid_index(x: f64, n_points: usize) -> usize {
    if n_points <= 1 {
        return 0;
    }
    let ratio = ((x - X_MIN) / (X_MAX - X_MIN)).clamp(0.0, 1.0);
    (ratio * (n_points - 1) as f64).round() as usize
}

fn read_emitter_tuple(ctx: &UnifiedCanvasContext<'_>) -> (f64, f64, f64) {
    let guard = ctx.params.read().unwrap_or_else(|e| e.into_inner());
    let x0 = guard.get("x0").copied().unwrap_or(DEFAULT_X0);
    let k0 = guard.get("k0").copied().unwrap_or(DEFAULT_K0);
    let sigma = guard.get("sigma").copied().unwrap_or(DEFAULT_SIGMA);
    clamp_wavepacket_params(x0, k0, sigma)
}

fn record_stroke_and_dispatch(
    ctx: &mut UnifiedCanvasContext<'_>,
    base_potential: &[f64],
    i0: usize,
    v0: f64,
    i1: usize,
    v1: f64,
) {
    {
        let mut series_guard = ctx.custom_series.write().unwrap_or_else(|e| e.into_inner());
        let entry = series_guard
            .entry(CUSTOM_POTENTIAL_KEY.to_string())
            .or_insert_with(|| base_potential.to_vec());
        if entry.len() != base_potential.len() {
            *entry = base_potential.to_vec();
        }
        let mut dvec = DVector::from_column_slice(entry);
        apply_bresenham_potential_stroke(&mut dvec, i0, v0, i1, v1);
        for (dst, &src) in entry.iter_mut().zip(dvec.iter()) {
            *dst = quantize_url_centi(src, MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL);
        }
    }
    let qv0 = quantize_url_centi(v0, MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL);
    let qv1 = quantize_url_centi(v1, MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL);
    let _ = ctx.controller.send_command(SimCommand::SculptStroke1D { i0, v0: qv0, i1, v1: qv1 });
}

fn draw_hud_badges(ui: &mut egui::Ui, frame: &WaveData) {
    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    let barrier_x = frame.x_axis.get(frame.barrier_idx).copied().unwrap_or(0.0);
    let pt_pct = frame.p_transmitted * 100.0;
    let pr_pct = frame.p_reflected * 100.0;
    ui.horizontal_wrapped(|ui| {
        ui.label(format!("Time: {:.2}", frame.time));
        ui.separator();
        ui.colored_label(palette.series_color(0), "━ |ψ(x)|²");
        ui.colored_label(palette.series_color(1), "━ V(x) (drag sculpt, right-drag erase)");
        ui.separator();
        ui.colored_label(palette.success, format!("Transmission P_T: {pt_pct:.1}%"));
        ui.colored_label(palette.info, format!("Reflection P_R: {pr_pct:.1}%"));
        ui.label(format!("(x_b = {barrier_x:.2})"));
    });
}

fn marker_point(name: &str, xy: [f64; 2], r: f32, shape: MarkerShape, c: egui::Color32) -> Points<'_> {
    Points::new(name, vec![xy]).radius(r).shape(shape).color(c)
}

fn draw_wave_curves(
    plot_ui: &mut egui_plot::PlotUi<'_>,
    frame: &WaveData,
    palette: &crate::app::SemanticPalette,
) {
    let psi_pts: Vec<[f64; 2]> = frame.x_axis.iter().zip(&frame.prob_density).map(|(&x, &p)| [x, p]).collect();
    let v_pts: Vec<[f64; 2]> = frame
        .x_axis
        .iter()
        .zip(&frame.potential)
        .map(|(&x, &v)| [x, v * POTENTIAL_DISPLAY_SCALE])
        .collect();
    plot_ui.line(Line::new("Probability |ψ|²", PlotPoints::new(psi_pts)).color(palette.series_color(0)).width(2.0_f32));
    plot_ui.line(Line::new("Potential V(x) (scaled)", PlotPoints::new(v_pts)).color(palette.series_color(1)).width(2.0_f32));
    if let Some(&xb) = frame.x_axis.get(frame.barrier_idx) {
        plot_ui.vline(VLine::new("Barrier Partition x_b", xb).style(LineStyle::Dashed { length: 5.0_f32 }).color(palette.muted_text));
    }
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
    plot_ui.line(Line::new("Packet Width Envelope", PlotPoints::new(span)).style(LineStyle::Dashed { length: 4.0_f32 }).color(palette.accent));
    plot_ui.arrows(Arrows::new("Momentum Vector k₀", vec![[x0, EMITTER_HANDLE_Y]], vec![[k0_tip_x, EMITTER_HANDLE_Y]]).color(palette.warning));
    plot_ui.points(marker_point("Emitter x₀", [x0, EMITTER_HANDLE_Y], 6.0, MarkerShape::Diamond, palette.accent));
    plot_ui.points(marker_point("Momentum Tip k₀", [k0_tip_x, EMITTER_HANDLE_Y], 5.0, MarkerShape::Circle, palette.warning));
    plot_ui.points(marker_point("Width Handle σ", [x0 + sigma, EMITTER_SIGMA_Y], 5.0, MarkerShape::Square, palette.success));
    if has_focus {
        if let (Some(&rx), Some(&rv)) = (frame.x_axis.get(reticle_idx), frame.potential.get(reticle_idx)) {
            plot_ui.vline(VLine::new("Sculpt Reticle", rx).style(LineStyle::Dotted { spacing: 4.0_f32 }).color(palette.accent));
            plot_ui.points(marker_point("Reticle Cursor", [rx, rv * POTENTIAL_DISPLAY_SCALE], 6.5, MarkerShape::Cross, palette.accent));
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
        (ActiveCanvasDrag::EmitterK0, PlotPoint::new(x0 + k0 * K0_ARROW_SCALE, EMITTER_HANDLE_Y)),
        (ActiveCanvasDrag::EmitterSigma, PlotPoint::new(x0 + sigma, EMITTER_SIGMA_Y)),
        (ActiveCanvasDrag::EmitterX0, PlotPoint::new(x0, EMITTER_HANDLE_Y)),
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
    ActiveCanvasDrag::Sculpt { last_idx: idx, last_v: v, erase: false }
}

pub(super) fn apply_emitter_drag(
    ctx: &mut UnifiedCanvasContext<'_>,
    drag: ActiveCanvasDrag,
    plot_x: f64,
    emitter: (f64, f64, f64),
) -> bool {
    let (mut x0, mut k0, mut sigma) = emitter;
    match drag {
        ActiveCanvasDrag::EmitterX0 => x0 = plot_x,
        ActiveCanvasDrag::EmitterK0 => k0 = (plot_x - x0) / K0_ARROW_SCALE,
        ActiveCanvasDrag::EmitterSigma => sigma = (plot_x - x0).abs(),
        _ => return false,
    }
    let (sx0, sk0, ssigma) = clamp_wavepacket_params(x0, k0, sigma);
    {
        let mut guard = ctx.params.write().unwrap_or_else(|e| e.into_inner());
        guard.insert("x0".to_string(), sx0);
        guard.insert("k0".to_string(), sk0);
        guard.insert("sigma".to_string(), ssigma);
    }
    let _ = ctx.controller.send_command(SimCommand::SetWavepacket { x0: sx0, k0: sk0, sigma: ssigma });
    true
}

fn handle_pointer_interactions(
    plot_ui: &egui_plot::PlotUi<'_>,
    egui_ctx: &egui::Context,
    ctx: &mut UnifiedCanvasContext<'_>,
    frame: &WaveData,
    state: &mut QuantumCanvasState,
    emitter: (f64, f64, f64),
) {
    let (pressed, p_down, s_down, pos_opt) = egui_ctx.input(|i| {
        (i.pointer.primary_pressed(), i.pointer.primary_down(), i.pointer.secondary_down(), i.pointer.interact_pos())
    });
    if !p_down && !s_down {
        state.drag = ActiveCanvasDrag::None;
        return;
    }
    let Some(screen_pos) = pos_opt else { return };
    let plot_pt = plot_ui.plot_from_screen(screen_pos);
    if matches!(state.drag, ActiveCanvasDrag::None) && plot_ui.response().rect.contains(screen_pos) {
        if s_down {
            plot_ui.response().request_focus();
            let idx = x_to_grid_index(plot_pt.x, frame.x_axis.len());
            state.reticle_idx = idx;
            state.drag = ActiveCanvasDrag::Sculpt { last_idx: idx, last_v: 0.0, erase: true };
        } else if pressed {
            plot_ui.response().request_focus();
            state.drag = pick_pointer_drag_target(plot_ui, screen_pos, emitter, frame.x_axis.len());
        }
    }
    match state.drag {
        ActiveCanvasDrag::Sculpt { last_idx, last_v, erase } => {
            let idx = x_to_grid_index(plot_pt.x, frame.x_axis.len());
            let next_v = if erase { 0.0 } else { (plot_pt.y / POTENTIAL_DISPLAY_SCALE).clamp(MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL) };
            state.reticle_idx = idx;
            record_stroke_and_dispatch(ctx, &frame.potential, last_idx, last_v, idx, next_v);
            state.drag = ActiveCanvasDrag::Sculpt { last_idx: idx, last_v: next_v, erase };
            egui_ctx.request_repaint();
        }
        ActiveCanvasDrag::EmitterX0 | ActiveCanvasDrag::EmitterK0 | ActiveCanvasDrag::EmitterSigma => {
            if apply_emitter_drag(ctx, state.drag, plot_pt.x, emitter) {
                egui_ctx.request_repaint();
            }
        }
        ActiveCanvasDrag::None => {}
    }
}

fn consume_reticle_key_deltas(egui_ctx: &egui::Context) -> (isize, f64, bool) {
    egui_ctx.input_mut(|i| {
        let mut dx: isize = 0;
        let mut dv: f64 = 0.0;
        for (mods, key, step) in [
            (egui::Modifiers::SHIFT, egui::Key::ArrowLeft, -5),
            (egui::Modifiers::NONE, egui::Key::ArrowLeft, -1),
            (egui::Modifiers::SHIFT, egui::Key::ArrowRight, 5),
            (egui::Modifiers::NONE, egui::Key::ArrowRight, 1),
        ] {
            if dx == 0 && i.consume_key(mods, key) {
                dx = step;
            }
        }
        for (mods, key, step) in [
            (egui::Modifiers::SHIFT, egui::Key::ArrowUp, 5.0),
            (egui::Modifiers::NONE, egui::Key::ArrowUp, 2.0),
            (egui::Modifiers::SHIFT, egui::Key::ArrowDown, -5.0),
            (egui::Modifiers::NONE, egui::Key::ArrowDown, -2.0),
        ] {
            if dv == 0.0 && i.consume_key(mods, key) {
                dv = step;
            }
        }
        let clear = i.consume_key(egui::Modifiers::NONE, egui::Key::Delete)
            || i.consume_key(egui::Modifiers::NONE, egui::Key::Backspace);
        (dx, dv, clear)
    })
}

fn handle_reticle_keyboard(
    egui_ctx: &egui::Context,
    ctx: &mut UnifiedCanvasContext<'_>,
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
    state.reticle_idx = (state.reticle_idx.min(n - 1) as isize + dx_idx).clamp(0, (n - 1) as isize) as usize;
    let cur_v = ctx
        .custom_series
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .get(CUSTOM_POTENTIAL_KEY)
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
        record_stroke_and_dispatch(ctx, &frame.potential, idx, next_v, idx, next_v);
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
        announce_live_status(ui.ctx(), format!("Quantum Potential Sculpting Canvas — {summary}."));
    }
}

fn base_quantum_plot() -> Plot<'static> {
    Plot::new("quantum_plot")
        .legend(Legend::default())
        .x_axis_label("Position (x)")
        .y_axis_label("Probability Density / Scaled Potential")
        .view_aspect(2.0)
}

pub(super) fn render_readonly_wave_panel(ui: &mut egui::Ui, snapshot_opt: Option<&StateSnapshot>) -> bool {
    let Some(frame) = extract_wave_frame(snapshot_opt) else {
        crate::async_sim::ui::render_empty_plot_guidance(ui, "Wavepacket ready — click ▶ Run in the side panel to evolve ψ(x, t).");
        return true;
    };
    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    draw_hud_badges(ui, &frame);
    base_quantum_plot().show(ui, |plot_ui| {
        draw_wave_curves(plot_ui, &frame, &palette);
        draw_emitter_and_reticle_overlays(plot_ui, &frame, (DEFAULT_X0, DEFAULT_K0, DEFAULT_SIGMA), 100, false, &palette);
    });
    true
}

pub(super) fn render_interactive_wave_panel(ui: &mut egui::Ui, ctx: &mut UnifiedCanvasContext<'_>) -> bool {
    let Some(frame) = extract_wave_frame(ctx.snapshot()) else {
        crate::async_sim::ui::render_empty_plot_guidance(ui, "Wavepacket ready — click ▶ Run in the side panel to evolve ψ(x, t).");
        return true;
    };
    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    draw_hud_badges(ui, &frame);
    let state_id = ui.make_persistent_id("quantum_plot_canvas_state");
    let mut state = ui.ctx().data_mut(|d| d.get_temp::<QuantumCanvasState>(state_id)).unwrap_or_default();
    let emitter = read_emitter_tuple(ctx);
    let egui_ctx = ui.ctx().clone();
    let mut pending_announcement = None;

    let plot_resp = base_quantum_plot().allow_drag(false).allow_boxed_zoom(false).show(ui, |plot_ui| {
        let focused = plot_ui.response().has_focus();
        if focused {
            pending_announcement = handle_reticle_keyboard(&egui_ctx, ctx, &frame, &mut state);
        }
        handle_pointer_interactions(plot_ui, &egui_ctx, ctx, &frame, &mut state, emitter);
        draw_wave_curves(plot_ui, &frame, &palette);
        draw_emitter_and_reticle_overlays(plot_ui, &frame, read_emitter_tuple(ctx), state.reticle_idx, focused, &palette);
    });
    ui.ctx().data_mut(|d| d.insert_temp(state_id, state));
    attach_canvas_accessibility(ui, &plot_resp.response, &frame, state.reticle_idx, pending_announcement);
    true
}

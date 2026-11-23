//! Live 1D potential $V(x)$ sculpting, resonant double-barrier cavity presets,
//! sparse tuple persistence, and central-panel plotting for [`super::wave_sim::WaveUnified`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::wave_sim::{WaveData, WaveUnified, WAVE_N_POINTS, WAVE_X_MAX, WAVE_X_MIN};
use crate::async_sim::unified::{CompiledFormulaPayload, UnifiedModel};
use crate::async_sim::{SculptStatePayload, SimCommand, StateSnapshot};
use crate::framework::{
    apply_sparse_potential, encode_sparse_potential, interpolate_1d_span, BrushInteractionMode,
    BrushMode, BrushStroke, SpatialBrushKit,
};
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, Plot, PlotPoints, VLine};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub(crate) fn evaluate_formula_potential(
    payload: &CompiledFormulaPayload,
    x_axis: &[f64],
) -> Vec<f64> {
    let p = &payload.params[..payload.param_len];
    x_axis
        .iter()
        .map(|&x| {
            let raw_v = payload.compiled.eval_1d(x, p);
            if raw_v.is_finite() {
                raw_v.clamp(-1e4, 1e4)
            } else {
                0.0
            }
        })
        .collect()
}

pub(crate) fn build_base_potential(potential_type: f64, x_axis: &[f64]) -> Vec<f64> {
    let kind = potential_type.round() as i32;
    x_axis
        .iter()
        .map(|&x| match kind {
            1 => 0.5 * x * x,
            2 if (0.0..=0.45).contains(&x) => 16.0,
            // Symmetric Fabry-Perot double-barrier resonant cavity around x = 0.
            3 if (-0.52..=-0.28).contains(&x) || (0.28..=0.52).contains(&x) => 16.0,
            4 if x >= 0.2 => 8.0,
            _ => 0.0,
        })
        .collect()
}

pub(crate) fn apply_discrete_brush_to_potential(
    potential: &mut [f64],
    base_potential: &[f64],
    cx: i32,
    r: i32,
    is_obstacle: bool,
) {
    let n = potential.len();
    if n == 0 {
        return;
    }
    let idx = (cx.max(0) as usize).min(n - 1);
    let half_w = (r.max(1) as usize).min(n / 4);
    let Some((lo, hi)) = interpolate_1d_span(idx, idx, half_w, n) else {
        return;
    };
    if is_obstacle {
        for v in &mut potential[lo..=hi] {
            *v = 16.0;
        }
    } else {
        for (dst, &base_v) in potential[lo..=hi].iter_mut().zip(&base_potential[lo..=hi]) {
            *dst = base_v;
        }
    }
}

pub(crate) fn x_to_grid_index(x: f64, x_min: f64, x_max: f64, n_points: usize) -> usize {
    if n_points <= 1 {
        return 0;
    }
    let frac = ((x - x_min) / (x_max - x_min)).clamp(0.0, 1.0);
    (frac * (n_points - 1) as f64).round() as usize
}

pub(crate) fn apply_stroke_to_potential_slice(
    potential: &mut [f64],
    base_potential: &[f64],
    x_min: f64,
    x_max: f64,
    stroke: &BrushStroke,
) {
    let n = potential.len();
    if n < 2 {
        return;
    }
    let dx = (x_max - x_min) / (n - 1) as f64;
    let i0 = x_to_grid_index(stroke.start[0], x_min, x_max, n);
    let i1 = x_to_grid_index(stroke.end[0], x_min, x_max, n);
    let half_w = ((stroke.radius / dx).round() as usize).clamp(1, n / 4);
    let Some((lo, hi)) = interpolate_1d_span(i0, i1, half_w, n) else {
        return;
    };

    // Plot scales V(x) by 0.2 visually; convert vertical plot coordinate back to V(x) when dragged.
    let plot_v = stroke.end[1] * 5.0;
    let is_stamp = stroke.drag_vector[0].abs() < 1e-9 && stroke.drag_vector[1].abs() < 1e-9;

    match stroke.mode {
        BrushMode::PotentialBarrier | BrushMode::Solid => {
            let v_bar = if is_stamp {
                16.0
            } else {
                plot_v.clamp(2.0, 25.0)
            };
            for v in &mut potential[lo..=hi] {
                *v = v_bar;
            }
        }
        BrushMode::PotentialWell => {
            let v_well = if is_stamp {
                -8.0
            } else if plot_v < -0.5 {
                plot_v.clamp(-20.0, -0.5)
            } else {
                -8.0
            };
            for v in &mut potential[lo..=hi] {
                *v = v_well;
            }
        }
        BrushMode::PotentialStep => {
            let v_step = if is_stamp {
                8.0
            } else {
                plot_v.clamp(1.0, 20.0)
            };
            for v in &mut potential[lo..n] {
                *v = v_step;
            }
        }
        BrushMode::Erase => {
            for (dst, &base_v) in potential[lo..=hi].iter_mut().zip(&base_potential[lo..=hi]) {
                *dst = base_v;
            }
        }
        _ => {}
    }
}

pub(crate) fn record_wave_stroke_sparse(
    stroke: &BrushStroke,
    params: &HashMap<String, f64>,
    formula: Option<&CompiledFormulaPayload>,
    sculpt_state: &mut SculptStatePayload,
) {
    let dx = (WAVE_X_MAX - WAVE_X_MIN) / (WAVE_N_POINTS as f64 - 1.0);
    let x_axis: Vec<f64> = (0..WAVE_N_POINTS)
        .map(|i| WAVE_X_MIN + i as f64 * dx)
        .collect();
    let base = if let Some(payload) = formula {
        evaluate_formula_potential(payload, &x_axis)
    } else {
        let p_type = *params.get("potential_type").unwrap_or(&0.0);
        build_base_potential(p_type, &x_axis)
    };
    let mut current = base.clone();
    let _ = apply_sparse_potential(&mut current, &sculpt_state.sparse_potential);
    apply_stroke_to_potential_slice(&mut current, &base, WAVE_X_MIN, WAVE_X_MAX, stroke);
    sculpt_state.sparse_potential = encode_sparse_potential(&base, &current, 1e-6);
}

pub(crate) fn stamp_double_barrier_into(potential: &mut [f64], x_axis: &[f64]) {
    for (v, &x) in potential.iter_mut().zip(x_axis.iter()) {
        *v = if (-0.52..=-0.28).contains(&x) || (0.28..=0.52).contains(&x) {
            16.0
        } else {
            0.0
        };
    }
}

pub(crate) fn detect_barrier_split_index(potential: &[f64], base_potential: &[f64]) -> usize {
    let n = potential.len();
    if n == 0 {
        return 0;
    }
    // Locate rightmost elevated barrier cell in the central scattering region [20%..85%].
    let start = n / 5;
    let end = (n * 85) / 100;
    let mut rightmost = None;
    for i in start..end {
        if potential[i] - base_potential[i] > 0.5 || potential[i] > 1.0 {
            rightmost = Some(i + 1);
        }
    }
    rightmost.unwrap_or(n / 2).min(n - 1)
}

pub(crate) fn draw_wave_brush_controls(
    ui: &mut egui::Ui,
    brush_kit: &mut SpatialBrushKit,
    params: &Arc<RwLock<HashMap<String, f64>>>,
    sculpt_state: &Arc<RwLock<SculptStatePayload>>,
    snapshot: Option<&StateSnapshot>,
) -> Vec<SimCommand> {
    let mut cmds = Vec::new();
    ui.label("Potential V(x) Sculpting");

    ui.horizontal(|ui| {
        ui.selectable_value(
            &mut brush_kit.interaction_mode,
            BrushInteractionMode::Sculpt,
            "🖌 Sculpt",
        );
        ui.selectable_value(
            &mut brush_kit.interaction_mode,
            BrushInteractionMode::Pan,
            "✋ Pan",
        );
    });

    ui.horizontal_wrapped(|ui| {
        ui.selectable_value(&mut brush_kit.mode, BrushMode::PotentialBarrier, "Barrier");
        ui.selectable_value(&mut brush_kit.mode, BrushMode::PotentialWell, "Well");
        ui.selectable_value(&mut brush_kit.mode, BrushMode::PotentialStep, "Step");
        ui.selectable_value(&mut brush_kit.mode, BrushMode::Erase, "Erase");
    });

    ui.add(egui::Slider::new(&mut brush_kit.radius, 0.10..=1.20).text("Brush Width"));

    ui.horizontal_wrapped(|ui| {
        if ui.button("⛩ Stamp Double Barrier").clicked() {
            let dx = (WAVE_X_MAX - WAVE_X_MIN) / (WAVE_N_POINTS as f64 - 1.0);
            let x_axis: Vec<f64> = (0..WAVE_N_POINTS)
                .map(|i| WAVE_X_MIN + i as f64 * dx)
                .collect();
            let p_type = *params
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .get("potential_type")
                .unwrap_or(&0.0);
            let base = build_base_potential(p_type, &x_axis);
            let mut target = base.clone();
            stamp_double_barrier_into(&mut target, &x_axis);
            sculpt_state
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .sparse_potential = encode_sparse_potential(&base, &target, 1e-6);
            cmds.push(SimCommand::Custom("stamp_double_barrier".to_string()));
        }
        if ui.button("🚀 Re-launch ψ(x,0)").clicked() {
            cmds.push(SimCommand::Custom("relaunch_wavepacket".to_string()));
        }
    });

    draw_wave_tunneling_readout(ui, snapshot);
    cmds
}

fn draw_wave_tunneling_readout(ui: &mut egui::Ui, snapshot: Option<&StateSnapshot>) {
    let Some(wdata) = snapshot
        .and_then(|s| s.structured_data.as_ref())
        .and_then(|d| d.downcast_ref::<WaveData>())
    else {
        return;
    };
    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    ui.separator();
    ui.label("Tunneling & Unitarity");
    ui.colored_label(
        palette.info,
        format!(
            "Transmission T: {:.1}%   Reflection R: {:.1}%",
            wdata.transmission_prob * 100.0,
            (1.0 - wdata.transmission_prob).clamp(0.0, 1.0) * 100.0
        ),
    );
    ui.colored_label(
        palette.success,
        format!("Unitary Norm ||ψ||²: {:.6}", wdata.norm_sq),
    );
}

pub(crate) fn render_wave_central_panel(
    ui: &mut egui::Ui,
    snapshot_opt: Option<&StateSnapshot>,
    brush_kit: &mut SpatialBrushKit,
) -> (bool, Vec<BrushStroke>) {
    let Some(wdata) = snapshot_opt
        .and_then(|s| s.structured_data.as_ref())
        .and_then(|d| d.downcast_ref::<WaveData>())
    else {
        crate::async_sim::ui::render_empty_plot_guidance(
            ui,
            "Wavepacket ready — click ▶ Run in the side panel to evolve ψ(x, t).",
        );
        return (true, Vec::new());
    };

    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    draw_wave_header_strip(ui, wdata, &palette);
    let (psi_points, v_points) = build_wave_plot_series(wdata);
    let allow_pan = !brush_kit.is_sculpting();
    let mut strokes = Vec::new();
    Plot::new("quantum_plot")
        .legend(Legend::default())
        .x_axis_label("Position (x)")
        .y_axis_label("Probability Density / Scaled Potential")
        .view_aspect(2.0)
        .allow_drag(allow_pan)
        .allow_zoom(allow_pan)
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new("Probability |ψ|²", PlotPoints::new(psi_points))
                    .color(palette.series_color(0))
                    .width(2.0_f32),
            );
            plot_ui.line(
                Line::new("Potential V(x) (scaled)", PlotPoints::new(v_points))
                    .color(palette.series_color(1))
                    .width(1.8_f32),
            );
            plot_ui.vline(
                VLine::new("Transmission Detector", wdata.barrier_split_x)
                    .color(palette.muted_text)
                    .style(LineStyle::Dashed { length: 5.0_f32 }),
            );
            strokes = brush_kit.sample_plot(
                plot_ui,
                [WAVE_X_MIN, -2.5],
                [WAVE_X_MAX, 4.5],
                Some(BrushMode::Erase),
            );
        });

    let _ = WaveUnified::domain();
    (true, strokes)
}

fn build_wave_plot_series(wdata: &WaveData) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let psi_points = wdata
        .x_axis
        .iter()
        .zip(&wdata.prob_density)
        .map(|(&x, &p)| [x, p])
        .collect();
    let v_points = wdata
        .x_axis
        .iter()
        .zip(&wdata.potential)
        .map(|(&x, &v)| [x, v * 0.2])
        .collect();
    (psi_points, v_points)
}

fn draw_wave_header_strip(
    ui: &mut egui::Ui,
    wdata: &WaveData,
    palette: &crate::app::SemanticPalette,
) {
    ui.horizontal_wrapped(|ui| {
        ui.label(format!("Time: {:.2}", wdata.time));
        ui.separator();
        ui.colored_label(palette.series_color(0), "━ |ψ(x)|²");
        ui.colored_label(palette.series_color(1), "━ V(x) (×0.2)");
        ui.separator();
        ui.colored_label(
            palette.info,
            format!("Transmission T = {:.2}%", wdata.transmission_prob * 100.0),
        );
    });
}

impl WaveUnified {
    pub(crate) fn handle_wave_command(&mut self, cmd: SimCommand, params: &HashMap<String, f64>) {
        let dt = *params.get("dt").unwrap_or(&0.05);
        match cmd {
            SimCommand::Reset => {
                self.diverged = false;
                self.diagnostic_report = None;
                self.step_count = 0;
                self.init_system(*params.get("potential_type").unwrap_or(&0.0));
            }
            SimCommand::ApplyBrush {
                cx,
                r,
                is_obstacle,
                ..
            } => {
                let slice = self.potential.as_mut_slice();
                apply_discrete_brush_to_potential(slice, &self.base_potential, cx, r, is_obstacle);
                self.rebuild_operators(dt);
            }
            SimCommand::ApplyBrushStroke(ref stroke) => {
                let (x_min, x_max) = (self.x_min, self.x_max);
                let slice = self.potential.as_mut_slice();
                apply_stroke_to_potential_slice(slice, &self.base_potential, x_min, x_max, stroke);
                self.rebuild_operators(dt);
            }
            SimCommand::ClearObstacles => self.restore_base_potential(dt),
            SimCommand::Custom(ref s) if s == "Clear Potential Edits" => {
                self.restore_base_potential(dt);
            }
            SimCommand::Custom(ref s) if s == "stamp_double_barrier" => {
                stamp_double_barrier_into(self.potential.as_mut_slice(), &self.x_axis);
                self.rebuild_operators(dt);
            }
            SimCommand::Custom(ref s) if s == "relaunch_wavepacket" => {
                self.psi = math_explorer::physics::quantum::gaussian_wavepacket(
                    &self.x_axis,
                    -2.0,
                    5.0,
                    0.5,
                );
                self.time = 0.0;
                self.step_count = 0;
            }
            _ => {}
        }
    }

    fn restore_base_potential(&mut self, dt: f64) {
        self.potential
            .as_mut_slice()
            .copy_from_slice(&self.base_potential);
        self.rebuild_operators(dt);
    }
}

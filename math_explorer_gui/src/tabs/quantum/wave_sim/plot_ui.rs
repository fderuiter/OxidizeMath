#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
// [cite:quantum_mechanics]

//! Parameter-atlas panel and `egui_plot` wavepacket renderer for [`super::WaveSimulatorTool`].

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, Plot, PlotPoints};

use super::atlas::QuantumTunnelingCartographer;
use super::presets::{
    wave_parameter_constraints, WavePotentialPreset, GRID_X_MAX, GRID_X_MIN, POTENTIAL_CLAMP_MAX,
    POTENTIAL_CLAMP_MIN, POTENTIAL_DISPLAY_SCALE,
};
use super::runner::WaveData;
use super::WaveSimulatorTool;
use crate::framework::{
    BrushMode, CalloutManager, PlotBrushController, PlotBrushFrame, SpatialBrushKit,
};

impl WaveSimulatorTool {
    pub(super) fn render_wave_atlas_panel(&mut self, ui: &mut egui::Ui) {
        if !self.atlas.is_visible() {
            return;
        }
        let ctx = ui.ctx().clone();
        let (xk, yk) = self.atlas.axis_keys();
        let get_val = |k: &str| match k {
            "k0" => self.k0,
            "v0" => self.v0,
            "width" => self.width,
            _ => 0.0,
        };
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.preset as u8).hash(&mut h);
        self.sigma.to_bits().hash(&mut h);
        if xk != "v0" && yk != "v0" {
            self.v0.to_bits().hash(&mut h);
        }
        if xk != "width" && yk != "width" {
            self.width.to_bits().hash(&mut h);
        }
        if self.preset == WavePotentialPreset::Custom {
            self.math_input.raw_text.hash(&mut h);
        }
        for &(idx, val) in &self.sparse_potential {
            idx.hash(&mut h);
            val.to_bits().hash(&mut h);
        }
        let params = wave_parameter_constraints();
        let (preset, k0, v0, width, sigma) =
            (self.preset, self.k0, self.v0, self.width, self.sigma);
        let (expr, sparse) = (self.last_valid_expr.clone(), self.sparse_potential.clone());
        if let Some(ev) = self.atlas.show_panel(
            ui,
            "wave_sim",
            get_val(xk),
            get_val(yk),
            h.finish(),
            move |keys| {
                Arc::new(QuantumTunnelingCartographer::new(
                    &params, keys, preset, k0, v0, width, sigma, &expr, &sparse,
                ))
            },
        ) {
            self.apply_atlas_teleport(ev, Some(&ctx));
        }
    }
}

fn draw_wave_plot_header(
    ui: &mut egui::Ui,
    wave_data: &WaveData,
    psi_color: egui::Color32,
    potential_color: egui::Color32,
    energy_color: egui::Color32,
) {
    ui.horizontal_wrapped(|ui| {
        ui.label(format!("Time: {:.3}", wave_data.time));
        ui.separator();
        ui.colored_label(psi_color, "━ |ψ(x, t)|²");
        ui.colored_label(potential_color, "━ V(x) (scaled)");
        ui.colored_label(
            energy_color,
            format!("┅ E0 = {:.2} (scaled)", wave_data.kinetic_energy),
        );
    });
}

fn build_wave_plot_series(wave_data: &WaveData) -> (Vec<[f64; 2]>, Vec<[f64; 2]>, [f64; 2]) {
    let len = wave_data
        .x_axis
        .len()
        .min(wave_data.prob_density.len())
        .min(wave_data.potential.len());
    let mut psi_points = Vec::with_capacity(len);
    let mut v_points = Vec::with_capacity(len);
    for i in 0..len {
        let x = wave_data.x_axis[i];
        psi_points.push([x, wave_data.prob_density[i]]);
        v_points.push([x, wave_data.potential[i] * POTENTIAL_DISPLAY_SCALE]);
    }
    let x_start = wave_data.x_axis.first().copied().unwrap_or(-5.0);
    let x_end = wave_data.x_axis.last().copied().unwrap_or(5.0);
    (psi_points, v_points, [x_start, x_end])
}

pub(super) fn render_wave_plot(
    ui: &mut egui::Ui,
    wave_data: &WaveData,
    brush_kit: &mut SpatialBrushKit,
    callouts: &mut CalloutManager,
    locked: bool,
) -> (egui::Response, Option<PlotBrushFrame>) {
    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    let (psi_color, potential_color, energy_color) = (
        palette.series_color(0),
        palette.series_color(1),
        palette.series_color(2),
    );
    draw_wave_plot_header(ui, wave_data, psi_color, potential_color, energy_color);
    let (psi_points, v_points, [x_start, x_end]) = build_wave_plot_series(wave_data);
    let e0_scaled = wave_data.kinetic_energy * POTENTIAL_DISPLAY_SCALE;
    let allow_pan = locked || !brush_kit.is_sculpting();
    let mut brush_frame = None;
    let (y_min, y_max) = (
        POTENTIAL_CLAMP_MIN * POTENTIAL_DISPLAY_SCALE,
        POTENTIAL_CLAMP_MAX * POTENTIAL_DISPLAY_SCALE,
    );

    let plot_resp = Plot::new("quantum_plot")
        .legend(Legend::default())
        .x_axis_label("Position (x)")
        .y_axis_label("Probability Density |ψ|² / Scaled Energy")
        .view_aspect(2.0)
        .allow_drag(allow_pan)
        .allow_zoom(allow_pan)
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new("Probability |ψ(x, t)|²", PlotPoints::new(psi_points))
                    .color(psi_color)
                    .width(2.0_f32),
            );
            plot_ui.line(
                Line::new("Potential V(x) (scaled)", PlotPoints::new(v_points))
                    .color(potential_color)
                    .width(1.75_f32),
            );
            plot_ui.line(
                Line::new(
                    format!(
                        "Kinetic Energy E0 = {:.2} (scaled)",
                        wave_data.kinetic_energy
                    ),
                    PlotPoints::new(vec![[x_start, e0_scaled], [x_end, e0_scaled]]),
                )
                .style(LineStyle::Dashed { length: 6.0_f32 })
                .color(energy_color),
            );
            let b = plot_ui.plot_bounds();
            if b.width() > 0.0 && b.height() > 0.0 {
                callouts.render_on_plot(plot_ui);
            }
            if !locked {
                brush_frame = Some(PlotBrushController::capture_plot(
                    brush_kit,
                    plot_ui,
                    [GRID_X_MIN, y_min],
                    [GRID_X_MAX, y_max],
                    Some(BrushMode::Erase),
                ));
            }
        });
    (plot_resp.response, brush_frame)
}

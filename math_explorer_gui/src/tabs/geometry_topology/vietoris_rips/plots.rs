//! Dual `egui_plot` views (2D/3D complex + persistence diagram/barcode), live Betti badges, and exports.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::io::MAX_VIETORIS_RIPS_POINTS;
use super::presets::{GeneratorDescriptor, ShapePreset};
use super::state::{SculptBrush, VrSourceMode};
use super::VietorisRipsTool;
use crate::accessibility::AccessibleTheoryHover;
use crate::app::SemanticPalette;
use crate::framework::bounded_drag_value;
use eframe::egui;
use egui_plot::{Line, LineStyle, Plot, Points, Polygon};
use math_explorer::pure_math::statistics::tda::{RipsInterval, MAX_FILTRATION_TRIANGLES};

/// Visualization mode for the right-hand persistent homology plot.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum PersistencePlotMode {
    /// Birth-vs-death scatter diagram with $y = x$ diagonal and active $\epsilon$ quadrant.
    #[default]
    Diagram,
    /// Horizontal persistence barcode ordered by homology dimension and birth scale.
    Barcode,
}

impl PersistencePlotMode {
    /// Returns `true` when set to the default [`Self::Diagram`] mode.
    #[must_use]
    pub const fn is_default(self) -> bool {
        matches!(self, Self::Diagram)
    }
}

/// Renders the full Vietoris-Rips TDA workbench inside `ui`.
pub fn render_workbench(tool: &mut VietorisRipsTool, ui: &mut egui::Ui) {
    super::io::poll_tool_io(tool);
    tool.poll_incoming_landmarks(ui.ctx());
    egui::CentralPanel::default().show_inside(ui, |ui| {
        ui.heading("Vietoris-Rips Complex & Persistent Homology (TDA)");
        render_cloud_toolbar(tool, ui);
        if matches!(tool.mode, VrSourceMode::Lorenz3D | VrSourceMode::Takens3D) {
            tool.render_chaos_toolbar(ui);
        } else if !tool.mode.is_3d() && tool.ambient_dimension() > 2 {
            render_nd_projection_bar(tool, ui);
        }
        render_filtration_toolbar(tool, ui);
        ui.separator();
        let palette = SemanticPalette::from_visuals(ui.visuals());
        render_topology_readouts(tool, ui, &palette);
        render_banners(tool, ui, &palette);
        render_dual_plots(tool, ui, &palette);
    });
}

fn render_cloud_toolbar(tool: &mut VietorisRipsTool, ui: &mut egui::Ui) {
    ui.horizontal_wrapped(|ui| {
        tool.render_source_mode_combo(ui);
        if !tool.mode.is_3d() {
            render_2d_preset_controls(tool, ui);
        } else {
            ui.menu_button("🎥 3D Camera", |ui| {
                tool.camera.ui(ui);
            });
            tool.refresh_projected_3d_export_geometry();
        }
        tool.render_sculpt_brush_controls(ui);
    });
}

fn render_2d_preset_controls(tool: &mut VietorisRipsTool, ui: &mut egui::Ui) {
    let active_preset = tool.active_generator().map(|g| g.preset);
    egui::ComboBox::from_label("Shape")
        .selected_text(active_preset.map_or("Custom Cloud", ShapePreset::label))
        .show_ui(ui, |ui| {
            for preset in ShapePreset::ALL {
                if ui
                    .selectable_label(active_preset == Some(preset), preset.label())
                    .clicked()
                {
                    tool.select_shape_preset(preset);
                }
            }
        });

    if let Some(mut desc) = tool.active_generator() {
        let mut count_f = desc.count as f64;
        ui.label("Points:");
        let resp = ui.add(bounded_drag_value(
            &mut count_f,
            3.0..=(MAX_VIETORIS_RIPS_POINTS as f64),
            1.0,
        ));
        let new_count = (count_f.round() as usize).clamp(3, MAX_VIETORIS_RIPS_POINTS);
        if resp.changed() && new_count != desc.count {
            desc.count = new_count;
            tool.apply_generator(desc);
        }
        if desc.preset != ShapePreset::UniformRandom {
            let prev = desc.noise;
            ui.add(egui::Slider::new(&mut desc.noise, 0.0..=0.5).text("Noise"));
            if (desc.noise - prev).abs() > 1e-9 {
                tool.apply_generator(desc);
            }
        }
        if ui.button("Reseed").clicked() {
            desc.seed = desc.seed.wrapping_add(1);
            tool.apply_generator(desc);
        }
    } else if ui.button("Reset to Annulus").clicked() {
        tool.apply_generator(GeneratorDescriptor::default());
    }

    if ui.button("Import CSV / JSON…").clicked() {
        tool.io_tasks.trigger_point_cloud_open();
    }
}

fn render_nd_projection_bar(tool: &mut VietorisRipsTool, ui: &mut egui::Ui) {
    let dim = tool.ambient_dimension();
    let (mut ax_x, mut ax_y) = tool.projection_axes();
    let prev = (ax_x, ax_y);
    ui.horizontal_wrapped(|ui| {
        ui.label(format!(
            "Ambient space: ℝ^{dim} (distances & persistence in {dim}D)"
        ));
        egui::ComboBox::from_id_salt("vr_nd_axis_x")
            .selected_text(format!("X: Axis {ax_x}"))
            .show_ui(ui, |ui| {
                for axis in 0..dim {
                    ui.selectable_value(&mut ax_x, axis, format!("Axis {axis}"));
                }
            });
        egui::ComboBox::from_id_salt("vr_nd_axis_y")
            .selected_text(format!("Y: Axis {ax_y}"))
            .show_ui(ui, |ui| {
                for axis in 0..dim {
                    ui.selectable_value(&mut ax_y, axis, format!("Axis {axis}"));
                }
            });
    });
    if (ax_x, ax_y) != prev {
        tool.set_projection_axes(ax_x, ax_y);
    }
}

fn render_filtration_toolbar(tool: &mut VietorisRipsTool, ui: &mut egui::Ui) {
    ui.horizontal_wrapped(|ui| {
        let max_r = tool.max_slider_radius;
        let mut eps = tool.radius;
        ui.add(egui::Slider::new(&mut eps, 0.0..=max_r).text("Scale (ε)"));
        if (eps - tool.radius).abs() > 1e-12 {
            tool.set_radius(eps);
        }
        ui.separator();
        if ui
            .selectable_label(tool.plot_mode == PersistencePlotMode::Diagram, "Diagram")
            .clicked()
        {
            tool.set_plot_mode(PersistencePlotMode::Diagram);
        }
        if ui
            .selectable_label(tool.plot_mode == PersistencePlotMode::Barcode, "Barcode")
            .clicked()
        {
            tool.set_plot_mode(PersistencePlotMode::Barcode);
        }
        ui.label("Min Δ:");
        let mut min_p = tool.min_persistence;
        if ui
            .add(bounded_drag_value(&mut min_p, 0.0..=max_r, 0.02))
            .changed()
        {
            tool.set_min_persistence(min_p);
        }
        if ui.button("Download .csv").clicked() {
            tool.io_tasks.trigger_intervals_csv_save(&tool.persistence);
        }
        if ui.button("Download .json").clicked() {
            if let Err(err) = tool.io_tasks.trigger_intervals_json_save(&tool.persistence) {
                tool.error_msg = Some(err);
            }
        }
    });
}

fn render_topology_readouts(tool: &VietorisRipsTool, ui: &mut egui::Ui, palette: &SemanticPalette) {
    let (beta0, beta1) = tool.betti;
    let total_h1 = tool
        .persistence
        .intervals
        .iter()
        .filter(|iv| iv.dimension == 1 && iv.persistence() >= tool.min_persistence)
        .count();
    ui.horizontal_wrapped(|ui| {
        badge_pill(
            ui,
            format!("β₀ (Components): {beta0}"),
            palette.series_color(0),
        );
        badge_pill(
            ui,
            format!("β₁ (1-Cycles): {beta1}"),
            palette.series_color(1),
        );
        ui.separator();
        ui.label(format!("Vertices: {}", tool.complex.count_simplices(0)));
        ui.label(format!("Edges: {}", tool.complex.count_simplices(1)));
        ui.label(format!("Triangles: {}", tool.complex.count_simplices(2)));
        ui.label(format!("H₁ Bars: {total_h1}"));
        tool.render_chaos_invariant_badges(ui, palette);
    });
}

pub(super) fn badge_pill(ui: &mut egui::Ui, text: String, color: egui::Color32) {
    let bg = egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 36);
    egui::Frame::NONE
        .fill(bg)
        .stroke(egui::Stroke::new(1.0_f32, color))
        .corner_radius(egui::CornerRadius::same(6))
        .inner_margin(egui::Margin::symmetric(8, 3))
        .show(ui, |ui| {
            ui.colored_label(color, egui::RichText::new(text).strong());
        });
}

fn render_banners(tool: &VietorisRipsTool, ui: &mut egui::Ui, palette: &SemanticPalette) {
    if tool.is_triangle_truncated() {
        ui.colored_label(
            palette.warning,
            format!("Dense filtration capped at {MAX_FILTRATION_TRIANGLES} 2-simplices to preserve interactive responsiveness."),
        );
    }
    if tool.exceeds_url_bounds {
        ui.colored_label(
            palette.info,
            "Custom cloud exceeds the 4 KiB URL bookmark budget; use Preset (.json) or Session (.mws) export to save all points.",
        );
    }
    if let Some(err) = &tool.error_msg {
        ui.colored_label(palette.error, err);
    } else if let Some(msg) = &tool.status_msg {
        ui.colored_label(palette.success, msg);
    }
}

fn render_dual_plots(tool: &mut VietorisRipsTool, ui: &mut egui::Ui, palette: &SemanticPalette) {
    let avail = ui.available_size();
    let plot_h = (avail.y - 8.0).max(220.0);
    if avail.x >= 620.0 {
        ui.columns(2, |cols| {
            render_active_complex_viewport(tool, &mut cols[0], palette, plot_h);
            render_persistence_plot(tool, &mut cols[1], palette, plot_h);
        });
    } else {
        let stacked_h = (plot_h * 0.5).max(180.0);
        render_active_complex_viewport(tool, ui, palette, stacked_h);
        render_persistence_plot(tool, ui, palette, stacked_h);
    }
}

fn render_active_complex_viewport(
    tool: &mut VietorisRipsTool,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
    height: f32,
) {
    if tool.mode.is_3d() {
        tool.draw_3d_simplex_viewport(ui, palette, height);
    } else {
        render_complex_plot_2d(tool, ui, palette, height);
    }
}

fn render_complex_plot_2d(
    tool: &mut VietorisRipsTool,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
    height: f32,
) {
    let s0 = palette.series_color(0);
    let tri_fill = egui::Color32::from_rgba_unmultiplied(s0.r(), s0.g(), s0.b(), 55);
    let allow_pan = tool.brush == SculptBrush::OrbitOrPan;
    let resp = Plot::new("vietoris_rips_plot")
        .height(height)
        .data_aspect(1.0)
        .allow_drag(allow_pan)
        .allow_zoom(true)
        .show(ui, |plot_ui| {
            let cursor = tool.handle_canvas_sculpt(plot_ui, false);
            for t in &tool.cached_triangles {
                plot_ui.polygon(
                    Polygon::new("2-Simplex", &t[..])
                        .fill_color(tri_fill)
                        .stroke(egui::Stroke::new(0.0_f32, egui::Color32::TRANSPARENT)),
                );
            }
            for e in &tool.cached_edges {
                plot_ui.line(
                    Line::new("1-Simplex", &e[..])
                        .color(palette.graph_edge)
                        .width(1.5_f32),
                );
            }
            plot_ui.points(
                Points::new("Vertices", &tool.cached_vertices[..])
                    .color(s0)
                    .radius(4.0_f32),
            );
            if let Some(xy) = cursor {
                super::ui_3d::draw_brush_ring(
                    plot_ui,
                    xy,
                    tool.brush_radius,
                    palette.series_color(2),
                );
            }
        })
        .response;
    resp.accessible_theory_hover(tool);
}

fn render_persistence_plot(
    tool: &VietorisRipsTool,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
    height: f32,
) {
    match tool.plot_mode {
        PersistencePlotMode::Diagram => render_diagram_plot(tool, ui, palette, height),
        PersistencePlotMode::Barcode => render_barcode_plot(tool, ui, palette, height),
    }
}

pub(super) fn visible_intervals(tool: &VietorisRipsTool) -> Vec<&RipsInterval> {
    tool.persistence
        .intervals
        .iter()
        .filter(|iv| iv.is_essential || iv.persistence() >= tool.min_persistence)
        .collect()
}

fn render_diagram_plot(
    tool: &VietorisRipsTool,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
    height: f32,
) {
    let eps = tool.radius;
    let top = tool.max_slider_radius.max(eps).max(1.0);
    let (mut h0_act, mut h0_inact, mut h1_act, mut h1_inact) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for iv in visible_intervals(tool) {
        let pt = [iv.birth, iv.death.max(iv.birth)];
        match (iv.dimension, iv.is_alive_at(eps)) {
            (0, true) => h0_act.push(pt),
            (0, false) => h0_inact.push(pt),
            (1, true) => h1_act.push(pt),
            _ => h1_inact.push(pt),
        }
    }

    Plot::new("vietoris_rips_diagram_plot")
        .height(height)
        .data_aspect(1.0)
        .x_axis_label("Birth (ε)")
        .y_axis_label("Death (ε)")
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new("y = x", vec![[0.0, 0.0], [top, top]])
                    .color(palette.muted_text)
                    .style(LineStyle::dashed_dense())
                    .width(1.2_f32),
            );
            plot_ui.line(
                Line::new(
                    "Active ε Quadrant",
                    vec![[0.0, eps], [eps, eps], [eps, top]],
                )
                .color(palette.accent)
                .width(1.6_f32),
            );
            plot_ui.points(
                Points::new("H₀ (inactive)", h0_inact)
                    .color(palette.muted_text)
                    .radius(3.2_f32),
            );
            plot_ui.points(
                Points::new("H₁ (inactive)", h1_inact)
                    .color(palette.graph_edge)
                    .radius(3.8_f32),
            );
            plot_ui.points(
                Points::new("H₀ (active)", h0_act)
                    .color(palette.series_color(0))
                    .radius(5.5_f32),
            );
            plot_ui.points(
                Points::new("H₁ (active)", h1_act)
                    .color(palette.series_color(1))
                    .radius(6.2_f32),
            );
        });
}

fn render_barcode_plot(
    tool: &VietorisRipsTool,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
    height: f32,
) {
    let eps = tool.radius;
    let intervals = visible_intervals(tool);
    let bar_top = (intervals.len().max(1) as f64) + 1.0;

    Plot::new("vietoris_rips_barcode_plot")
        .height(height)
        .x_axis_label("Filtration Scale (ε)")
        .y_axis_label("Homology Feature Index")
        .show(ui, |plot_ui| {
            for (idx, iv) in intervals.iter().enumerate() {
                let y = (idx + 1) as f64;
                let active = iv.is_alive_at(eps);
                let base = palette.series_color(iv.dimension.min(1));
                let color = if active {
                    base
                } else {
                    egui::Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), 110)
                };
                let label = if iv.dimension == 0 {
                    "H₀ Bar"
                } else {
                    "H₁ Bar"
                };
                plot_ui.line(
                    Line::new(label, vec![[iv.birth, y], [iv.death.max(iv.birth), y]])
                        .color(color)
                        .width(if active { 3.2_f32 } else { 1.6_f32 }),
                );
            }
            plot_ui.line(
                Line::new("ε Threshold", vec![[eps, 0.0], [eps, bar_top]])
                    .color(palette.accent)
                    .width(2.0_f32),
            );
        });
}

// [cite:graph_parameters_rust]

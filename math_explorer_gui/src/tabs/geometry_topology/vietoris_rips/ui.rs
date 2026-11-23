#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::handoff::{poll_live_attractor_landmarks, take_chaos_landmark_handoff};
use super::state::{SculptBrush, VrSourceMode, MAX_VR_RADIUS};
use super::{VietorisRipsTool, DEFAULT_2D_POINTS, MAX_VIETORIS_RIPS_POINTS};
use crate::accessibility::AccessibleTheoryHover;
use crate::app::SemanticPalette;
use crate::framework::Plot3DViewport;
use eframe::egui;
use egui_plot::{Line, Plot, Points, Polygon};
use rand::Rng;

impl VietorisRipsTool {
    pub(super) fn render_tool_ui(&mut self, ui: &mut egui::Ui) {
        self.poll_incoming_landmarks(ui.ctx());
        self.draw_side_controls(ui);
        self.draw_main_viewports(ui);
    }

    fn poll_incoming_landmarks(&mut self, ctx: &egui::Context) {
        if let Some(handoff) = take_chaos_landmark_handoff() {
            let _ = self.ingest_chaos_handoff(handoff);
        } else if self.mode == VrSourceMode::Lorenz3D && self.chaos.playing {
            if let Some((seq, live)) = poll_live_attractor_landmarks(self.chaos.live_stream_seq) {
                self.chaos.live_stream_seq = seq;
                let _ = self.ingest_live_stream_landmarks(live);
            } else {
                self.chaos.advance_live_steps(4);
                let _ = self.rebuild_chaos_landmarks();
            }
            ctx.request_repaint();
        } else if self.mode == VrSourceMode::Takens3D && self.chaos.playing {
            self.chaos.advance_live_steps(4);
            let _ = self.rebuild_chaos_landmarks();
            ctx.request_repaint();
        }
    }

    fn draw_side_controls(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "vietoris_rips_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Vietoris-Rips Homology");
            ui.separator();
            self.draw_source_and_filtration_section(ui);
            if matches!(self.mode, VrSourceMode::Lorenz3D | VrSourceMode::Takens3D) {
                ui.separator();
                self.draw_chaos_and_takens_section(ui);
            }
            ui.separator();
            self.draw_sculpt_brush_section(ui);
            if self.mode.is_3d() {
                ui.separator();
                egui::CollapsingHeader::new("3D Camera")
                    .default_open(true)
                    .show(ui, |ui| self.camera.ui(ui));
            }
            ui.separator();
            self.draw_invariants_readout(ui);
        });
    }

    fn draw_source_and_filtration_section(&mut self, ui: &mut egui::Ui) {
        ui.label("Point Cloud Source");
        let prev_mode = self.mode;
        egui::ComboBox::from_id_salt("vr_source_mode")
            .selected_text(match self.mode {
                VrSourceMode::Points2D => "2D Point Cloud",
                VrSourceMode::Points3D => "3D Sculpted Landmarks",
                VrSourceMode::Lorenz3D => "Live 3D Lorenz Attractor",
                VrSourceMode::Takens3D => "Takens 3D Delay Embedding",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.mode, VrSourceMode::Points2D, "2D Point Cloud");
                ui.selectable_value(
                    &mut self.mode,
                    VrSourceMode::Points3D,
                    "3D Sculpted Landmarks",
                );
                ui.selectable_value(
                    &mut self.mode,
                    VrSourceMode::Lorenz3D,
                    "Live 3D Lorenz Attractor",
                );
                ui.selectable_value(
                    &mut self.mode,
                    VrSourceMode::Takens3D,
                    "Takens 3D Delay Embedding",
                );
            });
        if self.mode != prev_mode {
            self.on_source_mode_switched();
        }

        let max_r = if self.mode.is_3d() {
            MAX_VR_RADIUS
        } else {
            15.0
        };
        let prev_radius = self.radius;
        ui.add(egui::Slider::new(&mut self.radius, 0.0..=max_r).text("Radius (ε)"));
        if (self.radius - prev_radius).abs() > f64::EPSILON {
            self.update_complex();
        }

        if self.mode == VrSourceMode::Points2D && ui.button("Generate Random Points").clicked() {
            let mut rng = oxidize_core::rng::OxidizeRng::new(self.revision);
            self.points.clear();
            for _ in 0..DEFAULT_2D_POINTS {
                self.points
                    .push(math_explorer::pure_math::statistics::tda::Point2D::new(
                        rng.gen_range(0.0..10.0),
                        rng.gen_range(0.0..10.0),
                    ));
            }
            self.update_complex();
        }
    }

    fn on_source_mode_switched(&mut self) {
        self.active_drag_vertex = None;
        match self.mode {
            VrSourceMode::Points2D => {
                self.radius = self.radius.clamp(0.0, 15.0).min(2.2);
                if self.points.is_empty() {
                    self.generate_points(DEFAULT_2D_POINTS);
                } else {
                    self.update_complex();
                }
            }
            VrSourceMode::Points3D => {
                if self.points_3d.is_empty() {
                    let _ = self.rebuild_chaos_landmarks();
                    self.mode = VrSourceMode::Points3D;
                } else {
                    self.update_view_center_3d();
                    self.update_complex();
                }
            }
            VrSourceMode::Lorenz3D => {
                self.radius = 7.5;
                let _ = self.rebuild_chaos_landmarks();
            }
            VrSourceMode::Takens3D => {
                self.radius = 7.0;
                let _ = self.rebuild_chaos_landmarks();
            }
        }
    }

    fn draw_chaos_and_takens_section(&mut self, ui: &mut egui::Ui) {
        ui.label("Strange Attractor & Delay Controls");
        ui.horizontal(|ui| {
            let play_label = if self.chaos.playing {
                "⏸ Pause"
            } else {
                "▶ Stream"
            };
            if ui.button(play_label).clicked() {
                self.chaos.playing = !self.chaos.playing;
            }
            if ui.button("⏭ Step").clicked() {
                self.chaos.playing = false;
                self.chaos.advance_live_steps(8);
                let _ = self.rebuild_chaos_landmarks();
            }
        });

        let mut changed = false;
        changed |= ui
            .add(egui::Slider::new(&mut self.chaos.rho, 0.0..=100.0).text("Rayleigh (ρ)"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.chaos.sigma, 0.0..=50.0).text("Prandtl (σ)"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.chaos.beta, 0.1..=10.0).text("Geometric (β)"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.chaos.landmark_count, 8..=MAX_VIETORIS_RIPS_POINTS)
                    .text("Landmarks"),
            )
            .changed();
        if self.mode == VrSourceMode::Takens3D {
            let lag_changed = ui
                .add(egui::Slider::new(&mut self.chaos.takens_lag, 1..=40).text("Takens Lag (τ)"))
                .changed();
            if lag_changed && !changed {
                let _ = self.rebuild_chaos_landmarks();
            }
        }
        if changed {
            self.chaos.regenerate_trajectory();
            let _ = self.rebuild_chaos_landmarks();
        }
    }

    fn draw_sculpt_brush_section(&mut self, ui: &mut egui::Ui) {
        ui.label("Tactile Landmark Brush");
        ui.horizontal_wrapped(|ui| {
            let orbit_lbl = if self.mode.is_3d() {
                "🌐 Orbit"
            } else {
                "✋ Pan"
            };
            ui.selectable_value(&mut self.brush, SculptBrush::OrbitOrPan, orbit_lbl);
            ui.selectable_value(&mut self.brush, SculptBrush::Move, "✥ Move");
            ui.selectable_value(&mut self.brush, SculptBrush::Add, "＋ Add");
            ui.selectable_value(&mut self.brush, SculptBrush::Erase, "✕ Erase");
        });
        if self.brush != SculptBrush::OrbitOrPan {
            ui.add(egui::Slider::new(&mut self.brush_radius, 0.2..=4.0).text("Brush Size"));
        }
    }

    fn draw_invariants_readout(&self, ui: &mut egui::Ui) {
        ui.label("Homology & Chaos Invariants");
        ui.label(format!("β₀ (Components): {}", self.beta_0));
        ui.label(format!("β₁ (1-Cycles / Loops): {}", self.beta_1));
        ui.label(format!(
            "Simplices: {} V | {} E | {} T",
            self.complex.count_simplices(0),
            self.complex.count_simplices(1),
            self.complex.count_simplices(2),
        ));
        #[cfg(feature = "physics")]
        if self.mode.is_3d() {
            if let Some(lyap) = self.chaos.lyapunov_exponent() {
                ui.label(format!("Lorenz Lyapunov λ₁: {lyap:+.3}"));
            }
            if let Some(corr) = self.chaos.correlation_sum() {
                let d2 = self.chaos.correlation_dimension_estimate().unwrap_or(corr);
                ui.label(format!("Correlation C(ε): {corr:.3} (D₂ ≈ {d2:.2})"));
            }
        }
    }

    fn draw_main_viewports(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let palette = SemanticPalette::from_visuals(ui.visuals());
            if let Some(err) = &self.error_msg {
                ui.colored_label(palette.error, err);
            }
            let total_h = ui.available_height().max(260.0);
            let main_h = (total_h * 0.64).max(160.0);
            let barcode_h = (total_h - main_h - 28.0).max(90.0);

            ui.allocate_ui(egui::Vec2::new(ui.available_width(), main_h), |ui| {
                if self.mode.is_3d() {
                    self.draw_3d_simplex_viewport(ui, &palette);
                } else {
                    self.draw_2d_simplex_viewport(ui, &palette);
                }
            });
            ui.separator();
            self.draw_persistence_barcode_chart(ui, &palette, barcode_h);
        });
    }

    fn draw_2d_simplex_viewport(&mut self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let allow_pan = self.brush == SculptBrush::OrbitOrPan;
        let plot = Plot::new("vietoris_rips_plot")
            .view_aspect(1.0)
            .data_aspect(1.0)
            .allow_drag(allow_pan)
            .allow_zoom(true);
        let resp = plot
            .show(ui, |plot_ui| {
                let cursor = self.handle_canvas_sculpt(plot_ui, false);
                let s0 = palette.series_color(0);
                let tri_fill = egui::Color32::from_rgba_unmultiplied(s0.r(), s0.g(), s0.b(), 55);
                for t in &self.cached_triangles {
                    plot_ui.polygon(
                        Polygon::new("Triangle", &t[..])
                            .fill_color(tri_fill)
                            .stroke(egui::Stroke::new(0.0_f32, egui::Color32::TRANSPARENT)),
                    );
                }
                for e in &self.cached_edges {
                    plot_ui.line(
                        Line::new("Edge", &e[..])
                            .color(palette.graph_edge)
                            .width(1.5_f32),
                    );
                }
                plot_ui.points(
                    Points::new("Vertices", &self.cached_vertices[..])
                        .color(s0)
                        .radius(4.5_f32),
                );
                if let Some(xy) = cursor {
                    draw_brush_ring(plot_ui, xy, self.brush_radius, palette.series_color(2));
                }
            })
            .response;
        resp.accessible_theory_hover(self);
    }

    fn draw_3d_simplex_viewport(&mut self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let mut camera = self.camera;
        let resp = Plot3DViewport::new("vietoris_rips_3d_plot", 28.0)
            .show(ui, &mut camera, |plot_ui| {
                let cursor = self.handle_canvas_sculpt(plot_ui, true);
                self.render_projected_3d_simplices(plot_ui, palette);
                if let Some(xy) = cursor {
                    draw_brush_ring(
                        plot_ui,
                        xy,
                        self.brush_radius * 2.5,
                        palette.series_color(2),
                    );
                }
            })
            .response;
        self.camera = camera;
        resp.accessible_theory_hover(self);
    }

    fn render_projected_3d_simplices(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: &SemanticPalette,
    ) {
        let s0 = palette.series_color(0);
        let tri_fill = egui::Color32::from_rgba_unmultiplied(s0.r(), s0.g(), s0.b(), 48);
        let mut sorted_tris: Vec<(f64, [[f64; 2]; 3])> = self
            .cached_triangles_3d
            .iter()
            .map(|&[a, b, c]| {
                let (pa, da) = self.project_3d_with_depth(a);
                let (pb, db) = self.project_3d_with_depth(b);
                let (pc, dc) = self.project_3d_with_depth(c);
                ((da + db + dc) / 3.0, [pa, pb, pc])
            })
            .collect();
        sorted_tris.sort_unstable_by(|x, y| x.0.total_cmp(&y.0));
        for (_, [pa, pb, pc]) in sorted_tris {
            plot_ui.polygon(
                Polygon::new("3D Triangle", vec![pa, pb, pc])
                    .fill_color(tri_fill)
                    .stroke(egui::Stroke::new(0.0_f32, egui::Color32::TRANSPARENT)),
            );
        }
        for &[a, b] in &self.cached_edges_3d {
            plot_ui.line(
                Line::new("3D Edge", vec![self.project_3d(a), self.project_3d(b)])
                    .color(palette.graph_edge)
                    .width(1.4_f32),
            );
        }
        let verts: Vec<[f64; 2]> = self.points_3d.iter().map(|&p| self.project_3d(p)).collect();
        plot_ui.points(Points::new("3D Landmarks", verts).color(s0).radius(4.2_f32));
    }

    fn draw_persistence_barcode_chart(
        &self,
        ui: &mut egui::Ui,
        palette: &SemanticPalette,
        height: f32,
    ) {
        ui.label(format!(
            "24-Step Persistence Barcode (β₀ blue, β₁ loops accent | active ε = {:.2})",
            self.radius
        ));
        let max_x = self
            .filtration_radii
            .last()
            .copied()
            .unwrap_or(15.0)
            .max(1.0);
        Plot::new("vietoris_rips_barcode")
            .height(height)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .show_axes([true, false])
            .default_x_bounds(0.0, max_x)
            .show(ui, |plot_ui| {
                let c0 = palette.series_color(0);
                let c1 = palette.series_color(1);
                let mut row = 0.0_f64;
                for iv in self.barcode_intervals.iter().filter(|iv| iv.dimension == 1) {
                    plot_ui.line(
                        Line::new("β₁ Interval", vec![[iv.birth, row], [iv.death, row]])
                            .color(c1)
                            .width(3.2_f32),
                    );
                    row -= 1.0;
                }
                for iv in self
                    .barcode_intervals
                    .iter()
                    .filter(|iv| iv.dimension == 0)
                    .take(16)
                {
                    plot_ui.line(
                        Line::new("β₀ Interval", vec![[iv.birth, row], [iv.death, row]])
                            .color(c0)
                            .width(1.6_f32),
                    );
                    row -= 0.7;
                }
                let y_min = (row - 1.0).min(-2.0);
                plot_ui.line(
                    Line::new("Active ε", vec![[self.radius, 1.0], [self.radius, y_min]])
                        .color(palette.series_color(3))
                        .width(1.8_f32),
                );
            });
    }
}

fn draw_brush_ring(
    plot_ui: &mut egui_plot::PlotUi<'_>,
    center: [f64; 2],
    radius: f64,
    color: egui::Color32,
) {
    let ring: Vec<[f64; 2]> = (0..=24)
        .map(|i| {
            let a = std::f64::consts::TAU * (i as f64 / 24.0);
            [center[0] + radius * a.cos(), center[1] + radius * a.sin()]
        })
        .collect();
    plot_ui.line(Line::new("Sculpt Brush", ring).color(color).width(1.2_f32));
}

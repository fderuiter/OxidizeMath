//! 3D chaos/Takens controls, live landmark polling, sculpt brush bar, and `Plot3DViewport` renderer.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::handoff::{poll_live_attractor_landmarks, take_chaos_landmark_handoff};
use super::presets::GeneratorDescriptor;
use super::state::{SculptBrush, VrSourceMode};
use super::{CloudSource, VietorisRipsTool, MAX_CHAOS_LANDMARKS};
use crate::accessibility::AccessibleTheoryHover;
use crate::app::SemanticPalette;
use crate::framework::Plot3DViewport;
use eframe::egui;
use egui_plot::{Line, Points, Polygon};

impl VietorisRipsTool {
    pub(super) fn poll_incoming_landmarks(&mut self, ctx: &egui::Context) {
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

    pub(super) fn render_source_mode_combo(&mut self, ui: &mut egui::Ui) {
        let prev_mode = self.mode;
        egui::ComboBox::from_label("Source")
            .selected_text(match self.mode {
                VrSourceMode::Points2D => "2D / N-D Cloud",
                VrSourceMode::Points3D => "3D Landmarks",
                VrSourceMode::Lorenz3D => "3D Lorenz Attractor",
                VrSourceMode::Takens3D => "Takens 3D Delay",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.mode, VrSourceMode::Points2D, "2D / N-D Cloud");
                ui.selectable_value(&mut self.mode, VrSourceMode::Points3D, "3D Landmarks");
                ui.selectable_value(
                    &mut self.mode,
                    VrSourceMode::Lorenz3D,
                    "3D Lorenz Attractor",
                );
                ui.selectable_value(&mut self.mode, VrSourceMode::Takens3D, "Takens 3D Delay");
            });
        if self.mode != prev_mode {
            self.on_source_mode_switched();
        }
    }

    fn on_source_mode_switched(&mut self) {
        self.active_drag_vertex = None;
        match self.mode {
            VrSourceMode::Points2D => {
                if self.points.is_empty() {
                    self.apply_generator(GeneratorDescriptor::default());
                } else {
                    let src = self.source.clone();
                    let pts = self.points.clone();
                    let r = self.radius.clamp(0.0, 15.0).min(2.2);
                    let _ = self.rebuild_from_2d(src, pts, r);
                }
            }
            VrSourceMode::Points3D => {
                if self.points_3d.is_empty() {
                    let _ = self.rebuild_chaos_landmarks();
                    self.mode = VrSourceMode::Points3D;
                } else {
                    let pts = self.points_3d.clone();
                    let _ = self.load_validated_3d_points(pts, self.radius);
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

    pub(super) fn render_chaos_toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
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
            let mut changed = false;
            changed |= ui
                .add(egui::Slider::new(&mut self.chaos.rho, 0.0..=100.0).text("ρ"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.chaos.sigma, 0.0..=50.0).text("σ"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.chaos.beta, 0.1..=10.0).text("β"))
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.chaos.landmark_count, 8..=MAX_CHAOS_LANDMARKS)
                        .text("Landmarks"),
                )
                .changed();
            if self.mode == VrSourceMode::Takens3D {
                let lag_changed = ui
                    .add(egui::Slider::new(&mut self.chaos.takens_lag, 1..=40).text("Lag τ"))
                    .changed();
                if lag_changed && !changed {
                    let _ = self.rebuild_chaos_landmarks();
                }
            }
            if changed {
                self.chaos.regenerate_trajectory();
                let _ = self.rebuild_chaos_landmarks();
            }
        });
    }

    pub(super) fn render_sculpt_brush_controls(&mut self, ui: &mut egui::Ui) {
        if !self.mode.is_3d() && matches!(self.source, CloudSource::CustomNd { .. }) {
            return;
        }
        ui.separator();
        let orbit_lbl = if self.mode.is_3d() {
            "🌐 Orbit"
        } else {
            "✋ Pan"
        };
        ui.selectable_value(&mut self.brush, SculptBrush::OrbitOrPan, orbit_lbl);
        ui.selectable_value(&mut self.brush, SculptBrush::Move, "✥ Move");
        ui.selectable_value(&mut self.brush, SculptBrush::Add, "＋ Add");
        ui.selectable_value(&mut self.brush, SculptBrush::Erase, "✕ Erase");
        if self.brush != SculptBrush::OrbitOrPan {
            ui.add(egui::Slider::new(&mut self.brush_radius, 0.2..=4.0).text("Brush"));
        }
    }

    pub(super) fn render_chaos_invariant_badges(
        &self,
        ui: &mut egui::Ui,
        palette: &SemanticPalette,
    ) {
        if self.mode.is_3d() {
            if let Some(lyap) = self.chaos.lyapunov_exponent() {
                super::plots::badge_pill(ui, format!("λ₁: {lyap:+.3}"), palette.series_color(2));
            }
            if let Some(corr) = self.chaos.correlation_sum() {
                let d2 = self.chaos.correlation_dimension_estimate().unwrap_or(corr);
                super::plots::badge_pill(
                    ui,
                    format!("C(ε): {corr:.3} (D₂ ≈ {d2:.2})"),
                    palette.series_color(3),
                );
            }
        }
        #[cfg(not(feature = "physics"))]
        let _ = (ui, palette);
    }

    pub(super) fn draw_3d_simplex_viewport(
        &mut self,
        ui: &mut egui::Ui,
        palette: &SemanticPalette,
        height: f32,
    ) {
        let mut camera = self.camera;
        let mut resp_opt = None;
        ui.allocate_ui(egui::Vec2::new(ui.available_width(), height), |ui| {
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
            resp_opt = Some(resp);
        });
        if self.camera != camera {
            self.camera = camera;
            self.refresh_projected_3d_export_geometry();
        }
        if let Some(resp) = resp_opt {
            resp.accessible_theory_hover(self);
        }
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
}

pub(super) fn draw_brush_ring(
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

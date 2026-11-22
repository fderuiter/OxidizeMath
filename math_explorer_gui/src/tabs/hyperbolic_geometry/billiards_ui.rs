#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::billiards_tool::HyperbolicBilliardsTool;
use crate::accessibility::{
    AccessibleHoverText, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility,
};
use crate::app::SemanticPalette;
use crate::framework::{
    PaneSidePanel, PlaybackAction, PlaybackTransport, scoped_panel_id,
};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::num_complex::Complex64;
use math_explorer::pure_math::differential_geometry::hyperbolic::{
    HyperbolicModel, project_disk_point, unproject_disk_point,
};

const WALL_ARC_PTS: usize = 17;

impl HyperbolicBilliardsTool {
    pub(crate) fn render_billiards_tool(&mut self, ui: &mut egui::Ui) {
        if self.is_playing {
            for _ in 0..self.steps_per_frame.clamp(1, 12) {
                self.system.step(self.step_ds);
            }
            ui.ctx().request_repaint();
        }

        let panel_id = scoped_panel_id(ui, "hyperbolic_billiards_controls");
        PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.render_billiards_controls(ui);
            });
        });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.columns(2, |cols| {
                self.render_billiard_disk_column(&mut cols[0]);
                self.render_lyapunov_plot_column(&mut cols[1]);
            });
        });
    }

    fn render_billiards_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Playback & Table");
        if let Some(action) = PlaybackTransport::in_thread(
            "hyperbolic_billiards_transport",
            "Hyperbolic Geodesic Billiards",
            self.is_playing,
        )
        .show(ui)
        {
            self.handle_playback_action(action);
        }

        ui.separator();
        ui.label("Disk Projection:");
        ui.horizontal_wrapped(|ui| {
            let p_sel = ui
                .selectable_value(&mut self.model, HyperbolicModel::Poincare, "Poincaré Disk")
                .accessible_hover_text("Conformal disk where geodesic billiard walls curve inward.")
                .changed();
            let k_sel = ui
                .selectable_value(&mut self.model, HyperbolicModel::Klein, "Beltrami-Klein")
                .accessible_hover_text("Projective disk where geodesic walls and rays are straight chords.")
                .changed();
            if p_sel || k_sel {
                self.revision = self.revision.wrapping_add(1);
            }
        });

        ui.label("Table Schläfli {p, q}:");
        ui.horizontal_wrapped(|ui| {
            for (tp, tq, label) in [
                (7, 3, "{7,3}"),
                (5, 4, "{5,4}"),
                (6, 4, "{6,4}"),
                (8, 3, "{8,3}"),
            ] {
                let selected = self.p == tp && self.q == tq;
                if ui.selectable_label(selected, label).clicked() {
                    self.p = tp;
                    self.q = tq;
                    let _ = self.rebuild_system(true);
                }
            }
        });

        ui.separator();
        self.render_launch_sliders(ui);
        ui.separator();
        self.render_lyapunov_metrics(ui);
    }

    fn handle_playback_action(&mut self, action: PlaybackAction) {
        match action {
            PlaybackAction::Play => {
                self.is_playing = true;
            }
            PlaybackAction::Pause => {
                self.is_playing = false;
            }
            PlaybackAction::Step => {
                self.is_playing = false;
                self.system.step(self.step_ds);
                self.revision = self.revision.wrapping_add(1);
            }
            PlaybackAction::Reset => {
                self.is_playing = false;
                self.system.reset_particles();
                self.revision = self.revision.wrapping_add(1);
            }
        }
    }

    fn render_launch_sliders(&mut self, ui: &mut egui::Ui) {
        ui.heading("Twin-Particle Perturbation");
        let mut changed = false;
        changed |= ui
            .add(
                egui::Slider::new(&mut self.log10_epsilon, -9.0..=-2.0)
                    .text("log10(epsilon)"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(
                    &mut self.initial_angle,
                    -std::f64::consts::PI..=std::f64::consts::PI,
                )
                .text("Launch Angle (rad)"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.step_ds, 0.01..=0.12).text("Arc Step ds"))
            .changed();
        ui.add(egui::Slider::new(&mut self.steps_per_frame, 1..=8).text("Steps / Frame"));
        if changed {
            let _ = self.rebuild_system(!self.is_playing);
        }
    }

    fn render_lyapunov_metrics(&self, ui: &mut egui::Ui) {
        let sample = self.system.latest_sample();
        ui.heading("Lyapunov Diagnostics");
        render_copyable_metric(ui, "Perturbation eps", &format!("{:.2e}", self.epsilon()));
        render_copyable_metric(
            ui,
            "Wall Reflections",
            &format!("{} bounces", self.system.primary_bounces()),
        );
        render_copyable_metric(
            ui,
            "Path Length s",
            &format!("{:.3}", sample.path_length),
        );
        render_copyable_metric(
            ui,
            "Separation d_H(z1, z2)",
            &format!("{:.3e}", sample.separation),
        );
        render_copyable_metric(
            ui,
            "ln(d_H / eps)",
            &format!("{:.4}", sample.log_ratio),
        );
        render_copyable_metric(ui, "FTLE lambda(s)", &format!("{:.4}", sample.ftle));
    }

    fn render_billiard_disk_column(&mut self, ui: &mut egui::Ui) {
        ui.heading("Geodesic Billiard Table");
        let avail = ui.available_size();
        let canvas_size = egui::vec2(avail.x.max(160.0), avail.y.max(220.0));
        let (response, painter) = ui.allocate_painter(canvas_size, egui::Sense::click_and_drag());
        self.last_canvas_id = Some(response.id);

        let side = response.rect.width().min(response.rect.height());
        let radius = (0.45 * side).max(24.0);
        let center = response.rect.center();

        self.handle_billiard_canvas_input(ui, &response, center, radius);

        let latest = self.system.latest_sample();
        let summary = format!(
            "{{{},{}}} billiard table, {} bounces, ln(d/eps) = {:.2}",
            self.p,
            self.q,
            self.system.primary_bounces(),
            latest.log_ratio
        );
        register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            "Hyperbolic Geodesic Billiards Canvas",
            Some(&summary),
        );
        lock_canvas_arrow_keys(ui, &response);

        let palette = SemanticPalette::from_visuals(ui.visuals());
        self.paint_billiard_scene(&painter, center, radius, &palette, ui.visuals());
        paint_canvas_focus_ring(ui, &response);
    }

    fn handle_billiard_canvas_input(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        center: egui::Pos2,
        radius: f32,
    ) {
        if response.clicked() || response.dragged() {
            if let Some(pos) = response.interact_pointer_pos() {
                response.request_focus();
                let raw = Complex64::new(
                    f64::from((pos.x - center.x) / radius),
                    f64::from((center.y - pos.y) / radius),
                );
                let mut poincare = unproject_disk_point(raw, self.model);
                let max_r = 0.85 * self.system.inradius_poincare();
                if poincare.norm() > max_r && poincare.norm() > 1e-12 {
                    poincare *= max_r / poincare.norm();
                }
                self.initial_re = poincare.re;
                self.initial_im = poincare.im;
                let _ = self.rebuild_system(!self.is_playing);
            }
        }

        if response.has_focus() {
            let mut d_angle = 0.0;
            ui.input(|i| {
                if i.key_pressed(egui::Key::ArrowLeft) {
                    d_angle -= 0.08;
                }
                if i.key_pressed(egui::Key::ArrowRight) {
                    d_angle += 0.08;
                }
                if i.key_pressed(egui::Key::ArrowUp) {
                    d_angle += 0.02;
                }
                if i.key_pressed(egui::Key::ArrowDown) {
                    d_angle -= 0.02;
                }
            });
            if d_angle != 0.0 {
                self.initial_angle = (self.initial_angle + d_angle)
                    .clamp(-std::f64::consts::PI, std::f64::consts::PI);
                let _ = self.rebuild_system(!self.is_playing);
            }
        }
    }

    fn paint_billiard_scene(
        &self,
        painter: &egui::Painter,
        center: egui::Pos2,
        radius: f32,
        palette: &SemanticPalette,
        visuals: &egui::Visuals,
    ) {
        let to_screen = |z: Complex64| -> egui::Pos2 {
            egui::pos2(
                center.x + (z.re as f32) * radius,
                center.y - (z.im as f32) * radius,
            )
        };
        painter.circle_filled(center, radius, visuals.extreme_bg_color);
        painter.circle_stroke(center, radius, egui::Stroke::new(1.5_f32, palette.graph_edge));

        let mut wall_buf = [Complex64::new(0.0, 0.0); WALL_ARC_PTS];
        for wall in self.system.walls() {
            wall.arc.sample_into(self.model, &mut wall_buf);
            let pts: Vec<egui::Pos2> = wall_buf.iter().map(|&z| to_screen(z)).collect();
            painter.add(egui::Shape::line(
                pts,
                egui::Stroke::new(2.2_f32, palette.accent),
            ));
        }

        let count = self.system.sample_count();
        let mut p1_pts = Vec::with_capacity(count);
        let mut p2_pts = Vec::with_capacity(count);
        for idx in 0..count {
            if let Some((z1, z2, _)) = self.system.sample_at(idx) {
                p1_pts.push(to_screen(project_disk_point(z1, self.model)));
                p2_pts.push(to_screen(project_disk_point(z2, self.model)));
            }
        }
        if p1_pts.len() >= 2 {
            painter.add(egui::Shape::line(
                p1_pts,
                egui::Stroke::new(1.7_f32, palette.info),
            ));
            painter.add(egui::Shape::line(
                p2_pts,
                egui::Stroke::new(1.5_f32, palette.warning),
            ));
        }

        let head1 = to_screen(project_disk_point(self.system.primary().pos, self.model));
        let head2 = to_screen(project_disk_point(self.system.twin().pos, self.model));
        painter.circle_filled(head1, 4.8_f32, palette.info);
        painter.circle_filled(head2, 4.5_f32, palette.error);
    }

    fn render_lyapunov_plot_column(&self, ui: &mut egui::Ui) {
        ui.heading("Logarithmic Geodesic Divergence");
        let datasets = self.lyapunov_plot_datasets();
        let palette = SemanticPalette::from_visuals(ui.visuals());
        Plot::new("hyperbolic_lyapunov_plot")
            .x_axis_label("Hyperbolic Path Length s")
            .y_axis_label("ln(d_H(z1, z2) / eps)")
            .include_y(0.0)
            .show(ui, |plot_ui| {
                if let Some((name, pts)) = datasets.first() {
                    plot_ui.line(
                        Line::new(name.clone(), PlotPoints::new(pts.clone()))
                            .color(palette.warning)
                            .width(2.2_f32),
                    );
                }
                if let Some((name, pts)) = datasets.get(1) {
                    plot_ui.line(
                        Line::new(name.clone(), PlotPoints::new(pts.clone()))
                            .color(palette.graph_edge)
                            .style(egui_plot::LineStyle::Dashed { length: 8.0_f32 }),
                    );
                }
            });
    }
}

// [cite:modular_polynomials_review]

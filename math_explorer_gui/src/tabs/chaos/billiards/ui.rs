//! Side-panel controls and live 2D Birkhoff `(s / L, sin φ)` phase portrait for [`DynamicalBilliardsTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{BilliardGeometryKind, DynamicalBilliardsTool, MAX_RAYS};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::framework::{scoped_panel_id, PaneSidePanel, PlaybackAction, PlaybackTransport};
use eframe::egui;
use egui_plot::{Corner, HLine, Legend, LineStyle, Plot, PlotPoints, Points};
use std::f64::consts::PI;

impl DynamicalBilliardsTool {
    pub(super) fn render_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = scoped_panel_id(parent_ui, "dynamical_billiards_controls");
        PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Dynamical Billiards Lab");
            ui.separator();

            self.render_transport_controls(ui);
            ui.separator();

            self.render_geometry_selector(ui);
            ui.separator();

            self.render_geometry_and_fan_sliders(ui);
            ui.separator();

            self.render_fan_actions(ui);
            ui.separator();

            self.render_invariant_telemetry(ui);
        });
    }

    fn render_transport_controls(&mut self, ui: &mut egui::Ui) {
        let action = PlaybackTransport::in_thread(
            "dynamical_billiards_transport",
            "Dynamical Billiards Lab",
            self.is_playing,
        )
        .show(ui);

        match action {
            Some(PlaybackAction::Play) => self.is_playing = true,
            Some(PlaybackAction::Pause) => self.is_playing = false,
            Some(PlaybackAction::Step) => {
                self.advance_bounces(self.bounces_per_frame.clamp(1, 64));
                self.state_rev = self.state_rev.wrapping_add(1);
            }
            Some(PlaybackAction::Reset) => {
                self.is_playing = false;
                self.rebuild_tables_and_fire_fan();
            }
            Some(PlaybackAction::StepBack | PlaybackAction::ScrubTo(_)) | None => {}
        }
    }

    fn render_geometry_selector(&mut self, ui: &mut egui::Ui) {
        ui.label("Euclidean Table Geometry:");
        let prev = self.geometry;
        egui::ComboBox::from_id_salt("billiards_geometry_selector")
            .selected_text(self.geometry.label())
            .show_ui(ui, |ui| {
                for candidate in [
                    BilliardGeometryKind::Circle,
                    BilliardGeometryKind::Ellipse,
                    BilliardGeometryKind::Stadium,
                    BilliardGeometryKind::Sinai,
                ] {
                    ui.selectable_value(&mut self.geometry, candidate, candidate.label());
                }
            });
        if self.geometry != prev {
            let chosen = self.geometry;
            self.switch_geometry(chosen);
        }
    }

    fn render_geometry_and_fan_sliders(&mut self, ui: &mut egui::Ui) {
        let geom_changed = self.render_geometry_dimension_sliders(ui);
        let fan_changed = self.render_ray_fan_sliders(ui);
        if geom_changed || fan_changed {
            self.rebuild_tables_and_fire_fan();
        }
    }

    fn render_geometry_dimension_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        match self.geometry {
            BilliardGeometryKind::Circle | BilliardGeometryKind::Ellipse => {
                self.render_smooth_billiard_sliders(ui)
            }
            BilliardGeometryKind::Stadium | BilliardGeometryKind::Sinai => {
                self.render_piecewise_billiard_sliders(ui)
            }
        }
    }

    fn render_smooth_billiard_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        if self.geometry == BilliardGeometryKind::Circle {
            return decimal_slider(
                ui,
                &mut self.circle_radius,
                0.4..=2.0,
                "Radius R",
                "Circular billiard radius R",
            );
        }
        let a = decimal_slider(
            ui,
            &mut self.ellipse_a,
            0.8..=2.2,
            "Semi-Major a",
            "Confocal ellipse semi-major axis a",
        );
        let max_b = self.ellipse_a;
        let b = decimal_slider(
            ui,
            &mut self.ellipse_b,
            0.4..=max_b,
            "Semi-Minor b",
            "Confocal ellipse semi-minor axis b ≤ a",
        );
        a || b
    }

    fn render_piecewise_billiard_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        if self.geometry == BilliardGeometryKind::Stadium {
            let r = decimal_slider(
                ui,
                &mut self.stadium_radius,
                0.3..=1.2,
                "Cap Radius R",
                "Bunimovich semicircle cap radius R",
            );
            let a = decimal_slider(
                ui,
                &mut self.stadium_half_straight,
                0.05..=1.5,
                "Half Straight a",
                "Half-length a of flat stadium walls",
            );
            return r || a;
        }
        let side = decimal_slider(
            ui,
            &mut self.sinai_half_side,
            0.8..=1.6,
            "Half Side L",
            "Sinai outer square half-side L",
        );
        let max_r = (self.sinai_half_side * 0.72).min(0.85);
        let disk = decimal_slider(
            ui,
            &mut self.sinai_scatterer_radius,
            0.10..=max_r,
            "Scatterer R",
            "Central dispersing scatterer disk radius R",
        );
        side || disk
    }

    fn render_ray_fan_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut fan_changed = false;
        fan_changed |= ui
            .add(
                egui::Slider::new(&mut self.launch_angle, -PI..=PI)
                    .text("Launch θ (rad)")
                    .max_decimals(3),
            )
            .accessible_hover_text("Central launch direction angle θ")
            .changed();
        fan_changed |= ui
            .add(egui::Slider::new(&mut self.fan_count, 1..=MAX_RAYS).text("Ray Fan Count"))
            .accessible_hover_text("Number of rays launched in angular fan")
            .changed();
        fan_changed |= ui
            .add(
                egui::Slider::new(&mut self.fan_spread, 0.0..=1.0)
                    .text("Fan Spread Δθ")
                    .max_decimals(3),
            )
            .accessible_hover_text("Total angular spread of the ray fan in radians")
            .changed();

        if ui
            .add(egui::Slider::new(&mut self.bounces_per_frame, 1..=64).text("Bounces / Frame"))
            .accessible_hover_text("Specular reflections computed per animation frame")
            .changed()
        {
            self.state_rev = self.state_rev.wrapping_add(1);
        }
        fan_changed
    }

    fn render_fan_actions(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("🎯 Fire Ray Fan")
                .accessible_hover_text("Launch fresh ray fan from current launch reticle")
                .clicked()
            {
                let pos = self.launch_pos;
                if let Ok(n) = self.fire_fan_at(pos) {
                    self.last_status_note = Some(format!(
                        "Fired {n}-ray fan from ({:.3}, {:.3}).",
                        pos[0], pos[1]
                    ));
                }
            }
            if ui
                .button("⚡ +500 Bounces")
                .accessible_hover_text("Advance all active rays by 500 exact specular collisions")
                .clicked()
            {
                self.advance_bounces(500);
                self.state_rev = self.state_rev.wrapping_add(1);
            }
            if ui
                .button("🗑 Clear Rays")
                .accessible_hover_text("Clear all active billiard rays and Birkhoff points")
                .clicked()
            {
                self.clear_rays();
            }
        });
        if let Some(note) = &self.last_status_note {
            let color = if note.starts_with("Cannot") {
                palette.warning
            } else {
                palette.info
            };
            ui.colored_label(color, note);
        }
    }

    fn render_invariant_telemetry(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let total_bounces: usize = self.rays.iter().map(|r| r.bounce_count).sum();
        let total_birkhoff: usize = self.rays.iter().map(|r| r.birkhoff_points.len()).sum();
        ui.label(format!(
            "Active Rays: {} / {}  ({} bounces)",
            self.rays.len(),
            MAX_RAYS,
            total_bounces
        ));
        ui.label(format!(
            "Birkhoff Samples: {}  ({:.2} ms/frame)",
            total_birkhoff, self.last_frame_ms
        ));
        if let Some(lead) = self.rays.last() {
            let speed_err = (lead.ray.speed() - 1.0).abs();
            ui.colored_label(
                palette.success,
                format!("Specular Unit Speed |‖v‖ - 1|: {:.2e}", speed_err),
            );
            if self.geometry == BilliardGeometryKind::Ellipse {
                let prod = self.ellipse.focal_angular_momentum_product(&lead.ray);
                ui.colored_label(
                    palette.info,
                    format!("Berry Focal Product L₁·L₂: {:+.6}", prod),
                );
            }
        }
    }

    pub(super) fn render_central_workspace(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let avail = ui.available_size();
            let wide = avail.x >= 680.0;
            if wide {
                let left_w = (avail.x * 0.50).max(240.0);
                let right_w = (avail.x - left_w - 10.0).max(220.0);
                ui.horizontal(|ui| {
                    self.render_billiard_canvas(ui, egui::vec2(left_w, avail.y));
                    ui.vertical(|ui| {
                        ui.set_width(right_w);
                        self.render_birkhoff_plot(ui, avail.y);
                    });
                });
            } else {
                let top_h = (avail.y * 0.52).max(190.0);
                let bot_h = (avail.y - top_h - 10.0).max(160.0);
                self.render_billiard_canvas(ui, egui::vec2(avail.x, top_h));
                ui.add_space(6.0);
                self.render_birkhoff_plot(ui, bot_h);
            }
        });
    }

    fn render_birkhoff_plot(&self, ui: &mut egui::Ui, height: f32) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        Plot::new("dynamical_billiards_birkhoff_plot")
            .height(height.max(140.0))
            .legend(Legend::default().position(Corner::LeftTop))
            .include_x(0.0)
            .include_x(1.0)
            .include_y(-1.0)
            .include_y(1.0)
            .x_axis_label("Normalized Perimeter Coordinate s / L")
            .y_axis_label("Tangential Momentum sin φ")
            .show(ui, |plot_ui| {
                plot_ui.hline(
                    HLine::new("Normal Incidence (sin φ = 0)", 0.0)
                        .color(palette.graph_edge)
                        .style(LineStyle::dashed_loose()),
                );
                for (idx, slot) in self.rays.iter().enumerate() {
                    if slot.birkhoff_points.is_empty() {
                        continue;
                    }
                    plot_ui.points(
                        Points::new(
                            format!("Ray {} (s/L, sin φ)", idx + 1),
                            PlotPoints::from(slot.birkhoff_points.clone()),
                        )
                        .color(palette.series_color(slot.color_index))
                        .radius(2.1_f32),
                    );
                }
            });
    }
}

fn decimal_slider(
    ui: &mut egui::Ui,
    value: &mut f64,
    range: std::ops::RangeInclusive<f64>,
    text: &str,
    hover: &str,
) -> bool {
    ui.add(egui::Slider::new(value, range).text(text).max_decimals(3))
        .accessible_hover_text(hover)
        .changed()
}

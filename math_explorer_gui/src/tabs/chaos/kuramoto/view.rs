//! Phase-network canvas, unit-circle phasor diagram, and `R(t)` coherence plot for [`super::KuramotoOscillatorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::KuramotoOscillatorTool;
use crate::accessibility::AccessibleHoverText;
use crate::app::workspace_bus::KURAMOTO_COHERENCE_SERIES;
use eframe::egui::{self, Pos2};
use egui_plot::{Line, Plot, PlotPoints};

impl KuramotoOscillatorTool {
    pub(super) fn render_tool_inside(&mut self, parent_ui: &mut egui::Ui) {
        if crate::app::workspace_bus::with_linked_bus(|bus| {
            self.ingest_topology_channel(&bus.topology);
        })
        .is_none()
        {
            self.on_bus_unlinked();
        }

        let mut reset_bus_series = false;
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "kuramoto_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Kuramoto Oscillators");
            ui.separator();
            self.render_controls(ui, &mut reset_bus_series);
        });

        if self.is_running {
            let _ = self.step_once();
            parent_ui.ctx().request_repaint();
        }

        let _ = crate::app::workspace_bus::with_linked_bus(|bus| {
            if reset_bus_series {
                bus.signals.clear_series(KURAMOTO_COHERENCE_SERIES);
                self.last_published_sample_time = None;
            }
            self.publish_latest_coherence(bus);
        });

        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let avail = ui.available_size();
            let top_h = (avail.y * 0.56).max(120.0).min((avail.y - 80.0).max(80.0));
            ui.allocate_ui(egui::vec2(avail.x, top_h), |top_ui| {
                self.render_phase_canvas(top_ui);
            });
            ui.separator();
            self.render_coherence_plot(ui);
        });
    }

    fn render_phase_canvas(&self, ui: &mut egui::Ui) {
        let (response, painter) =
            ui.allocate_painter(ui.available_size(), egui::Sense::focusable_noninteractive());
        let label = "Kuramoto Phase Oscillator Network";
        let _ = response.clone().accessible_hover_text(label);
        let (r, _) = self.system.order_parameter();
        let summary = format!(
            "{} oscillators, coherence R(t) {:.2}",
            self.system.num_oscillators(),
            r
        );
        crate::accessibility::register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            label,
            Some(&summary),
        );
        crate::accessibility::paint_canvas_focus_ring(ui, &response);

        let rect = response.rect;
        let mid_x = rect.min.x + rect.width() * 0.55;
        let net_rect = egui::Rect::from_min_max(rect.min, Pos2::new(mid_x, rect.max.y));
        let ring_rect = egui::Rect::from_min_max(Pos2::new(mid_x, rect.min.y), rect.max);

        self.paint_coupled_network(ui, &painter, net_rect);
        painter.line_segment(
            [
                Pos2::new(mid_x, rect.min.y + 6.0),
                Pos2::new(mid_x, rect.max.y - 6.0),
            ],
            ui.visuals().widgets.noninteractive.bg_stroke,
        );
        self.paint_order_parameter_circle(ui, &painter, ring_rect);
    }

    fn paint_coupled_network(&self, ui: &egui::Ui, painter: &egui::Painter, rect: egui::Rect) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let center = rect.center();
        let span = (0.38 * rect.width().min(rect.height())).max(16.0);
        let to_screen = |p: [f32; 2]| Pos2::new(center.x + p[0] * span, center.y + p[1] * span);

        for &(u, v, _) in &self.edges {
            if let (Some(&pu), Some(&pv)) = (self.positions.get(u), self.positions.get(v)) {
                painter.line_segment([to_screen(pu), to_screen(pv)], (1.4, palette.graph_edge));
            }
        }

        let phases = self.system.phases();
        let node_r = if self.positions.len() <= 32 { 7.5 } else { 4.5 };
        for (idx, &pos) in self.positions.iter().enumerate() {
            let p = to_screen(pos);
            let theta = phases.get(idx).copied().unwrap_or(0.0);
            let color = phase_to_color(theta);
            painter.circle_filled(p, node_r, color);
            painter.circle_stroke(p, node_r, (1.0, palette.node_stroke));
        }
    }

    fn paint_order_parameter_circle(
        &self,
        ui: &egui::Ui,
        painter: &egui::Painter,
        rect: egui::Rect,
    ) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let center = rect.center();
        let radius = (0.36 * rect.width().min(rect.height())).max(16.0);
        painter.circle_stroke(center, radius, (1.5, palette.graph_edge));

        for &theta in self.system.phases() {
            let (s, c) = (theta as f32).sin_cos();
            let bead = Pos2::new(center.x + radius * c, center.y - radius * s);
            painter.circle_filled(bead, 4.0, phase_to_color(theta));
        }

        let (r, psi) = self.system.order_parameter();
        let (ps, pc) = (psi as f32).sin_cos();
        let tip = Pos2::new(
            center.x + radius * (r as f32) * pc,
            center.y - radius * (r as f32) * ps,
        );
        painter.line_segment([center, tip], (2.8, palette.success));
        painter.circle_filled(tip, 5.0, palette.success);
    }

    fn render_coherence_plot(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let points: PlotPoints<'_> = self.coherence_trace.iter().map(|&[t, r]| [t, r]).collect();
        let line = Line::new("Coherence R(t)", points).color(palette.series_color(0));
        Plot::new(crate::framework::scoped_panel_id(
            ui,
            "kuramoto_coherence_plot",
        ))
        .include_y(0.0)
        .include_y(1.05)
        .x_axis_label("Time t")
        .y_axis_label("Order Parameter R(t)")
        .show(ui, |plot_ui| {
            plot_ui.line(line);
        });
    }
}

fn phase_to_color(theta: f64) -> egui::Color32 {
    let hue = (theta.rem_euclid(std::f64::consts::TAU) / std::f64::consts::TAU) as f32;
    egui::ecolor::Hsva::new(hue, 0.82, 0.92, 1.0).into()
}

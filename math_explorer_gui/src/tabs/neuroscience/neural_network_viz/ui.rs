//! Side-panel controls, accessible canvas rendering, and zero-allocation tests for [`super::NeuralNetworkVizTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{NeuralDynamicsMode, NeuralNetworkVizTool};
use crate::accessibility::AccessibleHoverText;
use crate::app::{contrasting_text_color, render_topology_source_bar, SemanticPalette};
use crate::framework::{PlaybackAction, PlaybackTransport};
use eframe::egui::{self, Color32};

impl NeuralNetworkVizTool {
    pub(super) fn draw_controls(&mut self, parent_ui: &mut egui::Ui) {
        let ctx = parent_ui.ctx().clone();
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "nn_viz_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Controls");
                if !self.upstream_linked {
                    let badge = self.clamp_badge_text();
                    if render_topology_source_bar(
                        ui,
                        "neural_network_viz",
                        &mut self.topology_source,
                        self.synced_bus_revision,
                        badge.as_deref(),
                    ) {
                        self.sync_shared_topology(&ctx);
                    }
                }
                ui.separator();
                self.render_nn_transport(ui, &ctx);
                ui.separator();
                self.draw_mode_and_kuramoto_controls(ui);
                if self.mode == NeuralDynamicsMode::HodgkinHuxley {
                    self.render_hh_controls(ui);
                }
            });
        });
    }

    fn render_hh_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        crate::audio::render_sonification_panel(
            ui,
            "neural_network_viz",
            &self.sonification,
            !self.is_running,
        );
        ui.separator();
        ui.heading("External Current (uA/cm^2)");
        let max_sliders = if self.upstream_linked {
            self.external_input.len().min(12)
        } else {
            self.external_input.len()
        };
        for i in 0..max_sliders {
            ui.horizontal(|ui| {
                if ui
                    .add(
                        egui::Slider::new(&mut self.external_input[i], -10.0..=50.0)
                            .text(format!("Neuron {i}")),
                    )
                    .changed()
                {
                    self.bump_revision();
                }
            });
        }
        ui.separator();
        ui.label(
            "Synaptic activity is modeled via CSR conductance pulses: when presynaptic V > 0 mV, scaled synaptic current drives postsynaptic targets.",
        );
    }

    fn render_nn_transport(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if let Some(action) =
            PlaybackTransport::in_thread("nn_viz_transport", "Neural Network Viz", self.is_running)
                .show(ui)
        {
            match action {
                PlaybackAction::Play => self.is_running = true,
                PlaybackAction::Pause => self.is_running = false,
                PlaybackAction::Step => self.step_once(),
                PlaybackAction::StepBack | PlaybackAction::ScrubTo(_) => {}
                PlaybackAction::Reset => {
                    self.reset();
                    if !self.upstream_linked {
                        self.sync_shared_topology(ctx);
                    }
                }
            }
            self.bump_revision();
        }
    }

    pub(super) fn draw_synaptic_edges(
        &self,
        painter: &egui::Painter,
        center: egui::Pos2,
        palette: &SemanticPalette,
    ) {
        let n = self.neurons.len();
        for pre in 0..n {
            let start = self.csr_row_ptr.get(pre).copied().unwrap_or(0);
            let end = self.csr_row_ptr.get(pre + 1).copied().unwrap_or(0);
            for idx in start..end {
                let post = self.csr_col_idx[idx];
                let weight = self.csr_weights[idx];
                if post < n && weight != 0.0 {
                    let p_pre = center + egui::vec2(self.positions[pre][0], self.positions[pre][1]);
                    let p_post =
                        center + egui::vec2(self.positions[post][0], self.positions[post][1]);
                    let base = if weight > 0.0 {
                        palette.success
                    } else {
                        palette.error
                    };
                    let color = Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), 170);
                    painter.line_segment(
                        [p_pre, p_post],
                        ((1.0 + weight.abs() as f32).min(4.0), color),
                    );
                }
            }
        }
    }

    pub(super) fn draw_network(&self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            if self.mode == NeuralDynamicsMode::Kuramoto {
                self.draw_kuramoto_view(ui);
                return;
            }
            ui.heading("Network Visualization");
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::focusable_noninteractive());
            let label = "Neural Network Visualization: Showing interconnected neurons";
            let _ = response.clone().accessible_hover_text(label);
            let summary = format!("{} interconnected neurons", self.neurons.len());
            crate::accessibility::register_canvas_accessibility(
                ui,
                &response,
                egui::accesskit::Role::GraphicsDocument,
                label,
                Some(&summary),
            );
            crate::accessibility::paint_canvas_focus_ring(ui, &response);

            let center = response.rect.center();
            let palette = SemanticPalette::from_visuals(ui.visuals());
            self.draw_synaptic_edges(&painter, center, &palette);

            for i in 0..self.neurons.len() {
                let v = self.neurons[i].v() as f32;
                let p = center + egui::vec2(self.positions[i][0], self.positions[i][1]);
                let normalized_v = ((v + 65.0) / 105.0).clamp(0.0, 1.0);
                let (low, high) = (palette.axis_z, palette.axis_x);
                let lerp = |a: u8, b: u8| {
                    (f32::from(a) + (f32::from(b) - f32::from(a)) * normalized_v).round() as u8
                };
                let color = Color32::from_rgb(
                    lerp(low.r(), high.r()),
                    lerp(low.g(), high.g()),
                    lerp(low.b(), high.b()),
                );
                painter.circle_filled(p, 15.0, color);
                painter.circle_stroke(p, 15.0, (1.0, palette.node_stroke));
                painter.text(
                    p,
                    egui::Align2::CENTER_CENTER,
                    format!("N{i}\n{v:.0}mV"),
                    egui::FontId::proportional(10.0),
                    contrasting_text_color(color),
                );
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_network_viz_step_once_keeps_paused() {
        let mut tool = NeuralNetworkVizTool::default();
        assert!(!tool.is_running);
        let initial_v0 = tool.neurons[0].v();
        tool.step_once();
        assert!(!tool.is_running);
        assert_ne!(tool.neurons[0].v(), initial_v0);
    }

    #[test]
    fn test_neural_network_viz_zero_alloc_1000_steps_under_verified_allocator() {
        let mut tool = NeuralNetworkVizTool::default();
        {
            let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let (v0, _metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
                for _ in 0..1_000 {
                    tool.step();
                }
                tool.neurons[0].v()
            });
            assert!(v0.is_finite());
        }

        let ctx = egui::Context::default();
        let bus = crate::app::SharedSimulationBus::from_context_or_insert(&ctx);
        bus.install_in_context(&ctx, true);
        let nodes: Vec<(usize, [f32; 2])> = (0..40)
            .map(|i| (i, [i as f32 * 5.0, (i % 8) as f32 * 5.0]))
            .collect();
        let edges: Vec<(usize, usize, f64)> = (0..40).map(|i| (i, (i + 1) % 40, 1.5)).collect();
        bus.publish_graph_if_changed(99, true, &nodes, &edges);
        tool.sync_shared_topology(&ctx);
        assert_eq!(tool.neuron_count(), 32);
        assert_eq!(
            tool.clamp_badge_text().as_deref(),
            Some("Clamped 40 -> 32 neurons")
        );

        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (v0_clamped, _metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            for _ in 0..1_000 {
                tool.step();
            }
            tool.neurons[0].v()
        });
        assert!(v0_clamped.is_finite());
    }
}

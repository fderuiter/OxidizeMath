//! Kuramoto phase-synchronization controls, $r(t)$ plotting, and `PaneDataflowBus` ingestion
//! for [`super::NeuralNetworkVizTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{MAX_ORDER_HISTORY, NeuralDynamicsMode, NeuralNetworkVizTool};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::app::dataflow_bus::{GraphTopologyPayload, MAX_DATAFLOW_GRAPH_NODES, poll_pane_graph};
use eframe::egui::{self, Color32, Pos2};
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::biology::neuroscience::HodgkinHuxleyNeuron;
use std::f64::consts::TAU;

fn phase_to_color(phase: f64, palette: &SemanticPalette) -> Color32 {
    let t = (phase.rem_euclid(TAU) / TAU) as f32;
    let idx = ((t * 6.0).floor() as usize) % 6;
    let next_idx = (idx + 1) % 6;
    let frac = t * 6.0 - (idx as f32);
    let a = palette.series[idx];
    let b = palette.series[next_idx];
    let lerp = |x: u8, y: u8| ((x as f32) * (1.0 - frac) + (y as f32) * frac).round() as u8;
    Color32::from_rgb(lerp(a.r(), b.r()), lerp(a.g(), b.g()), lerp(a.b(), b.b()))
}

impl NeuralNetworkVizTool {
    pub(super) fn sync_kuramoto_from_weight_matrix(&mut self, spread: f64) {
        let n = self.neurons.len();
        let mut edges = Vec::new();
        for (dst, row) in self.weights.iter().enumerate().take(n) {
            for (src, &w) in row.iter().enumerate().take(n) {
                if src != dst && w > 0.0 {
                    edges.push((src, dst, w));
                }
            }
        }
        self.kuramoto.reconfigure_topology(n, &edges, true, spread);
    }

    /// Polls `PaneDataflowBus` and resizes neural/Kuramoto buffers only when `revision` advances.
    pub(super) fn poll_upstream_graph(&mut self, ctx: &egui::Context) {
        let Some(payload) = poll_pane_graph(ctx) else {
            self.upstream_linked = false;
            return;
        };
        self.upstream_linked = true;
        if payload.revision != self.last_upstream_graph_rev {
            self.ingest_graph_topology(&payload);
        }
    }

    /// Applies a live [`GraphTopologyPayload`], resizing buffers only when vertex count changes.
    pub fn ingest_graph_topology(&mut self, payload: &GraphTopologyPayload) {
        let n = payload.node_positions.len().min(MAX_DATAFLOW_GRAPH_NODES);
        if n == 0 {
            self.last_upstream_graph_rev = payload.revision;
            return;
        }
        self.positions = payload.normalized_positions(n, 110.0);
        if self.neurons.len() != n {
            self.neurons
                .resize_with(n, || HodgkinHuxleyNeuron::new(-65.0));
            self.scratch_inputs.resize(n, 0.0);
            self.external_input.resize(n, 0.0);
            if self.external_input.iter().all(|&v| v == 0.0) {
                self.external_input[0] = 10.0;
            }
        }
        let mut next_weights = vec![vec![0.0; n]; n];
        for &(u, v, w) in &payload.edges {
            if u < n && v < n && u != v && w.is_finite() {
                next_weights[v][u] = w;
                if !payload.directed {
                    next_weights[u][v] = w;
                }
            }
        }
        self.weights = next_weights;
        let spread = self.kuramoto.frequency_spread();
        self.kuramoto
            .reconfigure_topology(n, &payload.edges, payload.directed, spread);
        self.last_upstream_graph_rev = payload.revision;
        self.bump_revision();
    }

    pub(super) fn record_kuramoto_history_zero_alloc(&mut self) {
        let (r, _) = self.kuramoto.order_parameter();
        if self.order_history.len() >= MAX_ORDER_HISTORY {
            self.order_history.pop_front();
        }
        self.order_history.push_back([self.kuramoto.time(), r]);
    }

    pub(super) fn draw_mode_and_kuramoto_controls(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        if self.upstream_linked {
            ui.colored_label(
                palette.accent,
                format!("⚡ Upstream Graph ({} oscillators)", self.neurons.len()),
            );
            ui.separator();
        }
        ui.label("Dynamics Regime:");
        let prev_mode = self.mode;
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(
                &mut self.mode,
                NeuralDynamicsMode::HodgkinHuxley,
                "Hodgkin-Huxley",
            );
            ui.radio_value(
                &mut self.mode,
                NeuralDynamicsMode::Kuramoto,
                "Kuramoto r(t)",
            );
        });
        if self.mode != prev_mode {
            self.bump_revision();
        }
        if self.mode != NeuralDynamicsMode::Kuramoto {
            return;
        }
        ui.separator();
        self.draw_kuramoto_sliders(ui, &palette);
    }

    fn draw_kuramoto_sliders(&mut self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let mut coupling = self.kuramoto.coupling();
        if ui
            .add(egui::Slider::new(&mut coupling, 0.0..=15.0).text("Coupling K"))
            .changed()
        {
            self.kuramoto.set_coupling(coupling);
            self.bump_revision();
        }
        let mut spread = self.kuramoto.frequency_spread();
        if ui
            .add(egui::Slider::new(&mut spread, 0.0..=3.0).text("Freq Spread Δω"))
            .changed()
        {
            self.kuramoto.set_frequency_spread(spread);
            self.bump_revision();
        }
        let (r, psi) = self.kuramoto.order_parameter();
        ui.colored_label(
            if r >= 0.7 {
                palette.success
            } else {
                palette.info
            },
            format!("Order Parameter r(t) = {r:.3}  (ψ = {psi:.2} rad)"),
        );
        if ui
            .button("↻ Reset Phases")
            .accessible_hover_text("Re-seed oscillator phases around the unit circle")
            .clicked()
        {
            self.kuramoto.reset_phases();
            self.order_history.clear();
            self.record_kuramoto_history_zero_alloc();
            self.bump_revision();
        }
    }

    pub(super) fn draw_kuramoto_view(&self, ui: &mut egui::Ui) {
        let avail = ui.available_size();
        let graph_h = (avail.y * 0.62).max(160.0);
        let (response, painter) = ui.allocate_painter(
            egui::vec2(avail.x, graph_h),
            egui::Sense::focusable_noninteractive(),
        );
        let label = "Neural Network Visualization: Kuramoto coupled phase-oscillator network";
        let _ = response.clone().accessible_hover_text(label);
        let (r, psi) = self.kuramoto.order_parameter();
        let summary = format!("{} oscillators, r(t) = {r:.3}", self.kuramoto.num_nodes());
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
        let stroke = ui.visuals().window_stroke();

        self.draw_synaptic_edges(&painter, center, &palette);
        let phases = self.kuramoto.phases();
        for (i, &theta) in phases.iter().enumerate() {
            let pos = center + egui::vec2(self.positions[i][0], self.positions[i][1]);
            let fill = phase_to_color(theta, &palette);
            painter.circle(pos, 15.0, fill, stroke);
            let dir = egui::vec2(theta.cos() as f32, theta.sin() as f32) * 12.0;
            painter.line_segment([pos, pos + dir], (2.2, stroke.color));
        }
        let mean_vec = egui::vec2(psi.cos() as f32, psi.sin() as f32) * (r as f32 * 48.0);
        painter.circle_stroke(center, 48.0, (1.0, palette.muted_text));
        painter.arrow(center, mean_vec, (2.8, palette.accent));
        painter.text(
            Pos2::new(response.rect.left() + 10.0, response.rect.top() + 14.0),
            egui::Align2::LEFT_TOP,
            format!("Kuramoto Synchronization r(t) = {r:.3}"),
            egui::FontId::proportional(14.0),
            palette.accent,
        );

        let pts: PlotPoints<'_> = self.order_history.iter().copied().collect();
        Plot::new(ui.id().with("kuramoto_order_plot"))
            .height((avail.y - graph_h - 8.0).max(90.0))
            .include_y(0.0)
            .include_y(1.05)
            .allow_scroll(false)
            .show(ui, |plot_ui| {
                plot_ui.line(Line::new("r(t)", pts).color(palette.success).width(2.0_f32));
            });
    }
}

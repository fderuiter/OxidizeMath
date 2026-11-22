#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{NetworkPropagationTool, NodeClickAction};
use crate::accessibility::AccessibleHoverText;
use crate::framework::{PlaybackAction, PlaybackTransport};
use eframe::egui;
use math_explorer::epidemiology::networks::NodeState;

impl NetworkPropagationTool {
    pub(super) fn draw_controls_column(&mut self, ui: &mut egui::Ui) {
        ui.heading("Controls");
        if self.using_bus_topology {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            ui.label(
                egui::RichText::new(format!(
                    "🔗 Linked Topology (rev {})",
                    self.synced_bus_revision.unwrap_or(0)
                ))
                .small()
                .color(palette.success),
            );
        }
        ui.separator();

        if let Some(action) = PlaybackTransport::in_thread(
            "network_prop_transport",
            "Network Propagation",
            self.is_running,
        )
        .reset_label("↻ Reset Network")
        .show(ui)
        {
            match action {
                PlaybackAction::Play => self.is_running = true,
                PlaybackAction::Pause => self.is_running = false,
                PlaybackAction::Step => self.step_once(),
                PlaybackAction::Reset => self.reset_network(),
            }
        }

        ui.separator();
        if !self.using_bus_topology {
            let mut num_nodes = self.standalone_num_nodes;
            if ui
                .add(egui::Slider::new(&mut num_nodes, 10..=200).text("Nodes"))
                .changed()
            {
                self.standalone_num_nodes = num_nodes;
                self.reset_network();
            }
        }
        self.draw_rate_and_click_controls(ui);
        self.draw_epidemic_readouts(ui);
    }

    fn draw_rate_and_click_controls(&mut self, ui: &mut egui::Ui) {
        let mut beta = self.model.beta();
        if ui
            .add(egui::Slider::new(&mut beta, 0.0..=1.0).text("Transmission (beta)"))
            .changed()
        {
            let _ = self.model.set_beta(beta);
            self.bump_state_rev();
        }

        let mut gamma = self.model.gamma();
        if ui
            .add(egui::Slider::new(&mut gamma, 0.0..=1.0).text("Recovery (gamma)"))
            .changed()
        {
            let _ = self.model.set_gamma(gamma);
            self.bump_state_rev();
        }

        ui.separator();
        ui.label("Node Click Action:");
        let prev = self.click_action;
        ui.radio_value(
            &mut self.click_action,
            NodeClickAction::CycleState,
            "Cycle (S → I → R)",
        );
        ui.radio_value(
            &mut self.click_action,
            NodeClickAction::SeedInfected,
            "Seed Infected (I)",
        );
        ui.radio_value(
            &mut self.click_action,
            NodeClickAction::QuarantineRecovered,
            "Quarantine Hub (R)",
        );
        if self.click_action != prev {
            self.bump_state_rev();
        }
    }

    fn draw_epidemic_readouts(&self, ui: &mut egui::Ui) {
        ui.separator();
        let r0 = self.current_heterogeneous_r0();
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let r0_color = if r0 >= 1.0 {
            palette.error
        } else {
            palette.success
        };
        ui.label(
            egui::RichText::new(format!("Heterogeneous R0: {r0:.3}"))
                .strong()
                .color(r0_color),
        );
        let count_state = |target: NodeState| {
            self.model
                .states()
                .iter()
                .filter(|&&s| s == target)
                .count()
        };
        ui.label(
            egui::RichText::new(format!(
                "Susceptible: {}",
                count_state(NodeState::Susceptible)
            ))
            .color(palette.axis_z),
        );
        ui.label(
            egui::RichText::new(format!("Infected: {}", count_state(NodeState::Infected)))
                .color(palette.axis_x),
        );
        ui.label(
            egui::RichText::new(format!(
                "Recovered: {}",
                count_state(NodeState::Recovered)
            ))
            .color(palette.axis_y),
        );
    }

    pub(super) fn draw_graph_column(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.heading("Network Visualization");
        let (response, painter) =
            ui.allocate_painter(ui.available_size(), egui::Sense::click());
        let label = "Network Propagation Visualization";
        let _ = response.clone().accessible_hover_text(
            "Network Propagation Visualization — click nodes to seed infection or quarantine",
        );
        let summary = format!(
            "{} network nodes, R0 {:.2}",
            self.model.num_nodes(),
            self.current_heterogeneous_r0()
        );
        crate::accessibility::register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            label,
            Some(&summary),
        );
        crate::accessibility::paint_canvas_focus_ring(ui, &response);

        let center = response.rect.center();
        let scale = canvas_position_scale(response.rect);
        let node_radius = if self.model.num_nodes() <= 40 {
            8.0
        } else {
            5.5
        };

        if response.clicked() {
            if let Some(click_pos) = response.interact_pointer_pos() {
                if let Some(hit_idx) = self.hit_test_node(center, scale, click_pos, node_radius + 8.0) {
                    let _ = self.apply_node_click(hit_idx);
                }
            }
        }

        self.paint_network_edges_and_nodes(&painter, center, scale, node_radius, &palette);
    }

    fn paint_network_edges_and_nodes(
        &self,
        painter: &egui::Painter,
        center: egui::Pos2,
        scale: f32,
        node_radius: f32,
        palette: &crate::app::SemanticPalette,
    ) {
        for i in 0..self.model.num_nodes() {
            for &j in &self.model.adjacency()[i] {
                if i < j {
                    let p1 = center
                        + scale * egui::vec2(self.model.positions()[i][0], self.model.positions()[i][1]);
                    let p2 = center
                        + scale * egui::vec2(self.model.positions()[j][0], self.model.positions()[j][1]);
                    painter.line_segment([p1, p2], (1.4, palette.graph_edge));
                }
            }
        }

        for i in 0..self.model.num_nodes() {
            let p =
                center + scale * egui::vec2(self.model.positions()[i][0], self.model.positions()[i][1]);
            let color = match self.model.states()[i] {
                NodeState::Susceptible => palette.axis_z,
                NodeState::Infected => palette.axis_x,
                NodeState::Recovered => palette.axis_y,
            };
            painter.circle_filled(p, node_radius, color);
            painter.circle_stroke(p, node_radius, (1.2, palette.node_stroke));
        }
    }

    fn hit_test_node(
        &self,
        center: egui::Pos2,
        scale: f32,
        click_pos: egui::Pos2,
        hit_radius: f32,
    ) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for i in 0..self.model.num_nodes() {
            let p =
                center + scale * egui::vec2(self.model.positions()[i][0], self.model.positions()[i][1]);
            let dist = (click_pos - p).length();
            if dist <= hit_radius && best.is_none_or(|(_, b_dist)| dist < b_dist) {
                best = Some((i, dist));
            }
        }
        best.map(|(idx, _)| idx)
    }
}

fn canvas_position_scale(rect: egui::Rect) -> f32 {
    let min_dim = rect.width().min(rect.height()).max(60.0);
    (min_dim / 440.0).clamp(0.35, 1.35)
}

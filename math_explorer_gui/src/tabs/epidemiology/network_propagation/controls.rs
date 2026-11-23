#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::analytics::{compute_network_analytics, render_compartment_and_analytics_panel};
use super::{ContainmentMode, NetworkPropagationTool, MAX_SLIDER_BUDGET};
use crate::accessibility::{announce_live_status, AccessibleHoverText};
use crate::framework::{PlaybackAction, PlaybackTransport};
use eframe::egui;

impl NetworkPropagationTool {
    pub(super) fn draw_controls_column(&mut self, ui: &mut egui::Ui) {
        ui.heading("Controls");
        ui.separator();

        self.render_transport_controls(ui);
        ui.separator();
        self.render_network_sliders(ui);
        ui.separator();
        self.render_containment_controls(ui);
        ui.separator();

        let summary = compute_network_analytics(&self.model);
        render_compartment_and_analytics_panel(ui, &summary, self.used_budget(), self.budget);
    }

    fn render_transport_controls(&mut self, ui: &mut egui::Ui) {
        if let Some(action) = PlaybackTransport::in_thread(
            "network_prop_transport",
            "Network Propagation",
            self.is_running,
        )
        .reset_label("↻ Reset Network")
        .show(ui)
        {
            match action {
                PlaybackAction::Play => {
                    if !self.is_running {
                        self.is_running = true;
                        self.bump_revision();
                    }
                }
                PlaybackAction::Pause => {
                    if self.is_running {
                        self.is_running = false;
                        self.bump_revision();
                    }
                }
                PlaybackAction::Step => self.step_once(),
                PlaybackAction::Reset => self.reset_network(),
                PlaybackAction::StepBack | PlaybackAction::Scrub(_) => {}
            }
        }
    }

    fn render_network_sliders(&mut self, ui: &mut egui::Ui) {
        let mut num_nodes = self.model.num_nodes();
        if ui
            .add(egui::Slider::new(&mut num_nodes, 10..=200).text("Nodes"))
            .changed()
        {
            let _ = self.model.set_num_nodes(num_nodes);
            self.vaccinated_nodes.clear();
            self.quarantined_edges.clear();
            self.rebuild_seeded_graph();
        }

        let mut beta = self.model.beta();
        if ui
            .add(egui::Slider::new(&mut beta, 0.0..=1.0).text("Transmission (beta)"))
            .changed()
        {
            let _ = self.model.set_beta(beta);
            self.bump_revision();
        }

        let mut gamma = self.model.gamma();
        if ui
            .add(egui::Slider::new(&mut gamma, 0.0..=1.0).text("Recovery (gamma)"))
            .changed()
        {
            let _ = self.model.set_gamma(gamma);
            self.bump_revision();
        }

        let slider_max = MAX_SLIDER_BUDGET.max(self.budget);
        let mut budget = self.budget;
        if ui
            .add(egui::Slider::new(&mut budget, 0..=slider_max).text("Containment Budget"))
            .changed()
        {
            self.budget = budget;
            self.enforce_budget_cap();
            self.bump_revision();
        }
    }

    fn render_containment_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Seed:");
            let mut seed = self.seed;
            if ui
                .add(egui::DragValue::new(&mut seed).speed(1.0))
                .accessible_hover_text("Deterministic network topology seed")
                .changed()
            {
                self.set_seed_and_reset(seed);
            }
            if ui
                .button("🎲 Reseed")
                .accessible_hover_text("Generate next deterministic network seed")
                .clicked()
            {
                let next = self.seed.wrapping_add(1);
                self.set_seed_and_reset(next);
                announce_live_status(ui.ctx(), format!("Network reseeded to {next}"));
            }
        });

        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(
                &mut self.interaction_mode,
                ContainmentMode::VaccinateNode,
                "💉 Vaccinate Node",
            )
            .accessible_hover_text("Click or press Space/Enter on canvas to vaccinate nodes");
            ui.selectable_value(
                &mut self.interaction_mode,
                ContainmentMode::QuarantineEdge,
                "✂ Quarantine Edge",
            )
            .accessible_hover_text("Click or press Space/Enter on canvas to quarantine edges");
        });

        if ui
            .add_enabled(
                self.used_budget() > 0,
                egui::Button::new("🧹 Clear Interventions"),
            )
            .accessible_hover_text("Remove all vaccinations and edge quarantines")
            .clicked()
        {
            self.clear_interventions();
            announce_live_status(ui.ctx(), "Cleared all network interventions");
        }
    }
}

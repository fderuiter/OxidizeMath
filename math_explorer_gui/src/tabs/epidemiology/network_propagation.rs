#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "network_propagation/bus_sync.rs"]
mod bus_sync;
#[path = "network_propagation/presets.rs"]
mod presets;

use crate::accessibility::AccessibleHoverText;
use crate::app::data_bus::ui::{render_follow_workspace_toggle, render_node_budget_banner};
use crate::app::data_bus::{
    SharedGraph, MAX_EPIDEMIC_NODE_BUDGET, SESSION_MAX_BUS_EDGES, SESSION_MAX_BUS_NODES,
};
use crate::framework::{InteractiveTool, PlaybackAction, PlaybackTransport};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use math_explorer::epidemiology::networks::{NetworkEpidemicModel, NodeState};
use std::sync::Arc;

fn default_true() -> bool {
    true
}

const fn is_false(value: &bool) -> bool {
    !*value
}

const fn is_true(value: &bool) -> bool {
    *value
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NetPropState {
    num_nodes: usize,
    beta: f64,
    gamma: f64,
    #[serde(default, skip_serializing_if = "is_false")]
    is_running: bool,
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    follow_workspace_graph: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seeded_graph: Option<SharedGraph>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    frozen_graph: Option<SharedGraph>,
}

pub struct NetworkPropagationTool {
    model: NetworkEpidemicModel,
    rng: oxidize_core::rng::OxidizeRng,
    is_running: bool,
    pub(crate) follow_workspace_graph: bool,
    pub(crate) applied_bus_revision: u64,
    clamped_source_nodes: Option<usize>,
    active_shared_graph: Option<Arc<SharedGraph>>,
    seeded_graph: Option<SharedGraph>,
    pending_seed_publish: bool,
    trace_step: usize,
    trace_s: Vec<[f64; 2]>,
    trace_i: Vec<[f64; 2]>,
    trace_r: Vec<[f64; 2]>,
    trace_dirty: bool,
    #[cfg(test)]
    sim_steps: u64,
}

impl Default for NetworkPropagationTool {
    fn default() -> Self {
        let mut tool = Self {
            model: NetworkEpidemicModel::new(50, 0.05, 0.02),
            rng: oxidize_core::rng::OxidizeRng::default(),
            is_running: false,
            follow_workspace_graph: true,
            applied_bus_revision: 0,
            clamped_source_nodes: None,
            active_shared_graph: None,
            seeded_graph: None,
            pending_seed_publish: false,
            trace_step: 0,
            trace_s: Vec::new(),
            trace_i: Vec::new(),
            trace_r: Vec::new(),
            trace_dirty: true,
            #[cfg(test)]
            sim_steps: 0,
        };
        tool.reset_empirical_trace();
        tool
    }
}

impl NetworkPropagationTool {
    fn reset_network(&mut self) {
        self.rng = oxidize_core::rng::OxidizeRng::default();
        if let Some(shared) = self.active_shared_graph.clone() {
            self.apply_shared_graph(&shared);
        } else {
            self.model.initialize_geometric_graph_with_rng(&mut self.rng);
            self.reset_empirical_trace();
        }
        self.is_running = false;
        #[cfg(test)]
        {
            self.sim_steps = 0;
        }
    }

    pub(crate) fn step_once(&mut self) {
        self.is_running = false;
        self.model.step_with_rng(&mut self.rng);
        self.trace_step = self.trace_step.saturating_add(1);
        self.record_trace_sample();
        #[cfg(test)]
        {
            self.sim_steps = self.sim_steps.saturating_add(1);
        }
    }

    #[cfg(test)]
    pub(crate) fn sim_steps(&self) -> u64 {
        self.sim_steps
    }

    fn draw_compartment_summary(&self, ui: &mut egui::Ui) {
        let (mean_k, var_k, r0_het) = self.degree_moments_and_r0();
        render_copyable_metric(ui, "Heterogeneous R₀", &format!("{r0_het:.2}"));
        ui.small(format!("⟨k⟩ = {mean_k:.2}, Var(k) = {var_k:.2}"));

        ui.separator();
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.label(
            egui::RichText::new(format!(
                "Susceptible: {}",
                self.count_compartment(NodeState::Susceptible)
            ))
            .color(palette.axis_z),
        );
        ui.label(
            egui::RichText::new(format!(
                "Infected: {}",
                self.count_compartment(NodeState::Infected)
            ))
            .color(palette.axis_x),
        );
        ui.label(
            egui::RichText::new(format!(
                "Recovered: {}",
                self.count_compartment(NodeState::Recovered)
            ))
            .color(palette.axis_y),
        );
    }

    fn draw_rate_sliders(&mut self, ui: &mut egui::Ui) {
        if self.active_shared_graph.is_none() || !self.follow_workspace_graph {
            let mut num_nodes = self.model.num_nodes().clamp(10, 200);
            if ui
                .add(egui::Slider::new(&mut num_nodes, 10..=200).text("Nodes"))
                .changed()
            {
                self.active_shared_graph = None;
                self.clamped_source_nodes = None;
                let _ = self.model.set_num_nodes(num_nodes);
                self.reset_network();
            }
        }

        let mut beta = self.model.beta();
        if ui
            .add(egui::Slider::new(&mut beta, 0.0..=1.0).text("Transmission (beta)"))
            .changed()
        {
            let _ = self.model.set_beta(beta);
            self.trace_dirty = true;
        }

        let mut gamma = self.model.gamma();
        if ui
            .add(egui::Slider::new(&mut gamma, 0.0..=1.0).text("Recovery (gamma)"))
            .changed()
        {
            let _ = self.model.set_gamma(gamma);
            self.trace_dirty = true;
        }
    }

    fn draw_controls_column(&mut self, ui: &mut egui::Ui) {
        ui.heading("Controls");
        ui.separator();

        let rev_opt = (self.applied_bus_revision > 0).then_some(self.applied_bus_revision);
        if render_follow_workspace_toggle(
            ui,
            &mut self.follow_workspace_graph,
            "Network Propagation",
            rev_opt,
        ) && self.follow_workspace_graph
        {
            self.applied_bus_revision = 0;
            self.sync_from_bus(ui.ctx());
        }
        render_node_budget_banner(
            ui,
            self.clamped_source_nodes,
            MAX_EPIDEMIC_NODE_BUDGET,
            "epidemic nodes",
        );
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
        self.draw_rate_sliders(ui);
        ui.separator();
        self.draw_compartment_summary(ui);
    }

    fn draw_graph_column(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.heading("Network Visualization");
        let (response, painter) = ui.allocate_painter(
            ui.available_size(),
            egui::Sense::focusable_noninteractive(),
        );
        let label = "Network Propagation Visualization";
        let _ = response.clone().accessible_hover_text(label);
        let (_, _, r0_het) = self.degree_moments_and_r0();
        let summary = format!(
            "{} network nodes, heterogeneous R0 {:.2}",
            self.model.num_nodes(),
            r0_het
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

        for i in 0..self.model.num_nodes() {
            for &j in &self.model.adjacency()[i] {
                if i < j {
                    let p1 = center
                        + egui::vec2(self.model.positions()[i][0], self.model.positions()[i][1]);
                    let p2 = center
                        + egui::vec2(self.model.positions()[j][0], self.model.positions()[j][1]);
                    painter.line_segment([p1, p2], (1.2, palette.graph_edge));
                }
            }
        }

        for i in 0..self.model.num_nodes() {
            let p =
                center + egui::vec2(self.model.positions()[i][0], self.model.positions()[i][1]);
            let color = match self.model.states()[i] {
                NodeState::Susceptible => palette.axis_z,
                NodeState::Infected => palette.axis_x,
                NodeState::Recovered => palette.axis_y,
            };
            painter.circle_filled(p, 5.5, color);
            painter.circle_stroke(p, 5.5, (1.0, palette.node_stroke));
        }
    }
}

impl InteractiveTool for NetworkPropagationTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Network Propagation"
    }
    fn domain(&self) -> &'static str {
        "epidemiology"
    }
    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::none()
    }

    fn save_state(&self) -> Option<String> {
        let frozen_graph = (!self.follow_workspace_graph)
            .then(|| self.active_shared_graph.as_deref().cloned())
            .flatten();
        serde_json::to_string(&NetPropState {
            num_nodes: self.model.num_nodes(),
            beta: self.model.beta(),
            gamma: self.model.gamma(),
            is_running: self.is_running,
            follow_workspace_graph: self.follow_workspace_graph,
            seeded_graph: self.seeded_graph.clone(),
            frozen_graph,
        })
        .ok()
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: NetPropState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid network propagation state: {e}"))?;
        let has_graph = s.seeded_graph.is_some() || s.frozen_graph.is_some();
        let nodes_ok = if has_graph {
            (1..=MAX_EPIDEMIC_NODE_BUDGET).contains(&s.num_nodes)
        } else {
            (10..=200).contains(&s.num_nodes)
        };
        if !nodes_ok
            || !s.beta.is_finite()
            || !(0.0..=1.0).contains(&s.beta)
            || !s.gamma.is_finite()
            || !(0.0..=1.0).contains(&s.gamma)
        {
            return Err("Network propagation parameters out of bounds".into());
        }
        if let Some(ref g) = s.seeded_graph {
            g.validate(SESSION_MAX_BUS_NODES, SESSION_MAX_BUS_EDGES)?;
        }
        if let Some(ref g) = s.frozen_graph {
            g.validate(SESSION_MAX_BUS_NODES, SESSION_MAX_BUS_EDGES)?;
        }

        self.follow_workspace_graph = s.follow_workspace_graph;
        self.seeded_graph = s.seeded_graph.clone();
        self.active_shared_graph = None;
        self.clamped_source_nodes = None;
        self.applied_bus_revision = 0;

        let mut candidate = NetworkEpidemicModel::new(50, s.beta, s.gamma);
        if let Some(g) = s.frozen_graph.or(s.seeded_graph) {
            self.model = candidate;
            self.apply_shared_graph(&Arc::new(g));
            self.pending_seed_publish = self.seeded_graph.is_some();
        } else {
            candidate
                .set_num_nodes(s.num_nodes)
                .map_err(|e| e.to_string())?;
            candidate.set_beta(s.beta).map_err(|e| e.to_string())?;
            candidate.set_gamma(s.gamma).map_err(|e| e.to_string())?;
            self.model = candidate;
            self.pending_seed_publish = false;
            self.reset_network();
        }
        self.is_running = s.is_running;
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.show_ui(ui);
        });
    }

    fn show_ui(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.sync_from_bus(&ctx);

        if self.is_running {
            self.model.step_with_rng(&mut self.rng);
            self.trace_step = self.trace_step.saturating_add(1);
            self.record_trace_sample();
            #[cfg(test)]
            {
                self.sim_steps = self.sim_steps.saturating_add(1);
            }
            ctx.request_repaint();
        }

        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                self.draw_controls_column(ui);
            });
            ui.separator();
            ui.vertical(|ui| {
                self.draw_graph_column(ui);
            });
        });

        self.flush_trace_to_bus(&ctx);
    }
}

// [cite:algorithmic_information_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "NetworkPropagationTool",
        display_name: "Network Propagation",
        domain: "epidemiology",
        tags: &[],
        build: || Box::new(NetworkPropagationTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for NetworkPropagationTool {
    fn theory_description(&self) -> String {
        let (mean_k, var_k, r0_het) = self.degree_moments_and_r0();
        format!(
            "Stochastic contact network epidemic propagation with heterogeneous reproduction number $R_0^{{\\text{{het}}}} = \\frac{{\\beta}}{{\\gamma}} \\frac{{\\langle k^2 \\rangle - \\langle k \\rangle}}{{\\langle k \\rangle}} = {r0_het:.2}$ ($\\langle k \\rangle = {mean_k:.2}$, $\\operatorname{{Var}}(k) = {var_k:.2}$)."
        )
    }
    fn phonetic_description(&self) -> String {
        "Network epidemic propagation and contact graph theory".into()
    }
    fn theory_citation(&self) -> String {
        "Epidemic processes in complex networks (Pastor-Satorras et al., 2015); Networks (Newman, 2010)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_propagation_step_once_keeps_paused() {
        let mut tool = NetworkPropagationTool::default();
        let _ = tool.model.set_beta(1.0);
        assert!(!tool.is_running);
        tool.step_once();
        assert!(!tool.is_running);
    }
}

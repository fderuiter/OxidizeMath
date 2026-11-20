use crate::accessibility::AccessibleHoverText;
use crate::framework::{InteractiveTool, PlaybackAction, PlaybackTransport};
use eframe::egui;
use math_explorer::epidemiology::networks::{NetworkEpidemicModel, NodeState};

pub struct NetworkPropagationTool {
    model: NetworkEpidemicModel,
    rng: oxidize_core::rng::OxidizeRng,
    is_running: bool,
    #[cfg(test)]
    sim_steps: u64,
}

impl Default for NetworkPropagationTool {
    fn default() -> Self {
        Self {
            model: NetworkEpidemicModel::new(50, 0.05, 0.02),
            rng: oxidize_core::rng::OxidizeRng::default(),
            is_running: false,
            #[cfg(test)]
            sim_steps: 0,
        }
    }
}

impl NetworkPropagationTool {
    fn reset_network(&mut self) {
        self.rng = oxidize_core::rng::OxidizeRng::default();
        self.model.initialize_geometric_graph_with_rng(&mut self.rng);
        self.is_running = false;
        #[cfg(test)]
        {
            self.sim_steps = 0;
        }
    }

    pub(crate) fn step_once(&mut self) {
        self.is_running = false;
        self.model.step_with_rng(&mut self.rng);
        #[cfg(test)]
        {
            self.sim_steps = self.sim_steps.saturating_add(1);
        }
    }

    #[cfg(test)]
    pub(crate) fn sim_steps(&self) -> u64 {
        self.sim_steps
    }

    fn draw_controls_column(&mut self, ui: &mut egui::Ui) {
        ui.heading("Controls");
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
        let mut num_nodes = self.model.num_nodes();
        if ui
            .add(egui::Slider::new(&mut num_nodes, 10..=200).text("Nodes"))
            .changed()
        {
            let _ = self.model.set_num_nodes(num_nodes);
            self.reset_network();
        }

        let mut beta = self.model.beta();
        ui.add(egui::Slider::new(&mut beta, 0.0..=1.0).text("Transmission (beta)"));
        let _ = self.model.set_beta(beta);

        let mut gamma = self.model.gamma();
        ui.add(egui::Slider::new(&mut gamma, 0.0..=1.0).text("Recovery (gamma)"));
        let _ = self.model.set_gamma(gamma);

        ui.separator();
        let count_state = |target: NodeState| {
            self.model
                .states()
                .iter()
                .filter(|&&s| s == target)
                .count()
        };
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.label(
            egui::RichText::new(format!("Susceptible: {}", count_state(NodeState::Susceptible)))
                .color(palette.axis_z),
        );
        ui.label(
            egui::RichText::new(format!("Infected: {}", count_state(NodeState::Infected)))
                .color(palette.axis_x),
        );
        ui.label(
            egui::RichText::new(format!("Recovered: {}", count_state(NodeState::Recovered)))
                .color(palette.axis_y),
        );
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
        let summary = format!("{} network nodes", self.model.num_nodes());
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
                    let p1 =
                        center + egui::vec2(self.model.positions()[i][0], self.model.positions()[i][1]);
                    let p2 =
                        center + egui::vec2(self.model.positions()[j][0], self.model.positions()[j][1]);
                    painter.line_segment([p1, p2], (1.0, palette.graph_edge));
                }
            }
        }

        for i in 0..self.model.num_nodes() {
            let p = center + egui::vec2(self.model.positions()[i][0], self.model.positions()[i][1]);
            let color = match self.model.states()[i] {
                NodeState::Susceptible => palette.axis_z,
                NodeState::Infected => palette.axis_x,
                NodeState::Recovered => palette.axis_y,
            };
            painter.circle_filled(p, 5.0, color);
            painter.circle_stroke(p, 5.0, (1.0, palette.node_stroke));
        }
    }
}

const fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NetPropState {
    num_nodes: usize,
    beta: f64,
    gamma: f64,
    #[serde(default, skip_serializing_if = "is_false")]
    is_running: bool,
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
        serde_json::to_string(&NetPropState {
            num_nodes: self.model.num_nodes(),
            beta: self.model.beta(),
            gamma: self.model.gamma(),
            is_running: self.is_running,
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
        if !(10..=200).contains(&s.num_nodes)
            || !s.beta.is_finite()
            || !(0.0..=1.0).contains(&s.beta)
            || !s.gamma.is_finite()
            || !(0.0..=1.0).contains(&s.gamma)
        {
            return Err("Network propagation parameters out of bounds".into());
        }
        let mut candidate = NetworkEpidemicModel::new(50, 0.05, 0.02);
        candidate
            .set_num_nodes(s.num_nodes)
            .map_err(|e| e.to_string())?;
        candidate.set_beta(s.beta).map_err(|e| e.to_string())?;
        candidate.set_gamma(s.gamma).map_err(|e| e.to_string())?;
        self.model = candidate;
        self.reset_network();
        self.is_running = s.is_running;
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        eframe::egui::CentralPanel::default().show_inside(ui, |ui| {
            self.show_ui(ui);
        });
    }

    fn show_ui(&mut self, ui: &mut egui::Ui) {
        if self.is_running {
            self.model.step_with_rng(&mut self.rng);
            #[cfg(test)]
            {
                self.sim_steps = self.sim_steps.saturating_add(1);
            }
            ui.ctx().request_repaint();
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
        "Stochastic contact network epidemic propagation: infection transmission across graph edges with per-contact probability $\\beta$, recovery rate $\\gamma$, and state transitions $S \\to I \\to R$ conditioned on local node degree.".into()
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

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Graph Epidemic Spread",
    "domain": "epidemiology",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "num_nodes": 60,
    "beta": 0.15,
    "gamma": 0.05
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "network_propagation_starter",
        title: "Graph Epidemic Spread",
        description: "Stochastic SIR contagion propagation across a random geometric contact network.",
        category: "Network Epidemiology",
        icon: "🕸️",
        domain: "epidemiology",
        tool_name: Some("NetworkPropagationTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

use crate::accessibility::AccessibleHoverText;
use crate::framework::InteractiveTool;
use eframe::egui;
use math_explorer::epidemiology::networks::{NetworkEpidemicModel, NodeState};

pub struct NetworkPropagationTool {
    model: NetworkEpidemicModel,
    is_running: bool,
    #[cfg(test)]
    sim_steps: u64,
}

impl Default for NetworkPropagationTool {
    fn default() -> Self {
        Self {
            model: NetworkEpidemicModel::new(50, 0.05, 0.02),
            is_running: false,
            #[cfg(test)]
            sim_steps: 0,
        }
    }
}

impl NetworkPropagationTool {
    fn reset_network(&mut self) {
        let mut rng = oxidize_core::rng::OxidizeRng::default();
        self.model.initialize_geometric_graph_with_rng(&mut rng);
        self.is_running = false;
    }
}

#[cfg(test)]
fn is_zero_u64(val: &u64) -> bool {
    *val == 0
}

#[derive(serde::Serialize, serde::Deserialize)]
struct NetPropState {
    num_nodes: usize,
    beta: f64,
    gamma: f64,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    is_running: bool,
    #[cfg(test)]
    #[serde(default, skip_deserializing, skip_serializing_if = "is_zero_u64")]
    sim_steps: u64,
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

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&NetPropState {
            num_nodes: self.model.num_nodes(),
            beta: self.model.beta(),
            gamma: self.model.gamma(),
            is_running: self.is_running,
            #[cfg(test)]
            sim_steps: self.sim_steps,
        })
        .ok()
    }

    fn load_state(&mut self, state: &str) {
        if let Ok(s) = serde_json::from_str::<NetPropState>(state) {
            let _ = self.model.set_num_nodes(s.num_nodes);
            let _ = self.model.set_beta(s.beta);
            let _ = self.model.set_gamma(s.gamma);
            self.reset_network();
            self.is_running = s.is_running;
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            self.show_ui(ui);
        });
    }
    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show_ui(&mut self, ui: &mut egui::Ui) {
        if self.is_running {
            let mut rng = oxidize_core::rng::OxidizeRng::default();
            self.model.step_with_rng(&mut rng);
            #[cfg(test)]
            {
                self.sim_steps = self.sim_steps.saturating_add(1);
            }
            ui.ctx().request_repaint();
        }

        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.heading("Controls");
                ui.separator();

                if ui
                    .button(if self.is_running {
                        "⏸ Pause"
                    } else {
                        "▶ Start"
                    })
                    .clicked()
                {
                    self.is_running = !self.is_running;
                }
                if ui.button("↻ Reset Network").clicked() {
                    self.reset_network();
                }

                ui.separator();
                let mut changed = false;
                let mut num_nodes = self.model.num_nodes();
                changed |= ui
                    .add(egui::Slider::new(&mut num_nodes, 10..=200).text("Nodes"))
                    .changed();
                if changed {
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
                // Statistics
                let s_count = self
                    .model
                    .states()
                    .iter()
                    .filter(|&&s| s == NodeState::Susceptible)
                    .count();
                let i_count = self
                    .model
                    .states()
                    .iter()
                    .filter(|&&s| s == NodeState::Infected)
                    .count();
                let r_count = self
                    .model
                    .states()
                    .iter()
                    .filter(|&&s| s == NodeState::Recovered)
                    .count();

                let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
                ui.label(
                    egui::RichText::new(format!("Susceptible: {}", s_count)).color(palette.axis_z),
                );
                ui.label(
                    egui::RichText::new(format!("Infected: {}", i_count)).color(palette.axis_x),
                );
                ui.label(
                    egui::RichText::new(format!("Recovered: {}", r_count)).color(palette.axis_y),
                );
            });

            ui.separator();

            ui.vertical(|ui| {
                let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
                ui.heading("Network Visualization");
                let (response, painter) =
                    ui.allocate_painter(ui.available_size(), egui::Sense::hover());
                let _ = response
                    .clone()
                    .accessible_hover_text("Network Propagation Visualization");
                let rect = response.rect;
                let center = rect.center();

                // Draw edges
                for i in 0..self.model.num_nodes() {
                    for &j in &self.model.adjacency()[i] {
                        if i < j {
                            let p1 = center
                                + egui::vec2(
                                    self.model.positions()[i][0],
                                    self.model.positions()[i][1],
                                );
                            let p2 = center
                                + egui::vec2(
                                    self.model.positions()[j][0],
                                    self.model.positions()[j][1],
                                );
                            painter.line_segment([p1, p2], (1.0, palette.graph_edge));
                        }
                    }
                }

                // Draw nodes
                for i in 0..self.model.num_nodes() {
                    let p = center
                        + egui::vec2(self.model.positions()[i][0], self.model.positions()[i][1]);
                    let color = match self.model.states()[i] {
                        NodeState::Susceptible => palette.axis_z,
                        NodeState::Infected => palette.axis_x,
                        NodeState::Recovered => palette.axis_y,
                    };
                    painter.circle_filled(p, 5.0, color);
                    painter.circle_stroke(p, 5.0, (1.0, palette.node_stroke));
                }
            });
        });
    }
}

// [cite:algorithmic_information_rust]


inventory::submit! {
    crate::framework::ToolMetadata {
        name: "NetworkPropagationTool",
        domain: "epidemiology",
        tags: &[],
        build: || Box::new(NetworkPropagationTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for NetworkPropagationTool {
    fn theory_description(&self) -> String {
        "Stochastic contact network epidemic propagation: infection transmission across graph edges with per-contact probability $\\beta$, recovery rate $\\gamma$, and state transitions $S \\to I \\to R$ conditioned on local node degree.".into()
    }
    fn phonetic_description(&self) -> String { "Network epidemic propagation and contact graph theory".into() }
    fn theory_citation(&self) -> String { "Epidemic processes in complex networks (Pastor-Satorras et al., 2015); Networks (Newman, 2010)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Graph Epidemic Spread",
    "domain": "epidemiology",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "infection_prob": 0.15,
    "recovery_prob": 0.05
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "network_propagation_starter",
        title: "Graph Epidemic Spread",
        description: "Stochastic infection contagion percolation across scale-free contact networks.",
        category: "Network Epidemiology",
        icon: "🕸️",
        domain: "epidemiology",
        tool_name: Some("NetworkPropagationTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

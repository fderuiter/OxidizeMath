#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::tabs::{ExplorerTab, TabDescriptor};
use eframe::egui;

pub mod algorithm_visualizer;
pub mod graph_editor;
pub mod io_helpers;
pub mod network_metrics;
pub mod plot_export;
pub mod shared_topology;
#[cfg(test)]
mod unification_tests;

#[cfg(test)]
pub use shared_topology::parse_topology_document;
pub use shared_topology::{SharedGraphTopology, TopologyHandoffTarget};

pub struct GraphTheoryTab {
    framework: crate::framework::SimulationFramework,
    /// Shared custom network topology buffer synchronized across Editor, Visualizer, and Metrics.
    pub shared_topology: SharedGraphTopology,
}

impl Default for GraphTheoryTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("graph_theory"),
            shared_topology: SharedGraphTopology::default(),
        }
    }
}

impl GraphTheoryTab {
    fn process_pending_handoff(&mut self, ctx: &egui::Context) {
        let Some(pending) = shared_topology::take_any_pending_handoff(ctx) else {
            return;
        };
        self.shared_topology = pending.topology.clone();
        let target_name = pending.target.display_name();
        let _ = self.framework.select_tool_by_name(target_name);
        if let Some(tool) = self.framework.active_tool.as_mut() {
            if let Some(json) = serialize_handoff_for_target(pending.target, &pending.topology) {
                let _ = tool.try_load_state(&json);
            }
        }
        ctx.request_repaint();
    }
}

fn serialize_handoff_for_target(
    target: TopologyHandoffTarget,
    topology: &SharedGraphTopology,
) -> Option<String> {
    match target {
        TopologyHandoffTarget::GraphEditor => graph_editor::editor_state_json_from_shared(topology),
        TopologyHandoffTarget::AlgorithmVisualizer => {
            algorithm_visualizer::visualizer_state_json_from_shared(topology)
        }
        TopologyHandoffTarget::NetworkMetrics => {
            network_metrics::metrics_state_json_from_shared(topology)
        }
    }
}

impl ExplorerTab for GraphTheoryTab {
    fn name(&self) -> &'static str {
        "Graph Theory"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        shared_topology::sync_shared_buffer_to_ctx(&ctx, &self.shared_topology);
        self.framework.show_inside(ui, "graph_theory");
        self.process_pending_handoff(&ctx);
    }

    fn state_revision(&self) -> u64 {
        self.framework.state_revision()
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        self.framework.load_state(state);
    }

    fn select_tool_by_name(&mut self, name: &str) -> bool {
        self.framework.select_tool_by_name(name)
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.framework.focused_canvas_id()
    }

    fn framework(&self) -> Option<&crate::framework::SimulationFramework> {
        Some(&self.framework)
    }

    fn framework_mut(&mut self) -> Option<&mut crate::framework::SimulationFramework> {
        Some(&mut self.framework)
    }
}

inventory::submit! {
    TabDescriptor {
        id: "graph_theory",
        order: 23,
        build: || Box::new(GraphTheoryTab::default()),
    }
}

// [cite:graph_parameters_rust]

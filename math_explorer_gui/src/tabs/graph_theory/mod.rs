// @explorer_feature = "pure_math"
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod algorithm_visualizer;
pub mod graph_editor;
pub mod network_metrics;
pub mod topology_sync;
#[cfg(test)]
mod topology_sync_tests;

pub struct GraphTheoryTab {
    framework: crate::framework::SimulationFramework,
    sync_hub: topology_sync::GraphTopologySyncHub,
}

impl Default for GraphTheoryTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("graph_theory"),
            sync_hub: topology_sync::GraphTopologySyncHub::default(),
        }
    }
}

#[cfg(test)]
impl GraphTheoryTab {
    /// Imports a `.dot`, `.graphml`, or `.csv`/`.tsv` topology directly into this tab and synchronizes active tools.
    pub fn import_topology_text(&mut self, filename: &str, content: &str) -> Result<(), String> {
        self.sync_hub
            .import_topology_into_framework(&mut self.framework, filename, content)
    }

    /// Returns the currently active shared [`math_explorer::pure_math::graph_theory::graph_io::GraphTopologyDocument`].
    #[must_use]
    pub fn shared_topology(
        &self,
    ) -> Option<&math_explorer::pure_math::graph_theory::graph_io::GraphTopologyDocument> {
        self.sync_hub.shared_document()
    }
}

impl ExplorerTab for GraphTheoryTab {
    fn name(&self) -> &'static str {
        "Graph Theory"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.sync_hub.show_sync_bar(ui, &mut self.framework);
        self.sync_hub.sync_before_frame(&mut self.framework);
        self.framework.show_inside(ui, "graph_theory");
        self.sync_hub.sync_after_frame(&self.framework);
    }

    fn state_revision(&self) -> u64 {
        self.framework.state_revision()
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        self.framework.load_state(state);
        self.sync_hub.sync_after_frame(&self.framework);
    }

    fn select_tool_by_name(&mut self, name: &str) -> bool {
        let selected = self.framework.select_tool_by_name(name);
        if selected {
            self.sync_hub.sync_before_frame(&mut self.framework);
        }
        selected
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

// [cite:graph_parameters_rust]

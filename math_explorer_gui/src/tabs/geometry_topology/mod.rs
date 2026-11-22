// @explorer_feature = "pure_math"
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod curvature_heatmap;
pub mod export_utils;
#[cfg(feature = "applied")]
pub mod implicit_cad;
pub mod surface_viewer;
pub mod vietoris_rips;

pub struct GeometryTopologyTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for GeometryTopologyTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("geometry_topology"),
        }
    }
}

impl ExplorerTab for GeometryTopologyTab {
    fn name(&self) -> &'static str {
        "Geometry & Topology"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "geometry_topology");
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

    fn framework(&self) -> Option<&crate::framework::SimulationFramework> {
        Some(&self.framework)
    }

    fn framework_mut(&mut self) -> Option<&mut crate::framework::SimulationFramework> {
        Some(&mut self.framework)
    }
}

// [cite:graph_parameters_rust]

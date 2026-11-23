// @explorer_feature = "physics"
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod lattice_boltzmann;
pub mod lbm_fairing;
pub mod lbm_fairing_ui;
pub mod potential_flow;
pub mod turbulence;

pub struct FluidDynamicsTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for FluidDynamicsTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("fluid_dynamics"),
        }
    }
}

impl ExplorerTab for FluidDynamicsTab {
    fn name(&self) -> &'static str {
        "Fluid Dynamics"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        #[cfg(all(feature = "physics", feature = "biology"))]
        if self
            .framework
            .active_tool
            .as_ref()
            .is_some_and(|t| t.name() == "Lattice Boltzmann (Demo)")
        {
            use crate::accessibility::AccessibleHoverText;
            let bar_id = crate::framework::scoped_panel_id(ui, "lbm_morphogenesis_crosslink");
            egui::TopBottomPanel::bottom(bar_id).show_inside(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label("🧬 Coupled Advection-Reaction-Diffusion:");
                    if ui
                        .button("Open Vortex Morphogenesis in Biology →")
                        .accessible_hover_text(
                            "Jump to the coupled D2Q9 Lattice Boltzmann and Turing morphogenesis laboratory",
                        )
                        .clicked()
                    {
                        crate::app::request_cross_tab_navigation(
                            ui.ctx(),
                            "Morphogenesis",
                            Some("VortexMorphogenesisTool"),
                        );
                    }
                });
            });
        }
        self.framework.show_inside(ui, "fluid_dynamics");
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

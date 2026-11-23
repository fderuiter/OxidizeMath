// @explorer_feature = "physics"
// @explorer_order = 12
#[cfg(feature = "biology")]
use crate::accessibility::{announce_live_status, AccessibleHoverText};
#[cfg(feature = "biology")]
use crate::framework::{
    request_open_advective_morphogenesis, AdvectiveMorphogenesisHandoff, FairingStamp,
    LBM_CELL_COUNT,
};
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod lattice_boltzmann;
pub mod lattice_boltzmann_sculpt;
pub mod lbm_fairing;
pub mod lbm_fairing_ui;
pub mod potential_flow;
mod potential_flow_brush;
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

impl FluidDynamicsTab {
    /// Extracts the active or saved `100 x 50` Lattice Boltzmann obstacle mask and viscosity
    /// and queues a cross-tab launch into `AdvectiveMorphogenesisTool`.
    #[cfg(feature = "biology")]
    pub fn launch_advective_morphogenesis_studio(&self, ctx: &egui::Context) {
        let mut handoff = AdvectiveMorphogenesisHandoff::default();
        let active_lbm_json = self
            .framework
            .active_tool
            .as_ref()
            .filter(|t| t.name() == "Lattice Boltzmann Fluid")
            .and_then(|t| t.save_state())
            .or_else(|| {
                self.framework
                    .saved_tool_states
                    .get("Lattice Boltzmann Fluid")
                    .cloned()
            });
        if let Some(json) = active_lbm_json {
            if let Ok(parsed) = serde_json::from_str::<lbm_fairing::LbmSerializedState>(&json) {
                handoff.viscosity = parsed.viscosity.clamp(0.018, 0.25);
                handoff.mask_hex = parsed.mask_hex;
            }
        }
        if handoff.mask_hex.is_none() {
            let mut mask = [false; LBM_CELL_COUNT];
            FairingStamp::BluffCylinder.write_mask(&mut mask);
            handoff.mask_hex = Some(crate::framework::encode_mask_hex(&mask));
        }
        request_open_advective_morphogenesis(ctx, handoff);
        announce_live_status(
            ctx,
            "Launching Advective Morphogenesis Studio on 100 by 50 coupled D2Q9 lattice",
        );
    }

    fn render_cross_domain_shortcut_strip(&self, ui: &mut egui::Ui) {
        #[cfg(feature = "biology")]
        ui.horizontal_wrapped(|ui| {
            let btn = ui
                .button("🧬 Launch Advective Morphogenesis (100×50)")
                .accessible_hover_text(
                    "Open the co-stepped D2Q9 + Schnakenberg / Gray-Scott studio in Morphogenesis with the current 100×50 obstacle wake.",
                );
            if btn.clicked() {
                self.launch_advective_morphogenesis_studio(ui.ctx());
            }
        });
        #[cfg(not(feature = "biology"))]
        let _ = ui;
    }
}

impl ExplorerTab for FluidDynamicsTab {
    fn name(&self) -> &'static str {
        "Fluid Dynamics"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.render_cross_domain_shortcut_strip(ui);
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

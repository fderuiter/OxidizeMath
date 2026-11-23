#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:biology]

use crate::tabs::{ExplorerTab, TabDescriptor};
use eframe::egui;

pub mod dynamics;
pub mod evolution;

/// Static-inventory Biology tab hosting [`dynamics::BiochemicalDynamicsTool`] and
/// [`evolution::EvolutionaryBiologyTool`].
pub struct BiochemicalTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for BiochemicalTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("biochemical"),
        }
    }
}

impl ExplorerTab for BiochemicalTab {
    fn name(&self) -> &'static str {
        "Biochemical Dynamics"
    }

    fn academic_domain(&self) -> &'static str {
        "Biology"
    }

    fn url_slug(&self) -> &'static str {
        "biochemical"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "biochemical");
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

inventory::submit! {
    TabDescriptor {
        id: "biochemical",
        order: 43,
        build: || Box::new(BiochemicalTab::default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::InteractiveTool;

    fn assert_tool_roundtrip(tool: &mut dyn InteractiveTool, custom_json: &str) {
        let default_saved = tool.save_state().expect("default save_state");
        tool.load_state(custom_json);
        let updated_saved = tool.save_state().expect("updated save_state");
        assert_ne!(updated_saved, default_saved);
        let expected: serde_json::Value = serde_json::from_str(custom_json).unwrap();
        let actual: serde_json::Value = serde_json::from_str(&updated_saved).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_biology_botanical_and_biochemical_state_roundtrips() {
        assert_tool_roundtrip(
            &mut crate::tabs::morphogenesis::PhyllotaxisLSystemTool::default(),
            r#"{"mode":"DualSplit","divergence_preset":"GoldenAngle","divergence_deg":137.507764,"primordia_count":350,"radial_scale":0.85,"show_parastichies":true,"lsystem_preset":"ParametricDichotomousTree","generations":6,"branch_angle_deg":24.0,"contraction_r1":0.78,"contraction_r2":0.66}"#,
        );
        assert_tool_roundtrip(
            &mut dynamics::BiochemicalDynamicsTool::default(),
            r#"{"view":"CoupledSplit","v_max":2.8,"k_m":0.9,"hill_n":3.2,"max_substrate":12.0,"substrate_reservoir":2.5,"feed_rate":0.25,"degradation_rate":0.4,"diffusion_activator":0.06,"diffusion_substrate":0.3,"grid_points":64,"dt":0.02,"playing":false}"#,
        );
        assert_tool_roundtrip(
            &mut evolution::EvolutionaryBiologyTool::default(),
            r#"{"view":"TrajectoryAndPhase","resource_value":2.0,"injury_cost":5.0,"initial_hawk_freq":0.8,"horizon":25.0,"dt":0.05,"show_ensemble":true}"#,
        );
        let mut biochem_tab = BiochemicalTab::default();
        assert!(biochem_tab.select_tool_by_name("EvolutionaryBiologyTool"));
        let saved = biochem_tab.save_state().expect("biochem save_state");
        let mut restored = BiochemicalTab::default();
        let default_saved = restored.save_state().expect("default tab save_state");
        restored.load_state(&saved);
        let updated_saved = restored.save_state().expect("updated tab save_state");
        assert_ne!(updated_saved, default_saved);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&updated_saved).unwrap(),
            serde_json::from_str::<serde_json::Value>(&saved).unwrap()
        );
    }
}

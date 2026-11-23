// @explorer_feature = "pure_math"
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod chladni_wave_tank;
pub mod complex_mapping;
pub mod ode;
pub mod riemann;
pub mod wave_unified;

pub struct AnalysisTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for AnalysisTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("analysis"),
        }
    }
}

impl ExplorerTab for AnalysisTab {
    fn name(&self) -> &'static str {
        "Analysis & Calculus"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let wave_active = self
            .framework
            .active_tool
            .as_ref()
            .map(|t| t.name() == "2D Acoustic Wave Tank (Unified)")
            .unwrap_or(false);
        ui.horizontal(|ui| {
            if ui
                .selectable_label(wave_active, "🌊 2D Wave Twin Lab")
                .on_hover_text(
                    "Launch the Unified 2D Acoustic Wave Tank with timeline scrubbing and raster shadow-twin difference maps",
                )
                .clicked()
            {
                self.framework.active_tool = Some(wave_unified::WaveUnified::build_tool());
            }
        });
        self.framework.show_inside(ui, "analysis");
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

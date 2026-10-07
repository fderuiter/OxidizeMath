// @explorer_feature = "pure_math"
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod bankroll_growth;
pub mod bet_size;

pub struct FinancialMathTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for FinancialMathTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("financial_math"),
        }
    }
}

impl ExplorerTab for FinancialMathTab {
    fn name(&self) -> &'static str {
        "Financial Math"
    }

    fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.framework.show(ctx, "financial_math");
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        self.framework.load_state(state);
    }
}

// [cite:clinical_trials_statistics]

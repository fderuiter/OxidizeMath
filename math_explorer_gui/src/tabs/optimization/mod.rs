// @explorer_feature = "applied"
use crate::accessibility::AccessibleHoverText;
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod constrained_tool;
pub mod geometry;
pub mod lp_tool;
pub mod pareto_tool;
pub mod presets;
pub mod qp_tool;

/// Interactive Optimization tab hosting Linear Programming (Simplex), Active-Set QP,
/// Nonlinear Constrained Contours, and Multi-Objective Pareto workbenches.
pub struct OptimizationTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for OptimizationTab {
    /// Instantiates the Optimization framework domain (`[cite:lorahub]`).
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("optimization"),
        }
    }
}

pub(super) fn render_tool_header(ui: &mut egui::Ui, tool_name: &str) {
    ui.horizontal(|ui| {
        ui.heading(tool_name);
        let help_btn = ui
            .add(egui::Button::new("❓").small())
            .accessible_hover_text(format!("Toggle Theory Context Portal for {tool_name}"));
        if help_btn.clicked() {
            let ctx = ui.ctx();
            let next = !ctx.data(|d| {
                d.get_temp::<bool>(egui::Id::new("SHOW_THEORY_PORTAL"))
                    .unwrap_or(false)
            });
            ctx.data_mut(|d| {
                d.insert_temp(egui::Id::new("SHOW_THEORY_PORTAL"), next);
                if let Some(inst_key) =
                    d.get_temp::<egui::Id>(egui::Id::new("CURRENT_THEORY_PORTAL_INST_ID"))
                {
                    d.insert_temp(inst_key, next);
                }
            });
        }
    });
    ui.separator();
}

impl ExplorerTab for OptimizationTab {
    fn name(&self) -> &'static str {
        "Optimization"
    }

    fn academic_domain(&self) -> &'static str {
        "Applied Math"
    }

    fn url_slug(&self) -> &'static str {
        "optimization"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "optimization");
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

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.framework.focused_canvas_id()
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

// [cite:lorahub]
// [cite:advanced_linear_algebra]

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::constrained_tool::ConstrainedContourTool;
    use super::lp_tool::LinearProgrammingTool;
    use super::pareto_tool::ParetoFrontierTool;
    use super::qp_tool::QuadraticProgrammingTool;
    use super::*;
    use crate::app::{MathExplorerApp, ThemeMode};
    use math_explorer::pure_math::analysis::optimization::LpStatus;

    #[test]
    fn test_optimization_tab_autodiscovery_and_workbenches() {
        let _guard = crate::reflective_ui::tests::lock_test_mutex();
        let app = MathExplorerApp::default();
        let tab = app
            .tabs
            .iter()
            .find(|t| t.url_slug() == "optimization")
            .expect("OptimizationTab must be auto-discovered by build.rs");
        assert_eq!(tab.name(), "Optimization");
        assert_eq!(tab.academic_domain(), "Applied Math");

        let mut local = OptimizationTab::default();
        for tool_id in [
            "LinearProgrammingTool",
            "QuadraticProgrammingTool",
            "ConstrainedContourTool",
            "ParetoFrontierTool",
        ] {
            assert!(local.select_tool_by_name(tool_id));
            assert_eq!(local.active_tool_id(), Some(tool_id));
        }
    }

    #[test]
    fn test_lp_qp_constrained_and_pareto_solvers_produce_valid_solutions() {
        let mut lp = LinearProgrammingTool::default();
        assert_eq!(lp.last_status(), LpStatus::Optimal);
        assert!((lp.optimal_objective() - 33.0).abs() < 1e-6);
        lp.apply_infeasible_preset();
        assert_eq!(lp.last_status(), LpStatus::Infeasible);
        lp.apply_unbounded_preset();
        assert_eq!(lp.last_status(), LpStatus::Unbounded);

        let qp = QuadraticProgrammingTool::default();
        assert!(qp.is_optimal());
        assert!(qp.optimal_point()[0] + qp.optimal_point()[1] <= 4.0 + 1e-6);

        let mut cc = ConstrainedContourTool::default();
        for idx in 0..4 {
            cc.set_solver_index(idx);
            assert!(cc.optimal_point()[0].is_finite() && cc.optimal_point()[1].is_finite());
        }

        let mut pf = ParetoFrontierTool::default();
        assert!(pf.frontier_len() >= 5);
        pf.set_problem_mode(2);
        assert!(pf.frontier_len() >= 3);
    }

    #[test]
    fn test_optimization_tab_renders_across_all_three_themes() {
        let _guard = crate::reflective_ui::tests::lock_test_mutex();
        let ctx = egui::Context::default();
        let mut frame = eframe::Frame::_new_kittest();
        let mut app = MathExplorerApp::default();
        let idx = app
            .tabs
            .iter()
            .position(|t| t.url_slug() == "optimization")
            .expect("OptimizationTab index");
        app.select_tab(&ctx, idx);
        for tool_id in [
            "LinearProgrammingTool",
            "QuadraticProgrammingTool",
            "ConstrainedContourTool",
            "ParetoFrontierTool",
        ] {
            assert!(app.tabs[idx].select_tool_by_name(tool_id));
            for theme in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
                app.theme = theme;
                let _ = ctx.run(egui::RawInput::default(), |ctx| {
                    eframe::App::update(&mut app, ctx, &mut frame);
                });
            }
        }
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn wasm_optimization_tab_workbenches_smoke() {
        let mut tab = OptimizationTab::default();
        assert_eq!(tab.url_slug(), "optimization");
        for id in [
            "LinearProgrammingTool",
            "QuadraticProgrammingTool",
            "ConstrainedContourTool",
            "ParetoFrontierTool",
        ] {
            assert!(tab.select_tool_by_name(id));
        }
    }
}

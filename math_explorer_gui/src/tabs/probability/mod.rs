#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::AccessibleHoverText;
use crate::tabs::{ExplorerTab, TabDescriptor};
use eframe::egui;

pub mod bayesian_tool;
pub mod dist_types;
pub mod distributions_tool;
pub mod factor_bp_tool;
pub mod presets;

/// Interactive Probability Workbench tab hosting `#![no_std]` distributions, conjugate Bayesian
/// inference, and factor-graph belief propagation powered by `engine_prob` and `engine_graph`.
pub struct ProbabilityWorkbenchTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for ProbabilityWorkbenchTab {
    /// Instantiates the Probability Workbench framework domain (`[cite:probability]`).
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("probability"),
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
            crate::framework::record_theory_portal_toggle(ctx, help_btn.id, next);
        }
    });
    ui.separator();
}

impl ExplorerTab for ProbabilityWorkbenchTab {
    fn name(&self) -> &'static str {
        "Probability Workbench"
    }

    fn academic_domain(&self) -> &'static str {
        "Pure Math"
    }

    fn url_slug(&self) -> &'static str {
        "probability"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "probability");
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
        id: "probability",
        order: 27,
        build: || Box::new(ProbabilityWorkbenchTab::default()),
    }
}

// [cite:probability]
// [cite:clinical_trials_statistics]
// [cite:graph_parameters_rust]

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::bayesian_tool::{ConjugateBayesianTool, ConjugateFamily};
    use super::distributions_tool::{DistFamily, DistributionsExplorerTool};
    use super::factor_bp_tool::{BpTopology, FactorBeliefPropagationTool};
    use super::*;
    use crate::app::{MathExplorerApp, ThemeMode};
    use crate::framework::InteractiveTool;

    #[test]
    fn test_probability_workbench_autodiscovery_and_tool_selection() {
        let _guard = crate::reflective_ui::tests::lock_test_mutex();
        let app = MathExplorerApp::default();
        let tab = app
            .tabs
            .iter()
            .find(|t| t.url_slug() == "probability")
            .expect("ProbabilityWorkbenchTab must register via inventory");
        assert_eq!(tab.name(), "Probability Workbench");
        assert_eq!(tab.academic_domain(), "Pure Math");

        let mut local = ProbabilityWorkbenchTab::default();
        for tool_id in [
            "DistributionsExplorerTool",
            "ConjugateBayesianTool",
            "FactorBeliefPropagationTool",
        ] {
            assert!(local.select_tool_by_name(tool_id));
            assert_eq!(local.active_tool_id(), Some(tool_id));
        }
    }

    #[test]
    fn test_distributions_and_conjugate_updates_match_exact_identities() {
        let _guard = crate::reflective_ui::tests::lock_test_mutex();
        let mut dist_tool = DistributionsExplorerTool::default();
        for fam in [
            DistFamily::Normal,
            DistFamily::Gamma,
            DistFamily::Beta,
            DistFamily::StudentsT,
            DistFamily::ChiSquared,
            DistFamily::Poisson,
            DistFamily::Binomial,
        ] {
            dist_tool.set_family_for_test(fam);
            assert!(dist_tool.metrics().quantile_x.is_finite());
            assert!(dist_tool.metrics().mean.is_finite());
            assert!(dist_tool.metrics().variance > 0.0);
        }

        let mut bayes_tool = ConjugateBayesianTool::default();
        for fam in [
            ConjugateFamily::BetaBinomial,
            ConjugateFamily::GammaPoisson,
            ConjugateFamily::NormalNormal,
            ConjugateFamily::NormalInverseGamma,
        ] {
            bayes_tool.set_family_for_test(fam);
            let m = bayes_tool.metrics();
            assert!(m.ci_low < m.posterior_mean && m.posterior_mean < m.ci_high);
            assert!(m.bayes_factor_10.is_finite() && m.bayes_factor_10 > 0.0);
        }
    }

    #[test]
    fn test_factor_bp_zero_alloc_under_verified_engine() {
        let _guard = crate::reflective_ui::tests::lock_test_mutex();
        let mut bp_tool = FactorBeliefPropagationTool::default();
        bp_tool.set_topology_for_test(BpTopology::StarTree5);
        let _ = verified_engine::engine::VerifiedEngine::run_verified(|| {
            bp_tool.step_bp_zero_alloc();
        });
        let m = bp_tool.metrics();
        assert!(m.converged);
        assert!(m.log_partition.is_finite());
    }

    #[test]
    fn test_probability_workbench_renders_across_all_three_themes() {
        let _guard = crate::reflective_ui::tests::lock_test_mutex();
        let ctx = egui::Context::default();
        let mut frame = eframe::Frame::_new_kittest();
        let mut app = MathExplorerApp::default();
        let idx = app
            .tabs
            .iter()
            .position(|t| t.url_slug() == "probability")
            .unwrap();
        app.select_tab(&ctx, idx);
        for tool_id in [
            "DistributionsExplorerTool",
            "ConjugateBayesianTool",
            "FactorBeliefPropagationTool",
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

    #[test]
    fn test_probability_workbench_png_exports() {
        let _guard = crate::reflective_ui::tests::lock_test_mutex();
        let opts = egui_plot::export::ExportOptions::default();

        let mut dist_tool = DistributionsExplorerTool::default();
        dist_tool.set_family_for_test(DistFamily::StudentsT);
        let dist_png = egui_plot::export::export_custom_png(&dist_tool.export_datasets(), &opts)
            .expect("DistributionsExplorerTool PNG export must succeed");
        assert!(dist_png.starts_with(b"\x89PNG"));
        let _ = std::fs::write("/tmp/probability_distributions_preview.png", &dist_png);

        let mut bayes_tool = ConjugateBayesianTool::default();
        bayes_tool.set_family_for_test(ConjugateFamily::BetaBinomial);
        let bayes_png = egui_plot::export::export_custom_png(&bayes_tool.export_datasets(), &opts)
            .expect("ConjugateBayesianTool PNG export must succeed");
        assert!(bayes_png.starts_with(b"\x89PNG"));
        let _ = std::fs::write(
            "/tmp/probability_conjugate_bayesian_preview.png",
            &bayes_png,
        );

        let mut bp_tool = FactorBeliefPropagationTool::default();
        bp_tool.set_topology_for_test(BpTopology::StarTree5);
        let bp_png = egui_plot::export::export_custom_png(&bp_tool.export_datasets(), &opts)
            .expect("FactorBeliefPropagationTool PNG export must succeed");
        assert!(bp_png.starts_with(b"\x89PNG"));
        let _ = std::fs::write("/tmp/probability_factor_bp_preview.png", &bp_png);
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::factor_bp_tool::FactorBeliefPropagationTool;
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn wasm_probability_workbench_tools_and_zero_alloc_bp() {
        let mut tab = ProbabilityWorkbenchTab::default();
        assert_eq!(tab.url_slug(), "probability");
        assert!(tab.select_tool_by_name("DistributionsExplorerTool"));
        assert!(tab.select_tool_by_name("ConjugateBayesianTool"));
        assert!(tab.select_tool_by_name("FactorBeliefPropagationTool"));

        let mut bp = FactorBeliefPropagationTool::default();
        let _ = verified_engine::engine::VerifiedEngine::run_verified(|| {
            bp.step_bp_zero_alloc();
        });
        assert!(bp.metrics().converged);
    }
}

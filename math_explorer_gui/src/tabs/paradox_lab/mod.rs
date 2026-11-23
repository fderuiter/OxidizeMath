#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::AccessibleHoverText;
use crate::tabs::{ExplorerTab, TabDescriptor};
use eframe::egui;

pub mod braess;
mod braess_ui;
pub mod parrondo;
mod parrondo_ui;

/// Interactive Paradox Lab tab hosting Parrondo's ratchet and Braess's congestion paradox.
pub struct ParadoxLabTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for ParadoxLabTab {
    /// Instantiates the Paradox Lab framework domain (`[cite:paradox_lab]`).
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("paradox_lab"),
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

impl ExplorerTab for ParadoxLabTab {
    fn name(&self) -> &'static str {
        "Paradox Lab"
    }

    fn academic_domain(&self) -> &'static str {
        "Applied Math"
    }

    fn url_slug(&self) -> &'static str {
        "paradox_lab"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "paradox_lab");
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

inventory::submit! {
    TabDescriptor {
        id: "paradox_lab",
        order: 100,
        build: || Box::new(ParadoxLabTab::default()),
    }
}

// [cite:paradox_lab]
// [cite:graph_parameters_rust]

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::braess::{check_braess_fixed_point, BraessNetworkParadoxTool};
    use super::parrondo::{ParrondoParadoxTool, ParrondoVariant};
    use super::*;
    use crate::app::{MathExplorerApp, ThemeMode};

    #[test]
    fn test_paradox_lab_autodiscovery_and_tool_selection() {
        let _guard = crate::reflective_ui::tests::lock_test_mutex();
        let app = MathExplorerApp::default();
        let tab = app
            .tabs
            .iter()
            .find(|t| t.url_slug() == "paradox_lab")
            .expect("ParadoxLabTab must be auto-discovered by build.rs");
        assert_eq!(tab.name(), "Paradox Lab");
        assert_eq!(tab.academic_domain(), "Applied Math");

        let mut local = ParadoxLabTab::default();
        assert!(local.select_tool_by_name("ParrondoParadoxTool"));
        assert_eq!(local.active_tool_id(), Some("ParrondoParadoxTool"));
        assert!(local.select_tool_by_name("BraessNetworkParadoxTool"));
        assert_eq!(local.active_tool_id(), Some("BraessNetworkParadoxTool"));
    }

    #[test]
    fn test_parrondo_capital_and_history_drift_convergence() {
        let _guard = crate::reflective_ui::tests::lock_test_mutex();
        for variant in [
            ParrondoVariant::CapitalDependent,
            ParrondoVariant::HistoryDependent,
        ] {
            let mut tool = ParrondoParadoxTool::default();
            tool.set_test_configuration(variant, 0.003, 2048, 180);
            let m = tool.metrics();
            assert!(m.drift_a < 0.0 && m.drift_b < 0.0, "{variant:?}: {m:?}");
            assert!(m.drift_abb > 0.0 && m.drift_mix > 0.0, "{variant:?}: {m:?}");
            assert!(m.tensor_avg_p01 > 0.0 && m.tensor_mid_p01 > 0.0);
            let mc_slope = tool.abb_terminal_capital() / 180.0;
            assert!((mc_slope - m.drift_abb).abs() < 0.04);
        }
    }

    #[test]
    fn test_parrondo_and_braess_zero_alloc_under_verified_engine() {
        let _guard = crate::reflective_ui::tests::lock_test_mutex();
        let mut parrondo = ParrondoParadoxTool::default();
        parrondo.prepare_stepping();
        let mut braess = BraessNetworkParadoxTool::default();
        let _ = verified_engine::engine::VerifiedEngine::run_verified(|| {
            for _ in 0..16 {
                parrondo.step_parrondo_zero_alloc();
                braess.step_braess_zero_alloc();
            }
        });
        assert_eq!(parrondo.completed_steps(), 16);
    }

    #[test]
    fn test_braess_latency_nash_fixed_point_and_mfg_cfl() {
        let _guard = crate::reflective_ui::tests::lock_test_mutex();
        let mut tool = BraessNetworkParadoxTool::default();
        let open_m = tool.metrics();
        assert!((open_m.wardrop_latency - 2.0).abs() < 1e-6);
        assert!(open_m.wardrop_latency > open_m.closed_latency);
        assert!((open_m.price_of_anarchy - 4.0 / 3.0).abs() < 1e-6);
        assert!(open_m.nash_fixed_point);
        assert!(!check_braess_fixed_point(tool.state_ref(), [0.5, 0.5, 0.0]));

        tool.apply_optimal_toll_and_mfg(0.22, 49);
        assert!((tool.metrics().wardrop_latency - 1.5).abs() < 1e-6);
        assert!(tool.metrics().mfg_cfl_ratio < 0.25);
        assert!(tool.has_finite_mfg_profiles());
    }

    #[test]
    fn test_paradox_lab_renders_across_all_three_themes() {
        let _guard = crate::reflective_ui::tests::lock_test_mutex();
        let ctx = egui::Context::default();
        let mut frame = eframe::Frame::_new_kittest();
        let mut app = MathExplorerApp::default();
        let idx = app
            .tabs
            .iter()
            .position(|t| t.url_slug() == "paradox_lab")
            .unwrap();
        app.select_tab(&ctx, idx);
        for tool_id in ["ParrondoParadoxTool", "BraessNetworkParadoxTool"] {
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

    use super::braess::BraessNetworkParadoxTool;
    use super::parrondo::ParrondoParadoxTool;
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn wasm_paradox_lab_tools_and_zero_alloc_steps() {
        let mut tab = ParadoxLabTab::default();
        assert_eq!(tab.url_slug(), "paradox_lab");
        assert!(tab.select_tool_by_name("ParrondoParadoxTool"));
        assert!(tab.select_tool_by_name("BraessNetworkParadoxTool"));

        let mut parrondo = ParrondoParadoxTool::default();
        parrondo.prepare_stepping();
        let mut braess = BraessNetworkParadoxTool::default();
        let _ = verified_engine::engine::VerifiedEngine::run_verified(|| {
            parrondo.step_parrondo_zero_alloc();
            braess.step_braess_zero_alloc();
        });
    }
}

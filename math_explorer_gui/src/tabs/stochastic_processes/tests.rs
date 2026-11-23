#![cfg_attr(any(), verified(opt_out = "tests"))]

//! Unit, accessibility, URL hydration, and export tests for [`super::StochasticProcessesTab`].

use super::ctmc_queue_workbench::CtmcQueueWorkbenchTool;
use super::ctmc_queue_workbench::state::{ContinuousModelMode, PoissonVariant};
use super::dtmc_workbench::DtmcTensorWorkbenchTool;
use super::dtmc_workbench::state::DtmcModelPreset;
use super::hmm_ou_workbench::HmmOuWorkbenchTool;
use super::hmm_ou_workbench::state::CalibrationMode;
use super::StochasticProcessesTab;
use crate::app::MathExplorerApp;
use crate::framework::{ExportFormat, ExportStatusSeverity, InteractiveTool};
use crate::tabs::ExplorerTab;
use eframe::egui;

fn test_raw_input() -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    }
}

#[test]
fn test_stochastic_processes_tab_discovers_all_three_workbenches_and_renders_a11y_canvases() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut tab = StochasticProcessesTab::default();
    assert_eq!(tab.name(), "Stochastic Processes");
    assert_eq!(tab.academic_domain(), "Pure Math");
    assert_eq!(tab.url_slug(), "stochastic_processes");

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut frame = eframe::Frame::_new_kittest();

    for tool_id in [
        "DtmcTensorWorkbenchTool",
        "CtmcQueueWorkbenchTool",
        "HmmOuWorkbenchTool",
    ] {
        assert!(
            tab.select_tool_by_name(tool_id),
            "Failed to select {tool_id}"
        );
        let output = ctx.run(test_raw_input(), |c| {
            egui::CentralPanel::default().show(c, |ui| tab.show_inside(ui, &mut frame));
        });
        let ak = output
            .platform_output
            .accesskit_update
            .as_ref()
            .expect("AccessKit enabled");
        let has_graphics_doc = ak
            .nodes
            .iter()
            .any(|(_, n)| n.role() == egui::accesskit::Role::GraphicsDocument);
        assert!(
            has_graphics_doc,
            "Workbench {tool_id} must expose AccessKit GraphicsDocument canvas"
        );
    }
}

#[test]
fn test_dtmc_workbench_regimes_stationary_and_absorbing_invariants() {
    let mut tool = DtmcTensorWorkbenchTool::default();
    tool.apply_model_preset(DtmcModelPreset::PeriodTwoWalk);
    let pi = tool.stationary.expect("Period-2 walk has unique stationary");
    assert!((pi[0] - 0.25).abs() < 1e-12);
    assert!((pi[1] - 0.50).abs() < 1e-12);
    assert!((pi[2] - 0.25).abs() < 1e-12);

    tool.apply_model_preset(DtmcModelPreset::AsymmetricAbsorbing);
    assert!(
        tool.stationary.is_none(),
        "Multi-absorbing chain must not fabricate uniform stationary distribution"
    );
    let lim = tool.limiting.expect("Limiting distribution from state 0");
    assert!(lim[0].abs() < 1e-12);
    assert!((lim[1] - 0.95).abs() < 1e-12);
    assert!((lim[2] - 0.05).abs() < 1e-12);

    tool.apply_model_preset(DtmcModelPreset::ErgodicChain);
    assert!(tool.stationary.is_some());

    tool.apply_model_preset(DtmcModelPreset::ShotClockTensor);
    assert!(tool.stationary.is_none());
    assert!(!tool.tensor_curve_s1.is_empty());
}

#[test]
fn test_ctmc_queue_and_hmm_ou_workbenches_modes_and_validation() {
    let mut ctmc_tool = CtmcQueueWorkbenchTool {
        rate_log10: -7.0,
        forward_weight: 1.0,
        backward_weight: 3.0,
        ..Default::default()
    };
    ctmc_tool.set_mode(ContinuousModelMode::SlowRateCtmc);
    let pi = ctmc_tool.ctmc_steady.expect("CTMC steady state at 1e-7");
    let sum: f64 = pi.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12);
    assert!(pi[0] > pi[1] && pi[1] > pi[2]);

    ctmc_tool.set_mode(ContinuousModelMode::MmcBirthDeathQueue);
    assert!(ctmc_tool.queue_metrics.is_some());
    assert!(!ctmc_tool.queue_analytic_pk.is_empty());

    ctmc_tool.poisson_variant = PoissonVariant::StepBurst;
    ctmc_tool.set_mode(ContinuousModelMode::InhomogeneousPoisson);
    assert!(ctmc_tool.expected_arrivals > 0.0);
    assert!(ctmc_tool.try_load_state(r#"{"bad_field":1}"#).is_err());

    let mut hmm_ou = HmmOuWorkbenchTool {
        seq_len: 600,
        ..Default::default()
    };
    hmm_ou.set_mode(CalibrationMode::HmmViterbiBaumWelch);
    assert!(hmm_ou.true_log_lik.is_finite());
    assert!(hmm_ou.em_log_liks.len() >= 2);
    for w in hmm_ou.em_log_liks.windows(2) {
        assert!(w[1].y + 1e-9 >= w[0].y);
    }

    hmm_ou.set_mode(CalibrationMode::OuExactAndCalibration);
    assert!(hmm_ou.estimated_ou.is_some());
    assert!(!hmm_ou.ou_exact_curve.is_empty());
    assert!(hmm_ou.try_load_state(r#"{"bad_field":1}"#).is_err());
}

#[test]
fn test_stochastic_workbenches_url_share_and_deferred_exports() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut app = MathExplorerApp::default();
    let tab_idx = app
        .tabs
        .iter()
        .position(|t| t.url_slug() == "stochastic_processes")
        .expect("Stochastic Processes tab registered");
    app.selected_tab = tab_idx;

    for tool_id in [
        "DtmcTensorWorkbenchTool",
        "CtmcQueueWorkbenchTool",
        "HmmOuWorkbenchTool",
    ] {
        assert!(app.tabs[tab_idx].select_tool_by_name(tool_id));
        let share_url = app.encode_current_share_url().expect("encode share url");
        let hash_idx = share_url.find('#').expect("hash fragment");
        let mut restored = MathExplorerApp::default();
        assert!(
            restored
                .apply_share_location_strings(&share_url[hash_idx..], "", None)
                .unwrap()
        );
        assert_eq!(restored.tabs[restored.selected_tab].active_tool_id(), Some(tool_id));
    }

    let ctx = egui::Context::default();
    let mut tab = StochasticProcessesTab::default();
    for tool_id in [
        "DtmcTensorWorkbenchTool",
        "CtmcQueueWorkbenchTool",
        "HmmOuWorkbenchTool",
    ] {
        assert!(tab.select_tool_by_name(tool_id));
        let fw = tab.framework_mut().expect("framework");
        fw.export_options.width = 480;
        fw.export_options.height = 320;
        for fmt in [ExportFormat::Png, ExportFormat::Svg, ExportFormat::Csv] {
            let _ = ctx.run(test_raw_input(), |c| {
                egui_plot::clear_last_plot_datasets();
                fw.show_export_modal = true;
                fw.queue_export(fmt);
                fw.show(c, "stochastic_export_test");
            });
            assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
        }
    }
}

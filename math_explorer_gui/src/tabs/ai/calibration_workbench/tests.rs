#![cfg_attr(any(), verified(opt_out = "tests"))]

use eframe::egui;

use super::presets::{PRESET_CLINICAL_ISOTONIC, PRESET_ENSEMBLE_CRPS, PRESET_OVERCONFIDENT_MLP};
use super::state::{ForecastScenario, RecalibratorKind, WorkbenchView};
use super::CalibrationWorkbenchTool;
use crate::framework::{ExportCapabilities, InteractiveTool};

#[test]
fn test_all_views_render_and_populate_plot_exports() {
    let mut tool = CalibrationWorkbenchTool::default();
    assert_eq!(tool.export_capabilities(), ExportCapabilities::plot());

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0));

    for view in WorkbenchView::ALL {
        tool.config.view = view;
        tool.recompute();
        egui_plot::clear_last_plot_datasets();
        let out = ctx.run(
            egui::RawInput {
                screen_rect: Some(screen),
                ..Default::default()
            },
            |ctx| {
                tool.show(ctx);
            },
        );
        let ak = out
            .platform_output
            .accesskit_update
            .as_ref()
            .expect("AccessKit tree");
        assert!(
            ak.nodes
                .iter()
                .any(|(_, n)| n.role() == egui::accesskit::Role::GraphicsDocument),
            "View {:?} must register GraphicsDocument accessibility node",
            view
        );
        let has_series =
            !tool.export_datasets().is_empty() || !tool.export_plot_primitives().is_empty();
        assert!(has_series, "View {:?} must populate plot export data", view);
    }
}

#[test]
fn test_recalibrators_reduce_held_out_ece_and_log_loss_in_workbench() {
    let mut tool = CalibrationWorkbenchTool::default();
    tool.config.scenario = ForecastScenario::OverconfidentMlp;
    tool.config.logit_scale = 3.2;
    tool.config.logit_bias = 0.85;
    tool.config.sample_count = 320;

    for recal in [
        RecalibratorKind::Platt,
        RecalibratorKind::Temperature,
        RecalibratorKind::IsotonicPava,
    ] {
        tool.config.recalibrator = recal;
        tool.recompute();
        assert!(
            tool.evaluation.cal_metrics.log_loss < tool.evaluation.raw_metrics.log_loss,
            "{:?} failed to lower log_loss ({} vs {})",
            recal,
            tool.evaluation.cal_metrics.log_loss,
            tool.evaluation.raw_metrics.log_loss
        );
        assert!(
            tool.evaluation.cal_metrics.ece < tool.evaluation.raw_metrics.ece,
            "{:?} failed to lower ECE ({} vs {})",
            recal,
            tool.evaluation.cal_metrics.ece,
            tool.evaluation.raw_metrics.ece
        );
    }
}

#[test]
fn test_presets_roundtrip_and_unknown_or_invalid_fields_rejected() {
    let mut tool = CalibrationWorkbenchTool::default();
    for preset in [
        PRESET_OVERCONFIDENT_MLP,
        PRESET_CLINICAL_ISOTONIC,
        PRESET_ENSEMBLE_CRPS,
    ] {
        tool.import_preset_json(preset).expect("valid preset");
        let saved = tool.save_state().expect("saved state");
        let mut restored = CalibrationWorkbenchTool::default();
        restored.try_load_state(&saved).expect("restored state");
        assert_eq!(tool.config, restored.config);
    }

    // Unknown fields and out-of-range parameters must fail cleanly.
    assert!(tool
        .try_load_state(r#"{"logit_scale":2.0,"unknown":1}"#)
        .is_err());
    let mut bad = tool.config.clone();
    bad.logit_scale = 999.0;
    let bad_json = serde_json::to_string(&bad).unwrap();
    assert!(tool.try_load_state(&bad_json).is_err());
}

// [cite:clinical_trials_statistics]

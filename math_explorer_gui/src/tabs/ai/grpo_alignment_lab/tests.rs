#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{GrpoAlignmentLabTool, GrpoScenarioPreset};
use crate::framework::InteractiveTool;

#[test]
fn grpo_alignment_lab_roundtrip_and_bounds() {
    let mut tool = GrpoAlignmentLabTool::default();
    assert!(tool.has_finite_parameters());
    let saved = tool.save_state().expect("serializes");
    assert!(tool.state_bounds().validate_json(&saved).is_ok());

    tool.epsilon = 0.35;
    tool.beta = 0.12;
    tool.scenario = GrpoScenarioPreset::HighEntropyExploration;
    let updated = tool.save_state().expect("serializes updated");

    let mut restored = GrpoAlignmentLabTool::default();
    restored
        .try_load_state(&updated)
        .expect("loads valid state");
    assert!((restored.epsilon - 0.35).abs() < 1e-9);
    assert!((restored.beta - 0.12).abs() < 1e-9);
    assert_eq!(
        restored.scenario,
        GrpoScenarioPreset::HighEntropyExploration
    );

    assert!(restored
        .try_load_state(r#"{"epsilon": 9.9, "beta": 0.1, "solver_accuracy": 0.5, "repetition_lambda": 0.5, "cluster_size": 2, "confidence_skew": 0.5}"#)
        .is_err());
    assert!(restored
        .try_load_state(r#"{"epsilon": 0.2, "unknown": 1}"#)
        .is_err());
}

#[test]
fn grpo_alignment_lab_evaluates_rewards_and_kl_distillation() {
    let mut tool = GrpoAlignmentLabTool {
        solver_accuracy: 0.5,
        cluster_size: 1,
        repetition_lambda: 0.0,
        ..GrpoAlignmentLabTool::default()
    };
    let snap = tool.evaluate_snapshot();
    assert!(snap.grpo_loss.is_finite());
    assert!((snap.active_uncertainty_reward - 1.0).abs() < 1e-9);
    assert!((snap.target_distribution.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    assert!(snap.calibration_kl_loss >= 0.0);

    let datasets = tool.export_datasets();
    assert_eq!(datasets.len(), 3);
    assert!(datasets.iter().all(|(_, pts)| !pts.is_empty()));
}

#[test]
fn grpo_alignment_lab_renders_headless_without_panic() {
    let mut tool = GrpoAlignmentLabTool::default();
    let ctx = eframe::egui::Context::default();
    let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            tool.show_inside(ui);
        });
    });
}

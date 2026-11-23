#![cfg_attr(any(), verified(opt_out = "tests"))]

use eframe::egui;
use egui_plot::{ImportedDataset, PlotOverlaySnapshot};

use super::{InteractiveTool, SimulationFramework};
use crate::tabs::battery_degradation::capacity_fade::CapacityFadeTool;
use crate::tabs::battery_degradation::lifetime_estimator::LifetimeEstimatorTool;
use crate::tabs::clinical_trials::pharmacokinetics::PharmacokineticsTool;
use crate::tabs::clinical_trials::sample_size::SampleSizeCalculatorTool;
use crate::tabs::financial_math::bankroll_growth::BankrollGrowthTool;
use crate::tabs::game_theory::glicko2::Glicko2TournamentTool;
use crate::tabs::game_theory::replicator::ReplicatorDynamicsTool;

fn framework_for_tool(tool: Box<dyn InteractiveTool>) -> SimulationFramework {
    let mut fw = SimulationFramework::new("__empty_cooperative_test_domain__");
    fw.domain = tool.domain().to_string();
    fw.baseline_tool_state = tool.save_state();
    fw.active_tool = Some(tool);
    fw
}

fn converge_framework_sensitivity(fw: &mut SimulationFramework) {
    fw.cooperative_sensitivity.uncertainty_enabled = true;
    fw.cooperative_sensitivity.seed = 0x5EED_2026;
    fw.cooperative_sensitivity.target_samples = 24;
    let mut guards = 0_usize;
    while !fw.cooperative_sensitivity.engine.is_converged()
        || !fw.cooperative_sensitivity.engine.is_tornado_converged()
        || !fw.cooperative_sensitivity.engine.is_monte_carlo_converged()
    {
        assert!(fw.step_active_tool_sensitivity().is_some());
        guards += 1;
        assert!(guards < 64, "Cooperative sensitivity failed to converge");
    }
}

#[test]
fn test_applied_tools_cooperative_sensitivity_tornado_and_monte_carlo() {
    let tools: Vec<Box<dyn InteractiveTool>> = vec![
        Box::new(BankrollGrowthTool::default()),
        Box::new(PharmacokineticsTool::default()),
        Box::new(SampleSizeCalculatorTool::default()),
        Box::new(ReplicatorDynamicsTool::default()),
        Box::new(Glicko2TournamentTool::default()),
        Box::new(CapacityFadeTool::default()),
        Box::new(LifetimeEstimatorTool::default()),
    ];

    for tool in tools {
        let tool_name = tool.name();
        let mut fw = framework_for_tool(tool);
        assert!(
            fw.active_tool_supports_sensitivity(),
            "{tool_name} must expose sensitivity parameters"
        );
        converge_framework_sensitivity(&mut fw);

        let engine = &fw.cooperative_sensitivity.engine;
        assert!(
            !engine.metrics.is_empty() && engine.metrics.iter().any(|m| m.elasticity > 0.0),
            "{tool_name} must produce non-zero local elasticity"
        );
        assert!(
            engine.tornado_intervals().iter().any(|iv| iv.swing > 0.0),
            "{tool_name} must produce non-zero tornado swing"
        );
        let mc = engine
            .monte_carlo_summary()
            .expect("converged Monte Carlo summary");
        assert_eq!(mc.sample_count, 24);
        assert!(
            mc.p05 <= mc.p50 && mc.p50 <= mc.p95,
            "{tool_name} quantiles out of order: {mc:?}"
        );
    }
}

#[test]
fn test_plot_scope_imported_datasets_exposure_and_csv_calibration() {
    let mut target_cf = CapacityFadeTool::default();
    target_cf
        .try_load_state(r#"{"dod":35.0,"temperature":25.0,"cycles_to_simulate":2000.0}"#)
        .unwrap();
    let target_pts = target_cf.export_datasets().remove(0).1;

    let mut fw = framework_for_tool(Box::new(CapacityFadeTool::default()));
    let snap = PlotOverlaySnapshot {
        imported_datasets: vec![ImportedDataset::new("bench_cell.csv", target_pts)],
        ..PlotOverlaySnapshot::default()
    };
    fw.saved_plot_states
        .entry("Capacity Fade".to_string())
        .or_default()
        .insert("capacity_fade_plot".to_string(), snap);
    fw.bump_plot_sync_generation();

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            fw.show_inside(ui, "cf_calib_pane");
        });
    });

    let imported = fw.latest_imported_datasets();
    assert_eq!(imported.len(), 1);
    assert_eq!(imported[0].name, "bench_cell.csv");
    assert!(fw.calibrate_active_tool_from_datasets(&imported).unwrap());

    let saved = fw.active_tool.as_ref().unwrap().save_state().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&saved).unwrap();
    let fitted_dod = parsed["dod"].as_f64().unwrap();
    assert!(
        (fitted_dod - 35.0).abs() < 0.5,
        "Expected calibrated DoD ≈ 35.0%, got {fitted_dod}"
    );
}

#[test]
fn test_pharmacokinetics_csv_calibration_and_non_finite_rollback() {
    let mut reference_pk = PharmacokineticsTool {
        ka: 1.8,
        ke: 0.16,
        volume_l: 55.0,
        ..Default::default()
    };
    reference_pk.recalculate();
    let pk_pts = reference_pk
        .export_datasets()
        .into_iter()
        .find(|(n, _)| n.contains("Total") || n.contains("Plasma") || !n.is_empty())
        .unwrap()
        .1;

    let mut fw = framework_for_tool(Box::new(PharmacokineticsTool::default()));
    let baseline_state = fw.active_tool.as_ref().unwrap().save_state().unwrap();

    // Non-finite CSV dataset must fail cleanly and roll back to exact pre-calibration state
    let bad_ds = vec![ImportedDataset::new(
        "corrupt.csv",
        vec![[0.0, 1.0], [2.0, f64::NAN], [4.0, 2.5]],
    )];
    assert!(fw.calibrate_active_tool_from_datasets(&bad_ds).is_err());
    assert!(!fw.is_diverged);
    assert_eq!(
        fw.active_tool.as_ref().unwrap().save_state().unwrap(),
        baseline_state
    );

    // Valid CSV dataset recovers bounded PK parameters (ka, ke, volume_l)
    let valid_ds = vec![ImportedDataset::new("subject_01.csv", pk_pts)];
    assert!(fw.calibrate_active_tool_from_datasets(&valid_ds).unwrap());
    let pk_after = fw
        .active_tool
        .as_ref()
        .unwrap()
        .as_any()
        .unwrap()
        .downcast_ref::<PharmacokineticsTool>()
        .unwrap();
    assert!((pk_after.ka - 1.8).abs() < 0.15, "ka={}", pk_after.ka);
    assert!((pk_after.ke - 0.16).abs() < 0.03, "ke={}", pk_after.ke);
    assert!(
        (pk_after.volume_l - 55.0).abs() < 3.0,
        "volume_l={}",
        pk_after.volume_l
    );
    assert_eq!(pk_after.scenario_kpis().len(), 4);
}

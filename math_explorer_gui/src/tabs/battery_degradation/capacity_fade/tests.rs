#![cfg_attr(any(), verified(opt_out = "tests"))]

// [cite:battery_degradation]

use super::io::{parse_calibration_csv, MAX_CALIBRATION_CSV_ROWS};
use super::*;
use crate::framework::InteractiveTool;
use crate::tabs::battery_degradation::cell_card::ChemistryChoice;
use math_explorer::applied::battery_degradation::{Cycles, DepthOfDischarge};

#[test]
fn test_first_life_80_threshold_error_and_power_law_fit() {
    let mut tool = CapacityFadeTool::default();
    let (n80, n70) = CapacityFadeTool::projected_life_cycles(tool.dod);
    assert!(n80 > 0.0 && n80 < n70);
    tool.cycles_to_simulate = (n70 * 1.2).max(2000.0);

    let datasets = tool.export_datasets();
    let mut crossings = [0.0_f64; 4];
    let count = math_explorer::poi::threshold_crossings_into(&datasets[0].1, 0.8, &mut crossings);
    assert_eq!(count, 1);
    assert!(
        (crossings[0] - n80).abs() < 0.5,
        "80% threshold cycle error = {} cycles",
        (crossings[0] - n80).abs()
    );

    let n70_curve: Vec<[f64; 2]> = (10..=100)
        .step_by(5)
        .map(|dod| {
            let (_, life70) = CapacityFadeTool::projected_life_cycles(dod as f64);
            [dod as f64, life70]
        })
        .collect();
    let fit =
        math_explorer::fit::fit_model(&n70_curve, math_explorer::fit::FitModel::PowerLaw, false)
            .expect("power law fit");
    assert!(fit.r_squared >= 0.99);
}

#[test]
fn test_capacity_fade_animated_dod_sweep() {
    use crate::reflective_ui::slider::{save_slider_state, AnimatedSliderState, SweepMode};

    let mut tool = CapacityFadeTool::default();
    let initial_dod = tool.dod;
    let ctx = egui::Context::default();
    save_slider_state(
        &ctx,
        egui::Id::new("capacity_fade_dod"),
        AnimatedSliderState {
            playing: true,
            direction: 1.0,
            sweep_mode: SweepMode::Oscillate,
        },
    );
    let _ = ctx.run(
        egui::RawInput {
            predicted_dt: 0.1,
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
    assert!(tool.dod > initial_dod);
    let saved = tool.save_state().expect("save_state");
    assert!(tool.try_load_state(&saved).is_ok());
}

#[test]
fn test_temperature_c_rate_and_preset_controls_drive_arrhenius_aging() {
    let mut tool = CapacityFadeTool::default();
    let (n80_25, n70_25) = tool.projected_life_cycles_coupled();

    tool.temperature = 45.0;
    let (n80_45, n70_45) = tool.projected_life_cycles_coupled();
    assert!(n70_45 < n70_25 * 0.5);
    assert!(n80_45 < n80_25 * 0.5);
    assert!(((n70_25 / n70_45) - 2.2229).abs() < 5e-3);

    tool.c_rate = 2.5;
    let (_, n70_fast) = tool.projected_life_cycles_coupled();
    assert!(n70_fast < n70_45);

    tool.apply_chemistry_preset(ChemistryChoice::Lto);
    let (_, n70_lto) = tool.projected_life_cycles_coupled();
    assert!(n70_lto > n70_25);

    let custom_json = tool.save_state().expect("save custom state");
    let mut restored = CapacityFadeTool::default();
    let rev_default = restored.state_revision();
    let sig_default = restored.atlas_context_signature(("dod", "cycles_to_simulate"));
    let cart_25 = restored
        .build_atlas_cartographer(("dod", "cycles_to_simulate"))
        .expect("25C cartographer");
    restored.try_load_state(&custom_json).expect("load custom");
    assert_ne!(restored.state_revision(), rev_default);
    assert_ne!(
        restored.atlas_context_signature(("dod", "cycles_to_simulate")),
        sig_default
    );
    let cart_lto = restored
        .build_atlas_cartographer(("dod", "cycles_to_simulate"))
        .expect("LTO cartographer");
    assert!(
        cart_lto.evaluate_cell(80.0, 2000.0).primary > cart_25.evaluate_cell(80.0, 2000.0).primary
    );
    assert!(cart_25.overlay_contour().is_some());
    assert_eq!(restored.chemistry, ChemistryChoice::Lto);
    assert!((restored.c_rate - 2.5).abs() < 1e-9);
    assert!((restored.temperature - 45.0).abs() < 1e-9);
}

#[test]
fn test_csv_bom_2000_row_cap_lm_calibration_rollback_and_cell_card() {
    let mut tool = CapacityFadeTool {
        dod: 70.0,
        temperature: 30.0,
        ..Default::default()
    };

    let truth = ElectrochemicalAgingModel::from_preset(ChemistryPreset::Lfp)
        .with_temperature_c(30.0)
        .with_exponent_z(0.54);
    let dod = DepthOfDischarge::new_clamped(70.0);

    let mut csv = String::from("\u{feff}cycles,soh\n");
    for idx in 0..2_050 {
        let cyc = (idx as f64) * 2.0;
        let soh = truth.capacity(Cycles::new_clamped(cyc), dod).as_f64();
        csv.push_str(&format!("{cyc:.1},{soh:.6}\n"));
    }

    let parsed = parse_calibration_csv(&csv).expect("parse BOM CSV");
    assert_eq!(parsed.len(), MAX_CALIBRATION_CSV_ROWS);

    let summary = tool
        .import_calibration_csv("lfp_lab.csv", &csv)
        .expect("LM fit succeeds");
    assert_eq!(summary.sample_count, MAX_CALIBRATION_CSV_ROWS);
    assert!(summary.rmse < 1e-4);
    assert!((tool.exponent_z - 0.54).abs() < 2e-2);
    assert_eq!(tool.residual_points().len(), MAX_CALIBRATION_CSV_ROWS);
    assert_eq!(tool.export_datasets().len(), 3);

    // Verify rollback on both CSV parse failure and LM solver domain failure (all zero cycles).
    let saved_alpha = tool.calibrated_alpha;
    let saved_z = tool.exponent_z;
    assert!(tool
        .import_calibration_csv("short.csv", "cycles,soh\n10,0.99\n")
        .is_err());
    assert!(tool
        .import_calibration_csv("zero_cyc.csv", "cycles,soh\n0,1.0\n0,0.99\n0,0.98\n")
        .is_err());
    assert_eq!(tool.calibrated_alpha, saved_alpha);
    assert_eq!(tool.exponent_z, saved_z);

    let card_json = tool
        .build_cell_card()
        .to_pretty_json()
        .expect("serialize cell card");
    assert!(card_json.contains("\"tool\": \"CapacityFadeTool\""));
    assert!(card_json.contains("\"calibration_rmse\""));
}

#[test]
fn test_capacity_fade_deferred_png_export_and_residuals_view() {
    let mut tool = CapacityFadeTool {
        dod: 70.0,
        temperature: 40.0,
        ..Default::default()
    };
    let csv = "cycles,soh\n0,1.0\n200,0.93\n500,0.87\n900,0.81\n1400,0.75\n";
    tool.import_calibration_csv("nmc_40c.csv", csv)
        .expect("calibrate CSV");

    let mut fw = crate::framework::SimulationFramework::new("__capacity_fade_export__");
    fw.active_tool = Some(Box::new(tool));
    let ctx = eframe::egui::Context::default();
    let raw = eframe::egui::RawInput {
        screen_rect: Some(eframe::egui::Rect::from_min_size(
            eframe::egui::Pos2::ZERO,
            eframe::egui::Vec2::new(1100.0, 760.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw, |ctx| {
        egui_plot::clear_last_plot_datasets();
        fw.show_export_modal = true;
        fw.queue_export(crate::framework::ExportFormat::Png);
        fw.show(ctx, "capacity_fade_export_test");
    });
    let png = fw
        .export_queue
        .last_exported_bytes
        .as_deref()
        .expect("PNG exported");
    assert!(png.len() > 200);
}

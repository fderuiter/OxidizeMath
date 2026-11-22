//! End-to-end calibration, residual CSV, live-region accessibility, and JSON/URL state round-trip
//! tests for [`OdeSolverTool`], [`SirTool`], [`CapacityFadeTool`], and [`EnzymeKineticsTool`].

#[cfg(feature = "epidemiology")]
use crate::accessibility::{all_announcements, clear_test_announcements};
use crate::framework::empirical_csv::EmpiricalCsvTable;
#[cfg(feature = "epidemiology")]
use crate::framework::empirical_csv::{handle_dropped_empirical_files, MAX_EMPIRICAL_CSV_BYTES};
use crate::framework::InteractiveTool;
#[cfg(feature = "pure_math")]
use crate::presets::OdePreset;
#[cfg(feature = "pure_math")]
use crate::tabs::analysis::ode::OdeSolverTool;
#[cfg(feature = "applied")]
use crate::tabs::battery_degradation::capacity_fade::CapacityFadeTool;
#[cfg(feature = "epidemiology")]
use crate::tabs::epidemiology::sir::SirTool;
#[cfg(feature = "biology")]
use crate::tabs::morphogenesis::kinetics::{EnzymeKineticsTool, EnzymeKineticsVariant};
use crate::url_state::{encode_share_fragment, parse_share_location};
#[cfg(feature = "applied")]
use math_explorer::applied::battery_degradation::{Cycles, DepthOfDischarge, PowerLawModel};
#[cfg(feature = "biology")]
use math_explorer::biology::kinetics::{HillKinetics, KineticsModel};
#[cfg(feature = "epidemiology")]
use math_explorer::epidemiology::compartmental::SIRModel;

#[cfg(feature = "pure_math")]
fn build_logistic_csv(r_true: f64, k_true: f64, y0_true: f64) -> String {
    let mut csv = String::from("time,population\n");
    for i in 0..=20 {
        let t = (i as f64) * 0.5;
        let denom = 1.0 + ((k_true - y0_true) / y0_true) * (-r_true * t).exp();
        let y = k_true / denom;
        csv.push_str(&format!("{t:.4},{y:.6}\n"));
    }
    csv
}

#[cfg(feature = "pure_math")]
fn build_custom_ast_csv(p1: f64, p2: f64, y0: f64) -> String {
    let mut csv = String::from("t,y\n");
    let y_eq = p2 / p1;
    for i in 0..=20 {
        let t = (i as f64) * 0.25;
        let y = y_eq + (y0 - y_eq) * (-p1 * t).exp();
        csv.push_str(&format!("{t:.4},{y:.6}\n"));
    }
    csv
}

#[cfg(feature = "epidemiology")]
fn build_sir_outbreak_csv(n: f64, beta: f64, gamma: f64, i0: f64) -> String {
    let mut model = SIRModel::new(n, i0, beta, gamma).expect("valid SIR setup");
    let dt = 0.05;
    let mut csv = String::from("day,infected\n");
    for step in 0..=600 {
        if step % 20 == 0 {
            let day = (step as f64) * dt;
            csv.push_str(&format!("{day:.2},{:.6}\n", model.state().i));
        }
        model.step(dt);
    }
    csv
}

#[cfg(feature = "applied")]
fn build_multi_dod_battery_csv(alpha: f64, beta: f64) -> String {
    let model = PowerLawModel::new(alpha, beta);
    let mut csv = String::from("cycles,soh,dod\n");
    for &(dod_raw, step) in &[(80.0_f64, 150.0_f64), (50.0_f64, 250.0_f64)] {
        let dod = DepthOfDischarge::new(dod_raw).unwrap();
        for k in 0..8 {
            let cyc_raw = (k as f64) * step;
            let cap = model.capacity(Cycles::new(cyc_raw).unwrap(), dod).as_f64();
            csv.push_str(&format!("{cyc_raw:.1},{cap:.6},{dod_raw:.1}\n"));
        }
    }
    csv
}

#[cfg(feature = "biology")]
fn build_hill_kinetics_csv(vmax: f64, km: f64, hill_n: f64) -> String {
    let model = HillKinetics::new(vmax, km, hill_n).unwrap();
    let mut csv = String::from("substrate,velocity\n");
    for i in 1..=18 {
        let s = (i as f64) * 8.0;
        let v = model.reaction_velocity(s).unwrap();
        csv.push_str(&format!("{s:.3},{v:.6}\n"));
    }
    csv
}

fn assert_url_and_preset_roundtrip(
    tab_slug: &str,
    tool_id: &str,
    source: &dyn InteractiveTool,
    restored: &mut dyn InteractiveTool,
) {
    let preset_json = source
        .export_preset_json()
        .expect("calibrated tool should export URL preset envelope");
    let fragment = encode_share_fragment(tab_slug, Some(tool_id), Some(&preset_json))
        .expect("calibrated state should fit within 4 KiB / 8 KiB URL caps");
    let parsed = parse_share_location(&fragment, "")
        .expect("valid share fragment")
        .expect("share parameters present");
    assert_eq!(parsed.tool.as_deref(), Some(tool_id));
    let state_json = parsed.state_json.expect("decoded JSON state present");
    restored
        .import_preset_json(&state_json)
        .expect("calibrated preset JSON must import cleanly");
}

#[cfg(feature = "pure_math")]
#[test]
fn test_ode_solver_logistic_and_custom_ast_calibration_and_roundtrip() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::LogisticGrowth,
        param_r: 0.25,
        param_cap_k: 90.0,
        ic_y0: 8.0,
        ..Default::default()
    };

    let table =
        EmpiricalCsvTable::parse_str("logistic.csv", &build_logistic_csv(0.45, 140.0, 12.0))
            .unwrap();
    tool.ingest_empirical_dataset(table).unwrap();
    tool.calibrate_from_loaded_csv(None)
        .expect("logistic ODE calibration should converge");

    assert!((tool.param_r - 0.45).abs() / 0.45 < 1e-2);
    assert!((tool.param_cap_k - 140.0).abs() / 140.0 < 1e-2);
    assert!((tool.ic_y0 - 12.0).abs() / 12.0 < 1e-2);

    let report = tool
        .empirical_csv
        .report
        .as_ref()
        .expect("report populated");
    assert!(report.r_squared > 0.999);
    assert!(report.aic.is_finite());
    let residuals_csv = report.residuals_csv().expect("finite residuals CSV");
    assert!(residuals_csv.starts_with("x,y_observed,y_fitted,residual\n"));

    let mut restored = OdeSolverTool::default();
    assert_url_and_preset_roundtrip("analysis", "OdeSolverTool", &tool, &mut restored);
    assert!((restored.param_r - tool.param_r).abs() < 1e-9);
    assert!((restored.param_cap_k - tool.param_cap_k).abs() < 1e-9);

    let mut custom_tool = OdeSolverTool {
        preset: OdePreset::Custom,
        ..Default::default()
    };
    custom_tool.math_input.set_expression("-k * y + r");
    let ast_table =
        EmpiricalCsvTable::parse_str("custom_ode.csv", &build_custom_ast_csv(0.6, 3.0, 1.0))
            .unwrap();
    custom_tool.ingest_empirical_dataset(ast_table).unwrap();
    custom_tool
        .calibrate_from_loaded_csv(None)
        .expect("DynamicAstOde calibration should converge");
    assert!((custom_tool.param_k - 0.6).abs() / 0.6 < 1e-2);
    assert!((custom_tool.param_r - 3.0).abs() / 3.0 < 1e-2);
}

#[cfg(feature = "epidemiology")]
#[test]
fn test_sir_outbreak_calibration_r0_confidence_interval_and_roundtrip() {
    let mut sir = SirTool::default();
    sir.n = 1_000.0;
    sir.beta = 0.25;
    sir.gamma = 0.2;
    sir.i0 = 5.0;

    let csv = build_sir_outbreak_csv(1_000.0, 0.42, 0.14, 8.0);
    let table = EmpiricalCsvTable::parse_str("outbreak.csv", &csv).unwrap();
    sir.ingest_empirical_dataset(table).unwrap();
    sir.calibrate_from_loaded_csv(None)
        .expect("SIR RK4 shooting calibration should converge");

    assert!((sir.beta - 0.42).abs() / 0.42 < 1e-2);
    assert!((sir.gamma - 0.14).abs() / 0.14 < 1e-2);
    assert!((sir.i0 - 8.0).abs() / 8.0 < 1e-2);

    let report = sir
        .empirical_csv
        .report
        .as_ref()
        .expect("SIR report present");
    assert!(report.r_squared > 0.999);
    assert!(
        report
            .derived_metrics
            .iter()
            .any(|(label, val)| label.contains("R₀") && val.contains("3.00")),
        "Expected propagated R0 ≈ 3.00 in derived metrics: {:?}",
        report.derived_metrics
    );
    assert!(report.residuals_csv().unwrap().lines().count() > 10);

    let mut restored = SirTool::default();
    assert_url_and_preset_roundtrip("epidemiology", "SirTool", &sir, &mut restored);
    assert!((restored.beta - sir.beta).abs() < 1e-9);
    assert!((restored.gamma - sir.gamma).abs() < 1e-9);
}

#[cfg(feature = "applied")]
#[test]
fn test_capacity_fade_calibration_and_legacy_preset_compatibility() {
    let mut fade = CapacityFadeTool::default();
    let legacy_json = r#"{"dod":80.0,"temperature":25.0,"cycles_to_simulate":2000.0}"#;
    fade.try_load_state(legacy_json)
        .expect("legacy 3-field CapacityFadeState must deserialize cleanly");
    let default_saved = fade.save_state().unwrap();
    assert!(!default_saved.contains("alpha"));

    let two_col = "cycles,capacity\n0.0,1.0000\n200.0,0.8870\n500.0,0.7410\n800.0,0.6190\n";
    fade.ingest_empirical_dataset(EmpiricalCsvTable::parse_str("two_col.csv", two_col).unwrap())
        .unwrap();
    fade.calibrate_from_loaded_csv(None)
        .expect("2-column single-DoD capacity fade CSV should calibrate cleanly");
    assert!(fade.empirical_csv.report.as_ref().unwrap().r_squared > 0.99);

    let csv = build_multi_dod_battery_csv(1.6e5, -1.75);
    let table = EmpiricalCsvTable::parse_str("cycling.csv", &csv).unwrap();
    fade.ingest_empirical_dataset(table).unwrap();
    assert_eq!(fade.empirical_csv.aux_col, Some(2));
    fade.calibrate_from_loaded_csv(None)
        .expect("battery capacity fade calibration should converge");

    assert!((fade.alpha - 1.6e5).abs() / 1.6e5 < 1e-2);
    assert!((fade.beta - (-1.75)).abs() / 1.75 < 1e-2);
    let calibrated_saved = fade.save_state().unwrap();
    assert!(calibrated_saved.contains("alpha") && calibrated_saved.contains("beta"));

    let mut restored = CapacityFadeTool::default();
    assert_url_and_preset_roundtrip(
        "battery_degradation",
        "CapacityFadeTool",
        &fade,
        &mut restored,
    );
    assert!((restored.alpha - fade.alpha).abs() < 1e-6);
    assert!((restored.beta - fade.beta).abs() < 1e-6);
}

#[cfg(feature = "biology")]
#[test]
fn test_enzyme_kinetics_aic_model_comparison_and_roundtrip() {
    let csv = build_hill_kinetics_csv(150.0, 35.0, 2.4);
    let table = EmpiricalCsvTable::parse_str("hill_rates.csv", &csv).unwrap();

    let mut tool = EnzymeKineticsTool::default();
    tool.variant = EnzymeKineticsVariant::MichaelisMenten;
    tool.ingest_empirical_dataset(table).unwrap();
    tool.calibrate_from_loaded_csv(None)
        .expect("Michaelis-Menten fit should run");
    let aic_mm = tool.empirical_csv.report.as_ref().unwrap().aic;

    tool.variant = EnzymeKineticsVariant::Hill;
    tool.calibrate_from_loaded_csv(None)
        .expect("Hill kinetics fit should converge");
    let hill_report = tool.empirical_csv.report.as_ref().unwrap();
    assert!(
        hill_report.aic < aic_mm,
        "Hill AIC ({}) should beat non-cooperative MM AIC ({}) on cooperative data",
        hill_report.aic,
        aic_mm
    );
    assert!((tool.v_max - 150.0).abs() / 150.0 < 1e-3);
    assert!((tool.k_m - 35.0).abs() / 35.0 < 1e-3);
    assert!((tool.hill_n - 2.4).abs() / 2.4 < 1e-3);

    let mut restored = EnzymeKineticsTool::default();
    assert_url_and_preset_roundtrip("biology", "EnzymeKineticsTool", &tool, &mut restored);
    assert_eq!(restored.variant, EnzymeKineticsVariant::Hill);
    assert!((restored.v_max - tool.v_max).abs() < 1e-9);
    assert!((restored.hill_n - tool.hill_n).abs() < 1e-9);
}

#[cfg(feature = "epidemiology")]
#[test]
fn test_malformed_and_oversized_dropped_csv_preserve_state_and_fire_assertive_alert() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    clear_test_announcements();

    let mut sir = SirTool::default();
    let baseline_beta = sir.beta;
    let baseline_gamma = sir.gamma;
    let baseline_points = sir.export_datasets();

    let ctx = eframe::egui::Context::default();
    let bad_payloads: Vec<(&str, Vec<u8>)> = vec![
        ("empty.csv", b"".to_vec()),
        ("single.csv", b"t,i\n0.0,10.0\n".to_vec()),
        ("nan.csv", b"t,i\n0.0,10.0\n1.0,NaN\n".to_vec()),
        ("oversized.csv", vec![b'1'; MAX_EMPIRICAL_CSV_BYTES + 16]),
    ];

    for (name, bytes) in bad_payloads {
        let mut raw = eframe::egui::RawInput::default();
        raw.dropped_files.push(eframe::egui::DroppedFile {
            name: name.to_string(),
            bytes: Some(bytes.into()),
            ..Default::default()
        });
        ctx.begin_pass(raw);
        handle_dropped_empirical_files(&ctx, &mut sir);
        let _ = ctx.end_pass();
        assert_eq!(sir.beta, baseline_beta);
        assert_eq!(sir.gamma, baseline_gamma);
        assert_eq!(sir.export_datasets(), baseline_points);
        assert!(sir.empirical_csv.is_error);
    }

    let assertive_count = all_announcements()
        .into_iter()
        .filter(|(_, prio)| prio == "assertive")
        .count();
    assert!(
        assertive_count >= 4,
        "Expected assertive alerts for all 4 bad CSV inputs"
    );
}

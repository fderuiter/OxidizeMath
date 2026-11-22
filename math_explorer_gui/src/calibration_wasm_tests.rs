//! Headless browser (`wasm32-unknown-unknown`) integration tests for empirical CSV ingestion,
//! Levenberg-Marquardt calibration, `PlatformInstant` timing, `TwinSlotLiveRegion` DOM alerts,
//! and residual CSV browser downloads.

use crate::accessibility::init_accessibility_bridge;
use crate::framework::empirical_csv::{handle_dropped_empirical_files, EmpiricalCsvTable};
use crate::framework::InteractiveTool;
use crate::presets::OdePreset;
use crate::tabs::analysis::ode::OdeSolverTool;
use crate::tabs::battery_degradation::capacity_fade::CapacityFadeTool;
use crate::tabs::epidemiology::sir::SirTool;
use crate::tabs::morphogenesis::kinetics::{EnzymeKineticsTool, EnzymeKineticsVariant};
use math_explorer::applied::battery_degradation::{Cycles, DepthOfDischarge, PowerLawModel};
use math_explorer::biology::kinetics::{HillKinetics, KineticsModel};
use math_explorer::epidemiology::compartmental::SIRModel;
use wasm_bindgen_test::wasm_bindgen_test;

fn wasm_hill_csv() -> String {
    let model = HillKinetics::new(125.0, 28.0, 2.1).unwrap();
    let mut out = String::from("s,v\n");
    for i in 1..=14 {
        let s = (i as f64) * 10.0;
        let v = model.reaction_velocity(s).unwrap();
        out.push_str(&format!("{s:.2},{v:.6}\n"));
    }
    out
}

fn wasm_battery_csv() -> String {
    let model = PowerLawModel::new(1.4e5, -1.7);
    let mut out = String::from("cycles,soh,dod\n");
    for &(dod_raw, step) in &[(80.0_f64, 160.0_f64), (45.0_f64, 240.0_f64)] {
        let dod = DepthOfDischarge::new(dod_raw).unwrap();
        for k in 0..7 {
            let c = (k as f64) * step;
            let soh = model.capacity(Cycles::new(c).unwrap(), dod).as_f64();
            out.push_str(&format!("{c:.1},{soh:.6},{dod_raw:.1}\n"));
        }
    }
    out
}

fn wasm_sir_csv() -> String {
    let mut model = SIRModel::new(1_000.0, 10.0, 0.38, 0.12).unwrap();
    let mut out = String::from("t,i\n");
    for step in 0..=400 {
        if step % 20 == 0 {
            out.push_str(&format!(
                "{:.2},{:.6}\n",
                (step as f64) * 0.05,
                model.state().i
            ));
        }
        model.step(0.05);
    }
    out
}

#[wasm_bindgen_test]
fn wasm_calibration_fits_enzyme_and_battery_and_downloads_residuals() {
    init_accessibility_bridge();
    let ctx = eframe::egui::Context::default();

    let mut kin = EnzymeKineticsTool::default();
    kin.variant = EnzymeKineticsVariant::Hill;
    kin.ingest_empirical_dataset(
        EmpiricalCsvTable::parse_str("hill.csv", &wasm_hill_csv()).unwrap(),
    )
    .unwrap();
    kin.calibrate_from_loaded_csv(Some(&ctx))
        .expect("wasm Hill calibration should succeed");
    assert!((kin.v_max - 125.0).abs() / 125.0 < 1e-2);
    assert!((kin.k_m - 28.0).abs() / 28.0 < 1e-2);
    kin.empirical_csv
        .trigger_save_residuals_dialog(Some(&ctx), "hill_residuals.csv")
        .expect("wasm residuals CSV browser download should succeed");

    let mut fade = CapacityFadeTool::default();
    fade.ingest_empirical_dataset(
        EmpiricalCsvTable::parse_str("fade.csv", &wasm_battery_csv()).unwrap(),
    )
    .unwrap();
    fade.empirical_csv.aux_col = Some(2);
    fade.calibrate_from_loaded_csv(Some(&ctx))
        .expect("wasm battery calibration should succeed");
    assert!((fade.alpha - 1.4e5).abs() / 1.4e5 < 2e-2);
    assert!((fade.beta - (-1.7)).abs() / 1.7 < 2e-2);
}

#[wasm_bindgen_test]
fn wasm_calibration_fits_ode_and_sir_and_updates_assertive_live_region_on_error() {
    init_accessibility_bridge();
    let ctx = eframe::egui::Context::default();

    let mut ode = OdeSolverTool {
        preset: OdePreset::Exponential,
        ..Default::default()
    };
    let mut exp_csv = String::from("t,y\n");
    for i in 0..=12 {
        let t = (i as f64) * 0.25;
        let y = 2.5 * (-0.8 * t).exp();
        exp_csv.push_str(&format!("{t:.2},{y:.6}\n"));
    }
    ode.ingest_empirical_dataset(EmpiricalCsvTable::parse_str("exp.csv", &exp_csv).unwrap())
        .unwrap();
    ode.calibrate_from_loaded_csv(Some(&ctx))
        .expect("wasm ODE shooting should succeed");
    assert!((ode.param_k - (-0.8)).abs() < 1e-2);

    let mut sir = SirTool::default();
    sir.ingest_empirical_dataset(EmpiricalCsvTable::parse_str("sir.csv", &wasm_sir_csv()).unwrap())
        .unwrap();
    sir.calibrate_from_loaded_csv(Some(&ctx))
        .expect("wasm SIR shooting should succeed");
    assert!((sir.beta - 0.38).abs() / 0.38 < 2e-2);

    let beta_after_fit = sir.beta;
    let mut raw = eframe::egui::RawInput::default();
    raw.dropped_files.push(eframe::egui::DroppedFile {
        name: "corrupt.csv".to_string(),
        bytes: Some(b"t,i\n0,NaN\n1,5\n".to_vec().into()),
        ..Default::default()
    });
    ctx.begin_pass(raw);
    handle_dropped_empirical_files(&ctx, &mut sir);
    let _ = ctx.end_pass();
    assert_eq!(sir.beta, beta_after_fit);

    let doc = web_sys::window().unwrap().document().unwrap();
    let s0 = doc
        .get_element_by_id("aria-live-assertive-slot-0")
        .and_then(|e| e.text_content())
        .unwrap_or_default();
    let s1 = doc
        .get_element_by_id("aria-live-assertive-slot-1")
        .and_then(|e| e.text_content())
        .unwrap_or_default();
    assert!(
        s0.contains("NaN") || s1.contains("NaN"),
        "Expected assertive live-region slot to announce NaN CSV error, got ({s0:?}, {s1:?})"
    );
}

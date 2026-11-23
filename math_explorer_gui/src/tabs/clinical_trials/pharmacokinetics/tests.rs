//! Analytical parity, zero-allocation (`VerifiedAllocator`), and state-validation tests.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{InteractiveTool, TraceSeries};
use math_commons::registry::TOLERANCE_FAST;
use math_explorer::applied::pharmacokinetics::{
    concentration_bateman, concentration_superposition, half_life, solve_ka, t_max,
    EnantiomerModel, PKParameters,
};

#[test]
fn test_analytical_bateman_single_and_equal_rates() {
    let mut tool = PharmacokineticsTool {
        bioavailability: 1.0,
        dose_mg: 20.0,
        ka: 1.2,
        ke: std::f64::consts::LN_2 / 10.0,
        volume_l: 1.0,
        n_doses: 1,
        horizon_hours: 20.0,
        sample_points: 201,
        ..PharmacokineticsTool::default()
    };
    tool.recalculate();

    let params = PKParameters::new(1.0, 20.0, tool.ka, tool.ke, 1.0).unwrap();
    for pt in &tool.total_points {
        let expected = concentration_bateman(&params, pt.x);
        assert!((pt.y - expected).abs() < TOLERANCE_FAST);
    }
    assert!((tool.metrics.half_life_h - half_life(tool.ke)).abs() < TOLERANCE_FAST);
    assert!((tool.metrics.half_life_h - 10.0).abs() < TOLERANCE_FAST);
    assert!((tool.metrics.t_max_h - t_max(tool.ka, tool.ke)).abs() < TOLERANCE_FAST);
    assert!((tool.metrics.clearance_l_h - tool.ke * tool.volume_l).abs() < TOLERANCE_FAST);

    tool.ka = tool.ke;
    tool.recalculate();
    let eq_params = PKParameters::new(1.0, 20.0, tool.ke, tool.ke, 1.0).unwrap();
    let sample_5h = tool
        .total_points
        .iter()
        .find(|p| (p.x - 5.0).abs() < 1.0e-9)
        .expect("t = 5h sample present");
    assert!((sample_5h.y - concentration_bateman(&eq_params, 5.0)).abs() < TOLERANCE_FAST);
}

#[test]
fn test_solve_ka_calibration_and_non_convergence_diagnostic() {
    let ke = std::f64::consts::LN_2 / 10.0;
    let mut tool = PharmacokineticsTool {
        absorption_mode: AbsorptionMode::SolveFromTMax,
        ke,
        target_t_max: 3.0,
        n_doses: 1,
        ..PharmacokineticsTool::default()
    };
    tool.recalculate();
    assert!(tool.solver_error.is_none());
    let ref_ka = solve_ka(3.0, ke, 1.0, 100, TOLERANCE_FAST).unwrap();
    assert!((tool.effective_ka - ref_ka).abs() < 1.0e-5);
    assert!((t_max(tool.effective_ka, ke) - 3.0).abs() < TOLERANCE_FAST);

    tool.target_t_max = 0.0;
    tool.recalculate();
    assert!(tool.solver_error.is_some());
    assert!(tool.total_points.is_empty());

    tool.ke = 1.8;
    tool.target_t_max = 45.0;
    tool.recalculate();
    assert!(tool.solver_error.is_some());
}

#[test]
fn test_superposition_loading_and_missed_dose_parity() {
    let mut tool = PharmacokineticsTool {
        bioavailability: 0.9,
        dose_mg: 100.0,
        loading_dose_mg: 200.0,
        use_loading_dose: true,
        ka: 1.4,
        ke: 0.12,
        volume_l: 30.0,
        interval_hours: 12.0,
        n_doses: 4,
        missed_slots: vec![2],
        horizon_hours: 48.0,
        sample_points: 241,
        ..PharmacokineticsTool::default()
    };
    tool.recalculate();

    let maint_p = PKParameters::new(0.9, 100.0, 1.4, 0.12, 30.0).unwrap();
    let load_p = PKParameters::new(0.9, 200.0, 1.4, 0.12, 30.0).unwrap();
    for pt in &tool.total_points {
        let expected = concentration_bateman(&load_p, pt.x)
            + concentration_superposition(&maint_p, &[12.0, 36.0], pt.x);
        assert!((pt.y - expected).abs() < 1.0e-10);
    }
}

#[test]
fn test_two_pulse_and_enantiomer_ir_xr_parity() {
    let ke_d = std::f64::consts::LN_2 / 10.0;
    let ke_l = std::f64::consts::LN_2 / 13.0;
    let ka = solve_ka(3.0, ke_d, 1.0, 100, TOLERANCE_FAST).unwrap();
    let d_p = PKParameters::new(1.0, 20.0, ka, ke_d, 1.0).unwrap();
    let l_p = PKParameters::new(1.0, 20.0, ka, ke_l, 1.0).unwrap();
    let enantiomer = EnantiomerModel::new(d_p, l_p, 0.75, 0.25).unwrap();

    let mut tool = PharmacokineticsTool {
        entity_mode: EntityMode::Enantiomer,
        release_mode: ReleaseMode::ImmediateRelease,
        bioavailability: 1.0,
        dose_mg: 20.0,
        ka,
        ke: ke_d,
        ka_l: ka,
        ke_l,
        volume_l: 1.0,
        f_d: 0.75,
        f_l: 0.25,
        n_doses: 1,
        horizon_hours: 20.0,
        sample_points: 101,
        ..PharmacokineticsTool::default()
    };
    tool.recalculate();
    for i in 0..tool.total_points.len() {
        let t = tool.total_points[i].x;
        let expected = enantiomer.concentration_ir_single_dose(t);
        assert!((tool.total_points[i].y - expected).abs() < 1.0e-10);
        let sum_iso = tool.d_points[i].y + tool.l_points[i].y;
        assert!((tool.total_points[i].y - sum_iso).abs() < 1.0e-10);
    }

    tool.release_mode = ReleaseMode::TwoPulse;
    tool.n_doses = 2;
    tool.interval_hours = 24.0;
    tool.horizon_hours = 40.0;
    tool.lag_time_hours = 4.0;
    tool.f1 = 0.5;
    tool.f2 = 0.5;
    tool.recalculate();
    for i in 0..tool.total_points.len() {
        let t = tool.total_points[i].x;
        let expected = enantiomer.concentration_xr_multiple_doses(&[0.0, 24.0], 4.0, 0.5, 0.5, t);
        assert!((tool.total_points[i].y - expected).abs() < 1.0e-10);
        let sum_iso = tool.d_points[i].y + tool.l_points[i].y;
        assert!((tool.total_points[i].y - sum_iso).abs() < 1.0e-10);
    }
}

#[test]
fn test_trapezoidal_auc_accumulation_and_therapeutic_window() {
    let mut tool = PharmacokineticsTool {
        bioavailability: 1.0,
        dose_mg: 100.0,
        ka: 2.0,
        ke: 0.2,
        volume_l: 10.0,
        n_doses: 1,
        interval_hours: 12.0,
        horizon_hours: 80.0,
        sample_points: 1_601,
        mec: 0.0,
        mtc: 100.0,
        ..PharmacokineticsTool::default()
    };
    tool.recalculate();

    let analytical_auc = tool.bioavailability * tool.dose_mg / (tool.ke * tool.volume_l);
    assert!((tool.metrics.auc_trapezoidal - analytical_auc).abs() < 0.02);
    let expected_rac = 1.0 / (1.0 - (-tool.ke * tool.interval_hours).exp());
    assert!((tool.metrics.accumulation_ratio - expected_rac).abs() < TOLERANCE_FAST);
    assert!((tool.metrics.therapeutic_window_pct - 100.0).abs() < TOLERANCE_FAST);

    tool.mec = 50.0;
    tool.mtc = 100.0;
    tool.recalculate();
    assert!(tool.metrics.therapeutic_window_pct.abs() < TOLERANCE_FAST);
}

#[test]
fn test_zero_alloc_hot_path_across_all_four_formulations() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    for (entity, release, log10) in [
        (EntityMode::SingleEntity, ReleaseMode::ImmediateRelease, false),
        (EntityMode::SingleEntity, ReleaseMode::TwoPulse, true),
        (EntityMode::Enantiomer, ReleaseMode::ImmediateRelease, false),
        (EntityMode::Enantiomer, ReleaseMode::TwoPulse, true),
    ] {
        let mut tool = PharmacokineticsTool {
            entity_mode: entity,
            release_mode: release,
            use_loading_dose: true,
            n_doses: 8,
            missed_slots: vec![3],
            sample_points: 501,
            log10_y: log10,
            ..PharmacokineticsTool::default()
        };
        tool.recalculate();

        verified_engine::engine::VerifiedEngine::run_verified(|| {
            tool.evaluate_sampled_regimen();
        });
        assert_eq!(tool.total_points.len(), 501);
        assert!(tool.metrics.is_finite());
    }
}

#[test]
fn test_state_validation_rejects_corrupt_inputs_without_mutating() {
    let mut tool = PharmacokineticsTool {
        dose_mg: 225.0,
        ..PharmacokineticsTool::default()
    };
    tool.recalculate();
    let baseline = tool.save_state().unwrap();

    for bad_patch in [
        r#"{"unknown_extra":1,"#,
        r#"{"dose_mg":-5.0,"#,
        r#"{"bioavailability":1.5,"#,
        r#"{"volume_l":0.0,"#,
        r#"{"ka":-0.2,"#,
        r#"{"n_doses":65,"#,
        r#"{"sample_points":5000,"#,
        r#"{"mec":8.0,"mtc":4.0,"#,
        r#"{"missed_slots":[99],"#,
    ] {
        let candidate = baseline.replacen('{', bad_patch, 1);
        assert!(
            tool.try_load_state(&candidate).is_err(),
            "Expected rejection for patch {bad_patch}"
        );
        assert!((tool.dose_mg - 225.0).abs() < TOLERANCE_FAST);
    }
    assert!(tool.try_load_state("{\"dose_mg\":NaN}").is_err());
}

#[test]
fn test_recovery_preserves_pinned_snapshots_and_clamp_recovers_nan() {
    let mut tool = PharmacokineticsTool::default();
    let recovery = tool.save_recovery_state().unwrap();
    assert!(!recovery.contains("snapshot_manager"));

    assert!(tool.snapshot_manager.add_snapshot(
        "Baseline",
        "D=100mg",
        vec![TraceSeries::new("Total C(t)", vec![[0.0, 0.0], [1.0, 1.5]])],
    ));
    tool.dose_mg = 300.0;
    tool.recalculate();
    tool.try_load_recovery_state(&recovery).unwrap();
    assert!((tool.dose_mg - 100.0).abs() < TOLERANCE_FAST);
    assert_eq!(tool.snapshot_manager.len(), 1);

    tool.dose_mg = f64::NAN;
    tool.volume_l = f64::INFINITY;
    tool.mec = 10.0;
    tool.mtc = 2.0;
    assert!(!tool.has_finite_parameters());
    assert!(tool.clamp_parameters());
    assert!(tool.has_finite_parameters());
}

// [cite:clinical_trials_statistics]

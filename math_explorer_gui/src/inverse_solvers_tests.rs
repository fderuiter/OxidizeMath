//! Ground-truth solver verification, keyboard accessibility, and `VerifiedAllocator` zero-heap-allocation
//! tests across MRI Bloch pulses, pharmacokinetic superposition, Lattice Boltzmann fairings, and IMRT beams.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use crate::async_sim::unified::UnifiedSimRunner;
use crate::async_sim::{SimCommand, SimulationRunner};
use crate::framework::InteractiveTool;
use crate::tabs::clinical_trials::pharmacokinetics::challenge::{
    PkRegimenState, PkWorkspace, CHALLENGE_PK_THERAPEUTIC_WINDOW,
};
use crate::tabs::clinical_trials::pharmacokinetics::PharmacokineticsTool;
use crate::tabs::fluid_dynamics::lattice_boltzmann::LbmUnified;
use crate::tabs::fluid_dynamics::lbm_fairing::{
    FairingStamp, LbmFairingEvaluator, CHALLENGE_LBM_FAIRING, CHALLENGE_LBM_VISCOSITY,
    LBM_CELL_COUNT, LBM_CORE_RADIUS, LBM_CORE_X, LBM_CORE_Y, LBM_MIN_SOLID_CELLS,
};
use crate::tabs::fluid_dynamics::lbm_fairing_ui::LbmUiExtension;
use crate::tabs::medical::dose::DoseCalculationTool;
use crate::tabs::medical::imrt::{
    ImrtWorkspace, CHALLENGE_IMRT_BETA, CHALLENGE_IMRT_OAR_LIMIT_GY, CHALLENGE_IMRT_OAR_SPARING,
    CHALLENGE_IMRT_PRESCRIPTION_GY,
};
use crate::tabs::mri::pulse::{
    MriPulseMode, MriPulseWorkspace, RfPulseConfig, CHALLENGE_BLOCH_HAHN_ECHO,
    CHALLENGE_BLOCH_PI_INVERSION, CHALLENGE_MRI_B0, CHALLENGE_MRI_T1, CHALLENGE_MRI_T2,
    CHALLENGE_MRI_TE,
};
use crate::tabs::mri::MriTool;
use math_explorer::applied::pharmacokinetics::{solve_ka, t_max};
use math_explorer::physics::mri::proton;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[test]
fn test_ground_truth_mri_pi_inversion_and_detuned_failure() {
    let duration_tau = 0.002_f64;
    let b1_pi = std::f64::consts::PI / (proton::GYROMAGNETIC_RATIO * duration_tau);

    let mut mri = MriTool::default();
    mri.apply_challenge(Some(CHALLENGE_BLOCH_PI_INVERSION))
        .unwrap();
    mri.pulse_config = RfPulseConfig {
        mode: MriPulseMode::InversionPulse,
        b1_amp: b1_pi,
        duration_tau,
        phase_deg: 0.0,
        detuning_hz: 0.0,
        echo_time_te: CHALLENGE_MRI_TE,
    };
    mri.recompute_pulse_sequence();

    let eval_pi = mri.evaluate_challenge().expect("pi inversion evaluation");
    assert!(eval_pi.residual_value <= -0.90 && eval_pi.passed && eval_pi.score >= 95.0);

    mri.pulse_config.detuning_hz = 2500.0;
    mri.recompute_pulse_sequence();
    let eval_detuned = mri.evaluate_challenge().unwrap();
    assert!(!eval_detuned.passed, "Detuned RF pulse must fail inversion");
}

#[test]
fn test_ground_truth_mri_hahn_spin_echo_and_zero_alloc() {
    let duration_tau = 0.002_f64;
    let b1_pi = std::f64::consts::PI / (proton::GYROMAGNETIC_RATIO * duration_tau);
    let mut mri = MriTool::default();
    mri.apply_challenge(Some(CHALLENGE_BLOCH_HAHN_ECHO))
        .unwrap();
    mri.pulse_config = RfPulseConfig {
        mode: MriPulseMode::HahnSpinEcho,
        b1_amp: b1_pi,
        duration_tau,
        phase_deg: 90.0,
        detuning_hz: 0.0,
        echo_time_te: CHALLENGE_MRI_TE,
    };
    mri.recompute_pulse_sequence();
    let eval_echo = mri.evaluate_challenge().expect("Hahn spin-echo evaluation");
    let min_echo = 0.75 * (-CHALLENGE_MRI_TE / CHALLENGE_MRI_T2).exp();
    assert!(eval_echo.residual_value >= min_echo && eval_echo.passed);

    let mut ws = MriPulseWorkspace::default();
    let cfg = mri.pulse_config;
    {
        let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            let inv_m = ws.evaluate_inversion(
                CHALLENGE_MRI_B0,
                CHALLENGE_MRI_T1,
                CHALLENGE_MRI_T2,
                1.0,
                &cfg,
            );
            let echo_m = ws.evaluate_hahn_echo(
                CHALLENGE_MRI_B0,
                CHALLENGE_MRI_T1,
                CHALLENGE_MRI_T2,
                1.0,
                &cfg,
            );
            assert!(inv_m.z.is_finite() && echo_m.x.is_finite());
        });
    }
}

#[test]
fn test_ground_truth_pharmacokinetics_solve_ka_superposition_and_zero_alloc() {
    // 1. Verify solve_ka recovers true ka from analytical T_max within 1e-6
    let ke = 0.12_f64;
    let ka_true = 1.85_f64;
    let target_tmax = t_max(ka_true, ke);
    let ka_recovered = solve_ka(target_tmax, ke, 2.0, 100, 1e-11).expect("solve_ka must converge");
    assert!(
        (ka_recovered - ka_true).abs() < 1e-6,
        "solve_ka error |{ka_recovered} - {ka_true}| >= 1e-6"
    );

    // 2. Valid multi-dose schedule scores 100.0 and passes therapeutic window
    let mut pk_tool = PharmacokineticsTool::default();
    pk_tool
        .apply_challenge(Some(CHALLENGE_PK_THERAPEUTIC_WINDOW))
        .unwrap();
    pk_tool.dose_mg = 165.0;
    pk_tool.interval_hours = 6.0;
    pk_tool.use_loading_dose = false;
    pk_tool.recalculate();
    let valid_eval = pk_tool.evaluate_challenge().unwrap();
    assert!(valid_eval.passed);
    assert!((valid_eval.score - 100.0).abs() < 1e-9);

    // 3. Toxic regimen fails with E_tox > 0
    pk_tool.dose_mg = 450.0;
    pk_tool.interval_hours = 4.0;
    pk_tool.recalculate();
    let toxic_eval = pk_tool.evaluate_challenge().unwrap();
    assert!(!toxic_eval.passed);
    assert!(pk_tool.workspace.summary.e_tox > 5.0);
    assert_eq!(toxic_eval.score, 0.0);

    // 4. Zero dynamic heap allocations inside VerifiedEngine::run_verified
    let mut ws = PkWorkspace::default();
    let valid_state = PkRegimenState::default();
    {
        let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            let summary = ws.evaluate(&valid_state);
            assert!(summary.passed);
            assert!((summary.score - 100.0).abs() < 1e-9);
        });
    }
}

#[test]
fn test_ground_truth_lbm_reset_preserves_fairing_and_restores_core() {
    let mut params_map = HashMap::from([("viscosity".to_string(), CHALLENGE_LBM_VISCOSITY)]);
    let shared = Arc::new(RwLock::new(params_map.clone()));
    let mut runner = UnifiedSimRunner::<LbmUnified>::new(Arc::clone(&shared));

    runner.process_command(SimCommand::ApplyBrush {
        cx: 38,
        cy: 25,
        r: 1,
        is_obstacle: true,
    });
    runner.process_command(SimCommand::ApplyBrush {
        cx: LBM_CORE_X,
        cy: LBM_CORE_Y,
        r: 1,
        is_obstacle: false,
    });
    runner.step();
    runner.step();
    let snap_stepped = runner.get_snapshot();
    assert_eq!(snap_stepped.custom_data[5], 2.0);
    assert_eq!(snap_stepped.custom_data[4], 0.0);

    runner.process_command(SimCommand::Reset);
    let snap_reset = runner.get_snapshot();
    assert_eq!(snap_reset.custom_data[5], 0.0);
    assert_eq!(snap_reset.custom_data[4], 1.0);
    assert!(snap_reset.custom_data[3] > 81.0);

    let mut ui_ext = LbmUiExtension::default();
    let _ = crate::async_sim::unified::UnifiedUiExtension::apply_challenge(
        &mut ui_ext,
        Some(CHALLENGE_LBM_FAIRING),
        &mut params_map,
    )
    .unwrap();
    let start_pos = (ui_ext.cursor_x, ui_ext.cursor_y);
    assert_eq!(
        ui_ext.nudge_cursor(2, 1),
        (start_pos.0 + 2, start_pos.1 + 1)
    );
    let (became_solid, cmds) = ui_ext.toggle_cursor_cell(CHALLENGE_LBM_VISCOSITY).unwrap();
    assert!(became_solid && !cmds.is_empty());

    ui_ext.cursor_x = LBM_CORE_X;
    ui_ext.cursor_y = LBM_CORE_Y + LBM_CORE_RADIUS - 1;
    assert!(ui_ext.toggle_cursor_cell(CHALLENGE_LBM_VISCOSITY).is_err());
}

#[test]
fn test_ground_truth_lbm_teardrop_drag_and_zero_alloc() {
    let mut evaluator = LbmFairingEvaluator::default();
    let mut cyl_mask = [false; LBM_CELL_COUNT];
    FairingStamp::BluffCylinder.write_mask(&mut cyl_mask);
    let cyl_metrics = evaluator.evaluate_mask_in_place(&cyl_mask, CHALLENGE_LBM_VISCOSITY);
    assert!(cyl_metrics.n_solid >= LBM_MIN_SOLID_CELLS && cyl_metrics.core_intact);
    assert!(!cyl_metrics.passed);

    let mut teardrop_mask = [false; LBM_CELL_COUNT];
    FairingStamp::TeardropFairing.write_mask(&mut teardrop_mask);
    let mut teardrop_metrics = Default::default();
    {
        let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            teardrop_metrics =
                evaluator.evaluate_mask_in_place(&teardrop_mask, CHALLENGE_LBM_VISCOSITY);
        });
    }
    assert!(teardrop_metrics.n_solid >= LBM_MIN_SOLID_CELLS && teardrop_metrics.core_intact);
    assert!(teardrop_metrics.c_d <= 0.82 * cyl_metrics.c_d_cylinder);
    assert!(teardrop_metrics.passed && teardrop_metrics.score >= 60.0);
}

#[test]
fn test_ground_truth_imrt_beams_cost_gradient_hint_and_zero_alloc() {
    let mut dose = DoseCalculationTool::default();
    // Forward point-source starter preset still loads unchanged
    assert_eq!(
        dose.save_state().unwrap(),
        r#"{"amplitude":1.0,"beta":1.0,"width":200,"height":200}"#
    );

    dose.apply_challenge(Some(CHALLENGE_IMRT_OAR_SPARING))
        .unwrap();
    let base_eval = dose.evaluate_challenge().expect("IMRT baseline eval");
    assert!(
        !base_eval.passed,
        "Baseline 0°/180° beams hit OAR and must fail"
    );
    let cost_before_hint = dose.imrt_workspace.summary.cost_j;

    // 1-step projected gradient descent hint via optimization::update_weights must strictly reduce cost J
    dose.apply_gradient_weight_hint();
    let cost_after_hint = dose.imrt_workspace.summary.cost_j;
    assert!(
        cost_after_hint < cost_before_hint,
        "update_weights hint must reduce cost J: after={cost_after_hint} vs before={cost_before_hint}"
    );

    // Oblique OAR-sparing 4-beam arrangement (`[65, 115, 245, 295]` deg) + tuned weights
    dose.beam_angles_deg = [65.0, 115.0, 245.0, 295.0];
    dose.beam_weights = [1.56, 1.56, 1.56, 1.56];
    dose.refresh_model();
    for _ in 0..4 {
        dose.apply_gradient_weight_hint();
    }
    let summary = dose.imrt_workspace.summary;
    assert!(
        summary.oar_max_gy.0 <= CHALLENGE_IMRT_OAR_LIMIT_GY,
        "OAR max dose {:.2} Gy must be <= {:.2} Gy",
        summary.oar_max_gy.0,
        CHALLENGE_IMRT_OAR_LIMIT_GY
    );
    assert!(
        summary.cost_j <= 0.15 * summary.baseline_cost_j,
        "Optimized cost J ({:.2}) must be <= 0.15 * J_base ({:.2})",
        summary.cost_j,
        0.15 * summary.baseline_cost_j
    );
    let solved_eval = dose.evaluate_challenge().unwrap();
    assert!(solved_eval.passed);
    assert!(solved_eval.score >= 85.0);

    // Zero dynamic heap allocations inside VerifiedEngine::run_verified
    let mut ws = ImrtWorkspace::default();
    let angles = dose.beam_angles_deg;
    let weights = dose.beam_weights;
    {
        let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            let s = ws.evaluate_in_place(
                &angles,
                &weights,
                CHALLENGE_IMRT_BETA,
                CHALLENGE_IMRT_PRESCRIPTION_GY,
                CHALLENGE_IMRT_OAR_LIMIT_GY,
            );
            assert!(s.passed);
        });
    }
}

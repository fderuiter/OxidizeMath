//! Ground-truth solver verification, keyboard accessibility, and `VerifiedAllocator` zero-heap-allocation
//! tests across MRI Bloch pulses, pharmacokinetic superposition, Lattice Boltzmann fairings, and IMRT beams.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use crate::async_sim::unified::UnifiedSimRunner;
use crate::async_sim::{SimCommand, SimulationRunner};
use crate::framework::InteractiveTool;
use crate::tabs::clinical_trials::pharmacokinetics::PharmacokineticsTool;
use crate::tabs::clinical_trials::pharmacokinetics::challenge::{
    CHALLENGE_PK_THERAPEUTIC_WINDOW, PkRegimenState, PkWorkspace,
};
use crate::tabs::fluid_dynamics::lattice_boltzmann::LbmUnified;
use crate::tabs::fluid_dynamics::lbm_fairing::{
    CHALLENGE_LBM_FAIRING, CHALLENGE_LBM_VISCOSITY, FairingStamp, LBM_CELL_COUNT, LBM_CORE_RADIUS,
    LBM_CORE_X, LBM_CORE_Y, LBM_MIN_SOLID_CELLS, LbmFairingEvaluator,
};
use crate::tabs::fluid_dynamics::lbm_fairing_ui::LbmUiExtension;
use crate::tabs::medical::dose::DoseCalculationTool;
use crate::tabs::medical::imrt::{
    CHALLENGE_IMRT_BETA, CHALLENGE_IMRT_OAR_LIMIT_GY, CHALLENGE_IMRT_OAR_SPARING,
    CHALLENGE_IMRT_PRESCRIPTION_GY, ImrtWorkspace,
};
use crate::tabs::mri::MriTool;
use crate::tabs::mri::pulse::{
    CHALLENGE_BLOCH_HAHN_ECHO, CHALLENGE_BLOCH_PI_INVERSION, CHALLENGE_MRI_B0, CHALLENGE_MRI_T1,
    CHALLENGE_MRI_T2, CHALLENGE_MRI_TE, MriPulseMode, MriPulseWorkspace, RfPulseConfig,
};
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

#[test]
fn test_ground_truth_auction_cartographer_and_workspace_zero_alloc() {
    use crate::async_sim::atlas::RegimeCartographer;
    use crate::tabs::game_theory::auction::MyersonAuctionTool;
    use crate::tabs::game_theory::auction::atlas::AuctionCartographer;
    use crate::tabs::game_theory::auction::math::{
        AuctionDistEval, AuctionDistributionKind, AuctionEvalInputs, AuctionEvalWorkspace,
        AuctionRuleMode,
    };
    use scientific_metadata::theory::TheoryDescribable;

    let auction_tool = MyersonAuctionTool::default();
    let dist =
        AuctionDistEval::from_params(AuctionDistributionKind::Uniform, 0.0, 100.0, 0.1, 2.0, 2.0);
    let auction_cart = AuctionCartographer::new(
        &auction_tool.theory_parameters(),
        ("reserve_price", "n_bidders"),
        dist.clone(),
        2,
        AuctionRuleMode::FirstPrice,
    );
    let mut auction_ws = AuctionEvalWorkspace::default();

    let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (ok, _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        let ac = auction_cart.evaluate_cell(50.0, 2.0);
        let am = auction_ws.evaluate_analytical(
            &dist,
            AuctionEvalInputs {
                rule: AuctionRuleMode::FirstPrice,
                reserve: 50.0,
                n_bidders: 2,
                focal_v: 100.0,
                candidate_bid: 62.5,
                fixed_point_tol: 0.5,
            },
        );
        ac.primary.is_finite() && am.is_equilibrium_fixed_point
    });
    assert!(ok);
}

#[test]
fn test_ground_truth_ising_cartographer_and_workspace_zero_alloc() {
    use crate::async_sim::atlas::RegimeCartographer;
    use crate::tabs::solid_state::ising::atlas::IsingCartographer;
    use crate::tabs::solid_state::ising::challenge::IsingChallengeWorkspace;
    use crate::tabs::solid_state::ising::{DropletSeed, IsingModelTool};
    use scientific_metadata::theory::TheoryDescribable;

    let ising_tool = IsingModelTool::default();
    let ising_theory = ising_tool.theory_parameters();
    let ising_cart = IsingCartographer::new(
        &ising_theory,
        ("T", "S"),
        0.08,
        1.0,
        0.0,
        [1.0, -0.1, 1.25, 0.0],
    );
    let mut ising_ws = IsingChallengeWorkspace::default();
    let seed = [DropletSeed {
        cx: 50,
        cy: 50,
        r: 8,
        spin: 1,
    }];

    let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (passed_pair, _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        let ic = ising_cart.evaluate_cell(1.25, -0.1);
        let inuc = ising_ws.evaluate_ising_nucleation(&seed);
        let fnuc = ising_ws.evaluate_fermi_nucleation(&seed);
        assert!(ic.primary.is_finite());
        (inuc.passed, fnuc.passed)
    });
    assert_eq!(passed_pair, (true, true));
}

#[test]
fn test_auction_and_ising_federated_challenge_registry_and_stateless_urls() {
    use crate::app::MathExplorerApp;
    use crate::starter_gallery::{find_challenge_metadata, get_challenge_cards_for_domain};
    use crate::url_state::{
        MAX_DECODED_URL_STATE_BYTES, MAX_URL_FRAGMENT_BYTES, parse_share_location,
    };

    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    assert_eq!(get_challenge_cards_for_domain("game_theory").len(), 2);
    assert_eq!(get_challenge_cards_for_domain("solid_state").len(), 2);

    for (slug, challenge_id) in [
        ("game_theory", "myerson_welfare_constrained_reserve"),
        ("game_theory", "first_price_shading_equilibrium"),
        ("solid_state", "ising_critical_droplet_nucleation"),
        ("solid_state", "fermi_prisoners_dilemma_nucleation"),
    ] {
        let meta = find_challenge_metadata(challenge_id).expect("ChallengeMetadata present");
        let clamped = meta
            .clamp_state_json(&meta.generate_baseline_state())
            .expect("baseline state passes clamp_state_json");
        assert!(clamped.len() <= MAX_DECODED_URL_STATE_BYTES);

        let bare_hash = format!("#tab={slug}&challenge={challenge_id}");
        let mut app = MathExplorerApp::default();
        assert!(
            app.apply_share_location_strings(&bare_hash, "", None)
                .unwrap()
        );
        let active_fw = app.tabs[app.selected_tab].framework().unwrap();
        assert_eq!(active_fw.active_challenge_id(), Some(challenge_id));
        let full_url = app.encode_current_share_url().unwrap();
        let fragment = &full_url[full_url.find('#').unwrap()..];
        assert!(fragment.len() <= MAX_URL_FRAGMENT_BYTES);
        let full_params = parse_share_location(fragment, "").unwrap().unwrap();
        assert_eq!(full_params.challenge.as_deref(), Some(challenge_id));
        assert!(full_params.state_json.unwrap().len() <= MAX_DECODED_URL_STATE_BYTES);
    }
}

#[test]
fn test_tampered_auction_challenge_url_clamps_locked_parameters() {
    use crate::app::MathExplorerApp;
    use crate::url_state::{encode_base64url, parse_share_location};

    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut app = MathExplorerApp::default();
    app.apply_share_location_strings(
        "#tab=game_theory&challenge=myerson_welfare_constrained_reserve",
        "",
        None,
    )
    .unwrap();
    let url = app.encode_current_share_url().unwrap();
    let params = parse_share_location(&url[url.find('#').unwrap()..], "")
        .unwrap()
        .unwrap();
    let mut env: serde_json::Value =
        serde_json::from_str(params.state_json.as_ref().unwrap()).unwrap();
    env["config"]["reserve_price"] = serde_json::json!(38.0);
    env["config"]["n_bidders"] = serde_json::json!(9);
    env["config"]["uniform_max"] = serde_json::json!(500.0);
    let b64 = encode_base64url(env.to_string().as_bytes());
    let tampered = format!(
        "#tab=game_theory&tool=MyersonAuctionTool&challenge=myerson_welfare_constrained_reserve&state={b64}"
    );
    let mut dest = MathExplorerApp::default();
    assert!(
        dest.apply_share_location_strings(&tampered, "", None)
            .unwrap()
    );
    let fw = dest.tabs[dest.selected_tab].framework().unwrap();
    let cfg: serde_json::Value =
        serde_json::from_str(&fw.active_tool.as_ref().unwrap().save_state().unwrap()).unwrap();
    assert_eq!(cfg["reserve_price"], serde_json::json!(38.0));
    assert_eq!(cfg["n_bidders"], serde_json::json!(2));
    assert_eq!(cfg["uniform_max"], serde_json::json!(100.0));
    assert!(
        fw.active_tool
            .as_ref()
            .unwrap()
            .evaluate_challenge()
            .unwrap()
            .passed
    );
}

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::wcag_contrast_on_black;
use crate::async_sim::atlas::{AtlasTeleportEvent, RegimeCartographer};
use crate::framework::InteractiveTool;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;

fn assert_finite_constraints(params: &HashMap<String, ParameterConstraint>, expected: &[&str]) {
    for &key in expected {
        let c = params
            .get(key)
            .unwrap_or_else(|| panic!("missing theory parameter '{key}'"));
        assert!(
            c.min.is_finite() && c.max.is_finite() && c.step.is_finite(),
            "non-finite constraint for '{key}': {c:?}"
        );
        assert!(
            c.max > c.min && c.step > 0.0,
            "invalid range for '{key}': {c:?}"
        );
    }
}

fn verify_cartographer_zero_alloc_and_wcag(cart: &dyn RegimeCartographer) {
    let axes = cart.axes();
    assert!(axes.x.constraint.max > axes.x.constraint.min);
    assert!(axes.y.constraint.max > axes.y.constraint.min);
    for (_, swatch) in cart.legend_entries(true) {
        assert!(wcag_contrast_on_black(swatch) >= 3.0);
    }
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        for sy in 0..24 {
            for sx in 0..24 {
                let (p1, p2) = axes.pixel_to_coords(sx, sy, 24, 24);
                let cell = cart.evaluate_cell(p1, p2);
                assert!(cell.primary.is_finite() || cell.regime >= 3);
                let hc = cart.color_for_cell(cell, true);
                assert!(
                    wcag_contrast_on_black(hc) >= 3.0,
                    "HighContrast WCAG < 3.0 at ({p1}, {p2}): {hc:?}"
                );
            }
        }
    });
}

#[cfg(feature = "biology")]
#[test]
fn test_biology_theory_bounds_and_gray_scott_regimes() {
    use crate::presets::MorphogenesisKineticsMode;
    use crate::tabs::morphogenesis::{
        GRAY_SCOTT_ATLAS_AXIS_PAIRS, GrayScottCartographer, GrayScottRegime, LeniaTool,
        MorphogenesisTheory, classify_gray_scott_regime,
    };

    let mut m_theory = MorphogenesisTheory::default();
    assert_finite_constraints(
        &m_theory.theory_parameters(),
        &["a", "b", "d_u", "d_v", "dt"],
    );
    m_theory.mode = MorphogenesisKineticsMode::GrayScott;
    let gs_theory = m_theory.theory_parameters();
    assert_finite_constraints(&gs_theory, &["f", "k", "d_u", "d_v", "dt"]);

    let lenia = LeniaTool::default();
    assert_finite_constraints(
        &lenia.theory().theory_parameters(),
        &[
            "mu",
            "sigma",
            "radius",
            "dt",
            "perturbation",
            "brush_radius",
        ],
    );

    let cases = [
        (
            (0.04, 0.09, 0.16, 0.08, 1.0),
            GrayScottRegime::SaddleNodeWashout,
        ),
        (
            (0.0367, 0.0649, 0.16, 0.08, 25.0),
            GrayScottRegime::EulerCflUnstable,
        ),
        ((0.014, 0.045, 0.16, 0.08, 0.5), GrayScottRegime::HopfWave),
        (
            (0.046, 0.059, 0.20, 0.04, 0.5),
            GrayScottRegime::TuringMitosis,
        ),
        (
            (0.055, 0.055, 0.10, 0.10, 0.5),
            GrayScottRegime::BistableHomogeneous,
        ),
    ];
    for ((f, k, du, dv, dt), expected) in cases {
        assert_eq!(
            GrayScottRegime::from_u8(classify_gray_scott_regime(f, k, du, dv, dt).regime),
            expected
        );
    }

    for &pair in GRAY_SCOTT_ATLAS_AXIS_PAIRS {
        let cart = GrayScottCartographer::new(&gs_theory, pair, 0.0367, 0.0649, 0.16, 0.08, 1.0);
        if pair == ("f", "k") {
            let (lbl, pts) = cart.overlay_contour().expect("Pearson fold contour");
            assert!(lbl.contains("4(f + k)² = f") && pts.len() >= 10);
        }
        verify_cartographer_zero_alloc_and_wcag(&cart);
    }
}

#[cfg(feature = "biology")]
#[test]
fn test_gray_scott_washout_auto_reseed_and_mode_preserving_teleport() {
    use crate::async_sim::SimulationRunner;
    use crate::presets::MorphogenesisKineticsMode;
    use crate::tabs::morphogenesis::MorphogenesisTool;

    let mut tool = MorphogenesisTool::default();
    {
        let mut ks = tool.kinetics_state.write().unwrap();
        ks.mode = MorphogenesisKineticsMode::GrayScott;
        ks.f = 0.04;
        ks.k = 0.09;
        let mut p = tool.params.write().unwrap();
        p.d_u = 0.16;
        p.d_v = 0.08;
        p.dt = 1.0;
    }

    let mut runner = MorphogenesisTool::create_kinetics_runner(
        &tool.params,
        &tool.kinetics_state,
        &tool.mask_rle,
        24,
        24,
    );
    runner.system.u_mut().fill(1.0);
    runner.system.v_mut().fill(0.0);
    runner.step();
    tool.kinetics_state.write().unwrap().f = 0.0367;
    tool.kinetics_state.write().unwrap().k = 0.0649;
    runner.step();
    let max_v = runner.system.v().iter().copied().fold(0.0_f64, f64::max);
    assert!(
        max_v > 0.05,
        "Gray-Scott v washout must auto-reseed catalyst in runner, got {max_v}"
    );

    runner.system.u_mut().fill(1.0);
    runner.system.v_mut().fill(0.0);
    tool.width = 24;
    tool.height = 24;
    tool.step_runner = Some(runner);
    tool.apply_atlas_teleport(AtlasTeleportEvent {
        x_key: "f",
        x_val: 0.0367,
        y_key: "k",
        y_val: 0.0649,
        regime_label: "Turing Mitosis",
    });
    assert_eq!(
        tool.kinetics_state.read().unwrap().mode,
        MorphogenesisKineticsMode::GrayScott
    );
    let reseeded_v = tool
        .step_runner
        .as_ref()
        .unwrap()
        .system
        .v()
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    assert!(reseeded_v > 0.05);
}

#[cfg(feature = "biology")]
#[test]
fn test_lenia_cartographer_zero_alloc_and_contour() {
    use crate::tabs::morphogenesis::lenia::{
        LENIA_ATLAS_AXIS_PAIRS, LeniaCartographer, LeniaRegime, LeniaSpeciesPreset, LeniaTool,
    };
    use math_explorer::biology::lenia::GrowthParams;

    let tool = LeniaTool::default();
    let theory = tool.theory().theory_parameters();
    for &pair in LENIA_ATLAS_AXIS_PAIRS {
        let cart = LeniaCartographer::new(
            &theory,
            pair,
            LeniaSpeciesPreset::Orbium,
            13.0,
            &[1.0],
            GrowthParams::new(0.15, 0.015),
            0.1,
        );
        if pair == ("mu", "sigma") {
            let (lbl, pts) = cart.overlay_contour().expect("G(0)=0 contour");
            assert_eq!(lbl, "G(0, μ, σ) = 0");
            assert!(pts.len() >= 2);
            let orbium_cell = cart.evaluate_cell(0.15, 0.015);
            assert!(matches!(
                LeniaRegime::from_u8(orbium_cell.regime),
                LeniaRegime::StableSoliton | LeniaRegime::ExpandingWave
            ));
            let extinct_cell = cart.evaluate_cell(0.52, 0.01);
            assert_eq!(
                LeniaRegime::from_u8(extinct_cell.regime),
                LeniaRegime::Extinction
            );
        }
        verify_cartographer_zero_alloc_and_wcag(&cart);
    }
}

#[cfg(feature = "biology")]
#[test]
fn test_lenia_url_bounds_and_extinct_auto_reseed() {
    use crate::framework::SimulationFramework;
    use crate::tabs::morphogenesis::lenia::LeniaTool;

    let mut tool = LeniaTool::default();
    let baseline_json = tool.save_state().expect("Lenia JSON");
    crate::url_state::URL_JSON_BOUNDS
        .validate_json(&baseline_json)
        .expect("Lenia state fits URL_JSON_BOUNDS");

    tool.mu = 0.55;
    tool.sigma = 0.01;
    for _ in 0..40 {
        tool.step_simulation();
    }
    assert!(tool.is_finite_and_extinct_for_test());

    tool.apply_atlas_teleport(AtlasTeleportEvent {
        x_key: "mu",
        x_val: 0.15,
        y_key: "sigma",
        y_val: 0.015,
        regime_label: "Stable Soliton Niche",
    });
    assert!(!tool.is_finite_and_extinct_for_test());

    let mut fw = SimulationFramework::new("biology");
    assert!(fw.select_tool_by_name("Lenia Studio"));
    assert!(fw.active_tool_supports_atlas());
    fw.set_parameter_atlas_visible(true);
    let ctx = eframe::egui::Context::default();
    let _ = ctx.run(eframe::egui::RawInput::default(), |egui_ctx| {
        eframe::egui::CentralPanel::default().show(egui_ctx, |ui| {
            fw.show_inside(ui, "test_bio_lenia_fw");
        });
    });
    let post_json = fw.active_tool.as_ref().unwrap().save_state().unwrap();
    crate::url_state::URL_JSON_BOUNDS
        .validate_json(&post_json)
        .expect("Framework atlas must not bloat LeniaToolState beyond URL_JSON_BOUNDS");
}

#[cfg(feature = "pure_math")]
#[test]
fn test_parrondo_9_state_stack_solver_parity_zero_alloc_and_wedge_teleport() {
    use crate::tabs::financial_math::parrondo::{
        PARRONDO_ATLAS_AXIS_PAIRS, ParrondoCartographer, ParrondoParadoxTool,
        solve_parrondo_ratchet_stack,
    };
    use math_explorer::pure_math::statistics::markov::parrondo::{
        ALLOWED_MODULI, ParrondoParams, ParrondoSchedule, solve_parrondo_ratchet,
    };

    let mut tool = ParrondoParadoxTool::default();
    assert_finite_constraints(
        &tool.theory().theory_parameters(),
        &["p_a_base", "p_a", "p_b0_base", "p_b1_base", "epsilon"],
    );

    for &m in &ALLOWED_MODULI {
        for sched_txt in ["ABB", "AABB", "ABAB"] {
            let sched = ParrondoSchedule::parse(sched_txt).expect("valid schedule");
            let params = ParrondoParams::new(m, 0.5, 0.1, 0.75, 0.005).expect("valid params");
            let exact = solve_parrondo_ratchet(&params, &sched).expect("exact solver");
            let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let (stack, _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
                solve_parrondo_ratchet_stack(m, 0.5, 0.1, 0.75, 0.005, sched.steps())
                    .expect("stack solver")
            });
            drop(_guard);
            assert!((stack.gain_a - exact.gain_a).abs() < 1e-9);
            assert!((stack.gain_b - exact.gain_b).abs() < 1e-9);
            assert!((stack.gain_mix - exact.gain_mix).abs() < 1e-9);
            assert!((stack.gain_schedule - exact.gain_schedule).abs() < 1e-9);
            assert!((stack.pi_b_trap - exact.pi_b[0]).abs() < 1e-9);
            assert!((stack.pi_sched_trap - exact.pi_schedule_mean[0]).abs() < 1e-9);
            assert_eq!(stack.is_paradox, exact.is_paradox());
        }
    }

    for &pair in PARRONDO_ATLAS_AXIS_PAIRS {
        let cart = ParrondoCartographer::new(&tool, pair).expect("Parrondo cartographer");
        verify_cartographer_zero_alloc_and_wcag(&cart);
    }

    let ctx = eframe::egui::Context::default();
    let rev_before = tool.state_revision();
    assert!(tool.teleport_phase_wedge(0.494, 0.006, Some(&ctx)));
    assert_ne!(tool.state_revision(), rev_before);
    let coords = tool.active_atlas_coords(("p_a", "epsilon"));
    assert!((coords.0 - 0.494).abs() < 1e-12 && (coords.1 - 0.006).abs() < 1e-12);
}

#[cfg(feature = "physics")]
#[test]
fn test_lbm_cartographer_zero_alloc_and_challenge_locks() {
    use crate::async_sim::unified::{UnifiedModel, UnifiedSimTool};
    use crate::tabs::fluid_dynamics::lattice_boltzmann::LbmUnified;
    use crate::tabs::fluid_dynamics::lbm_fairing::{
        CHALLENGE_LBM_FAIRING, CHALLENGE_LBM_VISCOSITY,
    };

    let mut lbm_tool = UnifiedSimTool::<LbmUnified>::new();
    assert_finite_constraints(
        &lbm_tool.theory().theory_parameters(),
        &["viscosity", "inlet_velocity", "angle_of_attack"],
    );
    let init_params = lbm_tool.params.read().unwrap().clone();
    let lbm_cart =
        LbmUnified::build_atlas_cartographer(&init_params, ("viscosity", "inlet_velocity"))
            .expect("LBM cartographer");
    verify_cartographer_zero_alloc_and_wcag(lbm_cart.as_ref());

    lbm_tool
        .apply_challenge(Some(CHALLENGE_LBM_FAIRING))
        .expect("activate fairing challenge");
    lbm_tool.apply_atlas_teleport(
        AtlasTeleportEvent {
            x_key: "viscosity",
            x_val: 0.25,
            y_key: "inlet_velocity",
            y_val: 0.16,
            regime_label: "Kármán Street",
        },
        None,
    );
    assert!(
        (lbm_tool.params.read().unwrap()["viscosity"] - CHALLENGE_LBM_VISCOSITY).abs() < 1e-12,
        "locked challenge viscosity must not change on atlas teleport"
    );
    lbm_tool.apply_challenge(None).expect("clear challenge");
    lbm_tool.apply_atlas_teleport(
        AtlasTeleportEvent {
            x_key: "viscosity",
            x_val: 0.08,
            y_key: "inlet_velocity",
            y_val: 0.14,
            regime_label: "Steady Wake",
        },
        None,
    );
    assert!((lbm_tool.params.read().unwrap()["inlet_velocity"] - 0.14).abs() < 1e-12);
}

#[cfg(feature = "physics")]
#[test]
fn test_turbulence_cartographer_zero_alloc_and_teleport() {
    use crate::tabs::fluid_dynamics::turbulence::TurbulenceTool;

    let mut turb = TurbulenceTool::default();
    assert_finite_constraints(
        &turb.theory().theory_parameters(),
        &["density", "dynamic_viscosity", "velocity", "length"],
    );
    for &pair in turb.atlas_axis_pairs() {
        let cart = turb
            .build_atlas_cartographer(pair)
            .expect("Turbulence cartographer");
        verify_cartographer_zero_alloc_and_wcag(cart.as_ref());
    }
    turb.apply_atlas_teleport(AtlasTeleportEvent {
        x_key: "velocity",
        x_val: 4.5,
        y_key: "length",
        y_val: 0.25,
        regime_label: "Turbulent",
    });
    assert!((turb.velocity - 4.5).abs() < 1e-12);
    let json = turb.save_state().expect("turbulence state");
    turb.try_load_state(&json)
        .expect("deny_unknown_fields intact");
}

#[cfg(feature = "applied")]
#[test]
fn test_pharmacokinetics_cartographer_and_challenge_locks() {
    use crate::tabs::clinical_trials::pharmacokinetics::PharmacokineticsTool;
    use crate::tabs::clinical_trials::pharmacokinetics::challenge::{
        CHALLENGE_PK_KE, CHALLENGE_PK_THERAPEUTIC_WINDOW,
    };

    let mut pk = PharmacokineticsTool::default();
    assert_finite_constraints(
        &pk.theory().theory_parameters(),
        &[
            "dose_mg",
            "bioavailability",
            "ka",
            "ke",
            "volume_l",
            "interval_hours",
            "mec",
            "mtc",
        ],
    );
    for &pair in pk.atlas_axis_pairs() {
        let cart = pk.build_atlas_cartographer(pair).expect("PK cartographer");
        verify_cartographer_zero_alloc_and_wcag(cart.as_ref());
    }
    pk.apply_challenge(Some(CHALLENGE_PK_THERAPEUTIC_WINDOW))
        .expect("activate PK challenge");
    pk.apply_atlas_teleport(AtlasTeleportEvent {
        x_key: "ka",
        x_val: 3.5,
        y_key: "ke",
        y_val: 0.5,
        regime_label: "Subtherapeutic",
    });
    assert!((pk.ke - CHALLENGE_PK_KE).abs() < 1e-12);
    pk.apply_atlas_teleport(AtlasTeleportEvent {
        x_key: "dose_mg",
        x_val: 180.0,
        y_key: "interval_hours",
        y_val: 8.0,
        regime_label: "Therapeutic",
    });
    assert!((pk.dose_mg - 180.0).abs() < 1e-12);
}

#[cfg(feature = "applied")]
#[test]
fn test_capacity_fade_cartographer_and_teleport() {
    use crate::tabs::battery_degradation::capacity_fade::CapacityFadeTool;

    let mut fade = CapacityFadeTool::default();
    assert_finite_constraints(
        &fade.theory().theory_parameters(),
        &["dod", "temperature", "cycles_to_simulate"],
    );
    let fade_cart = fade
        .build_atlas_cartographer(("dod", "cycles_to_simulate"))
        .expect("CapacityFade cartographer");
    verify_cartographer_zero_alloc_and_wcag(fade_cart.as_ref());
    fade.apply_atlas_teleport(AtlasTeleportEvent {
        x_key: "dod",
        x_val: 55.0,
        y_key: "cycles_to_simulate",
        y_val: 3500.0,
        regime_label: "Second-Life",
    });
    assert!((fade.dod - 55.0).abs() < 1e-12);
    let fade_json = fade.save_state().expect("fade json");
    fade.try_load_state(&fade_json)
        .expect("deny_unknown_fields intact");
}

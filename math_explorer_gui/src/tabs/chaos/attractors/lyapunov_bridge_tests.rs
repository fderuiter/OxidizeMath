#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::atlas::AttractorRegime;
use super::*;
use crate::async_sim::atlas::{AtlasTeleportEvent, RegimeCartographer};
use math_explorer::physics::chaos::fractals::correlation_dimension_in_place;
use math_explorer::physics::chaos::metrics::lorenz_lyapunov;
use math_explorer::pure_math::analysis::ode::RungeKutta4;
use scientific_metadata::theory::TheoryDescribable;
use verified_engine::engine::VerifiedEngine;

#[test]
fn test_evaluate_attractor_metrics_in_place_zero_allocations_and_signs() {
    let chaotic = LorenzBuilder::new()
        .sigma(10.0)
        .rho(28.0)
        .beta(8.0 / 3.0)
        .build(LorenzState::new(1.0, 1.0, 1.0));
    let stable = LorenzBuilder::new()
        .sigma(10.0)
        .rho(10.0)
        .beta(8.0 / 3.0)
        .build(LorenzState::new(1.0, 1.0, 1.0));

    let mut scratch = [Vector3::zeros(); 32];
    let mut sim = chaotic.clone();
    for slot in &mut scratch {
        sim.step(0.01);
        *slot = sim.state.vec;
    }
    let mut solver = RungeKutta4::new(&Vector3::zeros());

    let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ((lam_chaos, corr_chaos, lam_stable), _) = VerifiedEngine::run_verified(|| {
        let lc = lorenz_lyapunov(
            &chaotic,
            &mut solver,
            Vector3::new(10.0, 10.0, 10.0),
            0.01,
            40,
            0.25,
        )
        .unwrap();
        let cc = correlation_dimension_in_place(&mut scratch, 5.0);
        let ls = lorenz_lyapunov(
            &stable,
            &mut solver,
            Vector3::new(1.0, 1.0, 1.0),
            0.01,
            40,
            0.25,
        )
        .unwrap();
        (lc, cc, ls)
    });

    assert!(lam_chaos > 0.0);
    assert!(lam_stable < 0.0);
    assert!((0.0..=1.0).contains(&corr_chaos));
}

#[test]
fn test_diverged_attractor_halts_lyapunov_evaluation() {
    let mut plotter = AttractorPlotter::default();
    assert!(plotter.current_lyapunov_exponent().is_some());
    assert!(plotter.current_correlation_invariants().is_some());

    plotter.system.state.vec.x = f64::NAN;
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |c| plotter.show(c));

    assert!(plotter.diverged());
    assert!(plotter.current_lyapunov_exponent().is_none());
    assert!(plotter.current_correlation_invariants().is_none());
}

#[test]
fn test_lorenz_regime_cartographer_and_atlas_teleport() {
    let mut plotter = AttractorPlotter::default();
    let theory_params = plotter.theory_parameters();
    let cart = AttractorCartographer::new(
        &theory_params,
        ("sigma", "rho"),
        AttractorMode::Lorenz,
        CompiledFlowJob::from_bank(&plotter.flow_bank),
        (10.0, 28.0, 8.0 / 3.0),
        (0.2, 0.2),
    );

    let sink_cell = cart.evaluate_cell(10.0, 0.5);
    assert_eq!(sink_cell.regime, AttractorRegime::EquilibriumSink as u8);
    assert!(sink_cell.primary <= 0.04);

    let chaos_cell = cart.evaluate_cell(10.0, 28.0);
    assert_eq!(chaos_cell.regime, AttractorRegime::StrangeAttractor as u8);
    assert!(chaos_cell.primary > 0.04);

    plotter.step_simulation();
    assert!(!plotter.history.is_empty());
    assert!(plotter.current_lyapunov_exponent().expect("chaotic lyap") > 0.0);

    plotter.apply_atlas_teleport(AtlasTeleportEvent {
        x_key: "sigma",
        x_val: 10.0,
        y_key: "rho",
        y_val: 14.0,
        regime_label: "Equilibrium Sink",
    });
    assert!((plotter.system.rho - 14.0).abs() < 1e-9);
    assert!((plotter.system.sigma - 10.0).abs() < 1e-9);
    assert!(plotter.current_lyapunov_exponent().expect("stable lyap") < 0.0);
}

#[test]
fn test_multi_regime_invariants_and_ast_validation_guardrail() {
    let mut plotter = AttractorPlotter::default();
    for kind in [
        AttractorSystemKind::Rossler,
        AttractorSystemKind::DoublePendulum,
    ] {
        plotter.switch_system_kind(kind);
        let lyap = plotter
            .current_lyapunov_exponent()
            .expect("finite Lyapunov for active chaos model");
        assert!(
            lyap > 0.0,
            "Expected chaotic lambda_1 > 0 for {kind:?}, got {lyap}"
        );
        let (c_eps, d2) = plotter
            .current_correlation_invariants()
            .expect("finite correlation invariants");
        assert!((0.0..=1.0).contains(&c_eps) && (0.0..=3.0).contains(&d2));
    }

    let mut capped_bar = crate::framework::MathExpressionInputBar::new("δx₀", "epsilon", vec![])
        .with_max_instructions(3);
    capped_bar.set_expression("epsilon * (sin(x0) + cos(y0) + sin(z0))");
    assert!(!capped_bar.is_valid());
    let err = super::twin_state::evaluate_perturbation_bar(&capped_bar, [1.0, 1.0, 1.0], 1e-9)
        .expect_err("over-budget AST must be rejected");
    assert!(err.contains("Validation Error"));
}

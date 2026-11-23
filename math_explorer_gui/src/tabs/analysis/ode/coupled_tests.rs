#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::systems::CustomOdeSpec;
use super::*;
use crate::framework::InteractiveTool;
use math_explorer::pure_math::analysis::ode::{OdeModel, RungeKutta4, TimeStepper, VecState};

#[test]
fn test_coupled_ode_lotka_volterra_solvers_comparison_and_zero_alloc() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::Custom,
        custom_coupled: true,
        ic_y0: 2.0,
        ic_v0: 1.0,
        dt: 0.05,
        total_time: 4.0,
        ..Default::default()
    };
    tool.coupled_bank
        .set_expressions(&["a * y - b * y * v", "d * y * v - g * v"]);
    tool.coupled_bank.set_param("a", 1.1);
    tool.coupled_bank.set_param("b", 0.4);
    tool.coupled_bank.set_param("d", 0.1);
    tool.coupled_bank.set_param("g", 0.4);
    tool.set_comparison_mode(true);

    assert!(!tool.diverged());
    assert_eq!(tool.y_series.len(), tool.v_series.len());
    assert!(!tool.v_series.is_empty());

    // Comparison snapshots include both y(t) and v(t) across RK4, Euler, and Heun
    for snap in tool.comparison_manager().snapshots() {
        assert_eq!(snap.series.len(), 2);
        assert_eq!(snap.series[0].name, "y(t)");
        assert_eq!(snap.series[1].name, "v(t)");
    }
    assert!(tool.discrepancy_euler_rk4().max_error > tool.discrepancy_heun_rk4().max_error);

    // Transient syntax edit retains last valid CompiledExpr without FiniteStateGuard rollback
    let y_last_valid = *tool.y_series.last().expect("last y");
    tool.coupled_bank.set_expression(0, "a * y -");
    tool.recalculate();
    assert!(!tool.coupled_bank.is_valid());
    assert!(!tool.diverged());
    assert_eq!(*tool.y_series.last().expect("retained y"), y_last_valid);

    // Restore valid expression and verify zero heap allocations in RK4 step loop
    tool.coupled_bank.set_expression(0, "a * y - b * y * v");
    let spec = tool.build_custom_ode_spec().expect("coupled spec");
    let init = VecState(vec![2.0, 1.0]);
    let mut model = OdeModel::new(init.clone(), spec, RungeKutta4::new(&init));
    let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (out, _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        for _ in 0..64 {
            model.step(0.01);
        }
        model.state.0[0] + model.state.0[1]
    });
    assert!(out.is_finite());
}

#[test]
fn test_coupled_ode_persistence_and_url_roundtrip() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::Custom,
        custom_coupled: true,
        ic_y0: 1.5,
        ic_v0: -0.5,
        method: IntegrationMethod::Heun,
        comparison_mode: true,
        ..Default::default()
    };
    tool.coupled_bank.set_expressions(&["v", "-k * y - c * v"]);
    tool.coupled_bank.set_param("k", 2.25);
    tool.coupled_bank.set_param("c", 0.35);
    tool.recalculate();

    let json = tool.save_state().expect("save state");
    assert!(json.len() < 4096);
    let mut restored = OdeSolverTool::default();
    restored.try_load_state(&json).expect("load state");
    assert_eq!(restored.preset, OdePreset::Custom);
    assert!(restored.custom_coupled);
    assert_eq!(restored.coupled_bank.raw_text(1), "-k * y - c * v");
    assert_eq!(restored.coupled_bank.params_map().get("k"), Some(&2.25));
    assert_eq!(restored.y_series, tool.y_series);
    assert_eq!(restored.v_series, tool.v_series);

    // Unknown field must be rejected
    let bad = json.replacen('{', r#"{"bogus":1,"#, 1);
    assert!(restored.try_load_state(&bad).is_err());

    // CustomOdeSpec scalar fallback check
    let scalar_tool = OdeSolverTool::default();
    assert!(CustomOdeSpec::from_scalar_bar(&scalar_tool.math_input, 1.0, 1.0, 10.0).is_some());
}

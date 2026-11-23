#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::InteractiveTool;
use math_commons::ast::{parse, CompiledExpr};
use math_explorer::physics::quantum::evolve_state;
use std::collections::BTreeMap;

#[test]
fn test_wave_unified_dt_caching_and_trajectory_parity() {
    let mut params = HashMap::new();
    params.insert("potential_type".to_string(), 0.0);
    params.insert("dt".to_string(), 0.05);

    let mut sim = WaveUnified::new(&params);
    sim.n_points = 20;
    sim.init_system(0.0);
    assert_eq!(sim.cached_dt, None);
    assert_eq!(sim.scratch.len(), 20);

    let mut expected_psi = sim.psi.clone();
    for _ in 0..3 {
        sim.step(&params);
        expected_psi = evolve_state(&expected_psi, &sim.hamiltonian, 0.05, 1.0).normalize();
    }
    assert_eq!(sim.cached_dt, Some(0.05));
    let ptr_before = sim.propagator.matrix.as_ptr();
    sim.step(&params);
    expected_psi = evolve_state(&expected_psi, &sim.hamiltonian, 0.05, 1.0).normalize();
    assert_eq!(sim.propagator.matrix.as_ptr(), ptr_before);
    assert!((&sim.psi.vector - &expected_psi.vector).norm() < 1e-12);

    params.insert("dt".to_string(), 0.02);
    sim.step(&params);
    expected_psi = evolve_state(&expected_psi, &sim.hamiltonian, 0.02, 1.0).normalize();
    assert_eq!(sim.cached_dt, Some(0.02));
    assert!((&sim.psi.vector - &expected_psi.vector).norm() < 1e-12);
}

#[test]
fn test_wave_unified_reset_invalidates_propagator_on_potential_change() {
    let mut params = HashMap::new();
    params.insert("potential_type".to_string(), 0.0);
    params.insert("dt".to_string(), 0.04);

    let mut sim = WaveUnified::new(&params);
    sim.n_points = 20;
    sim.init_system(0.0);
    sim.step(&params);
    assert_eq!(sim.cached_dt, Some(0.04));
    let well_propagator = sim.propagator.matrix.clone();

    params.insert("potential_type".to_string(), 1.0);
    sim.process_command(SimCommand::Reset, &params);
    assert_eq!(sim.cached_dt, None);
    assert_eq!(sim.step_count, 0);
    assert_eq!(sim.time, 0.0);

    let init_psi = sim.psi.clone();
    sim.step(&params);
    assert_eq!(sim.cached_dt, Some(0.04));
    assert_ne!(sim.propagator.matrix, well_propagator);
    let expected = evolve_state(&init_psi, &sim.hamiltonian, 0.04, 1.0).normalize();
    assert!((&sim.psi.vector - &expected.vector).norm() < 1e-12);
}

#[test]
fn test_wave_unified_custom_vx_hamiltonian_and_tool_roundtrip() {
    let mut params = HashMap::new();
    params.insert("potential_type".to_string(), 0.0);
    params.insert("dt".to_string(), 0.03);

    let mut sim = WaveUnified::new(&params);
    sim.n_points = 24;
    sim.init_system(0.0);

    let expr = parse("0.1 * x^4 - a * x^2 + 3.0").unwrap();
    let compiled = Arc::new(CompiledExpr::compile_with_vars(&expr, &["x"]).unwrap());
    let mut param_buf = [0.0; 6];
    param_buf[0] = 1.2;
    let payload = CompiledFormulaPayload {
        revision: 2,
        compiled,
        params: param_buf,
        param_len: 1,
    };

    sim.apply_compiled_formula(Some(&payload), &params);
    assert_eq!(sim.cached_dt, None);

    let dx = (sim.x_max - sim.x_min) / (sim.n_points as f64 - 1.0);
    let expected_v = DVector::from_iterator(
        sim.n_points,
        sim.x_axis
            .iter()
            .map(|&x| 0.1 * x.powi(4) - 1.2 * x * x + 3.0),
    );
    let expected_h = construct_1d_hamiltonian(&expected_v, dx, 1.0, 1.0);
    assert!((&sim.hamiltonian.matrix - &expected_h.matrix).norm() < 1e-12);

    sim.step(&params);
    assert!(FiniteStateGuard::is_finite_state(&sim));
    assert!((sim.psi.vector.norm() - 1.0).abs() < 1e-10);
}

#[test]
fn test_wave_unified_custom_vx_and_brush_tool_roundtrip() {
    let mut tool = UnifiedSimTool::<WaveUnified>::new();
    tool.custom_formula_enabled = true;
    {
        let mut p = tool.params.write().unwrap();
        p.insert("potential_type".to_string(), 1.0);
        p.insert("dt".to_string(), 0.025);
    }
    let bar = tool.formula_bar.as_mut().expect("formula bar configured");
    bar.set_expression("b * cos(x)^2");
    let mut pmap = BTreeMap::new();
    pmap.insert("b".to_string(), 3.5);
    bar.set_params_map(&pmap);
    tool.publish_formula_update();

    let saved = tool.save_state().expect("wave tool saves custom V(x) state");
    assert!(saved.len() < crate::url_state::MAX_DECODED_URL_STATE_BYTES);

    let mut restored = UnifiedSimTool::<WaveUnified>::new();
    restored
        .try_load_state(&saved)
        .expect("wave tool loads custom V(x) state");
    assert!(restored.custom_formula_enabled);
    assert_eq!(
        *restored.params.read().unwrap().get("potential_type").unwrap(),
        1.0
    );
    let restored_bar = restored.formula_bar.as_ref().unwrap();
    assert_eq!(restored_bar.raw_text, "b * cos(x)^2");
    assert_eq!(restored_bar.params_map(), pmap);

    restored.dispatch_brush_stroke(
        BrushStroke {
            start: [0.0, 3.0],
            end: [0.25, 3.0],
            radius: 0.2,
            mode: BrushMode::PotentialBarrier,
            drag_vector: [0.25, 0.0],
        },
        None,
    );
    assert!(!restored.sculpt_state.read().unwrap().sparse_potential.is_empty());
    let combined_saved = restored.save_state().expect("combined wave state");
    assert!(combined_saved.len() < crate::url_state::MAX_DECODED_URL_STATE_BYTES);

    let mut combined_restored = UnifiedSimTool::<WaveUnified>::new();
    combined_restored
        .try_load_state(&combined_saved)
        .expect("restore combined wave state");
    assert!(combined_restored.custom_formula_enabled);
    assert_eq!(
        combined_restored.sculpt_state.read().unwrap().sparse_potential,
        restored.sculpt_state.read().unwrap().sparse_potential
    );
}

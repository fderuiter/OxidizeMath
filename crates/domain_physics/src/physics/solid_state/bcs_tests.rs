use super::*;

#[test]
#[verified_engine::verified]
fn test_bcs_probability_conservation() {
    let xi = ElectronVolts::new(1.5);
    let delta = ElectronVolts::new(0.2);
    let (u, v) = coherence_factors_strong(xi, delta).unwrap();
    let prob = u * u + v * v;
    assert!((prob - 1.0).abs() < math_commons::registry::TOLERANCE_STANDARD);
}

#[test]
#[verified_engine::verified]
fn test_solver_convergence() {
    let energies: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) * 0.1).collect();
    let potential = 0.5;
    let debye = 5.0;

    let model = IsotropicBCSModel::new(energies.clone(), potential, debye).unwrap();
    let solver = BcsGapSolver::default();
    let initial = ElectronVolts::new(0.05);

    let result = solver.solve(&model, initial);
    assert!(result.is_ok());
    let delta = result.unwrap();
    assert!((delta / ElectronVolts::new(1.0)) > 0.0);
}

#[test]
#[verified_engine::verified]
fn test_bcs_gap_solver_direct() {
    let energies: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) * 0.1).collect();
    let potential = 0.5;
    let debye = 5.0;

    let model = IsotropicBCSModel::new(energies, potential, debye).unwrap();
    let solver = BcsGapSolver::default()
        .with_max_iterations(1000)
        .with_tolerance(math_commons::registry::TOLERANCE_STANDARD)
        .with_mixing_param(0.5);

    let initial_guess = ElectronVolts::new(0.01 * debye);
    let res = solver.solve(&model, initial_guess);

    let delta = match res {
        Ok(delta) => delta,
        Err(SolidStateError::ConvergenceFailure(_, last_val)) => last_val,
        Err(e) => panic!("Unexpected solver error: {:?}", e),
    };
    assert!((delta / ElectronVolts::new(1.0)) > 0.0);
}

#[test]
#[verified_engine::verified]
fn test_weak_coupling_universal_ratio_and_tc_closure() {
    // Symmetric midpoint band across [-hw_D, +hw_D] with dimensionless coupling lambda = N(0)*V = 0.25
    let n_levels = 800usize;
    let hw_d = 0.05_f64;
    let d_xi = (2.0 * hw_d) / (n_levels as f64);
    let energies: Vec<f64> = (0..n_levels)
        .map(|i| -hw_d + ((i as f64) + 0.5) * d_xi)
        .collect();
    let lambda = 0.25_f64;
    let potential = lambda * d_xi;

    let model_t0 = IsotropicBCSModel::new(energies, potential, hw_d).unwrap();
    let tc = model_t0.critical_temperature().unwrap();
    assert!(tc.as_f64() > 1.0);

    let solver = BcsGapSolver::new()
        .with_max_iterations(2_000)
        .with_tolerance(1e-11)
        .with_mixing_param(0.8);
    let delta_0 = solver.solve(&model_t0, ElectronVolts::new(0.001)).unwrap();

    // Weak-coupling universal BCS ratio: 2 * Delta(0) / (k_B * T_c) ≈ 3.5278
    let ratio = (2.0 * delta_0.as_f64()) / tc.thermal_energy_ev().as_f64();
    assert!(
        (ratio - 3.528).abs() < 0.02,
        "Expected weak-coupling BCS ratio ~3.528, got {ratio:.5}"
    );

    // Monotonic thermal suppression below T_c and exact 0.0 eV closure for all T >= T_c
    let model_half = model_t0
        .clone()
        .with_temperature(Kelvin::new(0.5 * tc.as_f64()));
    let delta_half = solver.solve(&model_half, delta_0).unwrap();
    assert!(delta_half.as_f64() > 0.0 && delta_half.as_f64() < delta_0.as_f64());

    for factor in [1.0, 1.05, 1.5] {
        let model_above = model_t0
            .clone()
            .with_temperature(Kelvin::new(factor * tc.as_f64()));
        let delta_above = solver
            .solve(&model_above, ElectronVolts::new(0.002))
            .unwrap();
        assert_eq!(
            delta_above.as_f64(),
            0.0,
            "Gap must vanish identically at T = {factor} * T_c"
        );
    }
}

#[test]
#[verified_engine::verified]
fn test_zero_seed_recovery_below_tc_and_normal_state_stability() {
    let n_levels = 200usize;
    let hw_d = 0.04_f64;
    let d_xi = (2.0 * hw_d) / (n_levels as f64);
    let energies: Vec<f64> = (0..n_levels)
        .map(|i| -hw_d + ((i as f64) + 0.5) * d_xi)
        .collect();
    let model = IsotropicBCSModel::new(energies, 0.30 * d_xi, hw_d).unwrap();
    let tc = model.critical_temperature().unwrap();
    let solver = BcsGapSolver::new()
        .with_max_iterations(2_000)
        .with_tolerance(1e-11)
        .with_mixing_param(0.8);

    let from_positive = solver.solve(&model, ElectronVolts::new(0.005)).unwrap();
    let from_zero = solver.solve(&model, ElectronVolts::new(0.0)).unwrap();
    assert!(from_zero.as_f64() > 0.0);
    assert!((from_zero.as_f64() - from_positive.as_f64()).abs() < 1e-9);

    let sub_tc = model
        .clone()
        .with_temperature(Kelvin::new(0.6 * tc.as_f64()));
    let sub_from_zero = solver.solve(&sub_tc, ElectronVolts::new(0.0)).unwrap();
    let sub_from_pos = solver.solve(&sub_tc, ElectronVolts::new(0.005)).unwrap();
    assert!((sub_from_zero.as_f64() - sub_from_pos.as_f64()).abs() < 1e-9);

    let super_tc = model.with_temperature(Kelvin::new(1.2 * tc.as_f64()));
    let super_from_zero = solver.solve(&super_tc, ElectronVolts::new(0.0)).unwrap();
    assert_eq!(super_from_zero.as_f64(), 0.0);
}

#[test]
#[verified_engine::verified]
fn test_normal_state_coherence_limits_and_dynes_nis_conductance() {
    let zero = ElectronVolts::new(0.0);
    let (u_fermi, v_fermi) = coherence_factors_normal_limit(zero, zero).unwrap();
    assert!((u_fermi * u_fermi - 0.5).abs() < 1e-14);
    assert!((v_fermi * v_fermi - 0.5).abs() < 1e-14);

    let (u_hole, v_hole) = coherence_factors_normal_limit(ElectronVolts::new(-0.02), zero).unwrap();
    assert!((u_hole - 0.0).abs() < 1e-14 && (v_hole - 1.0).abs() < 1e-14);

    let (u_elec, v_elec) = coherence_factors_normal_limit(ElectronVolts::new(0.02), zero).unwrap();
    assert!((u_elec - 1.0).abs() < 1e-14 && (v_elec - 0.0).abs() < 1e-14);

    // Dynes DOS & NIS conductance invariants
    let delta = ElectronVolts::new(0.0015);
    let gamma = ElectronVolts::new(0.000_15);
    let dos_zero_bias = dynes_dos_ratio(zero, delta, gamma).unwrap();
    let expected_zbc = gamma.as_f64() / gamma.as_f64().hypot(delta.as_f64());
    assert!((dos_zero_bias - expected_zbc).abs() < 1e-13);

    let peak_g = nis_tunneling_conductance(delta, delta, gamma, Kelvin::new(0.5)).unwrap();
    assert!(
        peak_g > 1.5,
        "Coherence peak near eV = Delta must exceed normal-state 1.0"
    );

    let normal_g =
        nis_tunneling_conductance(ElectronVolts::new(0.002), zero, gamma, Kelvin::new(15.0))
            .unwrap();
    assert!((normal_g - 1.0).abs() < 1e-14);
}

#[test]
fn test_zero_heap_allocation_in_bcs_and_nis_hot_loops() {
    let n_levels = 120usize;
    let hw_d = 0.04_f64;
    let d_xi = (2.0 * hw_d) / (n_levels as f64);
    let energies: Vec<f64> = (0..n_levels)
        .map(|i| -hw_d + ((i as f64) + 0.5) * d_xi)
        .collect();
    let model = IsotropicBCSModel::new(energies, 0.30 * d_xi, hw_d)
        .unwrap()
        .with_temperature(Kelvin::new(4.2));
    let solver = BcsGapSolver::new()
        .with_max_iterations(500)
        .with_mixing_param(0.8);

    verified_engine::engine::VerifiedEngine::run_verified(|| {
        let tc = model.critical_temperature().unwrap();
        let gap = solver.solve(&model, ElectronVolts::new(0.0)).unwrap();
        let (u, v) = coherence_factors_normal_limit(ElectronVolts::new(0.0), gap).unwrap();
        let g = nis_tunneling_conductance(
            gap,
            gap,
            ElectronVolts::new(0.0002),
            Kelvin::new(0.2 * tc.as_f64()),
        )
        .unwrap();
        assert!(gap.as_f64() >= 0.0 && (u * u + v * v - 1.0).abs() < 1e-12 && g > 0.0);
    });
}

// [cite:mmwave_radiotherapy_setup]

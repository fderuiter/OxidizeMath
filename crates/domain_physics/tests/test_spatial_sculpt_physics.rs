//! Integration tests for 2D Lattice Boltzmann aerodynamics (NACA 4-digit airfoils,
//! Venturi nozzles, vorticity wake shedding, $C_L$/$C_D$ diagnostics) and 1D quantum
//! potential sculpting (double-barrier resonant tunneling, unitarity, zero heap allocations).

use domain_physics::physics::fluid_dynamics::lattice_boltzmann::{
    BgkCollision, LatticeBoltzmannD2Q9, Naca4Airfoil, advance_wind_tunnel_step,
    compute_aerodynamic_coefficients, compute_vorticity_into, inject_velocity_jet,
    stamp_naca4_airfoil, stamp_venturi_nozzle,
};
use domain_physics::physics::quantum::{
    CrankNicolson1D, HydrogenicSuperposition3D, OrbitalQuantumNumbers, QuantumFieldObservable,
    QuantumOperator, QuantumState, construct_1d_hamiltonian, construct_1d_hamiltonian_into,
    evolve_state, evolve_state_in_place, gaussian_wavepacket, gaussian_wavepacket_into, sigma_x,
    sigma_y, sigma_z, state_norm_squared, time_evolution_operator,
    time_evolution_operator_2x2_into, transmission_probability,
};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use std::sync::Mutex;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

static VERIFIED_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn test_naca4_airfoil_stall_vorticity_and_aerodynamic_coefficients() {
    let w = 100;
    let h = 50;
    let u_in = 0.08;
    let chord = 24.0;

    let mut lbm_zero_aoa = LatticeBoltzmannD2Q9::<BgkCollision>::new(w, h, 0.62);
    lbm_zero_aoa.set_inlet(0, 0, w, h, u_in, 0.0);
    let symmetric = Naca4Airfoil::from_code(12, chord, 28.0, 25.0, 0.0);
    stamp_naca4_airfoil(&mut lbm_zero_aoa, &symmetric);
    let solid_zero = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .filter(|&(x, y)| lbm_zero_aoa.is_obstacle(x, y))
        .count();
    assert!(solid_zero > 20, "NACA 0012 must stamp solid cells");

    let mut lbm_stall = LatticeBoltzmannD2Q9::<BgkCollision>::new(w, h, 0.62);
    lbm_stall.set_inlet(0, 0, w, h, u_in, 0.0);
    let cambered = Naca4Airfoil::from_code(2412, chord, 28.0, 25.0, 18.0);
    stamp_naca4_airfoil(&mut lbm_stall, &cambered);
    let solid_stall = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .filter(|&(x, y)| lbm_stall.is_obstacle(x, y))
        .count();
    assert!(solid_stall > 20, "NACA 2412 must stamp solid cells");

    for _ in 0..80 {
        advance_wind_tunnel_step(&mut lbm_zero_aoa, u_in);
        advance_wind_tunnel_step(&mut lbm_stall, u_in);
    }

    let mut vort_stall = vec![0.0; w * h];
    compute_vorticity_into(&lbm_stall, &mut vort_stall);
    let max_pos_vort = vort_stall.iter().copied().fold(0.0, f64::max);
    let min_neg_vort = vort_stall.iter().copied().fold(0.0, f64::min);
    assert!(
        max_pos_vort > 1e-3 && min_neg_vort < -1e-3,
        "High-AoA NACA 2412 wake must shed both positive ({max_pos_vort}) and negative ({min_neg_vort}) signed vorticity"
    );

    let aero_zero = compute_aerodynamic_coefficients(&lbm_zero_aoa, u_in, chord, 3);
    let aero_stall = compute_aerodynamic_coefficients(&lbm_stall, u_in, chord, 3);
    assert!(aero_zero.drag_coeff.is_finite() && aero_zero.lift_coeff.is_finite());
    assert!(aero_stall.drag_coeff.is_finite() && aero_stall.lift_coeff.is_finite());
    assert!(
        aero_zero.lift_coeff.abs() < 0.08,
        "Symmetric NACA 0012 at 0 deg AoA should have near-zero lift, got {}",
        aero_zero.lift_coeff
    );
    assert!(
        aero_stall.drag_coeff > aero_zero.drag_coeff,
        "18 deg stall AoA must increase drag ({} vs {})",
        aero_stall.drag_coeff,
        aero_zero.drag_coeff
    );
    assert!(
        (aero_stall.lift_coeff - aero_zero.lift_coeff).abs() > 0.02,
        "18 deg AoA must alter lift coefficient ({} vs {})",
        aero_stall.lift_coeff,
        aero_zero.lift_coeff
    );
}

#[test]
fn test_venturi_nozzle_throat_acceleration_and_velocity_jet() {
    let w = 80;
    let h = 40;
    let u_in = 0.06;
    let mut lbm = LatticeBoltzmannD2Q9::<BgkCollision>::new(w, h, 0.65);
    lbm.set_inlet(0, 0, w, h, u_in, 0.0);
    stamp_venturi_nozzle(&mut lbm, 0.45);
    let walls = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .filter(|&(x, y)| lbm.is_obstacle(x, y))
        .count();
    assert!(walls > 100);

    for _ in 0..80 {
        advance_wind_tunnel_step(&mut lbm, u_in);
    }

    let inlet_ux = lbm.get_velocity(4, h / 2).0;
    let throat_x = (0.45 * w as f64) as usize;
    let throat_ux = lbm.get_velocity(throat_x, h / 2).0;
    assert!(
        throat_ux > inlet_ux * 1.05,
        "Venturi constriction must accelerate centerline u_x ({throat_ux} vs {inlet_ux})"
    );

    inject_velocity_jet(&mut lbm, 15, 20, 2, 0.14, 0.02);
    let jet_ux = lbm.get_velocity(15, 20).0;
    assert!((jet_ux - 0.14).abs() < 1e-2);
}

fn extreme_hill_well_potential(x_axis: &[f64]) -> DVector<f64> {
    let mut pot = DVector::from_element(x_axis.len(), 0.0_f64);
    for (i, &x) in x_axis.iter().enumerate() {
        if (-0.45..=-0.10).contains(&x) {
            pot[i] = 450.0;
        } else if (0.25..=0.65).contains(&x) {
            pot[i] = -420.0;
        }
    }
    pot
}

#[test]
fn test_double_barrier_quantum_sculpt_unitarity_transmission_and_zero_allocations() {
    let (n, x_min, x_max, dt) = (100, -5.0, 5.0, 0.025);
    let dx = (x_max - x_min) / (n as f64 - 1.0);
    let x_axis: Vec<f64> = (0..n).map(|i| x_min + i as f64 * dx).collect();

    let mut potential = DVector::from_element(n, 0.0_f64);
    for (i, &x) in x_axis.iter().enumerate() {
        if (-0.65..=-0.30).contains(&x) || (0.30..=0.65).contains(&x) {
            potential[i] = 6.0;
        }
    }

    let mut hamiltonian = construct_1d_hamiltonian(&potential, dx, 1.0, 1.0);
    let propagator = time_evolution_operator(&hamiltonian, dt, 1.0);
    let frob_err = (propagator.matrix.adjoint() * &propagator.matrix
        - DMatrix::<Complex<f64>>::identity(n, n))
    .norm();
    assert!(
        frob_err < 1e-6,
        "Propagator U(dt) must be unitary within 1e-6, got {frob_err}"
    );

    let mut psi = gaussian_wavepacket(&x_axis, -2.2, 4.2, 0.45);
    let mut scratch = DVector::from_element(n, Complex::new(0.0, 0.0));
    let split_idx = x_axis.partition_point(|&x| x <= 0.70);

    let mut lbm = LatticeBoltzmannD2Q9::<BgkCollision>::new(48, 24, 0.62);
    lbm.set_inlet(0, 0, 48, 24, 0.07, 0.0);
    stamp_naca4_airfoil(
        &mut lbm,
        &Naca4Airfoil::from_code(2412, 12.0, 15.0, 12.0, 8.0),
    );
    let mut vort = vec![0.0; 48 * 24];

    let extreme_pot = extreme_hill_well_potential(&x_axis);
    let mut extreme_ham = construct_1d_hamiltonian(&extreme_pot, dx, 1.0, 1.0);
    let extreme_prop = time_evolution_operator(&extreme_ham, dt, 1.0);
    let mut extreme_psi = gaussian_wavepacket(&x_axis, -2.2, 4.2, 0.45);

    let ((norm_sq, trans_prob, extreme_norm, aero), metrics) = VerifiedEngine::run_verified(|| {
        construct_1d_hamiltonian_into(&potential, dx, 1.0, 1.0, &mut hamiltonian);
        construct_1d_hamiltonian_into(&extreme_pot, dx, 1.0, 1.0, &mut extreme_ham);
        for _ in 0..90 {
            evolve_state_in_place(&mut psi, &propagator, &mut scratch);
            evolve_state_in_place(&mut extreme_psi, &extreme_prop, &mut scratch);
        }
        let norm = state_norm_squared(&psi);
        let ext_norm = state_norm_squared(&extreme_psi);
        let trans = transmission_probability(&psi, split_idx);

        advance_wind_tunnel_step(&mut lbm, 0.07);
        compute_vorticity_into(&lbm, &mut vort);
        let aero_coeffs = compute_aerodynamic_coefficients(&lbm, 0.07, 12.0, 2);
        (norm, trans, ext_norm, aero_coeffs)
    });

    assert!(
        metrics.function_calls > 0,
        "VerifiedEngine::run_verified enforces 0 heap allocations and records verified calls"
    );
    assert!(
        (norm_sq - 1.0).abs() < 1e-6 && (extreme_norm - 1.0).abs() < 1e-6,
        "Wavefunction norms ||psi||^2 must stay unitary within 1e-6 across [-500, 500] potentials ({norm_sq}, {extreme_norm})"
    );
    assert!(
        trans_prob > 1e-4 && trans_prob < 0.999,
        "Double-barrier partial tunneling transmission T must lie in (0, 1), got {trans_prob}"
    );
    assert!(aero.drag_coeff.is_finite() && aero.lift_coeff.is_finite());
}

#[test]
fn test_crank_nicolson_1d_tunneling_parity_sculpt_and_zero_allocations() {
    let (n, x_min, x_max, dt) = (100, -5.0, 5.0, 0.005);
    let dx = (x_max - x_min) / (n as f64 - 1.0);
    let x_axis: Vec<f64> = (0..n).map(|i| x_min + i as f64 * dx).collect();
    let potential: Vec<f64> = x_axis
        .iter()
        .map(|&x| {
            if (-0.65..=-0.30).contains(&x) || (0.30..=0.65).contains(&x) {
                6.0
            } else {
                0.0
            }
        })
        .collect();
    let split_idx = x_axis.partition_point(|&x| x <= 0.70);

    let pot_vec = DVector::from_vec(potential.clone());
    let dense_h = construct_1d_hamiltonian(&pot_vec, dx, 1.0, 1.0);
    let psi0_dense = gaussian_wavepacket(&x_axis, -2.2, 4.2, 0.45);
    let psi_dense = evolve_state(&psi0_dense, &dense_h, 80.0 * dt, 1.0);

    let mut solver = CrankNicolson1D::new(&potential, dx, dt, 1.0, 1.0);
    assert!(solver.is_valid());
    let mut psi_cn = QuantumState::new(DVector::from_element(n, Complex::new(0.0, 0.0)));
    let mut density = vec![0.0; n];
    let sculpted: Vec<f64> = potential
        .iter()
        .enumerate()
        .map(|(i, &v)| if i == 50 { 25.0 } else { v })
        .collect();

    let ((norm_cn, trans_cn, parity_err, norm_sculpt), metrics) = {
        let _guard = VERIFIED_LOCK.lock().expect("verified lock");
        VerifiedEngine::run_verified(|| {
            gaussian_wavepacket_into(&x_axis, -2.2, 4.2, 0.45, &mut psi_cn);
            for _ in 0..80 {
                assert!(solver.step_in_place(&mut psi_cn));
            }
            let norm = state_norm_squared(&psi_cn);
            let trans = transmission_probability(&psi_cn, split_idx);
            let l2_err = psi_cn
                .vector
                .as_slice()
                .iter()
                .zip(psi_dense.vector.as_slice().iter())
                .map(|(a, b)| (*a - *b).norm_sqr())
                .sum::<f64>()
                .sqrt();
            psi_cn.probability_density_in_place(&mut density);
            assert!(solver.update_potential(&sculpted, dx, 1.0, 1.0));
            assert!(solver.update_dt(0.008, 1.0));
            assert!(solver.step_in_place(&mut psi_cn));
            (norm, trans, l2_err, state_norm_squared(&psi_cn))
        })
    };

    assert!(metrics.function_calls > 0);
    assert!((norm_cn - 1.0).abs() < 1e-11 && (norm_sculpt - 1.0).abs() < 1e-11);
    let trans_dense = transmission_probability(&psi_dense, split_idx);
    assert!(
        (trans_cn - trans_dense).abs() < 5e-3,
        "CN vs spectral T: {trans_cn} vs {trans_dense}"
    );
    assert!(parity_err < 2e-2, "CN vs spectral L2 error: {parity_err}");
    assert!((density.iter().sum::<f64>() - 1.0).abs() < 1e-11);
}

#[test]
fn test_crank_nicolson_1d_max_grid_latency_and_zero_allocations() {
    let n = math_commons::registry::MAX_GRID_SIZE;
    let dx = 20.0 / (n as f64 - 1.0);
    let x_axis: Vec<f64> = (0..n).map(|i| -10.0 + i as f64 * dx).collect();
    let mut potential: Vec<f64> = x_axis.iter().map(|&x| 0.25 * x * x).collect();
    let mut solver = CrankNicolson1D::new(&potential, dx, 0.01, 1.0, 1.0);
    assert!(solver.is_valid());
    let mut psi = gaussian_wavepacket(&x_axis, -2.5, 5.0, 0.6);
    potential[n / 2] = 120.0;

    let (final_norm, metrics) = {
        let _guard = VERIFIED_LOCK.lock().expect("verified lock");
        VerifiedEngine::run_verified(|| {
            assert!(solver.update_potential(&potential, dx, 1.0, 1.0));
            assert!(solver.update_dt(0.012, 1.0));
            for _ in 0..50 {
                assert!(solver.step_in_place(&mut psi));
            }
            state_norm_squared(&psi)
        })
    };

    assert!(metrics.function_calls > 0);
    assert!((final_norm - 1.0).abs() < 1e-10);
}

#[test]
fn test_closed_form_2x2_and_hydrogenic_superposition_zero_allocations() {
    let custom_h = QuantumOperator::new(DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex::new(1.4, 0.0),
            Complex::new(0.6, -0.8),
            Complex::new(0.6, 0.8),
            Complex::new(-0.9, 0.0),
        ],
    ));
    let operators = [sigma_x(), sigma_y(), sigma_z(), custom_h];
    let refs: Vec<QuantumOperator> = operators
        .iter()
        .map(|op| time_evolution_operator(op, 0.37, 1.1))
        .collect();
    let mut out = QuantumOperator::new(DMatrix::zeros(2, 2));

    let mut studio = HydrogenicSuperposition3D::new(
        OrbitalQuantumNumbers::new(1, 0, 0).expect("1s"),
        OrbitalQuantumNumbers::new(2, 1, 0).expect("2pz"),
        8,
        8,
        12.0,
        0.05,
    );
    let mut grid = vec![0.0_f32; 8 * 8 * 8];

    let n2s = OrbitalQuantumNumbers::new(2, 0, 0).expect("2s");
    let n3d = OrbitalQuantumNumbers::new(3, 2, 1).expect("3d");

    let (max_err, metrics) = {
        let _guard = VERIFIED_LOCK.lock().expect("verified lock");
        VerifiedEngine::run_verified(|| {
            let mut worst = 0.0_f64;
            for (op, expected) in operators.iter().zip(refs.iter()) {
                assert!(time_evolution_operator_2x2_into(op, 0.37, 1.1, &mut out));
                let err_sq: f64 = out
                    .matrix
                    .as_slice()
                    .iter()
                    .zip(expected.matrix.as_slice().iter())
                    .map(|(a, b)| (*a - *b).norm_sqr())
                    .sum();
                worst = worst.max(err_sq.sqrt());
            }
            studio.reconfigure_basis(n2s, n3d, 8, 12.0, 1.0);
            studio.set_superposition(0.35, 0.8);
            studio.rebuild_propagator(0.08, 0.02);
            studio.step_in_place();
            studio.write_field_into(QuantumFieldObservable::ProbabilityDensity, &mut grid);
            worst
        })
    };

    assert!(metrics.function_calls > 0);
    assert!(
        max_err < 1e-13,
        "Closed-form 2x2 propagator mismatch: {max_err}"
    );
    assert!(grid.iter().any(|&v| v > 0.0) && grid.iter().all(|v| v.is_finite()));
}

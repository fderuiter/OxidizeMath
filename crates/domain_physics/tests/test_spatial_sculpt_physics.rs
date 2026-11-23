//! Integration tests for 2D Lattice Boltzmann aerodynamics (NACA 4-digit airfoils,
//! Venturi nozzles, vorticity wake shedding, $C_L$/$C_D$ diagnostics) and 1D quantum
//! potential sculpting (double-barrier resonant tunneling, unitarity, zero heap allocations).

use domain_physics::physics::fluid_dynamics::lattice_boltzmann::{
    BgkCollision, LatticeBoltzmannD2Q9, Naca4Airfoil, advance_wind_tunnel_step,
    compute_aerodynamic_coefficients, compute_vorticity_into, inject_velocity_jet,
    stamp_naca4_airfoil, stamp_venturi_nozzle,
};
use domain_physics::physics::quantum::{
    construct_1d_hamiltonian, construct_1d_hamiltonian_into, evolve_state_in_place,
    gaussian_wavepacket, state_norm_squared, time_evolution_operator, transmission_probability,
};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

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
    let max_vort = vort_stall.iter().copied().map(f64::abs).fold(0.0, f64::max);
    assert!(
        max_vort > 1e-3,
        "High-AoA NACA 2412 wake must shed non-trivial vorticity, got {max_vort}"
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

#[test]
fn test_double_barrier_quantum_sculpt_unitarity_transmission_and_zero_allocations() {
    let n = 100;
    let x_min = -5.0;
    let x_max = 5.0;
    let dx = (x_max - x_min) / (n as f64 - 1.0);
    let x_axis: Vec<f64> = (0..n).map(|i| x_min + i as f64 * dx).collect();

    let mut potential = DVector::from_element(n, 0.0_f64);
    for (i, &x) in x_axis.iter().enumerate() {
        if (-0.65..=-0.30).contains(&x) || (0.30..=0.65).contains(&x) {
            potential[i] = 6.0;
        }
    }

    let mut hamiltonian = construct_1d_hamiltonian(&potential, dx, 1.0, 1.0);
    let dt = 0.025;
    let propagator = time_evolution_operator(&hamiltonian, dt, 1.0);

    // Verify matrix unitarity ||U^H U - I||_F < 1e-6
    let u_dag_u = propagator.matrix.adjoint() * &propagator.matrix;
    let identity = DMatrix::<Complex<f64>>::identity(n, n);
    let frob_err = (u_dag_u - identity).norm();
    assert!(
        frob_err < 1e-6,
        "Propagator U(dt) must be unitary within 1e-6, got {frob_err}"
    );

    let mut psi = gaussian_wavepacket(&x_axis, -2.2, 4.2, 0.45);
    let mut scratch = DVector::from_element(n, Complex::new(0.0, 0.0));
    let split_idx = x_axis.partition_point(|&x| x <= 0.70);

    let mut lbm = LatticeBoltzmannD2Q9::<BgkCollision>::new(48, 24, 0.62);
    lbm.set_inlet(0, 0, 48, 24, 0.07, 0.0);
    let airfoil = Naca4Airfoil::from_code(2412, 12.0, 15.0, 12.0, 8.0);
    stamp_naca4_airfoil(&mut lbm, &airfoil);
    let mut vort = vec![0.0; 48 * 24];

    let ((norm_sq, trans_prob, aero), metrics) = VerifiedEngine::run_verified(|| {
        construct_1d_hamiltonian_into(&potential, dx, 1.0, 1.0, &mut hamiltonian);
        for _ in 0..90 {
            evolve_state_in_place(&mut psi, &propagator, &mut scratch);
        }
        let norm = state_norm_squared(&psi);
        let trans = transmission_probability(&psi, split_idx);

        advance_wind_tunnel_step(&mut lbm, 0.07);
        compute_vorticity_into(&lbm, &mut vort);
        let aero_coeffs = compute_aerodynamic_coefficients(&lbm, 0.07, 12.0, 2);
        (norm, trans, aero_coeffs)
    });

    assert!(
        metrics.function_calls > 0,
        "VerifiedEngine::run_verified enforces 0 heap allocations and records verified calls"
    );
    assert!(
        (norm_sq - 1.0).abs() < 1e-6,
        "Wavefunction norm ||psi||^2 must stay unitary within 1e-6, got {norm_sq}"
    );
    assert!(
        trans_prob > 1e-4 && trans_prob < 0.999,
        "Double-barrier partial tunneling transmission T must lie in (0, 1), got {trans_prob}"
    );
    assert!(aero.drag_coeff.is_finite() && aero.lift_coeff.is_finite());
}

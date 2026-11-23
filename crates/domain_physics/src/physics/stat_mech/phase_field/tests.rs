#![cfg_attr(any(), verified(opt_out = "tests"))]

use oxidize_core::rng::OxidizeRng;
use verified_engine::engine::VerifiedEngine;

use super::{PhaseFieldModel, PhaseFieldParams, PhaseFieldSolver2D};

#[test]
fn cahn_hilliard_conserves_mean_concentration_and_monotonically_decreases_free_energy_at_zero_noise()
 {
    let mut rng = OxidizeRng::from_seed(0xB0DE_1B00);
    let params_det = PhaseFieldParams {
        model: PhaseFieldModel::CahnHilliard,
        temperature: 0.2,
        critical_temperature: 1.0,
        mobility: 1.0,
        kappa: 0.6,
        noise_sigma: 0.0,
        dx: 1.0,
        dt: 0.08,
    };

    let target_mean = 0.12;
    let mut solver =
        PhaseFieldSolver2D::new_spinodal(48, 48, target_mean, 0.15, &mut rng, params_det);
    let initial_mean = solver.mean_concentration();
    assert!((initial_mean - target_mean).abs() < 1e-13);

    let mut prev_fe = solver.free_energy();
    let early_summary = solver.refresh_telemetry();

    for _ in 0..40 {
        solver.step(&mut rng);
        let curr_fe = solver.free_energy();
        assert!(
            curr_fe < prev_fe,
            "Deterministic Cahn-Hilliard free energy must strictly decrease: {prev_fe} -> {curr_fe}"
        );
        assert!(
            (solver.mean_concentration() - initial_mean).abs() < 1e-12,
            "Deterministic Model B violated mass conservation"
        );
        prev_fe = curr_fe;
    }

    let late_summary = solver.refresh_telemetry();
    assert!(
        late_summary.k_first_moment < early_summary.k_first_moment,
        "Spinodal coarsening must shift characteristic wavenumber k_1(t) downward: {} -> {}",
        early_summary.k_first_moment,
        late_summary.k_first_moment
    );

    // Enable conserved thermal noise and verify <phi> stays conserved within 1e-12.
    let mut noisy_params = params_det;
    noisy_params.noise_sigma = 0.08;
    solver.set_params(noisy_params);
    solver.step_n(&mut rng, 40);
    assert!(
        (solver.mean_concentration() - initial_mean).abs() < 1e-12,
        "Stochastic Model B conserved-flux noise drifted mean concentration: {} vs {}",
        solver.mean_concentration(),
        initial_mean
    );
}

#[test]
fn allen_cahn_relaxes_free_energy_without_conserving_non_zero_mean() {
    let mut rng = OxidizeRng::from_seed(0xA11E_4CA4);
    let params = PhaseFieldParams {
        model: PhaseFieldModel::AllenCahn,
        temperature: 0.2,
        critical_temperature: 1.0,
        mobility: 1.0,
        kappa: 0.5,
        noise_sigma: 0.0,
        dx: 1.0,
        dt: 0.08,
    };
    let mut solver = PhaseFieldSolver2D::new_spinodal(32, 32, 0.25, 0.1, &mut rng, params);
    let f0 = solver.free_energy();
    let m0 = solver.mean_concentration();
    for _ in 0..20 {
        solver.step(&mut rng);
    }
    assert!(solver.free_energy() < f0);
    // Model A is non-conserved: positive initial bias grows toward the +sqrt(Tc - T) well.
    assert!(solver.mean_concentration() > m0 + 0.1);
}

#[test]
fn allen_cahn_and_cahn_hilliard_steps_perform_zero_heap_allocations() {
    let mut rng = OxidizeRng::from_seed(0x5FDE_0A11);
    let ac_params = PhaseFieldParams {
        model: PhaseFieldModel::AllenCahn,
        noise_sigma: 0.05,
        ..PhaseFieldParams::default()
    };
    let ch_params = PhaseFieldParams {
        model: PhaseFieldModel::CahnHilliard,
        noise_sigma: 0.05,
        ..PhaseFieldParams::default()
    };
    let mut ac = PhaseFieldSolver2D::new_spinodal(32, 32, 0.0, 0.1, &mut rng, ac_params);
    let mut ch = PhaseFieldSolver2D::new_spinodal(32, 32, 0.05, 0.1, &mut rng, ch_params);

    VerifiedEngine::run_verified(|| {
        ac.paint_droplet(16, 16, 4, 0.35, false);
        ac.step_n(&mut rng, 2);
        let ac_sum = ac.refresh_telemetry();

        ch.paint_droplet(10, 10, 4, 0.35, true);
        ch.step_n(&mut rng, 2);
        let ch_sum = ch.refresh_telemetry();

        assert!(ac_sum.num_bins >= 2 && ch_sum.num_bins >= 2);
    });
}

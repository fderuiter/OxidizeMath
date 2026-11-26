//! VerifiedAllocator lock tests confirming zero dynamic heap allocations across
//! finite-temperature BCS gap solving, Bogoliubov coherence, Dynes NIS conductance,
//! CODATA SI `ElectronVolts`/`Kelvin` quantum occupancy, and canonical ensembles.

use domain_physics::physics::solid_state::bcs::{
    BcsGapSolver, IsotropicBCSModel, coherence_factors_normal_limit, dynes_dos_ratio,
    nis_tunneling_conductance, quasiparticle_energy,
};
use domain_physics::physics::solid_state::types::{ElectronVolts, Kelvin};
use domain_physics::physics::stat_mech::ensembles::canonical_thermodynamics_ev;
use domain_physics::physics::stat_mech::quantum_stats::{
    BoseEinstein, FermiDirac, MaxwellBoltzmann, calculate_occupancy_ev,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_bcs_and_quantum_stats_zero_heap_allocations_under_verified_allocator() {
    let n_levels = 160usize;
    let hw_d = 0.04_f64;
    let d_xi = (2.0 * hw_d) / (n_levels as f64);
    let energies: Vec<f64> = (0..n_levels)
        .map(|i| -hw_d + ((i as f64) + 0.5) * d_xi)
        .collect();
    let base_model = IsotropicBCSModel::new(energies, 0.30 * d_xi, hw_d)
        .unwrap()
        .with_temperature(Kelvin::new(4.2));
    let solver = BcsGapSolver::new()
        .with_max_iterations(400)
        .with_tolerance(1e-11)
        .with_mixing_param(0.8);

    let mut spectrum = [ElectronVolts::new(0.0); 16];
    for (idx, slot) in spectrum.iter_mut().enumerate() {
        *slot = ElectronVolts::new(0.05 + (idx as f64) * 0.015);
    }

    let ((gap_ev, nis_peak, fd_sum, be_sum, mb_sum, z_canon), _metrics) =
        VerifiedEngine::run_verified(|| {
            let tc = base_model.critical_temperature().unwrap();
            let k_zero = base_model.pairing_kernel(ElectronVolts::new(0.0)).unwrap();
            assert!(tc.as_f64() > 1.0 && k_zero > 1.0);

            let gap = solver.solve(&base_model, ElectronVolts::new(0.0)).unwrap();
            let qp = quasiparticle_energy(ElectronVolts::new(0.002), gap);
            let (u, v) = coherence_factors_normal_limit(ElectronVolts::new(0.0), gap).unwrap();
            assert!(qp.as_f64() >= gap.as_f64() && (u * u + v * v - 1.0).abs() < 1e-12);

            let gamma = ElectronVolts::new(0.0002);
            let dos = dynes_dos_ratio(gap, gap, gamma).unwrap();
            let g_nis = nis_tunneling_conductance(gap, gap, gamma, Kelvin::new(0.25 * tc.as_f64()))
                .unwrap();
            assert!(dos > 1.0 && g_nis > 1.0);

            let temp = Kelvin::new(300.0);
            let mu_fermi = ElectronVolts::new(0.20);
            let mu_bose = ElectronVolts::new(-0.02);
            let (mut fd_acc, mut be_acc, mut mb_acc) = (0.0, 0.0, 0.0);
            for step in 0..64 {
                let e = ElectronVolts::new(0.01 + (step as f64) * 0.01);
                fd_acc += calculate_occupancy_ev(FermiDirac, e, mu_fermi, temp).unwrap();
                be_acc += calculate_occupancy_ev(BoseEinstein, e, mu_bose, temp).unwrap();
                mb_acc += calculate_occupancy_ev(MaxwellBoltzmann, e, mu_fermi, temp).unwrap();
            }

            let canon = canonical_thermodynamics_ev(&spectrum, temp).unwrap();
            (
                gap.as_f64(),
                g_nis,
                fd_acc,
                be_acc,
                mb_acc,
                canon.partition_function,
            )
        });

    assert!(gap_ev > 0.0 && nis_peak > 1.0);
    assert!(fd_sum > 0.0 && be_sum > 0.0 && mb_sum > 0.0 && z_canon > 0.0);
}

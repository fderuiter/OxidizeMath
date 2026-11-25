//! Statistical Mechanics module.
//!
//! This module serves as the mathematical bridge between quantum/classical micro-physics
//! and macroscopic thermodynamics, covering Ensemble Theory, Quantum Statistics,
//! Phase Transitions, Self-Organized Criticality, and Continuum Non-Equilibrium SPDE dynamics
//! (Edwards-Wilkinson, Kardar-Parisi-Zhang, Allen-Cahn, and Cahn-Hilliard).

/// Boltzmann Constant in J/K.
pub const KB: f64 = 1.380649e-23;

pub mod allen_cahn;
pub mod cahn_hilliard;
#[allow(missing_docs)]
pub mod dynamics;
pub mod edwards_wilkinson;
#[allow(missing_docs)]
pub mod ensembles;
pub mod graph_ising;
pub mod ising;
pub mod kpz;
#[allow(missing_docs)]
pub mod quantum_stats;
pub mod sandpile;

pub use allen_cahn::{
    AllenCahn2D, ginzburg_landau_free_energy_2d, try_ginzburg_landau_free_energy_2d,
};
pub use cahn_hilliard::CahnHilliard2D;
pub use edwards_wilkinson::{
    EdwardsWilkinson1D, EdwardsWilkinson2D, interface_roughness, mean_field,
    theoretical_steady_state_roughness_sq_1d, try_theoretical_steady_state_roughness_sq_1d,
};
pub use graph_ising::GraphSpinSystem;
pub use kpz::{KpzModel1D, KpzModel2D};

// [cite:stat_mech]
// [cite:graph_parameters_rust]

use pure_math::theory_verification;

#[cfg(test)]
fn verify_spinodal_coarsening_and_mass_conservation(target_mean: f64) {
    use approx::assert_relative_eq;
    use engine_pde::BoundaryCondition;
    use math_commons::registry::TOLERANCE_STANDARD;
    use oxidize_core::rng::OxidizeRng;

    let mut rng_ch = OxidizeRng::from_seed(2026_1123);
    let bc = BoundaryCondition::Periodic;
    let mut ch_stencil = CahnHilliard2D::new(16, 16, 1.0, 1.0, 1.0, 0.5, 1.0e-4, bc);
    ch_stencil.init_spinodal_perturbation(target_mean, 0.15, &mut rng_ch);
    let m0 = ch_stencil.spatial_mean();
    assert_relative_eq!(m0, target_mean, epsilon = TOLERANCE_STANDARD);

    let mut ch_det = CahnHilliard2D::new(16, 16, 1.0, 1.0, 1.0, 0.5, 0.0, bc);
    ch_det.field_mut().copy_from_slice(ch_stencil.field());
    let e0_stencil = ch_det.ginzburg_landau_energy();

    let mut ch_fourier = CahnHilliard2D::new(16, 16, 1.0, 1.0, 1.0, 0.5, 1.0e-4, bc);
    ch_fourier.field_mut().copy_from_slice(ch_stencil.field());
    let mut ch_fourier_det = CahnHilliard2D::new(16, 16, 1.0, 1.0, 1.0, 0.5, 0.0, bc);
    ch_fourier_det
        .field_mut()
        .copy_from_slice(ch_stencil.field());
    let e0_fourier = ch_fourier_det.ginzburg_landau_energy();

    let mut ac = AllenCahn2D::new(16, 16, 1.0, 1.0, 1.0, 0.5, 0.0, BoundaryCondition::Neumann);
    ac.field_mut().copy_from_slice(ch_stencil.field());
    let ac_e0 = ac.ginzburg_landau_energy();

    for _ in 0..120 {
        ch_stencil.step_stencil(0.005, &mut rng_ch);
        ch_det.step_stencil(0.005, &mut rng_ch);
        ch_fourier.step_fourier_dealiased(0.05, &mut rng_ch);
        ch_fourier_det.step_fourier_dealiased(0.05, &mut rng_ch);
        ac.step_split(0.025, 2, &mut rng_ch);
    }

    assert!((ch_stencil.spatial_mean() - m0).abs() < TOLERANCE_STANDARD);
    assert!((ch_fourier.spatial_mean() - m0).abs() < TOLERANCE_STANDARD);
    assert!(ch_det.ginzburg_landau_energy() < e0_stencil);
    assert!(ch_fourier_det.ginzburg_landau_energy() < e0_fourier);
    assert!(ac.ginzburg_landau_energy() < ac_e0);
}

#[cfg(test)]
fn verify_ew_and_kpz_roughness_scaling() {
    use engine_pde::BoundaryCondition;
    use oxidize_core::rng::OxidizeRng;

    let n_sites = 64;
    let n_runs = 24;
    let mut ew_early = 0.0;
    let mut ew_late = 0.0;
    let mut ew_mean = 0.0;
    let mut kpz_early = 0.0;
    let mut kpz_late = 0.0;
    let mut kpz_mean = 0.0;

    for run in 0..n_runs {
        let seed = 90_000 + run as u64;
        let mut rng_ew = OxidizeRng::from_seed(seed);
        let mut rng_kpz = OxidizeRng::from_seed(seed);
        let mut ew = EdwardsWilkinson1D::new(n_sites, 1.0, 0.5, 0.4, BoundaryCondition::Periodic);
        let mut kpz = KpzModel1D::new(n_sites, 1.0, 0.5, 2.0, 0.4, BoundaryCondition::Periodic);

        for _ in 0..20 {
            ew.step_fourier(0.05, &mut rng_ew);
            kpz.step_fourier_dealiased(0.05, &mut rng_kpz);
        }
        ew_early += ew.interface_roughness();
        kpz_early += kpz.interface_roughness();

        for _ in 0..140 {
            ew.step_fourier(0.05, &mut rng_ew);
            kpz.step_fourier_dealiased(0.05, &mut rng_kpz);
        }
        ew_late += ew.interface_roughness();
        ew_mean += ew.mean_height();
        kpz_late += kpz.interface_roughness();
        kpz_mean += kpz.mean_height();
    }

    let inv = 1.0 / (n_runs as f64);
    let ew_ratio = (ew_late * inv) / (ew_early * inv);
    assert!(ew_early * inv > 0.05 && (1.15..=2.35).contains(&ew_ratio));
    assert!(kpz_early * inv > 0.05 && kpz_late > kpz_early);
    assert!(kpz_mean * inv > ew_mean * inv + 0.15);
}

theory_verification!(
    module = "stat_mech",
    constants = {
        EW_BETA_EXPONENT = 0.25;
        SPINODAL_MEAN_TARGET = 0.08;
    },
    test = {
        use approx::assert_relative_eq;
        use math_commons::registry::TOLERANCE_STANDARD;

        assert_relative_eq!(EW_BETA_EXPONENT, 0.25, epsilon = TOLERANCE_STANDARD);
        super::verify_spinodal_coarsening_and_mass_conservation(SPINODAL_MEAN_TARGET);
        super::verify_ew_and_kpz_roughness_scaling();
    }
);

#[cfg(test)]
mod spde_zero_alloc_tests {
    use super::*;
    use engine_pde::BoundaryCondition;
    use oxidize_core::rng::OxidizeRng;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn test_stat_mech_spdes_and_langevin_zero_heap_allocations() {
        let mut rng = OxidizeRng::from_seed(4242);
        let mut ew = EdwardsWilkinson1D::new(32, 1.0, 0.5, 0.1, BoundaryCondition::Periodic);
        let mut ew2d =
            EdwardsWilkinson2D::new(8, 8, 1.0, 1.0, 0.5, 0.1, BoundaryCondition::Neumann);
        let mut kpz = KpzModel1D::new(32, 1.0, 0.5, 1.2, 0.1, BoundaryCondition::Periodic);
        let mut kpz2d = KpzModel2D::new(8, 8, 1.0, 1.0, 0.5, 1.2, 0.1, BoundaryCondition::Periodic);
        let mut ac = AllenCahn2D::new(8, 8, 1.0, 1.0, 1.0, 0.5, 0.05, BoundaryCondition::Neumann);
        let mut ch =
            CahnHilliard2D::new(8, 8, 1.0, 1.0, 1.0, 0.5, 0.02, BoundaryCondition::Periodic);
        ch.init_spinodal_perturbation(0.0, 0.1, &mut rng);

        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let ((q_final, mean_ch), metrics) = VerifiedEngine::run_verified(|| {
            let mut q = 1.0;
            let mut p = 0.0;
            for _ in 0..5 {
                q = dynamics::step_overdamped_brownian_1d(q, 1.0, 1.0, 0.1, 0.01, &mut rng);
                (q, p) = dynamics::step_underdamped_langevin_1d(
                    q, p, 1.0, 1.0, 0.5, 0.1, 0.01, &mut rng,
                );
                ew.step_stencil(0.01, &mut rng);
                ew.step_fourier(0.01, &mut rng);
                ew2d.step_stencil(0.01, &mut rng);
                kpz.step_stencil(0.01, &mut rng);
                kpz.step_fourier_dealiased(0.01, &mut rng);
                kpz2d.step_stencil(0.01, &mut rng);
                kpz2d.step_fourier_dealiased(0.01, &mut rng);
                ac.step_explicit(0.01, &mut rng);
                ac.step_split(0.01, 2, &mut rng);
                ch.step_stencil(0.005, &mut rng);
                ch.step_fourier_dealiased(0.01, &mut rng);
            }
            (q, ch.spatial_mean())
        });

        assert!(q_final.is_finite());
        assert!(mean_ch.abs() < 1e-12);
        assert!(metrics.function_calls > 0);
    }
}

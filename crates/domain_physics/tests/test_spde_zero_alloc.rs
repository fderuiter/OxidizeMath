//! Standalone zero-allocation verification suite for 1D/2D KPZ, Allen-Cahn, and Cahn-Hilliard SPDE steps
//! under `verified_engine::allocator::VerifiedAllocator`.

use domain_physics::physics::stat_mech::interface_growth::{
    InterfaceGrowth1D, InterfaceGrowth2D, InterfaceGrowthParams, InterfaceScalingTelemetry,
};
use domain_physics::physics::stat_mech::phase_field::{
    PhaseFieldModel, PhaseFieldParams, PhaseFieldSolver2D,
};
use oxidize_core::rng::OxidizeRng;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn spde_kpz_allen_cahn_and_cahn_hilliard_run_with_zero_heap_allocations() {
    let mut rng = OxidizeRng::from_seed(0x0005_FDE0);
    let kpz_params = InterfaceGrowthParams::default();
    let mut kpz_1d = InterfaceGrowth1D::new(128, kpz_params);
    let mut kpz_2d = InterfaceGrowth2D::new(32, 32, kpz_params);
    let mut kpz_tel = InterfaceScalingTelemetry::new(32);

    let ac_params = PhaseFieldParams {
        model: PhaseFieldModel::AllenCahn,
        ..PhaseFieldParams::default()
    };
    let ch_params = PhaseFieldParams {
        model: PhaseFieldModel::CahnHilliard,
        ..PhaseFieldParams::default()
    };
    let mut allen_cahn = PhaseFieldSolver2D::new_spinodal(32, 32, 0.0, 0.1, &mut rng, ac_params);
    let mut cahn_hilliard =
        PhaseFieldSolver2D::new_spinodal(32, 32, 0.05, 0.1, &mut rng, ch_params);

    let ((w_1d, w_2d, ac_k, ch_k), _metrics) = VerifiedEngine::run_verified(|| {
        kpz_1d.step_n(&mut rng, 4);
        let w1 = kpz_1d.interface_width();
        kpz_tel.record_kpz_1d(kpz_1d.time(), w1, kpz_1d.len() as f64);

        kpz_2d.step_n(&mut rng, 2);
        let w2 = kpz_2d.interface_width();

        allen_cahn.step_n(&mut rng, 2);
        let ac_sum = allen_cahn.refresh_telemetry();

        cahn_hilliard.step_n(&mut rng, 2);
        let ch_sum = cahn_hilliard.refresh_telemetry();

        (w1, w2, ac_sum.k_first_moment, ch_sum.k_first_moment)
    });

    assert!(w_1d > 0.0 && w_2d > 0.0);
    assert!(ac_k > 0.0 && ch_k > 0.0);
}

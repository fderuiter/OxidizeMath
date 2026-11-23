//! Integration verification suite for the 2D Gross-Pitaevskii superfluid vortex solver.

#![cfg_attr(any(), verified(opt_out = "tests"))]

// [cite:quantum_mechanics]

use domain_physics::physics::quantum::gross_pitaevskii::{
    GpeParams2D, GpeTrapConfig, GpeTrapKind, GrossPitaevskiiSolver2D,
};
use verified_engine::engine::VerifiedEngine;

#[test]
fn integration_gpe_128_norm_conservation_and_quantized_vortex_charges() {
    let mut solver = GrossPitaevskiiSolver2D::new(128, 128, 20.0, 20.0);
    solver.configure_trap(GpeTrapConfig {
        kind: GpeTrapKind::BoxPeriodic,
        omega_x: 0.0,
        omega_y: 0.0,
        obstacle_height: 0.0,
        obstacle_width: 1.0,
    });
    let g = 150.0;
    solver.seed_condensate(g, 1.0);
    solver.imprint_vortex_dipole(-2.0, 0.0, 2.0, 0.0, 0.45);

    let initial_summary = solver.detect_vortices(0.04);
    assert_eq!(initial_summary.positive_count, 1);
    assert_eq!(initial_summary.negative_count, 1);
    assert_eq!(initial_summary.net_charge, 0);

    let inv0 = solver.compute_invariants(g);
    let params = GpeParams2D {
        interaction_g: g,
        imaginary_time_cooling: false,
    };

    let mut summary = initial_summary;
    VerifiedEngine::run_verified(|| {
        for _ in 0..40 {
            solver.step(&params, 0.0025);
        }
        summary = solver.detect_vortices(0.04);
    });

    let inv1 = solver.compute_invariants(g);
    let norm_err = (inv1.norm - inv0.norm).abs();
    assert!(
        norm_err < 1.0e-12,
        "2D Strang GPE norm drift {norm_err:e} exceeded 1.0e-12"
    );
    assert_eq!(summary.positive_count, 1);
    assert_eq!(summary.negative_count, 1);
    assert_eq!(summary.net_charge, 0);
}

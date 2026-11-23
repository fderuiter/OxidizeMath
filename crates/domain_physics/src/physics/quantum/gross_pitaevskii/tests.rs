#![cfg_attr(any(), verified(opt_out = "tests"))]

// [cite:quantum_mechanics]

use std::f64::consts::TAU;

use verified_engine::engine::VerifiedEngine;

use super::*;

#[test]
fn test_gpe_2d_strang_norm_conservation_and_quantized_vortex_dipole() {
    let mut solver = GrossPitaevskiiSolver2D::new(128, 128, 20.0, 20.0);
    solver.configure_trap(GpeTrapConfig {
        kind: GpeTrapKind::HarmonicIsotropic,
        omega_x: 1.0,
        omega_y: 1.0,
        ..Default::default()
    });
    let g = 220.0;
    solver.seed_condensate(g, 1.0);
    solver.imprint_vortex_dipole(-1.4, 0.0, 1.4, 0.0, 0.32);

    let summary_init = solver.detect_vortices(0.02);
    assert_eq!(summary_init.positive_count, 1);
    assert_eq!(summary_init.negative_count, 1);
    assert_eq!(summary_init.net_charge, 0);
    assert!(summary_init.net_circulation.abs() < 1.0e-12);

    let inv_init = solver.compute_invariants(g);
    assert!((inv_init.norm - 1.0).abs() < 1.0e-12);

    let real_params = GpeParams2D {
        interaction_g: g,
        imaginary_time_cooling: false,
    };

    VerifiedEngine::run_verified(|| {
        for _ in 0..40 {
            solver.step(&real_params, 0.004);
        }
        let _ = solver.detect_vortices(0.02);
        let _ = solver.compute_invariants(g);
    });

    let inv_after = solver.compute_invariants(g);
    let norm_drift = (inv_after.norm - inv_init.norm).abs();
    assert!(
        norm_drift < 1.0e-12,
        "2D Strang split-step norm drift {norm_drift:.3e} exceeded 1.0e-12"
    );

    let summary_after = solver.detect_vortices(0.02);
    assert_eq!(summary_after.positive_count, 1);
    assert_eq!(summary_after.negative_count, 1);
    assert_eq!(summary_after.net_charge, 0);
    for core in solver.vortices() {
        assert!(core.charge == 1 || core.charge == -1);
        assert!((core.circulation.abs() - TAU).abs() < 1.0e-12);
    }
}

#[test]
fn test_gpe_co_rotating_pair_quadrupole_and_imaginary_time_cooling() {
    let mut solver = GrossPitaevskiiSolver2D::new(96, 96, 18.0, 18.0);
    solver.configure_trap(GpeTrapConfig {
        kind: GpeTrapKind::CentralObstacle,
        omega_x: 1.0,
        omega_y: 1.1,
        obstacle_height: 8.0,
        obstacle_width: 0.9,
    });
    let g = 200.0;
    solver.seed_condensate(g, 1.0);
    solver.imprint_vortex(-1.6, 0.0, 1, 0.35);
    solver.imprint_vortex(1.6, 0.0, 1, 0.35);

    let co_summary = solver.detect_vortices(0.02);
    assert_eq!(co_summary.positive_count, 2);
    assert_eq!(co_summary.negative_count, 0);
    assert_eq!(co_summary.net_charge, 2);
    assert!((co_summary.net_circulation - 2.0 * TAU).abs() < 1.0e-11);

    let e_before = solver.compute_invariants(g).energy;
    let cool_params = GpeParams2D {
        interaction_g: g,
        imaginary_time_cooling: true,
    };
    VerifiedEngine::run_verified(|| {
        for _ in 0..10 {
            solver.step(&cool_params, 0.005);
        }
    });
    let inv_cooled = solver.compute_invariants(g);
    assert!(
        inv_cooled.energy < e_before,
        "Imaginary-time cooling must lower GPE energy ({} >= {e_before})",
        inv_cooled.energy
    );
    assert!((inv_cooled.norm - 1.0).abs() < 1.0e-12);
}

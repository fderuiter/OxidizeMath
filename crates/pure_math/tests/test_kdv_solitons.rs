//! Integration verification suite for the 1D KdV soliton and cubic NLS pseudospectral solver.

#![cfg_attr(any(), verified(opt_out = "tests"))]

// [cite:advanced_linear_algebra]
// [cite:graph_parameters_rust]

use pure_math::pure_math::analysis::pde::kdv::{
    KdvEquationMode, KdvParams, KdvSolver1D, SolitonSpec, kdv_two_soliton_phase_shifts,
};
use verified_engine::engine::VerifiedEngine;

#[test]
fn integration_kdv_512_two_soliton_overtaking_and_linear_airy_collapse() {
    let solitons = [
        SolitonSpec {
            speed_c: 16.0,
            x0: -8.0,
        },
        SolitonSpec {
            speed_c: 4.0,
            x0: -2.0,
        },
    ];
    let expected = kdv_two_soliton_phase_shifts(16.0, 4.0);
    assert!((expected.delta_fast - 0.5493061443340549).abs() < 1.0e-12);
    assert!((expected.delta_slow - (-1.0986122886681098)).abs() < 1.0e-12);

    let mut kdv = KdvSolver1D::new(512, 48.0);
    kdv.seed_kdv_solitons(&solitons);
    let kdv_params = KdvParams {
        mode: KdvEquationMode::KdvSoliton,
        nonlinear_enabled: true,
        nls_coupling: 1.0,
    };
    let inv0 = kdv.compute_invariants(&kdv_params);
    let peak0 = kdv.field().iter().copied().fold(0.0_f64, f64::max);

    VerifiedEngine::run_verified(|| {
        for _ in 0..75 {
            let _ = kdv.step(&kdv_params, 0.02);
        }
    });

    let inv1 = kdv.compute_invariants(&kdv_params);
    let [rel_m, rel_p, rel_h] = inv1.relative_drift(inv0);
    assert!(rel_m < 1.0e-5, "relative mass drift {rel_m:e} >= 1e-5");
    assert!(rel_p < 1.0e-5, "relative momentum drift {rel_p:e} >= 1e-5");
    assert!(
        rel_h < 1.0e-5,
        "relative Hamiltonian drift {rel_h:e} >= 1e-5"
    );

    let (d_fast, d_slow) = kdv.measure_two_soliton_phase_shifts(16.0, -8.0, 4.0, -2.0);
    assert!(
        (d_fast - 0.5493061443340549).abs() < 5.0e-3,
        "fast soliton phase shift {d_fast} outside 5e-3 of +0.5493"
    );
    assert!(
        (d_slow - (-1.0986122886681098)).abs() < 5.0e-3,
        "slow soliton phase shift {d_slow} outside 5e-3 of -1.0986"
    );

    let mut airy = KdvSolver1D::new(512, 48.0);
    airy.seed_kdv_solitons(&solitons);
    let airy_params = KdvParams {
        mode: KdvEquationMode::KdvSoliton,
        nonlinear_enabled: false,
        nls_coupling: 1.0,
    };
    for _ in 0..75 {
        let _ = airy.step(&airy_params, 0.02);
    }
    let airy_peak = airy.field().iter().copied().fold(0.0_f64, f64::max);
    let decay_fraction = 1.0 - airy_peak / peak0;
    assert!(
        decay_fraction > 0.60,
        "Linear Airy peak decay fraction {decay_fraction:.4} must exceed 0.60"
    );
}

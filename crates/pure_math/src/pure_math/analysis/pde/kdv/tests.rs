#![cfg_attr(any(), verified(opt_out = "tests"))]

// [cite:advanced_linear_algebra]

use verified_engine::engine::VerifiedEngine;

use super::*;

#[test]
fn test_kdv_two_soliton_collision_invariants_and_analytical_phase_shifts() {
    let mut solver = KdvSolver1D::new(512, 40.0);
    let c1 = 16.0;
    let c2 = 4.0;
    let x1_0 = -10.0;
    let x2_0 = -1.0;
    solver.seed_kdv_solitons(&[
        SolitonSpec {
            speed_c: c1,
            x0: x1_0,
        },
        SolitonSpec {
            speed_c: c2,
            x0: x2_0,
        },
    ]);

    let params = KdvParams {
        mode: KdvEquationMode::KdvSoliton,
        nonlinear_enabled: true,
        nls_coupling: 1.0,
    };
    let inv_initial = solver.compute_invariants(&params);

    // Advance to t = 1.40 so the fast soliton (c1 = 16) overtakes the slow soliton (c2 = 4)
    // and both separate cleanly before reaching the periodic seam at x = +/-20.
    for _ in 0..70 {
        solver.step(&params, 0.02);
    }
    let inv_final = solver.compute_invariants(&params);
    let drift = inv_final.relative_drift(inv_initial);
    assert!(
        drift[0] < 1.0e-5 && drift[1] < 1.0e-5 && drift[2] < 1.0e-5,
        "Relative drift of (M, P, H) must stay < 1.0e-5, got {drift:?}"
    );

    let exact = kdv_two_soliton_phase_shifts(c1, c2);
    assert!((exact.delta_fast - 0.5493061443340549).abs() < 1.0e-12);
    assert!((exact.delta_slow - (-1.0986122886681098)).abs() < 1.0e-12);

    let (meas_fast, meas_slow) = solver.measure_two_soliton_phase_shifts(c1, x1_0, c2, x2_0);
    assert!(
        (meas_fast - exact.delta_fast).abs() < 5.0e-3,
        "Fast soliton phase shift {meas_fast:.6} differed from exact {:.6} by >= 5e-3",
        exact.delta_fast
    );
    assert!(
        (meas_slow - exact.delta_slow).abs() < 5.0e-3,
        "Slow soliton phase shift {meas_slow:.6} differed from exact {:.6} by >= 5e-3",
        exact.delta_slow
    );
}

#[test]
fn test_linear_airy_dispersion_decays_over_sixty_percent() {
    let mut solver = KdvSolver1D::new(512, 40.0);
    solver.seed_kdv_solitons(&[
        SolitonSpec {
            speed_c: 16.0,
            x0: -10.0,
        },
        SolitonSpec {
            speed_c: 4.0,
            x0: -1.0,
        },
    ]);
    let initial_peak = solver.field().iter().copied().fold(0.0_f64, f64::max);
    assert!((initial_peak - 8.0).abs() < 0.05);

    let linear_params = KdvParams {
        mode: KdvEquationMode::KdvSoliton,
        nonlinear_enabled: false,
        nls_coupling: 1.0,
    };
    let inv0 = solver.compute_invariants(&linear_params);
    for _ in 0..70 {
        solver.step(&linear_params, 0.02);
    }
    let final_peak = solver.field().iter().copied().fold(0.0_f64, f64::max);
    let decay_fraction = 1.0 - (final_peak / initial_peak);
    assert!(
        decay_fraction > 0.60,
        "Linear Airy wave peak must decay by > 60%, got {:.2}% (peak {initial_peak:.3} -> {final_peak:.3})",
        decay_fraction * 100.0
    );

    let inv1 = solver.compute_invariants(&linear_params);
    let drift = inv1.relative_drift(inv0);
    assert!(drift[1] < 1.0e-12 && drift[2] < 1.0e-12);
}

#[test]
fn test_cubic_nls_bright_soliton_conservation_and_zero_heap_allocations() {
    let mut nls = KdvSolver1D::new(256, 32.0);
    nls.seed_nls_bright_solitons(&[(1.5, 2.0, -5.0), (1.0, -1.5, 5.0)]);
    let nls_params = KdvParams {
        mode: KdvEquationMode::CubicNls1D,
        nonlinear_enabled: true,
        nls_coupling: 1.0,
    };
    let inv0 = nls.compute_invariants(&nls_params);

    let mut kdv = KdvSolver1D::new(512, 40.0);
    kdv.seed_kdv_solitons(&[
        SolitonSpec {
            speed_c: 16.0,
            x0: -8.0,
        },
        SolitonSpec {
            speed_c: 4.0,
            x0: 0.0,
        },
    ]);
    let kdv_params = KdvParams::default();
    // Prime cached IFRK4 factors once outside VerifiedAllocator or verify dynamic update works zero-alloc:
    VerifiedEngine::run_verified(|| {
        for _ in 0..20 {
            kdv.step(&kdv_params, 0.005);
            let _ = kdv.compute_invariants(&kdv_params);
            let _ = kdv.measure_two_soliton_phase_shifts(16.0, -8.0, 4.0, 0.0);
            nls.step(&nls_params, 0.01);
            let _ = nls.compute_invariants(&nls_params);
        }
    });

    let inv1 = nls.compute_invariants(&nls_params);
    let drift = inv1.relative_drift(inv0);
    assert!(
        drift[0] < 1.0e-12,
        "Strang split-step 1D NLS norm drift must be < 1e-12, got {}",
        drift[0]
    );
    assert!(drift[2] < 1.0e-3);
}

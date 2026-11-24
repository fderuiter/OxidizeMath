#![cfg_attr(any(), verified(opt_out = "tests"))]

use std::f64::consts::PI;

use approx::assert_relative_eq;
use engine_pde::{
    BoundaryCondition, inject_additive_noise_2d, inject_conserved_flux_noise_1d,
    inject_conserved_flux_noise_2d,
};
use math_commons::math_kernel::types::{SolverWorkspace, flatten_2d_index};
use oxidize_core::rng::OxidizeRng;
use verified_engine::engine::VerifiedEngine;

use super::{ConservedBiharmonicParams, ReactionBiharmonicParams, SemiImplicitSpectral2D};

#[test]
fn engine_pde_conserved_flux_noise_sums_to_zero_on_non_radix2_grids() {
    let mut rng = OxidizeRng::from_seed(0x59DE_1001);
    let mut scratch_1d = vec![0.0; 96];
    let mut periodic_1d = vec![0.0; 96];
    let mut ws_1d = SolverWorkspace::new(&mut scratch_1d);
    inject_conserved_flux_noise_1d(
        &mut periodic_1d,
        0.5,
        0.05,
        1.25,
        BoundaryCondition::Periodic,
        &mut ws_1d,
        &mut rng,
    );
    assert!(periodic_1d.iter().sum::<f64>().abs() < 1e-13);
    assert!(periodic_1d.iter().any(|&v| v.abs() > 0.05));

    let (w, h) = (36, 24);
    let mut scratch_2d = vec![0.0; 2 * w * h];
    let mut periodic_2d = vec![0.0; w * h];
    let mut ws_2d = SolverWorkspace::new(&mut scratch_2d);
    inject_conserved_flux_noise_2d(
        &mut periodic_2d,
        w,
        h,
        0.5,
        0.75,
        0.05,
        0.8,
        BoundaryCondition::Periodic,
        &mut ws_2d,
        &mut rng,
    );
    assert!(periodic_2d.iter().sum::<f64>().abs() < 1e-13);
    assert!(periodic_2d.iter().any(|&v| v.abs() > 0.05));
}

#[test]
fn semi_implicit_spectral_2d_matches_discrete_biharmonic_eigenmode_decay() {
    let (w, h) = (36, 24);
    let (dx, dy) = (1.0, 1.0);
    let mut solver = SemiImplicitSpectral2D::new(w, h, dx, dy);
    let mut u = vec![0.0; w * h];
    let rhs = vec![0.0; w * h];

    let (mx, my) = (3usize, 2usize);
    for y in 0..h {
        for x in 0..w {
            let phase = 2.0 * PI * ((mx * x) as f64 / (w as f64) + (my * y) as f64 / (h as f64));
            u[flatten_2d_index(x, y, w)] = phase.cos();
        }
    }
    let u0 = u.clone();

    let qx2 = 4.0 * (PI * (mx as f64) / (w as f64)).sin().powi(2);
    let qy2 = 4.0 * (PI * (my as f64) / (h as f64)).sin().powi(2);
    let q2 = qx2 + qy2;
    let q4 = q2 * q2;

    let params = ReactionBiharmonicParams {
        dt: 0.1,
        alpha_0: 0.3,
        alpha_2: 1.2,
        alpha_4: 0.4,
    };
    solver.step_reaction_biharmonic(&mut u, &rhs, None, params);

    let expected_factor =
        1.0 / (1.0 + params.dt * (params.alpha_0 + params.alpha_2 * q2 + params.alpha_4 * q4));
    for (actual, init) in u.iter().zip(u0.iter()) {
        assert_relative_eq!(*actual, init * expected_factor, epsilon = 1e-12);
    }
}

#[test]
fn conserved_biharmonic_preserves_mean_and_detects_radial_structure_peak() {
    let (w, h) = (48, 48);
    let mut solver = SemiImplicitSpectral2D::new(w, h, 1.0, 1.0);
    let mut rng = OxidizeRng::from_seed(0xCAFE_2026);
    let mut noise_field = vec![0.0; w * h];
    let mut noise_scratch = vec![0.0; 2 * w * h];

    let target_mode = 4usize;
    let mut phi = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let wave = (2.0 * PI * (target_mode * x) as f64 / (w as f64)).cos();
            phi[flatten_2d_index(x, y, w)] = 0.25 + 0.4 * wave;
        }
    }
    let initial_mean = phi.iter().sum::<f64>() / ((w * h) as f64);

    let mut mu_nl = vec![0.0; w * h];
    let params = ConservedBiharmonicParams {
        dt: 0.05,
        mobility: 1.0,
        stabilizer_s: 2.0,
        kappa: 0.5,
    };

    for _ in 0..10 {
        for (dst, &p) in mu_nl.iter_mut().zip(phi.iter()) {
            *dst = p * p * p - 3.0 * p;
        }
        noise_field.fill(0.0);
        let mut ws = SolverWorkspace::new(&mut noise_scratch);
        inject_conserved_flux_noise_2d(
            &mut noise_field,
            w,
            h,
            1.0,
            1.0,
            params.dt,
            0.004,
            BoundaryCondition::Periodic,
            &mut ws,
            &mut rng,
        );
        solver.step_conserved_biharmonic(&mut phi, &mu_nl, Some(&noise_field), params);
    }

    let final_mean = phi.iter().sum::<f64>() / ((w * h) as f64);
    assert!(
        (final_mean - initial_mean).abs() < 1e-13,
        "Mean drifted: {initial_mean} -> {final_mean}"
    );

    let n_bins = solver.radial_bin_count();
    let mut k_bins = vec![0.0; n_bins];
    let mut s_k = vec![0.0; n_bins];
    let mut counts = vec![0usize; n_bins];
    let summary = solver.compute_radial_structure_factor(&phi, &mut k_bins, &mut s_k, &mut counts);
    let expected_k = 2.0 * PI * (target_mode as f64) / (w as f64);
    assert_relative_eq!(summary.k_max, expected_k, epsilon = 1e-12);
    assert!(summary.s_max > 0.0);
}

#[test]
fn spde_noise_and_spectral_stepping_perform_zero_heap_allocations() {
    let (w, h) = (36, 36);
    let mut rng = OxidizeRng::from_seed(777);
    let mut noise_field = vec![0.0; w * h];
    let mut noise_scratch = vec![0.0; 2 * w * h];
    let mut solver = SemiImplicitSpectral2D::new(w, h, 1.0, 1.0);
    let mut phi = vec![0.05; w * h];
    let mut mu_nl = vec![0.0; w * h];
    let n_bins = solver.radial_bin_count();
    let mut k_bins = vec![0.0; n_bins];
    let mut s_k = vec![0.0; n_bins];
    let mut counts = vec![0usize; n_bins];

    let params = ConservedBiharmonicParams {
        dt: 0.05,
        mobility: 1.0,
        stabilizer_s: 2.5,
        kappa: 0.5,
    };

    VerifiedEngine::run_verified(|| {
        noise_field.fill(0.0);
        {
            let mut ws = SolverWorkspace::new(&mut noise_scratch);
            inject_additive_noise_2d(
                &mut noise_field,
                w,
                h,
                1.0,
                1.0,
                params.dt,
                0.002,
                BoundaryCondition::Periodic,
                &mut ws,
                &mut rng,
            );
        }
        noise_field.fill(0.0);
        {
            let mut ws = SolverWorkspace::new(&mut noise_scratch);
            inject_conserved_flux_noise_2d(
                &mut noise_field,
                w,
                h,
                1.0,
                1.0,
                params.dt,
                0.002,
                BoundaryCondition::Periodic,
                &mut ws,
                &mut rng,
            );
        }
        for (dst, &p) in mu_nl.iter_mut().zip(phi.iter()) {
            *dst = p * p * p - 3.0 * p;
        }
        solver.step_conserved_biharmonic(&mut phi, &mu_nl, Some(&noise_field), params);
        let summary =
            solver.compute_radial_structure_factor(&phi, &mut k_bins, &mut s_k, &mut counts);
        assert!(summary.num_bins >= 2);
    });
}

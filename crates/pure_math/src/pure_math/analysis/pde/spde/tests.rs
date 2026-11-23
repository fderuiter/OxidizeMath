#![cfg_attr(any(), verified(opt_out = "tests"))]

use std::f64::consts::PI;

use approx::assert_relative_eq;
use math_commons::math_kernel::types::flatten_2d_index;
use oxidize_core::rng::OxidizeRng;
use verified_engine::engine::VerifiedEngine;

use super::{
    ConservedBiharmonicParams, ReactionBiharmonicParams, SemiImplicitSpectral2D, SpatialNoiseBuffer,
};

#[test]
fn spatial_noise_conserved_flux_sums_to_zero_in_1d_and_2d() {
    let mut rng = OxidizeRng::from_seed(0x59DE_1001);
    let mut buf_1d = SpatialNoiseBuffer::new_1d(128);
    let periodic_1d = buf_1d.fill_conserved_flux_1d_periodic(&mut rng, 128, 0.5, 1.25);
    assert!(periodic_1d.iter().sum::<f64>().abs() < 1e-13);
    assert!(periodic_1d.iter().any(|&v| v.abs() > 0.1));

    let neumann_1d = buf_1d.fill_conserved_flux_1d_neumann(&mut rng, 128, 0.5, 1.25);
    assert!(neumann_1d.iter().sum::<f64>().abs() < 1e-13);

    let mut buf_2d = SpatialNoiseBuffer::new_2d(32, 24);
    let periodic_2d = buf_2d.fill_conserved_flux_2d_periodic(&mut rng, 32, 24, 0.5, 0.75, 0.8);
    assert!(periodic_2d.iter().sum::<f64>().abs() < 1e-13);
    assert!(periodic_2d.iter().any(|&v| v.abs() > 0.1));
}

#[test]
fn semi_implicit_spectral_2d_matches_discrete_biharmonic_eigenmode_decay() {
    let (w, h) = (32, 32);
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
    let (w, h) = (32, 32);
    let mut solver = SemiImplicitSpectral2D::new(w, h, 1.0, 1.0);
    let mut rng = OxidizeRng::from_seed(0xCAFE_2026);
    let mut noise_buf = SpatialNoiseBuffer::new_2d(w, h);

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
        let noise = noise_buf.fill_conserved_flux_2d_periodic(&mut rng, w, h, 1.0, 1.0, 0.02);
        solver.step_conserved_biharmonic(&mut phi, &mu_nl, Some(noise), params);
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
    let (w, h) = (32, 32);
    let mut rng = OxidizeRng::from_seed(777);
    let mut noise = SpatialNoiseBuffer::new_2d(w, h);
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
        let _ = noise.fill_gaussian(&mut rng, w * h, 0.01);
        let inc = noise.fill_conserved_flux_2d_periodic(&mut rng, w, h, 1.0, 1.0, 0.01);
        for (dst, &p) in mu_nl.iter_mut().zip(phi.iter()) {
            *dst = p * p * p - 3.0 * p;
        }
        solver.step_conserved_biharmonic(&mut phi, &mu_nl, Some(inc), params);
        let summary =
            solver.compute_radial_structure_factor(&phi, &mut k_bins, &mut s_k, &mut counts);
        assert!(summary.num_bins >= 2);
    });
}

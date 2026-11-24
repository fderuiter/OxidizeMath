// [cite:engine_pde]
//! Stencil, boundary, conserved-flux noise, radix-2 2/3-dealiased Fourier, Cahn-Hilliard, and zero-alloc tests.

use engine_pde::{
    BoundaryCondition, ExplicitSpdeStencilStepper, PdeOwnedWorkspace, SemiImplicitFourierStepper,
    SpdeNoiseKind, SplitStepSpdeStencilStepper, StencilOperators, dealias_1d_spectrum,
    fft_1d_inplace, fft_2d_inplace,
};
use math_commons::math_kernel::types::{
    SolverWorkspace, StepSize, flatten_2d_index, flatten_3d_index,
};
use math_commons::registry::TOLERANCE_STANDARD;
use oxidize_core::rng::OxidizeRng;
use verified_engine::engine::VerifiedEngine;

fn ginzburg_landau_energy_2d(
    phi: &[f64],
    width: usize,
    height: usize,
    ops: &StencilOperators,
    kappa: f64,
    bc: BoundaryCondition,
) -> f64 {
    let mut energy = 0.0;
    let dv = (*ops.dx) * (*ops.dy);
    for y in 0..height {
        for x in 0..width {
            let idx = flatten_2d_index(x, y, width);
            let p = phi[idx];
            let bulk = 0.25 * (p * p - 1.0) * (p * p - 1.0);
            let grad_sq = ops.grad_sq_2d_at(phi, width, height, x, y, bc);
            energy += (bulk + 0.5 * kappa * grad_sq) * dv;
        }
    }
    energy
}

#[test]
fn test_1d_2d_3d_stencils_and_public_2d_biharmonic() {
    let n = 32_usize;
    let dx = core::f64::consts::TAU / (n as f64);
    let ops = StencilOperators::new_3d(StepSize(dx), StepSize(dx), StepSize(dx));

    // 1D sin(x): grad = cos(x), lap = -sin(x)
    let u1: Vec<f64> = (0..n).map(|i| ((i as f64) * dx).sin()).collect();
    let g1 = ops.gradient_1d_at(&u1, n, 4, BoundaryCondition::Periodic);
    let l1 = ops.laplacian_1d_at(&u1, n, 4, BoundaryCondition::Periodic);
    assert!((g1 - (4.0 * dx).cos()).abs() < 2e-2);
    assert!((l1 + (4.0 * dx).sin()).abs() < 2e-2);

    // 2D sin(x) + cos(y): biharmonic nabla^4 u = sin(x) + cos(y)
    let mut u2 = vec![0.0_f64; n * n];
    for y in 0..n {
        for x in 0..n {
            u2[flatten_2d_index(x, y, n)] = ((x as f64) * dx).sin() + ((y as f64) * dx).cos();
        }
    }
    let mut lap2 = vec![0.0_f64; n * n];
    ops.compute_laplacian_2d(&u2, n, n, BoundaryCondition::Periodic, &mut lap2);
    let bh_from_lap =
        ops.eval_biharmonic_2d_from_laplacian(&lap2, n, n, 5, 7, BoundaryCondition::Periodic);
    let bh_direct = ops.eval_biharmonic_2d(&u2, n, n, 5, 7, BoundaryCondition::Periodic);
    let exact_bh = u2[flatten_2d_index(5, 7, n)];
    assert!((bh_from_lap - bh_direct).abs() < 1e-12);
    assert!((bh_direct - exact_bh).abs() < 3e-2);

    // 3D Laplacian + upwind check on 8x8x8
    let m = 8_usize;
    let mut u3 = vec![0.0_f64; m * m * m];
    for z in 0..m {
        for y in 0..m {
            for x in 0..m {
                u3[flatten_3d_index(x, y, z, m, m)] = (x + y + z) as f64;
            }
        }
    }
    let lap3 = ops.laplacian_3d_at(&u3, m, m, m, 3, 3, 3, BoundaryCondition::Neumann);
    assert!(
        lap3.abs() < 1e-10,
        "linear function has zero 3D interior Laplacian"
    );
    let up3 = ops.upwind_3d_at(
        &u3,
        m,
        m,
        m,
        3,
        3,
        3,
        (1.0, 2.0, 3.0),
        BoundaryCondition::Neumann,
    );
    assert!((up3 - (1.0 + 2.0 + 3.0) / dx).abs() < 1e-10);
}

#[test]
fn test_boundary_steppers_and_conserved_flux_noise_mean_invariance() {
    let w = 16_usize;
    let h = 16_usize;
    let ops = StencilOperators::new_2d(StepSize(0.5), StepSize(0.5));
    let mut rng = OxidizeRng::new(0xC095_E27E_D000_1111);
    let mut scratch = vec![0.0_f64; ExplicitSpdeStencilStepper::scratch_len_2d(w, h)];
    let mut ws = SolverWorkspace::new(&mut scratch);

    // Periodic & Neumann conserved-flux noise must preserve spatial mean within TOLERANCE_STANDARD
    for bc in [BoundaryCondition::Periodic, BoundaryCondition::Neumann] {
        let mut field: Vec<f64> = (0..w * h).map(|i| 0.25 + 0.01 * ((i % 7) as f64)).collect();
        let initial_mean = field.iter().sum::<f64>() / ((w * h) as f64);
        let stepper = ExplicitSpdeStencilStepper::new(
            ops,
            bc,
            SpdeNoiseKind::ConservedFlux { mobility_temp: 0.4 },
        );
        for _ in 0..25 {
            stepper.step_cahn_hilliard_2d(&mut field, w, h, 0.5, 0.25, 0.001, &mut ws, &mut rng);
        }
        let final_mean = field.iter().sum::<f64>() / ((w * h) as f64);
        assert!(
            (final_mean - initial_mean).abs() < TOLERANCE_STANDARD,
            "conserved-flux Cahn-Hilliard drifted mean: {initial_mean} -> {final_mean}"
        );
    }

    // Dirichlet boundary stepper pins edges to prescribed value
    let mut u_dir = vec![1.0_f64; w * h];
    let split = SplitStepSpdeStencilStepper::new(
        ops,
        BoundaryCondition::Dirichlet(-0.5),
        SpdeNoiseKind::Additive { intensity: 0.05 },
        2,
    );
    split.step_reaction_diffusion_2d(
        &mut u_dir,
        w,
        h,
        0.2,
        0.01,
        |v| v - v * v * v,
        &mut ws,
        &mut rng,
    );
    assert_eq!(u_dir[flatten_2d_index(0, 5, w)], -0.5);
    assert_eq!(u_dir[flatten_2d_index(w - 1, 5, w)], -0.5);
}

#[test]
fn test_fft_roundtrip_and_two_thirds_dealiasing() {
    let mut buf1 = [0.0_f64; 32];
    for i in 0..16 {
        buf1[2 * i] = ((i as f64) * 0.7).cos() + 0.3 * ((i as f64) * 1.9).sin();
    }
    let orig1 = buf1;
    fft_1d_inplace(&mut buf1, 16, false);
    fft_1d_inplace(&mut buf1, 16, true);
    for i in 0..16 {
        assert!((buf1[2 * i] - orig1[2 * i]).abs() < 1e-13);
    }

    // High mode m = 6 on N = 16 satisfies 3 * 6 = 18 >= 16 -> zeroed by 2/3 rule
    let mut spec = [0.0_f64; 32];
    spec[2 * 2] = 5.0;
    spec[2 * 6] = 9.0;
    dealias_1d_spectrum(&mut spec, 16);
    assert_eq!(spec[2 * 2], 5.0);
    assert_eq!(spec[2 * 6], 0.0);

    let mut buf2 = vec![0.0_f64; 2 * 8 * 8];
    let mut col = vec![0.0_f64; 16];
    for i in 0..64 {
        buf2[2 * i] = (i as f64).sin();
    }
    let orig2 = buf2.clone();
    fft_2d_inplace(&mut buf2, 8, 8, &mut col, false);
    fft_2d_inplace(&mut buf2, 8, 8, &mut col, true);
    for i in 0..64 {
        assert!((buf2[2 * i] - orig2[2 * i]).abs() < 1e-13);
    }
}

#[test]
fn test_cahn_hilliard_fourier_spinodal_coarsening() {
    let (w, h, dx, kappa) = (16_usize, 16_usize, 1.0_f64, 0.5_f64);
    let ops = StencilOperators::new_2d(StepSize(dx), StepSize(dx));
    let fourier = SemiImplicitFourierStepper::new_2d(dx, dx);
    let mut rng = OxidizeRng::new(0xCA49_4111_2026_5555);
    let mut phi = vec![0.0_f64; w * h];
    for (i, slot) in phi.iter_mut().enumerate() {
        *slot = 0.05 + 0.08 * ((i as f64 * 1.37).sin() + (i as f64 * 0.73).cos());
    }
    let mean_0 = phi.iter().sum::<f64>() / ((w * h) as f64);
    let mut scratch = vec![0.0_f64; SemiImplicitFourierStepper::scratch_len_2d(w, h)];
    let mut ws = SolverWorkspace::new(&mut scratch);

    let e0 = ginzburg_landau_energy_2d(&phi, w, h, &ops, kappa, BoundaryCondition::Periodic);
    for _ in 0..30 {
        fourier.step_cahn_hilliard_2d(&mut phi, w, h, 1.0, kappa, 0.0, 0.05, &mut ws, &mut rng);
    }
    let e_mid = ginzburg_landau_energy_2d(&phi, w, h, &ops, kappa, BoundaryCondition::Periodic);
    for _ in 0..30 {
        fourier.step_cahn_hilliard_2d(&mut phi, w, h, 1.0, kappa, 0.0, 0.05, &mut ws, &mut rng);
    }
    let e_end = ginzburg_landau_energy_2d(&phi, w, h, &ops, kappa, BoundaryCondition::Periodic);
    let mean_end = phi.iter().sum::<f64>() / ((w * h) as f64);

    assert!(
        (mean_end - mean_0).abs() < TOLERANCE_STANDARD,
        "Fourier Cahn-Hilliard mean drift: {mean_0} -> {mean_end}"
    );
    assert!(
        e_mid < e0 && e_end < e_mid,
        "Ginzburg-Landau energy failed to decrease during spinodal coarsening: {e0} -> {e_mid} -> {e_end}"
    );
}

#[test]
fn test_engine_pde_steppers_zero_heap_allocations_under_verified_lock() {
    let w = 8_usize;
    let h = 8_usize;
    let ops = StencilOperators::new_2d(StepSize(1.0), StepSize(1.0));
    let stencil_ch = ExplicitSpdeStencilStepper::new(
        ops,
        BoundaryCondition::Periodic,
        SpdeNoiseKind::ConservedFlux {
            mobility_temp: 0.02,
        },
    );
    let fourier = SemiImplicitFourierStepper::new_2d(1.0, 1.0);
    let mut owned =
        PdeOwnedWorkspace::<f64>::with_capacity(SemiImplicitFourierStepper::scratch_len_2d(w, h));
    let mut phi = [0.1_f64; 64];
    let mut height_field = [0.0_f64; 64];
    let mut rng = OxidizeRng::new(0x0A11_0C00_F00D_CAFE);

    let ((), metrics) = VerifiedEngine::run_verified(|| {
        let mut ws = owned.as_workspace();
        for _ in 0..6 {
            stencil_ch.step_cahn_hilliard_2d(&mut phi, w, h, 0.2, 0.5, 0.002, &mut ws, &mut rng);
            fourier.step_cahn_hilliard_2d(&mut phi, w, h, 0.5, 0.5, 0.02, 0.02, &mut ws, &mut rng);
            fourier.step_kpz_2d(
                &mut height_field,
                w,
                h,
                0.5,
                1.0,
                0.1,
                0.02,
                &mut ws,
                &mut rng,
            );
        }
    });
    assert!(phi.iter().all(|v| v.is_finite()));
    assert!(height_field.iter().all(|v| v.is_finite()));
    assert!(metrics.function_calls > 0);
}

#[test]
fn test_fused_stepper_2d_coupled_periodic_mass_conservation_and_zero_alloc() {
    use engine_pde::FusedStencilStepper;
    let stepper = FusedStencilStepper::new_2d(StepSize(1.0), StepSize(1.0));
    let mut u = [0.0_f64; 16];
    let mut v = [0.0_f64; 16];
    let mut next_u = [0.0_f64; 16];
    let mut next_v = [0.0_f64; 16];
    u[0] = 4.0;
    v[15] = 2.0;
    let mass_u0: f64 = u.iter().sum();
    let mass_v0: f64 = v.iter().sum();

    let ((), metrics) = VerifiedEngine::run_verified(|| {
        stepper.step_2d_coupled_periodic(
            (4, 4),
            [&u, &v],
            [&mut next_u, &mut next_v],
            0.1,
            1.0,
            |_idx, curr, l, r, up, down, ops| {
                [
                    0.5 * ops.central_diff_2nd_2d(curr[0], l[0], r[0], up[0], down[0]),
                    0.25 * ops.central_diff_2nd_2d(curr[1], l[1], r[1], up[1], down[1]),
                ]
            },
        );
    });
    assert!(metrics.function_calls > 0);
    assert!(next_u[3] > 0.0 && next_u[12] > 0.0);
    assert!(next_v[12] > 0.0 && next_v[3] > 0.0);
    assert!((next_u.iter().sum::<f64>() - mass_u0).abs() < TOLERANCE_STANDARD);
    assert!((next_v.iter().sum::<f64>() - mass_v0).abs() < TOLERANCE_STANDARD);
}

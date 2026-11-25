// [cite:engine_pde]
//! Checked constructor, stepper, FFT, and noise-injection domain error and zero-allocation tests.

use engine_pde::{
    BoundaryCondition, ExplicitSpdeStencilStepper, FusedStencilStepper, SemiImplicitFourierStepper,
    SpdeNoiseKind, SplitStepSpdeStencilStepper, StencilOperators, try_dealias_1d_spectrum,
    try_dealias_2d_spectrum, try_fft_1d_inplace, try_fft_2d_inplace, try_inject_additive_noise_1d,
    try_inject_additive_noise_2d, try_inject_conserved_flux_noise_1d,
    try_inject_conserved_flux_noise_2d,
};
use math_commons::error::MathError;
use math_commons::math_kernel::types::{SolverWorkspace, StepSize};
use math_rng::OxidizeRng;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_checked_stencil_fused_fourier_and_fft_errors() {
    assert!(matches!(
        StencilOperators::try_new(StepSize(0.0)),
        Err(MathError::StrictlyOutOfBounds { .. })
    ));
    assert!(StencilOperators::try_new_2d(StepSize(1.0), StepSize(-1.0)).is_err());
    assert!(
        StencilOperators::try_new_3d(StepSize(1.0), StepSize(1.0), StepSize(f64::NAN)).is_err()
    );
    assert!(FusedStencilStepper::try_new(StepSize(0.0)).is_err());
    assert!(FusedStencilStepper::try_new_2d(StepSize(1.0), StepSize(0.0)).is_err());
    assert!(FusedStencilStepper::try_new_3d(StepSize(1.0), StepSize(1.0), StepSize(0.0)).is_err());
    assert!(SemiImplicitFourierStepper::try_new_1d(0.0).is_err());
    assert!(SemiImplicitFourierStepper::try_new_2d(1.0, -0.2).is_err());

    let mut buf = [0.0_f64; 16];
    let mut col = [0.0_f64; 8];
    assert_eq!(
        try_fft_1d_inplace(&mut buf, 6, false),
        Err(MathError::InvalidRadix2GridSize { size: 6, min: 1 })
    );
    assert_eq!(
        try_fft_2d_inplace(&mut buf, 4, 2, &mut col[..2], false),
        Err(MathError::WorkspaceTooSmall {
            required: 4,
            available: 2,
        })
    );
    assert_eq!(
        try_dealias_1d_spectrum(&mut buf, 6),
        Err(MathError::InvalidRadix2GridSize { size: 6, min: 1 })
    );
    assert_eq!(
        try_dealias_2d_spectrum(&mut buf[..10], 4, 2),
        Err(MathError::WorkspaceTooSmall {
            required: 16,
            available: 10,
        })
    );
    assert!(try_fft_1d_inplace(&mut buf, 1 << (usize::BITS - 1), false).is_err());
    assert!(try_dealias_2d_spectrum(&mut buf, 1 << (usize::BITS - 2), 8).is_err());
}

#[test]
fn test_checked_fourier_1d_zero_alloc_errors() {
    let fourier = SemiImplicitFourierStepper::try_new_1d(1.0).unwrap();
    let mut rng = OxidizeRng::new(0xCECC_ED00_2026_0001);
    let mut u8_grid = [0.1_f64; 8];
    let mut tiny = [0.0_f64; 2];

    VerifiedEngine::run_verified(|| {
        let mut ws = SolverWorkspace::new(&mut tiny);
        let e1 = fourier.try_step_cahn_hilliard_1d(
            &mut u8_grid[..6],
            1.0,
            0.5,
            0.0,
            0.01,
            &mut ws,
            &mut rng,
        );
        assert_eq!(
            e1,
            Err(MathError::InvalidRadix2GridSize { size: 6, min: 4 })
        );
        assert!(
            fourier
                .try_step_kpz_1d(&mut u8_grid, 1.0, 0.5, 0.0, 0.01, &mut ws, &mut rng)
                .is_err()
        );
        assert!(
            fourier
                .try_step_edwards_wilkinson_1d(&mut u8_grid, 1.0, 0.0, 0.01, &mut ws, &mut rng)
                .is_err()
        );
    });
}

#[test]
fn test_checked_fourier_2d_zero_alloc_errors() {
    let fourier = SemiImplicitFourierStepper::try_new_2d(1.0, 1.0).unwrap();
    let mut rng = OxidizeRng::new(0xCECC_ED00_2026_0002);
    let mut u4x4 = [0.1_f64; 16];
    let mut tiny = [0.0_f64; 2];
    let mut full = [0.0_f64; 256];

    VerifiedEngine::run_verified(|| {
        let mut ws = SolverWorkspace::new(&mut tiny);
        assert!(
            fourier
                .try_step_cahn_hilliard_2d(&mut u4x4, 4, 4, 1.0, 0.5, 0.0, 0.01, &mut ws, &mut rng)
                .is_err()
        );
        assert!(
            fourier
                .try_step_kpz_2d(&mut u4x4, 4, 4, 1.0, 0.5, 0.0, 0.01, &mut ws, &mut rng)
                .is_err()
        );
        assert!(
            fourier
                .try_step_edwards_wilkinson_2d(&mut u4x4, 4, 4, 1.0, 0.0, 0.01, &mut ws, &mut rng)
                .is_err()
        );
        let mut ok_ws = SolverWorkspace::new(&mut full);
        assert!(
            fourier
                .try_step_edwards_wilkinson_2d(
                    &mut u4x4, 4, 4, 0.5, 0.01, 0.01, &mut ok_ws, &mut rng
                )
                .is_ok()
        );
    });
}

#[test]
fn test_checked_spde_stencil_zero_alloc_errors() {
    let ops = StencilOperators::try_new_2d(StepSize(1.0), StepSize(1.0)).unwrap();
    let explicit = ExplicitSpdeStencilStepper::new(
        ops,
        BoundaryCondition::Periodic,
        SpdeNoiseKind::Additive { intensity: 0.1 },
    );
    let split = SplitStepSpdeStencilStepper::new(
        ops,
        BoundaryCondition::Periodic,
        SpdeNoiseKind::Additive { intensity: 0.1 },
        1,
    );
    let mut rng = OxidizeRng::new(0xCECC_ED00_2026_0011);
    let mut u8_grid = [0.1_f64; 8];
    let mut u4x4 = [0.1_f64; 16];
    let mut tiny = [0.0_f64; 2];

    VerifiedEngine::run_verified(|| {
        let mut ws = SolverWorkspace::new(&mut tiny);
        assert!(
            explicit
                .try_step_1d(&mut u8_grid, 0.01, |_, _, _, _| 0.0, &mut ws, &mut rng)
                .is_err()
        );
        assert!(
            explicit
                .try_step_2d(
                    &mut u4x4,
                    4,
                    4,
                    0.01,
                    |_, _, _, _, _| 0.0,
                    &mut ws,
                    &mut rng
                )
                .is_err()
        );
        assert!(
            explicit
                .try_step_cahn_hilliard_2d(&mut u4x4, 4, 4, 0.5, 0.5, 0.01, &mut ws, &mut rng)
                .is_err()
        );
        assert!(
            split
                .try_step_reaction_diffusion_2d(
                    &mut u4x4,
                    4,
                    4,
                    0.2,
                    0.01,
                    |v| -v,
                    &mut ws,
                    &mut rng
                )
                .is_err()
        );
    });
}

#[test]
fn test_checked_spde_noise_zero_alloc_errors() {
    let mut rng = OxidizeRng::new(0xCECC_ED00_2026_0012);
    let mut u8_grid = [0.1_f64; 8];
    let mut u4x4 = [0.1_f64; 16];
    let mut tiny = [0.0_f64; 2];

    VerifiedEngine::run_verified(|| {
        let mut ws = SolverWorkspace::new(&mut tiny);
        let bc = BoundaryCondition::Periodic;
        assert!(
            try_inject_additive_noise_1d(&mut u8_grid, 0.0, 0.1, -0.01, bc, &mut ws, &mut rng)
                .is_err()
        );
        assert!(
            try_inject_additive_noise_2d(
                &mut u4x4[..15],
                4,
                4,
                1.0,
                1.0,
                0.1,
                0.01,
                bc,
                &mut ws,
                &mut rng
            )
            .is_err()
        );
        assert!(
            try_inject_conserved_flux_noise_1d(&mut u8_grid, 1.0, 0.1, 0.01, bc, &mut ws, &mut rng)
                .is_err()
        );
        assert!(
            try_inject_conserved_flux_noise_2d(
                &mut u4x4, 4, 4, 1.0, 1.0, 0.1, 0.01, bc, &mut ws, &mut rng
            )
            .is_err()
        );
        assert!(
            try_inject_conserved_flux_noise_2d(
                &mut u4x4,
                usize::MAX,
                4,
                1.0,
                1.0,
                0.1,
                0.01,
                bc,
                &mut ws,
                &mut rng
            )
            .is_err()
        );
    });
}

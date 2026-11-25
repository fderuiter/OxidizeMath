// [cite:engine_pde]
use crate::{
    BoundaryCondition, ExplicitSpdeStencilStepper, FusedStencilStepper, OxidizeRng, PdeError,
    SemiImplicitFourierStepper, SpdeNoiseKind, SplitStepSpdeStencilStepper, StencilOperators,
    try_dealias_1d_spectrum, try_fft_1d_inplace, try_inject_additive_noise_1d,
};
use math_commons::error::MathError;
use math_commons::math_kernel::types::{SolverWorkspace, StepSize};
use math_diagnostics::{Diagnostic, Severity};

fn assert_stack_error_traits<
    E: Copy + Clone + PartialEq + Eq + core::fmt::Display + core::error::Error + Diagnostic,
>() {
}

#[test]
fn pde_error_is_copy_and_diagnostic() {
    assert_stack_error_traits::<PdeError>();
    let err = PdeError::NonRadix2Dimension { axis: "n", size: 6 };
    let copy_err = err;
    assert_eq!(err, copy_err);
    assert_eq!(err.code(), "PDE-002");
    assert_eq!(Diagnostic::severity(&err), Severity::Error);
}

#[test]
fn stencil_and_fused_try_new_available_in_no_std() {
    assert!(StencilOperators::try_new(StepSize(0.1)).is_ok());
    assert!(StencilOperators::try_new_2d(StepSize(0.1), StepSize(0.2)).is_ok());
    assert!(StencilOperators::try_new_3d(StepSize(0.1), StepSize(0.2), StepSize(0.3)).is_ok());
    assert!(StencilOperators::try_new(StepSize(0.0)).is_err());

    assert!(FusedStencilStepper::try_new(StepSize(0.1)).is_ok());
    assert!(FusedStencilStepper::try_new_2d(StepSize(0.1), StepSize(0.2)).is_ok());
    assert!(FusedStencilStepper::try_new_3d(StepSize(0.1), StepSize(0.2), StepSize(0.3)).is_ok());
    assert!(FusedStencilStepper::try_new(StepSize(-0.1)).is_err());
}

#[test]
fn stencil_eval_and_biharmonic_try_companions_reject_bad_bounds() {
    let ops = StencilOperators::try_new_2d(StepSize(0.5), StepSize(0.5)).unwrap();
    let u = [0.0_f64; 9];
    assert_eq!(
        ops.try_laplacian_1d_at(&u, 9, 9, BoundaryCondition::Periodic),
        Err(PdeError::IndexOutOfBounds {
            axis: "x",
            index: 9,
            len: 9
        })
    );
    assert_eq!(
        ops.try_eval_biharmonic_2d(&u, 3, 3, 0, 9, BoundaryCondition::Periodic),
        Err(PdeError::IndexOutOfBounds {
            axis: "y",
            index: 9,
            len: 3
        })
    );
    let mut out = [0.0_f64; 4];
    assert!(matches!(
        ops.try_compute_laplacian_2d(&u, 3, 3, BoundaryCondition::Periodic, &mut out),
        Err(PdeError::BufferTooSmall {
            required: 9,
            actual: 4,
            ..
        })
    ));
}

#[test]
fn fft_and_fourier_try_companions_validate_radix2_and_workspace() {
    let mut spec = [0.0_f64; 12];
    assert!(matches!(
        try_fft_1d_inplace(&mut spec, 6, false),
        Err(PdeError::NonRadix2Dimension { size: 6, .. })
    ));
    assert!(matches!(
        try_dealias_1d_spectrum(&mut spec[..2], 4),
        Err(PdeError::BufferTooSmall {
            required: 8,
            actual: 2,
            ..
        })
    ));

    assert_eq!(
        SemiImplicitFourierStepper::try_new_1d(0.0),
        Err(MathError::StrictlyOutOfBounds {
            value: 0.0,
            min: 0.0,
            max: f64::INFINITY,
        })
    );
    let stepper = SemiImplicitFourierStepper::try_new_1d(0.25).unwrap();
    let mut phi = [0.1_f64; 8];
    let mut tiny_buf = [0.0_f64; 4];
    let mut ws = SolverWorkspace::new(&mut tiny_buf);
    let mut rng = OxidizeRng::new(7);
    assert!(matches!(
        stepper.try_step_cahn_hilliard_1d(&mut phi, 1.0, 0.1, 0.0, 0.01, &mut ws, &mut rng),
        Err(PdeError::Math(MathError::WorkspaceTooSmall { .. }))
    ));
}

#[test]
fn spde_noise_and_stencil_steppers_validate_inputs() {
    let ops = StencilOperators::try_new_2d(StepSize(0.5), StepSize(0.5)).unwrap();
    let stepper =
        ExplicitSpdeStencilStepper::new(ops, BoundaryCondition::Periodic, SpdeNoiseKind::None);
    let mut u = [1.0_f64; 4];
    let mut scratch = [0.0_f64; 2];
    let mut ws = SolverWorkspace::new(&mut scratch);
    let mut rng = OxidizeRng::new(11);
    assert_eq!(
        stepper.try_step_1d(&mut u, -0.1, |_, v, _, _| -v, &mut ws, &mut rng),
        Err(PdeError::InvalidParameter { parameter: "dt" })
    );

    let split =
        SplitStepSpdeStencilStepper::new(ops, BoundaryCondition::Periodic, SpdeNoiseKind::None, 2);
    assert!(matches!(
        split.try_step_reaction_diffusion_2d(&mut u, 2, 2, 0.1, 0.01, |v| -v, &mut ws, &mut rng),
        Err(PdeError::Math(MathError::WorkspaceTooSmall { .. }))
    ));

    assert_eq!(
        try_inject_additive_noise_1d(
            &mut u,
            0.0,
            0.01,
            0.1,
            BoundaryCondition::Periodic,
            &mut ws,
            &mut rng
        ),
        Err(PdeError::InvalidParameter { parameter: "dx" })
    );
}

#[test]
fn pde_fft_stencils_and_steppers_support_f32_and_dual_ad() {
    use crate::fft_1d_inplace;
    use math_scalars::Dual;

    // 1. f32 FFT + inverse round-trip
    let orig_f32 = [1.0_f32, 0.0, -0.5, 0.25, 0.75, -0.25, 0.0, 0.5];
    let mut buf_f32 = orig_f32;
    fft_1d_inplace(&mut buf_f32, 4, false);
    fft_1d_inplace(&mut buf_f32, 4, true);
    for (got, exp) in buf_f32.iter().zip(orig_f32.iter()) {
        assert!((got - exp).abs() < 1e-6_f32);
    }

    // 2. f32 FusedStencilStepper + StencilOperators
    let stepper_f32 = FusedStencilStepper::<f32>::new(StepSize(0.5_f32));
    let src_f32 = [1.0_f32, 0.0, 1.0];
    let mut dst_f32 = [0.0_f32; 3];
    stepper_f32.step_1d_slice(
        &src_f32,
        &mut dst_f32,
        0.1_f32,
        1.0_f32,
        |_, prev, curr, next, ops| ops.central_diff_2nd(prev, curr, next),
    );
    assert!((dst_f32[1] - 0.8_f32).abs() < 1e-6_f32);

    // 3. Dual<f64> forward-mode AD wrt diffusivity D on 1D heat step u^{n+1}_i = u_i + dt * D * u_xx
    let d_dual = Dual::variable(0.35_f64);
    let ops_dual = StencilOperators::<Dual<f64>>::new(StepSize(Dual::constant(0.5)));
    let spde_dual =
        ExplicitSpdeStencilStepper::new(ops_dual, BoundaryCondition::Periodic, SpdeNoiseKind::None);
    let mut u_dual = [
        Dual::constant(0.0),
        Dual::constant(1.0),
        Dual::constant(0.0),
        Dual::constant(-1.0),
    ];
    let expected_lap_1 = (0.0 + 0.0 - 2.0 * 1.0) / (0.5 * 0.5); // -8.0
    let dt = 0.02_f64;
    let mut scratch_dual = [Dual::constant(0.0); 8];
    let mut ws_dual = SolverWorkspace::new(&mut scratch_dual);
    let mut rng = OxidizeRng::new(19);
    spde_dual.step_1d(
        &mut u_dual,
        Dual::constant(dt),
        |_, _u, _grad, lap| d_dual * lap,
        &mut ws_dual,
        &mut rng,
    );
    assert!((u_dual[1].re - (1.0 + dt * 0.35 * expected_lap_1)).abs() < 1e-14);
    assert!((u_dual[1].eps - (dt * expected_lap_1)).abs() < 1e-14);
}

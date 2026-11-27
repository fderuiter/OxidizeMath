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

#[cfg(feature = "std")]
#[test]
fn pde_error_preserves_root_math_error_code_and_type() {
    let wrapped = PdeError::from(MathError::WorkspaceTooSmall {
        required: 16,
        available: 4,
    });
    let meta = wrapped.metadata();
    assert_eq!(
        meta.get("code").map(std::string::String::as_str),
        Some("MATH-005")
    );
    assert_eq!(
        meta.get("wrapper_code").map(std::string::String::as_str),
        Some("PDE-007")
    );
    assert_eq!(
        meta.get("root_error_type").map(std::string::String::as_str),
        Some("MathError")
    );
    assert_eq!(
        meta.get("error_type").map(std::string::String::as_str),
        Some("PdeError")
    );
}

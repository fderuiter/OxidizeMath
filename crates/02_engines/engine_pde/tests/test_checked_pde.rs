//! Verifies `#![no_std]`-compatible checked constructors across `engine_pde` under `VerifiedAllocator`.

use engine_pde::{FusedStencilStepper, SemiImplicitFourierStepper, StencilOperators};
use math_commons::error::MathError;
use math_commons::math_kernel::types::StepSize;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn pde_stencil_checked_constructors_reject_invalid_spacings_without_allocating() {
    let (ok, _) = VerifiedEngine::run_verified(|| {
        assert_eq!(
            StencilOperators::try_new(StepSize(0.0)),
            Err(MathError::StrictlyOutOfBounds {
                value: 0.0,
                min: 0.0,
                max: f64::INFINITY,
            })
        );
        assert_eq!(
            StencilOperators::try_new_2d(StepSize(0.1), StepSize(-0.5)),
            Err(MathError::StrictlyOutOfBounds {
                value: -0.5,
                min: 0.0,
                max: f64::INFINITY,
            })
        );
        assert!(matches!(
            StencilOperators::try_new_3d(StepSize(0.1), StepSize(0.1), StepSize(f64::NAN)),
            Err(MathError::StrictlyOutOfBounds { .. })
        ));
        assert_eq!(
            FusedStencilStepper::try_new(StepSize(-1.0)),
            Err(MathError::StrictlyOutOfBounds {
                value: -1.0,
                min: 0.0,
                max: f64::INFINITY,
            })
        );
        assert!(matches!(
            FusedStencilStepper::try_new_3d(StepSize(0.2), StepSize(0.0), StepSize(0.2)),
            Err(MathError::StrictlyOutOfBounds { .. })
        ));
        StencilOperators::try_new_3d(StepSize(0.1), StepSize(0.2), StepSize(0.5)).is_ok()
            && FusedStencilStepper::try_new_2d(StepSize(0.25), StepSize(0.25)).is_ok()
    });
    assert!(ok);
}

#[test]
fn pde_fourier_checked_constructors_reject_invalid_spacings_without_allocating() {
    let (ok, _) = VerifiedEngine::run_verified(|| {
        assert_eq!(
            SemiImplicitFourierStepper::try_new_1d(0.0),
            Err(MathError::StrictlyOutOfBounds {
                value: 0.0,
                min: 0.0,
                max: f64::INFINITY,
            })
        );
        assert_eq!(
            SemiImplicitFourierStepper::try_new_2d(0.25, -0.1),
            Err(MathError::StrictlyOutOfBounds {
                value: -0.1,
                min: 0.0,
                max: f64::INFINITY,
            })
        );
        assert!(matches!(
            SemiImplicitFourierStepper::try_new_2d(0.25, f64::INFINITY),
            Err(MathError::StrictlyOutOfBounds { .. })
        ));
        let checked =
            SemiImplicitFourierStepper::try_new_2d(0.25, 0.5).expect("valid Fourier spacings");
        checked == SemiImplicitFourierStepper::new_2d(0.25, 0.5)
            && SemiImplicitFourierStepper::try_new_1d(0.25).expect("valid 1D spacing")
                == SemiImplicitFourierStepper::new_1d(0.25)
    });
    assert!(ok);
}

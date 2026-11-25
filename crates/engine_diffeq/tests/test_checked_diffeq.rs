// [cite:engine_diffeq]
//! Verifies `#![no_std]`-compatible `BaoabLangevinSolver::try_new` validation under `VerifiedAllocator`.

use engine_diffeq::{BaoabLangevinSolver, BaoabParams};
use math_commons::error::MathError;
#[cfg(feature = "std")]
use verified_engine::allocator::VerifiedAllocator;
#[cfg(feature = "std")]
use verified_engine::engine::VerifiedEngine;

#[cfg(feature = "std")]
#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

fn run_maybe_verified<F: FnOnce() -> bool>(f: F) -> bool {
    #[cfg(feature = "std")]
    {
        VerifiedEngine::run_verified(f).0
    }
    #[cfg(not(feature = "std"))]
    {
        f()
    }
}

#[test]
fn baoab_try_new_rejects_invalid_mass_and_dt_without_allocating() {
    let base = BaoabParams {
        mass: 1.0_f64,
        friction: 0.5,
        temperature: 1.0,
        dt: 0.01,
    };

    let ok = run_maybe_verified(|| {
        assert_eq!(
            BaoabLangevinSolver::try_new(BaoabParams { mass: 0.0, ..base }),
            Err(MathError::StrictlyOutOfBounds {
                value: 0.0,
                min: 0.0,
                max: f64::INFINITY,
            })
        );
        assert!(matches!(
            BaoabLangevinSolver::try_new(BaoabParams {
                mass: f64::INFINITY,
                ..base
            }),
            Err(MathError::StrictlyOutOfBounds { .. })
        ));
        assert_eq!(
            BaoabLangevinSolver::try_new(BaoabParams { dt: 0.0, ..base }),
            Err(MathError::StrictlyOutOfBounds {
                value: 0.0,
                min: 0.0,
                max: f64::INFINITY,
            })
        );
        assert!(matches!(
            BaoabLangevinSolver::try_new(BaoabParams {
                dt: f64::NAN,
                ..base
            }),
            Err(MathError::StrictlyOutOfBounds { .. })
        ));
        let checked = BaoabLangevinSolver::try_new(base).expect("valid BAOAB params");
        checked == BaoabLangevinSolver::new(base)
    });
    assert!(ok);
}

#[test]
fn baoab_try_new_rejects_negative_bath_parameters_without_allocating() {
    let base = BaoabParams {
        mass: 1.0_f64,
        friction: 0.5,
        temperature: 1.0,
        dt: 0.01,
    };

    let ok = run_maybe_verified(|| {
        assert_eq!(
            BaoabLangevinSolver::try_new(BaoabParams {
                friction: -0.1,
                ..base
            }),
            Err(MathError::OutOfBounds {
                value: -0.1,
                min: 0.0,
                max: f64::INFINITY,
            })
        );
        assert_eq!(
            BaoabLangevinSolver::try_new(BaoabParams {
                temperature: -1.0,
                ..base
            }),
            Err(MathError::OutOfBounds {
                value: -1.0,
                min: 0.0,
                max: f64::INFINITY,
            })
        );
        BaoabLangevinSolver::try_new(BaoabParams {
            friction: 0.0,
            temperature: 0.0,
            ..base
        })
        .is_ok()
    });
    assert!(ok);
}

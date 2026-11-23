//! Integration tests for checked APIs and typed `NumberTheoryError` / `MathError`
//! return values across `pure_math`.

use math_commons::error::MathError;
use math_commons::math_kernel::types::{Dimension, StepSize};
use num_complex::Complex64;
use pure_math::error::NumberTheoryError;
use pure_math::math_types::Integer;
use pure_math::pure_math::algebra::fields::Fp;
use pure_math::pure_math::algebra::group::{CyclicElement, Permutation, Zn};
use pure_math::pure_math::algebra::traits::Field;
use pure_math::pure_math::analysis::complex::mapping::SchwarzChristoffel;
use pure_math::pure_math::analysis::complex::surfaces::NthRootSurface;
use pure_math::pure_math::analysis::pde::fused_stepper::{FusedStencilStepper, StencilOperators};
use pure_math::pure_math::number_theory::q_series::QSeries;
use verified_engine::engine::VerifiedEngine;

#[test]
fn integer_modular_and_parsing_apis_return_stack_allocated_errors() {
    assert_eq!(
        Integer::from_str_radix("101", 1),
        Err(NumberTheoryError::InvalidRadix { radix: 1 })
    );
    assert_eq!(
        Integer::from_str_radix("101", 37),
        Err(NumberTheoryError::InvalidRadix { radix: 37 })
    );
    assert_eq!(
        Integer::parse("not_an_integer"),
        Err(NumberTheoryError::InvalidIntegerLiteral { radix: 10 })
    );

    let a = Integer::from(6);
    let m_zero = Integer::from(0);
    let m_nine = Integer::from(9);
    assert_eq!(
        a.pow_mod(&Integer::from(3), &m_zero),
        Err(NumberTheoryError::ZeroModulus)
    );
    assert_eq!(a.invert(&m_zero), Err(NumberTheoryError::ZeroModulus));
    assert_eq!(a.invert(&m_nine), Err(NumberTheoryError::NoModularInverse));
    assert_eq!(
        Integer::from(3).invert(&Integer::from(7)),
        Ok(Integer::from(5))
    );

    let (no_alloc_ok, _) = VerifiedEngine::run_verified(|| {
        Integer::from_str_radix("xyz", 10).is_err() && Integer::parse("bad").is_err()
    });
    assert!(no_alloc_ok);
}

#[test]
fn qseries_checked_div_and_fp_checked_inverse_handle_singular_divisors() {
    let num = QSeries::from_vec(vec![1_i64, 2, 3]);
    let empty: QSeries<i64> = QSeries::new();
    let zero_const = QSeries::from_vec(vec![0_i64, 1, 0]);
    let valid_den = QSeries::from_vec(vec![1_i64, -1, 0]);

    assert_eq!(
        num.checked_div(&empty),
        Err(NumberTheoryError::DivisionByZeroQSeries)
    );
    assert_eq!(
        num.checked_div(&zero_const),
        Err(NumberTheoryError::DivisionByZeroConstantTerm)
    );
    assert_eq!(num.checked_div(&valid_den), Ok(&num / &valid_den));

    let zero = Fp::<7>::new(0);
    let three = Fp::<7>::new(3);
    assert_eq!(
        zero.checked_inverse(),
        Err(NumberTheoryError::DivisionByZeroFieldElement)
    );
    assert_eq!(three.checked_inverse(), Ok(three.multiplicative_inverse()));
}

#[test]
fn group_complex_and_pde_try_constructors_reject_invalid_parameters() {
    assert_eq!(
        CyclicElement::try_new(3, 0),
        Err(NumberTheoryError::ZeroModulus)
    );
    assert_eq!(Zn::<0>::try_new(1), Err(NumberTheoryError::ZeroModulus));

    let c5 = CyclicElement::try_new(2, 5).unwrap();
    let c7 = CyclicElement::try_new(2, 7).unwrap();
    assert_eq!(
        CyclicElement::try_operate(&c5, &c7),
        Err(MathError::DimensionMismatch {
            expected: Dimension(5),
            actual: Dimension(7),
        })
    );

    assert_eq!(
        Permutation::try_new(vec![0, 2, 2]),
        Err(MathError::InvalidPermutation {
            index: 2,
            value: 2,
            len: 3,
        })
    );
    let p2 = Permutation::identity(2);
    let p3 = Permutation::identity(3);
    assert!(matches!(
        Permutation::try_operate(&p2, &p3),
        Err(MathError::DimensionMismatch { .. })
    ));

    assert!(matches!(
        NthRootSurface::try_new(0),
        Err(MathError::StrictlyOutOfBounds { .. })
    ));
    assert!(matches!(
        SchwarzChristoffel::try_new(
            vec![0.0, 1.0],
            vec![0.5],
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ),
        Err(MathError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        StencilOperators::try_new(StepSize(0.0)),
        Err(MathError::StrictlyOutOfBounds { .. })
    ));
    assert!(matches!(
        FusedStencilStepper::try_new_2d(StepSize(0.1), StepSize(-0.1)),
        Err(MathError::StrictlyOutOfBounds { .. })
    ));
}

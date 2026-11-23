//! Integration tests for checked `BoxSet::try_new` and typed GLM `EngineeringError` returns.

use domain_applied::applied::freesurfer::glm::{estimate_beta, t_statistic};
use domain_applied::applied::game_theory::equilibrium::{BoxSet, ConvexSet};
use domain_applied::error::{EngineeringError, GameTheoryError};
use nalgebra::{DMatrix, DVector};

#[test]
fn box_set_try_new_validates_bound_lengths_and_ordering() {
    assert_eq!(
        BoxSet::try_new(vec![0.0, 1.0], vec![1.0]),
        Err(GameTheoryError::BoundsDimensionMismatch {
            min_len: 2,
            max_len: 1,
        })
    );
    assert_eq!(
        BoxSet::try_new(vec![0.0, 2.0], vec![1.0, 1.0]),
        Err(GameTheoryError::InvalidBoxBounds {
            dimension: 1,
            min: 2.0,
            max: 1.0,
        })
    );

    let valid = BoxSet::try_new(vec![-1.0, 0.0], vec![1.0, 2.0]).expect("valid box");
    assert!(valid.contains(&DVector::from_vec(vec![0.0, 1.0])));
}

#[test]
fn glm_estimators_return_typed_engineering_errors() {
    let singular_x = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
    let y2 = DVector::from_vec(vec![1.0, 2.0]);
    let y3 = DVector::from_vec(vec![1.0, 2.0, 3.0]);

    assert!(matches!(
        estimate_beta(&singular_x, &y3),
        Err(EngineeringError::DimensionMismatch {
            expected: 2,
            actual: 3,
            ..
        })
    ));
    assert_eq!(
        estimate_beta(&singular_x, &y2),
        Err(EngineeringError::SingularDesignMatrix)
    );

    let x = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 1.0, 1.0, 1.0, 2.0]);
    let beta = DVector::from_vec(vec![1.0, 0.5]);
    let c = DVector::from_vec(vec![0.0, 1.0]);
    assert_eq!(
        t_statistic(&c, &beta, &x, 0.0),
        Err(EngineeringError::ZeroContrastVariance)
    );
}

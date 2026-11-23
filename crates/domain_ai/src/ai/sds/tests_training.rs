use crate::ai::sds::training::{AdamOptimizer, Optimizer, SgdOptimizer};
use approx::assert_relative_eq;
use math_commons::primitives::{PositiveFloat, UnitInterval};
use nalgebra::DMatrix;

#[test]
#[verified_engine::verified]
fn test_adam_step() {
    let rows = 2;
    let cols = 2;
    let mut optimizer = AdamOptimizer::new(PositiveFloat::new(0.1).unwrap());

    let params = DMatrix::from_element(rows, cols, 0.5);
    let grads = DMatrix::from_element(rows, cols, 0.1);

    // First step
    // m = 0.1 * (1-0.9) = 0.01
    // v = 0.01 * (1-0.999) = 0.00001
    // m_hat = 0.01 / (1-0.9) = 0.1
    // v_hat = 0.00001 / (1-0.999) = 0.01
    // update = 0.1 * 0.1 / (sqrt(0.01) + eps) = 0.01 / 0.1 = 0.1
    // new_params = 0.5 - 0.1 = 0.4

    let new_params_result = optimizer.step(&params, &grads);
    assert!(new_params_result.is_ok());
    let new_params = new_params_result.unwrap();

    // Exact values might differ slightly due to float precision and epsilon
    assert!(new_params[(0, 0)] < 0.5);
    assert_relative_eq!(new_params[(0, 0)], 0.4, epsilon = 1e-4);
}

#[test]
#[verified_engine::verified]
fn test_sgd_step() {
    let rows = 2;
    let cols = 2;
    // SGD with momentum 0.9, lr 0.1
    let mut optimizer = SgdOptimizer::new(
        PositiveFloat::new(0.1).unwrap(),
        UnitInterval::new(0.9).unwrap(),
    );

    let params = DMatrix::from_element(rows, cols, 0.5);
    let grads = DMatrix::from_element(rows, cols, 0.1);

    // First step
    // v = 0.9 * 0 + 0.1 = 0.1
    // theta = 0.5 - 0.1 * 0.1 = 0.49

    let new_params_result = optimizer.step(&params, &grads);
    assert!(new_params_result.is_ok());
    let new_params = new_params_result.unwrap();

    assert_relative_eq!(
        new_params[(0, 0)],
        0.49,
        epsilon = math_commons::registry::TOLERANCE_FAST
    );

    // Second step
    // grads = 0.1
    // v = 0.9 * 0.1 + 0.1 = 0.19
    // theta = 0.49 - 0.1 * 0.19 = 0.49 - 0.019 = 0.471

    let new_params_result_2 = optimizer.step(&new_params, &grads);
    assert!(new_params_result_2.is_ok());
    let new_params_2 = new_params_result_2.unwrap();

    assert_relative_eq!(
        new_params_2[(0, 0)],
        0.471,
        epsilon = math_commons::registry::TOLERANCE_FAST
    );
}

#[test]
#[verified_engine::verified]
fn test_sds_adam_matches_pure_math_adam_and_shares_shape_checks() {
    use pure_math::pure_math::analysis::optimization::{Adam, Optimizer as CoreOptimizer};

    let mut sds_adam = AdamOptimizer::new(PositiveFloat::new(0.05).unwrap());
    let mut core_adam = Adam::<f64, ()>::new(0.05).unwrap();

    let mut sds_params = DMatrix::from_row_slice(2, 2, &[0.8, -0.3, 1.2, 0.05]);
    let mut core_params = sds_params.clone();
    let grads = DMatrix::from_row_slice(2, 2, &[0.25, -0.4, 0.1, -0.15]);

    for step_idx in 1..=5 {
        sds_params = sds_adam.step(&sds_params, &grads).unwrap();
        core_adam
            .update_matrix((), &mut core_params, &grads)
            .unwrap();
        assert_eq!(sds_adam.t, step_idx);
        assert_eq!(sds_params, core_params);
    }

    let bad_grads = DMatrix::zeros(2, 3);
    assert!(sds_adam.step(&sds_params, &bad_grads).is_err());

    let mut sds_sgd = SgdOptimizer::new(
        PositiveFloat::new(0.05).unwrap(),
        UnitInterval::new(0.9).unwrap(),
    );
    assert!(sds_sgd.step(&sds_params, &bad_grads).is_err());
}

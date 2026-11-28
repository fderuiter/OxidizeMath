use super::super::evolution::EvolutionEngine;
use super::*;
use approx::assert_relative_eq;
use oxidize_core::rng::OxidizeRng;

#[test]
fn test_sgd_optimizer_applies_configured_learning_rate_and_validates_inputs() {
    let mut sgd = SGD::new(0.1_f64);
    let mut params = DVector::from_vec(vec![2.0_f64, -1.0_f64]);
    let grads = DVector::from_vec(vec![0.5_f64, -0.25_f64]);

    sgd.update_vector(0_u64, &mut params, &grads)
        .expect("valid SGD vector step");
    assert_relative_eq!(params[0], 1.95, epsilon = 1e-12);
    assert_relative_eq!(params[1], -0.975, epsilon = 1e-12);

    let mut mat = DMatrix::from_row_slice(2, 2, &[2.0_f64, 4.0, -1.0, 0.5]);
    let mat_grad = DMatrix::from_row_slice(2, 2, &[0.5_f64, 1.0, -2.0, 0.25]);
    sgd.update_matrix(1_u64, &mut mat, &mat_grad)
        .expect("valid SGD matrix step");
    assert_relative_eq!(mat[(0, 0)], 1.95, epsilon = 1e-12);
    assert_relative_eq!(mat[(0, 1)], 3.9, epsilon = 1e-12);
    assert_relative_eq!(mat[(1, 0)], -0.8, epsilon = 1e-12);
    assert_relative_eq!(mat[(1, 1)], 0.475, epsilon = 1e-12);
}

#[test]
fn test_sgd_and_adam_return_dimension_mismatch_on_shape_errors() {
    let mut sgd = SGD::new(0.1_f64);
    let mut p_vec = DVector::from_vec(vec![1.0_f64, 2.0, 3.0]);
    let bad_vec_grad = DVector::from_vec(vec![0.5_f64, 0.5]);
    assert_eq!(
        sgd.update_vector(0_u64, &mut p_vec, &bad_vec_grad),
        Err(OptimizationError::DimensionMismatch {
            expected: 3,
            actual: 2,
        })
    );

    let mut p_mat = DMatrix::from_element(2, 3, 1.0_f64);
    let transposed_grad = DMatrix::from_element(3, 2, 0.5_f64);
    assert_eq!(
        sgd.update_matrix(1_u64, &mut p_mat, &transposed_grad),
        Err(OptimizationError::DimensionMismatch {
            expected: 2,
            actual: 3,
        })
    );

    let mut adam = Adam::<f64, u64>::new(0.05).expect("valid adam");
    assert_eq!(
        adam.update_vector(0_u64, &mut p_vec, &bad_vec_grad),
        Err(OptimizationError::DimensionMismatch {
            expected: 3,
            actual: 2,
        })
    );
    assert_eq!(
        adam.update_matrix(1_u64, &mut p_mat, &transposed_grad),
        Err(OptimizationError::DimensionMismatch {
            expected: 2,
            actual: 3,
        })
    );
}

#[test]
fn test_sgd_and_adam_reject_non_finite_or_non_positive_steps() {
    let mut params = DVector::from_vec(vec![2.0_f64, -1.0_f64]);
    let grads = DVector::from_vec(vec![0.5_f64, -0.25_f64]);

    for bad_lr in [-0.1_f64, 0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut bad_sgd = SGD::new(bad_lr);
        assert!(matches!(
            bad_sgd.update_vector(0_u64, &mut params, &grads),
            Err(OptimizationError::InvalidInput { .. })
        ));
        assert!(matches!(
            Adam::<f64, u64>::new(bad_lr),
            Err(OptimizationError::InvalidInput { .. })
        ));
    }
    assert_eq!(params[0], 2.0);

    let mut sgd = SGD::new(0.1_f64);
    let nan_grad = DVector::from_vec(vec![f64::NAN, 0.1]);
    assert!(matches!(
        sgd.update_vector(0_u64, &mut params, &nan_grad),
        Err(OptimizationError::InvalidInput { .. })
    ));

    let mut rng = OxidizeRng::new(42);
    let mut evo_grad = grads.clone();
    assert_eq!(
        sgd.step(&mut params, &mut evo_grad, &mut rng, -0.05),
        Err(EvolutionError::InvalidState)
    );
    assert_eq!(
        sgd.step(&mut params, &mut evo_grad, &mut rng, f64::NAN),
        Err(EvolutionError::InvalidState)
    );
}

#[test]
fn test_adam_optimizer_respects_configured_learning_rate() {
    let mut low_lr = Adam::<f64, usize>::new(0.001).expect("low adam");
    let mut high_lr = Adam::<f64, usize>::new(0.05).expect("high adam");

    let mut p_low = DMatrix::from_element(2, 2, 1.0_f64);
    let mut p_high = DMatrix::from_element(2, 2, 1.0_f64);
    let grad = DMatrix::from_element(2, 2, 0.5_f64);

    low_lr.update_matrix(0, &mut p_low, &grad).expect("low lr");
    high_lr
        .update_matrix(0, &mut p_high, &grad)
        .expect("high lr");

    let delta_low = (1.0 - p_low[(0, 0)]).abs();
    let delta_high = (1.0 - p_high[(0, 0)]).abs();
    assert!((delta_high / delta_low - 50.0).abs() < 1e-6);

    let mut v_low = DVector::from_element(2, 1.0);
    let mut v_high = DVector::from_element(2, 1.0);
    let v_grad = DVector::from_element(2, 0.5);
    low_lr
        .update_vector(1, &mut v_low, &v_grad)
        .expect("low vec");
    high_lr
        .update_vector(1, &mut v_high, &v_grad)
        .expect("high vec");
    assert!(((1.0 - v_high[0]).abs() / (1.0 - v_low[0]).abs() - 50.0).abs() < 1e-6);
}

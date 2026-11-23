#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

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

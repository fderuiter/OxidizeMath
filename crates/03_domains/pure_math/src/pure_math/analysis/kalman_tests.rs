use super::*;

#[derive(Debug)]
struct MockCvModel {
    process_noise: f64,
    measurement_noise: f64,
}

impl KalmanModel<f64> for MockCvModel {
    #[verified_engine::verified]
    fn transition_matrix(&self, dt: f64) -> DMatrix<f64> {
        DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0])
    }

    #[verified_engine::verified]
    fn measurement_matrix(&self) -> DMatrix<f64> {
        DMatrix::from_row_slice(1, 2, &[1.0, 0.0])
    }

    #[verified_engine::verified]
    fn process_noise(&self, _dt: f64) -> DMatrix<f64> {
        DMatrix::identity(2, 2) * self.process_noise
    }

    #[verified_engine::verified]
    fn measurement_noise(&self) -> DMatrix<f64> {
        DMatrix::from_element(1, 1, self.measurement_noise)
    }
}

#[test]
#[verified_engine::verified]
fn test_kalman_predict_logic() -> Result<(), KalmanError> {
    let dt = 1.0;
    let model = MockCvModel {
        process_noise: 0.0,
        measurement_noise: 1.0,
    };
    let x_init = DVector::from_vec(vec![0.0, 10.0]);
    let p_init = DMatrix::identity(2, 2);
    let mut kf = KalmanFilter::builder(model, dt)
        .initial_state(x_init)
        .initial_covariance(p_init)
        .build()?;
    kf.predict();
    assert!((kf.state[0] - 10.0).abs() < math_commons::registry::TOLERANCE_FAST);
    assert!((kf.state[1] - 10.0).abs() < math_commons::registry::TOLERANCE_FAST);
    Ok(())
}

#[test]
#[verified_engine::verified]
fn test_extended_kalman_filter() -> Result<(), KalmanError> {
    struct NonLinearModel;
    impl KalmanSystem<f64> for NonLinearModel {
        #[verified_engine::verified]
        fn predict_state(&self, state: &DVector<f64>, _dt: f64) -> (DVector<f64>, DMatrix<f64>) {
            let x = state[0];
            let x_pred = x.sqrt();
            let f_jacobian = 0.5 / x.sqrt();
            (
                DVector::from_element(1, x_pred),
                DMatrix::from_element(1, 1, f_jacobian),
            )
        }

        #[verified_engine::verified]
        fn predict_measurement(&self, state: &DVector<f64>) -> (DVector<f64>, DMatrix<f64>) {
            let x = state[0];
            let z_pred = x.powi(2);
            let h_jacobian = 2.0 * x;
            (
                DVector::from_element(1, z_pred),
                DMatrix::from_element(1, 1, h_jacobian),
            )
        }

        #[verified_engine::verified]
        fn process_noise(&self, _dt: f64) -> DMatrix<f64> {
            DMatrix::from_element(1, 1, 0.1)
        }

        #[verified_engine::verified]
        fn measurement_noise(&self) -> DMatrix<f64> {
            DMatrix::from_element(1, 1, 0.1)
        }
    }

    let mut kf = KalmanFilter::builder(NonLinearModel, 1.0)
        .initial_state(DVector::from_element(1, 100.0))
        .initial_covariance(DMatrix::from_element(1, 1, 1.0))
        .build()?;
    kf.predict();
    assert!(
        (kf.state[0] - 10.0).abs() < math_commons::registry::TOLERANCE_FAST,
        "Prediction step failed"
    );
    let measurement = DVector::from_element(1, 100.0);
    kf.update(&measurement)?;
    assert!(
        (kf.state[0] - 10.0).abs() < 0.5,
        "Update step diverged: {}",
        kf.state[0]
    );
    Ok(())
}

#[test]
#[verified_engine::verified]
fn test_singular_covariance_error() -> Result<(), KalmanError> {
    struct SingularModel;
    impl KalmanSystem<f64> for SingularModel {
        #[verified_engine::verified]
        fn predict_state(&self, state: &DVector<f64>, _dt: f64) -> (DVector<f64>, DMatrix<f64>) {
            (state.clone(), DMatrix::identity(1, 1))
        }

        #[verified_engine::verified]
        fn predict_measurement(&self, _state: &DVector<f64>) -> (DVector<f64>, DMatrix<f64>) {
            (DVector::from_element(1, 0.0), DMatrix::zeros(1, 1))
        }

        #[verified_engine::verified]
        fn process_noise(&self, _dt: f64) -> DMatrix<f64> {
            DMatrix::zeros(1, 1)
        }

        #[verified_engine::verified]
        fn measurement_noise(&self) -> DMatrix<f64> {
            DMatrix::zeros(1, 1)
        }
    }

    let mut kf = KalmanFilter::builder(SingularModel, 1.0)
        .initial_state(DVector::from_element(1, 0.0))
        .initial_covariance(DMatrix::identity(1, 1))
        .build()?;
    let measurement = DVector::from_element(1, 1.0);
    let result = kf.update(&measurement);
    assert_eq!(result, Err(KalmanError::MatrixInversionError));
    Ok(())
}

#[test]
#[verified_engine::verified]
fn test_dimension_mismatch_and_missing_fields() {
    let model = MockCvModel {
        process_noise: 0.0,
        measurement_noise: 1.0,
    };
    let state = DVector::from_element(2, 0.0);
    let cov = DMatrix::from_element(3, 3, 1.0);
    let result = KalmanFilter::builder(model, 1.0)
        .initial_state(state)
        .initial_covariance(cov)
        .build();
    match result {
        Err(KalmanError::Math(math_commons::error::MathError::DimensionMismatch {
            expected,
            actual,
        })) => {
            assert_eq!(expected.0, 2);
            assert_eq!(actual.0, 9);
        }
        _ => panic!("Expected Math::DimensionMismatch error, got {:?}", result),
    }

    let missing = KalmanFilter::builder(
        MockCvModel {
            process_noise: 0.0,
            measurement_noise: 1.0,
        },
        1.0,
    )
    .build();
    assert!(matches!(
        missing,
        Err(KalmanError::InitializationError(ref field)) if field == "initial_state"
    ));
}

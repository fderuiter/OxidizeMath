//! Integration tests for `ClimateError` and checked constructors/kernels in `domain_climate`.

use diagnostics::{Diagnostic, Severity};
use domain_climate::climate::cera::Cera;
use domain_climate::climate::config::CeraConfig;
use domain_climate::climate::loss::{
    earth_movers_distance, mse_loss, try_earth_movers_distance, try_mse_loss,
};
use domain_climate::climate::tensor_ops::{
    conv_transpose1d, conv1d, try_conv_transpose1d, try_conv1d,
};
use domain_climate::error::ClimateError;
use math_commons::error::MathError;
use nalgebra::{DMatrix, DVector};

fn base_config() -> CeraConfig {
    CeraConfig {
        learning_rate: 1e-3,
        lambda_pred: 1.0,
        lambda_emd: 0.1,
        epochs: 2,
        batch_size: 2,
        in_channels: 3,
        latent_channels: 4,
        aligned_channels: 2,
        num_levels: 4,
        output_size: 2,
    }
}

#[test]
fn cera_constructors_return_typed_climate_errors() {
    let mut bad_align = base_config();
    bad_align.aligned_channels = 6;
    assert!(matches!(
        Cera::new(bad_align),
        Err(ClimateError::ChannelAlignmentExceeded {
            aligned_channels: 6,
            latent_channels: 4,
        })
    ));

    let mut zero_align = base_config();
    zero_align.aligned_channels = 0;
    assert!(matches!(
        Cera::new(zero_align),
        Err(ClimateError::ZeroAlignedChannels)
    ));

    let mut zero_levels = base_config();
    zero_levels.num_levels = 0;
    assert!(matches!(
        Cera::new(zero_levels),
        Err(ClimateError::ZeroLevels)
    ));
}

#[test]
fn checked_losses_and_convolutions_validate_shapes_and_match_unchecked() {
    let a = DMatrix::<f32>::from_element(2, 3, 1.0);
    let b = DMatrix::<f32>::from_element(3, 2, 1.0);
    let c = DMatrix::<f32>::from_element(2, 3, 0.5);

    assert!(matches!(
        try_mse_loss(&a, &b),
        Err(ClimateError::ShapeMismatch {
            expected: (2, 3),
            actual: (3, 2),
            ..
        })
    ));
    assert_eq!(try_mse_loss(&a, &c), Ok(mse_loss(&a, &c)));

    assert!(matches!(
        try_earth_movers_distance(&a, &b),
        Err(ClimateError::ShapeMismatch { .. })
    ));
    assert_eq!(
        try_earth_movers_distance(&a, &c),
        Ok(earth_movers_distance(&a, &c))
    );

    let kernel = DMatrix::<f32>::from_element(4, 3, 0.25);
    let bias4 = DVector::<f32>::from_element(4, 0.1);
    let bias3 = DVector::<f32>::from_element(3, 0.1);

    assert!(matches!(
        try_conv1d(&b, &kernel, &bias4),
        Err(ClimateError::ChannelMismatch {
            expected: 3,
            actual: 2,
            ..
        })
    ));
    assert_eq!(
        try_conv1d(&a, &kernel, &bias4),
        Ok(conv1d(&a, &kernel, &bias4))
    );

    let latent = DMatrix::<f32>::from_element(2, 4, 0.5);
    assert!(matches!(
        try_conv_transpose1d(&a, &kernel, &bias3),
        Err(ClimateError::ChannelMismatch {
            expected: 4,
            actual: 3,
            ..
        })
    ));
    assert_eq!(
        try_conv_transpose1d(&latent, &kernel, &bias3),
        Ok(conv_transpose1d(&latent, &kernel, &bias3))
    );

    let wrapped = ClimateError::from(MathError::OutOfBounds {
        value: 2.0,
        min: 0.0,
        max: 1.0,
    });
    assert_eq!(wrapped.severity(), Severity::Error);
}

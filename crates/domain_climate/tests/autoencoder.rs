#![allow(missing_docs)]
use domain_climate::climate::autoencoder::Autoencoder;
use nalgebra::DVector;
use pure_math::pure_math::analysis::optimization::{SGD, Trainable};

#[test]
#[verified_engine::verified]
fn test_autoencoder_gradient_finite_difference() {
    let mut model = Autoencoder::new(2, 3);
    for i in 0..3 {
        model.encoder.layers[i].kernel.fill(0.05);
        model.encoder.layers[i].bias.fill(0.1);
        model.decoder.layers[i].kernel.fill(0.05);
        model.decoder.layers[i].bias.fill(0.1);
    }

    let x = DVector::from_vec(vec![0.5, 0.3]);
    let target = DVector::from_vec(vec![0.2, 0.8]);

    // Save initial kernels and biases
    let init_enc_k = [
        model.encoder.layers[0].kernel.clone(),
        model.encoder.layers[1].kernel.clone(),
        model.encoder.layers[2].kernel.clone(),
    ];
    let init_enc_b = [
        model.encoder.layers[0].bias.clone(),
        model.encoder.layers[1].bias.clone(),
        model.encoder.layers[2].bias.clone(),
    ];
    let init_dec_k = [
        model.decoder.layers[0].kernel.clone(),
        model.decoder.layers[1].kernel.clone(),
        model.decoder.layers[2].kernel.clone(),
    ];
    let init_dec_b = [
        model.decoder.layers[0].bias.clone(),
        model.decoder.layers[1].bias.clone(),
        model.decoder.layers[2].bias.clone(),
    ];

    let out = Trainable::forward(&model, &x);
    let loss_grad = &out - &target; // gradient of 0.5 * ||out - target||^2

    let lr = 1e-4;
    let mut opt = SGD::new(lr);
    model.backward_update(&x, &loss_grad, &mut opt).unwrap();

    let effective_lr = 0.01;
    let analytical_grad_enc0_k = (&init_enc_k[0] - &model.encoder.layers[0].kernel) / effective_lr;
    let analytical_grad_dec2_k = (&init_dec_k[2] - &model.decoder.layers[2].kernel) / effective_lr;

    // Restore model weights to initial state
    for i in 0..3 {
        model.encoder.layers[i].kernel = init_enc_k[i].clone();
        model.encoder.layers[i].bias = init_enc_b[i].clone();
        model.decoder.layers[i].kernel = init_dec_k[i].clone();
        model.decoder.layers[i].bias = init_dec_b[i].clone();
    }

    let compute_loss = |m: &Autoencoder| -> f32 {
        let pred = Trainable::forward(m, &x);
        0.5 * (&pred - &target).norm_squared()
    };

    let eps = 1e-3;

    // Check layer 0 weight (0, 0)
    model.encoder.layers[0].kernel[(0, 0)] += eps;
    let loss_plus = compute_loss(&model);
    model.encoder.layers[0].kernel[(0, 0)] -= 2.0 * eps;
    let loss_minus = compute_loss(&model);
    model.encoder.layers[0].kernel[(0, 0)] += eps;
    let num_grad_00 = (loss_plus - loss_minus) / (2.0 * eps);

    // Check layer 5 (decoder layer 2) weight (0, 0)
    model.decoder.layers[2].kernel[(0, 0)] += eps;
    let loss_plus_dec = compute_loss(&model);
    model.decoder.layers[2].kernel[(0, 0)] -= 2.0 * eps;
    let loss_minus_dec = compute_loss(&model);
    model.decoder.layers[2].kernel[(0, 0)] += eps;

    let num_grad_dec = (loss_plus_dec - loss_minus_dec) / (2.0 * eps);
    let diff_dec = (analytical_grad_dec2_k[(0, 0)] - num_grad_dec).abs();
    assert!(
        diff_dec < 1e-2,
        "Gradient mismatch at decoder layer 2 (0,0): analytical={}, numerical={}",
        analytical_grad_dec2_k[(0, 0)],
        num_grad_dec
    );

    let diff_00 = (analytical_grad_enc0_k[(0, 0)] - num_grad_00).abs();
    assert!(
        diff_00 < 1e-2,
        "Gradient mismatch at encoder layer 0 (0,0): analytical={}, numerical={}",
        analytical_grad_enc0_k[(0, 0)],
        num_grad_00
    );
}

#[test]
#[verified_engine::verified]
fn test_conv_layer_analytical_gradient() {
    use domain_climate::climate::autoencoder::ConvLayer;
    use nalgebra::DMatrix;

    let mut layer = ConvLayer::new(2, 3);
    layer.kernel.fill(0.2);
    layer.bias.fill(0.1);

    let initial_kernel = layer.kernel.clone();
    let initial_bias = layer.bias.clone();

    let input = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let loss_grad = DMatrix::from_row_slice(2, 3, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);

    let mut opt = SGD::new(0.01);
    let grad_input = layer
        .backward_update(&input, &loss_grad, &mut opt, 0)
        .unwrap();

    assert_eq!(grad_input.nrows(), 2);
    assert_eq!(grad_input.ncols(), 2);
    let expected_grad_in = &loss_grad * &initial_kernel;
    assert!((&grad_input - expected_grad_in).norm() < 1e-5);

    assert_ne!(layer.kernel, initial_kernel);
    assert_ne!(layer.bias, initial_bias);

    let expected_grad_w = (&loss_grad.transpose() * &input) / 2.0;
    let effective_lr = 0.01;
    let actual_grad_w = (&initial_kernel - &layer.kernel) / effective_lr;
    assert!((actual_grad_w - expected_grad_w).norm() < 1e-4);
}

#[test]
#[verified_engine::verified]
fn test_conv_layer_edge_cases() {
    use domain_climate::climate::autoencoder::ConvLayer;
    use nalgebra::DMatrix;

    let mut layer = ConvLayer::new(2, 3);
    let initial_kernel = layer.kernel.clone();
    let initial_bias = layer.bias.clone();
    let mut opt = SGD::new(0.01);

    let empty_input = DMatrix::zeros(0, 2);
    let empty_grad = DMatrix::zeros(0, 3);
    let res = layer.backward_update(&empty_input, &empty_grad, &mut opt, 0);
    assert!(res.is_ok());
    assert_eq!(layer.kernel, initial_kernel);

    let input = DMatrix::from_row_slice(1, 2, &[1.0, 2.0]);
    let nan_grad = DMatrix::from_row_slice(1, 3, &[f32::NAN, 0.0, 1.0]);
    let res_nan = layer.backward_update(&input, &nan_grad, &mut opt, 0);
    assert!(res_nan.is_ok());
    assert_eq!(layer.kernel, initial_kernel);
    assert_eq!(layer.bias, initial_bias);
}

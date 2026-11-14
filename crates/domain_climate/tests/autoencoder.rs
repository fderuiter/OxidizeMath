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

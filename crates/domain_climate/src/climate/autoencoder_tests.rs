#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use nalgebra::DMatrix;

#[test]
fn test_autoencoder_forward_pass_and_zero_alloc_into() {
    let in_channels = 2;
    let latent_channels = 3;
    let num_levels = 30;
    let batch_size = 4;
    let n_samples = num_levels * batch_size;

    let autoencoder = Autoencoder::new(in_channels, latent_channels);
    let mut rng = oxidize_core::rng::OxidizeRng::default();
    let input = DMatrix::from_fn(n_samples, in_channels, |_, _| rng.r#gen());

    let (latent, reconstruction) = autoencoder.forward(&input);
    assert_eq!(latent.shape(), (n_samples, latent_channels));
    assert_eq!(reconstruction.shape(), (n_samples, in_channels));

    let mut scratch = AutoencoderScratch::new(n_samples, in_channels, latent_channels);
    let mut latent_into = DMatrix::zeros(n_samples, latent_channels);
    let mut recon_into = DMatrix::zeros(n_samples, in_channels);

    let (res, metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        autoencoder.forward_into(&input, &mut scratch, &mut latent_into, &mut recon_into)
    });
    res.expect("forward_into should succeed");
    assert_eq!(metrics.allocations, 0);
    assert_eq!(metrics.allocated_bytes, 0);
    assert!(
        (&latent - &latent_into).abs().max() < math_commons::registry::TOLERANCE_FAST_F32,
        "forward_into latent diverged from allocating forward"
    );
    assert!(
        (&reconstruction - &recon_into).abs().max() < math_commons::registry::TOLERANCE_FAST_F32,
        "forward_into reconstruction diverged from allocating forward"
    );
}

#[test]
fn test_leaky_relu() {
    let mut matrix = DMatrix::from_row_slice(2, 2, &[-1.0, 2.0, -3.0, 0.0]);
    let ((), metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        leaky_relu(&mut matrix, 0.1);
    });
    assert_eq!(metrics.allocations, 0);
    let expected = DMatrix::from_row_slice(2, 2, &[-0.1, 2.0, -0.3, 0.0]);
    assert!((matrix - expected).abs().max() < math_commons::registry::TOLERANCE_FAST_F32);
}

#[test]
fn test_autoencoder_backward_update_and_zero_alloc_into() {
    use pure_math::pure_math::analysis::optimization::SGD;
    use pure_math::pure_math::analysis::optimization::Trainable;

    let in_channels = 2;
    let latent_channels = 3;
    let mut autoencoder = Autoencoder::new(in_channels, latent_channels);

    let initial_kernel_0 = autoencoder.encoder.layers[0].kernel.clone();
    let initial_bias_0 = autoencoder.encoder.layers[0].bias.clone();
    let initial_decoder_kernel_2 = autoencoder.decoder.layers[2].kernel.clone();

    let x = nalgebra::DVector::from_vec(vec![1.0, -0.5]);
    let loss_grad = nalgebra::DVector::from_vec(vec![0.1, -0.2]);
    let mut optimizer = SGD::new(0.01);

    let mut scratch = AutoencoderScratch::from_autoencoder(1, &autoencoder);
    let (res, metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        autoencoder.backward_update_into(&x, &loss_grad, &mut optimizer, &mut scratch)
    });
    assert!(res.is_ok());
    assert_eq!(metrics.allocations, 0);
    assert_eq!(metrics.allocated_bytes, 0);

    assert_ne!(autoencoder.encoder.layers[0].kernel, initial_kernel_0);
    assert_ne!(autoencoder.encoder.layers[0].bias, initial_bias_0);
    assert_ne!(
        autoencoder.decoder.layers[2].kernel,
        initial_decoder_kernel_2
    );

    let wrapper_res = autoencoder.backward_update(&x, &loss_grad, &mut optimizer);
    assert!(wrapper_res.is_ok());
}

#[test]
fn test_conv_layer_shared_rng_produces_distinct_channels() {
    let layer = ConvLayer::new(2, 4);
    assert_ne!(layer.kernel.row(0), layer.kernel.row(1));
    assert_ne!(layer.kernel.row(1), layer.kernel.row(2));
    assert_ne!(layer.kernel[(0, 0)], layer.kernel[(0, 1)]);
    assert_ne!(layer.bias[0], layer.bias[1]);

    let autoencoder = Autoencoder::new(2, 4);
    let input = DMatrix::from_row_slice(2, 2, &[0.4, -0.2, 0.8, 0.1]);
    let (latent, recon) = autoencoder.forward(&input);
    assert_ne!(latent.column(0), latent.column(1));
    assert_ne!(recon[(0, 0)], recon[(0, 1)]);
}

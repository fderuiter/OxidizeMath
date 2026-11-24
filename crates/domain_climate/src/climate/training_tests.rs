#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::climate::autoencoder::{ConvLayer, Decoder, Encoder};
use crate::climate::scratch::CeraScratch;
use nalgebra::DMatrix;

fn sample_cera_config() -> CeraConfig {
    CeraConfig {
        learning_rate: 0.01,
        lambda_pred: 0.2,
        lambda_emd: 0.1,
        epochs: 2,
        batch_size: 2,
        in_channels: 2,
        latent_channels: 3,
        aligned_channels: 2,
        num_levels: 4,
        output_size: 2,
    }
}

fn clone_cera_model(src: &Cera<Autoencoder, Predictor>) -> Cera<Autoencoder, Predictor> {
    let clone_conv = |l: &ConvLayer| ConvLayer {
        kernel: l.kernel.clone(),
        bias: l.bias.clone(),
    };
    let enc = Encoder::new_from_layers([
        clone_conv(&src.autoencoder.encoder.layers[0]),
        clone_conv(&src.autoencoder.encoder.layers[1]),
        clone_conv(&src.autoencoder.encoder.layers[2]),
    ]);
    let dec = Decoder::new_from_layers([
        clone_conv(&src.autoencoder.decoder.layers[0]),
        clone_conv(&src.autoencoder.decoder.layers[1]),
        clone_conv(&src.autoencoder.decoder.layers[2]),
    ]);
    let pred = Predictor::new_from_layers([
        clone_conv(&src.predictor.layers[0]),
        clone_conv(&src.predictor.layers[1]),
        clone_conv(&src.predictor.layers[2]),
        clone_conv(&src.predictor.layers[3]),
        clone_conv(&src.predictor.layers[4]),
    ]);
    Cera::new_with_models(
        src.config.clone(),
        Autoencoder::new_from_components(enc, dec),
        pred,
    )
    .expect("matching cloned Cera model")
}

#[test]
fn test_cera_trainer_train_into_zero_alloc_and_matches_wrapper() {
    let config = sample_cera_config();
    let mut model_into = Cera::new(config.clone()).expect("valid Cera config");
    let mut model_wrap = clone_cera_model(&model_into);

    let control_inputs = DMatrix::from_fn(16, 2, |r, c| 0.25 + 0.05 * (r + c) as f32);
    let warm_inputs = DMatrix::from_fn(16, 2, |r, c| 0.45 + 0.04 * (r + c) as f32);
    let control_targets = DMatrix::from_fn(4, 2, |r, c| 0.1 + 0.03 * (r + c) as f32);

    let mut scratch = CeraTrainingScratch::from_config(&config).expect("valid CeraTrainingScratch");
    let mut epoch_losses_into = [0.0_f32; 2];

    let (epochs_ran, metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        let mut trainer = CeraTrainer::new(&mut model_into);
        trainer.train_into(
            &control_inputs,
            &control_targets,
            &warm_inputs,
            &mut scratch,
            &mut epoch_losses_into,
        )
    });
    assert_eq!(epochs_ran.expect("train_into should succeed"), 2);
    assert_eq!(metrics.allocations, 0);
    assert_eq!(metrics.allocated_bytes, 0);

    let mut trainer_wrap = CeraTrainer::new(&mut model_wrap);
    let losses_wrap = trainer_wrap.train(&control_inputs, &control_targets, &warm_inputs);
    assert_eq!(losses_wrap.len(), 2);
    for (a, b) in epoch_losses_into.iter().zip(losses_wrap.iter()) {
        assert!((a - b).abs() < 1e-5, "loss {a} != {b}");
    }
}

#[test]
fn test_cera_predict_into_zero_alloc_and_matches_wrapper() {
    let config = sample_cera_config();
    let model = Cera::new(config.clone()).expect("valid Cera config");
    let eval_batch = DMatrix::from_fn(8, 2, |r, c| 0.2 + 0.05 * (r + c) as f32);
    let mut inf_scratch = CeraScratch::from_config(&config).expect("valid inf scratch");
    let mut pred_out = DMatrix::zeros(2, 2);

    let (pred_res, pred_metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        model.predict_into(&eval_batch, &mut inf_scratch, &mut pred_out)
    });
    pred_res.expect("predict_into should succeed");
    assert_eq!(pred_metrics.allocations, 0);
    assert_eq!(pred_metrics.allocated_bytes, 0);
    let wrap_pred = model.predict(&eval_batch);
    assert!((&pred_out - &wrap_pred).abs().max() < 1e-5);
}

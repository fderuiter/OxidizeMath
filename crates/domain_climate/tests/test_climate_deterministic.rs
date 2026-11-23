//! Cross-platform deterministic tests for `domain_climate` CERA cross-layer RNG threading and datasets.

use domain_climate::climate::autoencoder::{Autoencoder, Decoder, Encoder};
use domain_climate::climate::cera::{Cera, CeraConfig};
use domain_climate::climate::dataset::{
    get_co2_projections_with_rng, get_temperature_anomalies_with_rng,
};
use domain_climate::climate::predictor::Predictor;
use nalgebra::DMatrix;
use oxidize_core::rng::{OxidizeRng, TrajectoryDigest};

const PINNED_CERA_DIGEST: &str = "d079238246ca273d6ac0bfc3388a078f78bc3ead8a606d8e8867d2c626b73916";
const PINNED_CLIMATE_SERIES_DIGEST: &str =
    "b13bdc59fa6435df21651c2558903295172eab4574ad72be82cd5a1bfd00d5a7";

fn sample_config() -> CeraConfig {
    CeraConfig {
        learning_rate: 1e-3,
        lambda_pred: 0.1,
        lambda_emd: 0.05,
        epochs: 1,
        batch_size: 2,
        in_channels: 3,
        latent_channels: 4,
        aligned_channels: 2,
        num_levels: 4,
        output_size: 3,
    }
}

fn collect_kernels(model: &Cera<Autoencoder, Predictor>) -> Vec<Vec<f32>> {
    model
        .autoencoder
        .encoder
        .layers
        .iter()
        .chain(model.autoencoder.decoder.layers.iter())
        .chain(model.predictor.layers.iter())
        .map(|l| l.kernel.as_slice().to_vec())
        .collect()
}

#[test]
#[verified_engine::verified]
fn test_cera_layers_are_pairwise_distinct_and_pinned() {
    let enc = Encoder::new(3, 4);
    let dec = Decoder::new(4, 3);
    assert_eq!(enc.layers[1].kernel.shape(), (64, 64));
    assert_eq!(dec.layers[1].kernel.shape(), (64, 64));
    assert_ne!(
        enc.layers[1].kernel, dec.layers[1].kernel,
        "Encoder::new and Decoder::new must produce distinct 64x64 kernels"
    );

    let pred = Predictor::new(8, 3);
    assert_eq!(pred.layers[1].kernel.shape(), (128, 128));
    assert_ne!(pred.layers[1].kernel, pred.layers[2].kernel);
    assert_ne!(pred.layers[1].kernel, pred.layers[3].kernel);
    assert_ne!(pred.layers[2].kernel, pred.layers[3].kernel);

    let ae = Autoencoder::new(3, 4);
    assert_ne!(ae.encoder.layers[1].kernel, ae.decoder.layers[1].kernel);

    let default_cera = Cera::new(sample_config()).unwrap();
    let all_kernels = collect_kernels(&default_cera);
    assert_eq!(all_kernels.len(), 11);
    for i in 0..all_kernels.len() {
        for j in (i + 1)..all_kernels.len() {
            assert_ne!(
                all_kernels[i], all_kernels[j],
                "Cera::new layer {i} and layer {j} must have distinct kernel weights"
            );
        }
    }

    let batch = DMatrix::from_fn(8, 3, |r, c| ((r * 3 + c) as f32) * 0.05 - 0.3);
    let run_seeded = |seed: u64| -> (Vec<f32>, String) {
        let mut rng = OxidizeRng::new(seed);
        let model = Cera::new_with_rng(sample_config(), &mut rng).unwrap();
        let out = model.predict(&batch);
        let mut digest = TrajectoryDigest::new();
        for k in collect_kernels(&model) {
            digest.update_f32_slice(&k);
        }
        digest.update_f32_slice(out.as_slice());
        (out.as_slice().to_vec(), digest.finish_hex())
    };

    let (p1, h1) = run_seeded(2026);
    let (p2, h2) = run_seeded(2026);
    let (p3, h3) = run_seeded(2027);
    assert_eq!(p1, p2);
    assert_eq!(h1, h2);
    assert_ne!(p1, p3);
    assert_ne!(h1, h3);
    assert_eq!(h1, PINNED_CERA_DIGEST);
}

#[test]
#[verified_engine::verified]
fn test_climate_synthetic_series_pinned_digest() {
    let run = |seed: u64| -> (Vec<[f64; 2]>, String) {
        let mut rng = OxidizeRng::new(seed);
        let temps = get_temperature_anomalies_with_rng(&mut rng);
        let (hist, proj) = get_co2_projections_with_rng(&mut rng, 0.45);
        let mut digest = TrajectoryDigest::new();
        for &[y, v] in temps.iter().chain(hist.iter()).chain(proj.iter()) {
            digest.update_f64(y);
            digest.update_f64(v);
        }
        (temps, digest.finish_hex())
    };

    let (t1, h1) = run(42);
    let (t2, h2) = run(42);
    let (t3, h3) = run(43);
    assert_eq!(t1, t2);
    assert_eq!(h1, h2);
    assert_ne!(t1, t3);
    assert_ne!(h1, h3);
    assert_eq!(h1, PINNED_CLIMATE_SERIES_DIGEST);
}

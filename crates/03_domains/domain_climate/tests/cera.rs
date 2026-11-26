//! Integration tests for cera.
use domain_climate::climate::cera::{Cera, CeraConfig};
use domain_climate::climate::training::CeraTrainer;
use nalgebra::DMatrix;
use rand::Rng;

const IN_CHANNELS: usize = 2;
const LATENT_CHANNELS: usize = 3;
const ALIGNED_CHANNELS: usize = 2;
const NUM_LEVELS: usize = 30;
const OUTPUT_SIZE: usize = 148;

#[verified_engine::verified]
fn generate_synthetic_data(n_samples: usize, offset: f32) -> (DMatrix<f32>, DMatrix<f32>) {
    let inputs = DMatrix::from_fn(n_samples * NUM_LEVELS, IN_CHANNELS, |_, _| {
        oxidize_core::rng::OxidizeRng::default().r#gen::<f32>() + offset
    });
    let targets = DMatrix::from_fn(n_samples, OUTPUT_SIZE, |_, _| {
        oxidize_core::rng::OxidizeRng::default().r#gen()
    });
    (inputs, targets)
}

#[test]
#[verified_engine::verified]
fn test_cera_integration() {
    // 1. Configure the CERA model
    let config = CeraConfig {
        learning_rate: 0.001,
        lambda_pred: 0.1,
        lambda_emd: 0.01,
        epochs: 3, // Use a few epochs for the integration test
        batch_size: 2,
        in_channels: IN_CHANNELS,
        latent_channels: LATENT_CHANNELS,
        aligned_channels: ALIGNED_CHANNELS,
        num_levels: NUM_LEVELS,
        output_size: OUTPUT_SIZE,
    };

    // 2. Initialize the CERA model
    let mut cera = Cera::new(config.clone()).unwrap();

    // 3. Generate synthetic data for training and testing
    let n_train_samples = 8;
    let (control_inputs, control_targets) = generate_synthetic_data(n_train_samples, 0.0);
    let (warm_inputs, _) = generate_synthetic_data(n_train_samples, 2.0); // Make the distribution shift larger

    // 4. Train the model
    // This will print the loss for each epoch to the console.
    let mut trainer = CeraTrainer::new(&mut cera);
    let losses = trainer.train(&control_inputs, &control_targets, &warm_inputs);

    // Verify that training decreases loss values across successive training epochs
    assert!(
        losses.len() >= 2,
        "Training should record at least 2 epoch losses"
    );
    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "Final epoch loss ({:?}) must be lower than initial epoch loss ({:?})",
        losses.last(),
        losses.first()
    );

    // 5. Generate test data and make a prediction
    let n_test_samples = 4;
    let (test_inputs, _original_test_targets) = generate_synthetic_data(n_test_samples, 0.5);
    let prediction = cera.predict(&test_inputs);

    // 6. Assertions to verify the output
    // Check that the prediction has the correct dimensions
    assert_eq!(
        prediction.nrows(),
        n_test_samples,
        "Prediction should have the same number of rows as test samples."
    );
    assert_eq!(
        prediction.ncols(),
        OUTPUT_SIZE,
        "Prediction should have the correct number of output features."
    );

    // Check that the prediction is not just zeros or NaNs
    assert!(
        prediction.iter().all(|&x| x.is_finite()),
        "Prediction contains non-finite values."
    );
    assert!(prediction.norm() > 0.0, "Prediction is a zero matrix.");

    // Optional: Check if the model has learned anything.
    // A simple check is to see if the predictions are different from what a fresh model would predict.
    let fresh_cera = Cera::new(config).unwrap();
    let fresh_prediction = fresh_cera.predict(&test_inputs);

    let difference = (&prediction - &fresh_prediction).abs().sum();
    assert!(
        difference > math_commons::registry::TOLERANCE_FAST_F32,
        "Trained model's prediction should be different from an untrained model's."
    );
}

//! Example target for cera demo.
use math_explorer::climate::cera::Cera;
use math_explorer::climate::config::CeraConfig;
use math_explorer::climate::training::CeraTrainer;
use math_explorer::nalgebra::DMatrix;
use math_explorer::rand::Rng;
use math_explorer::rng::OxidizeRng;

fn main() {
    // 1. Configure the architecture
    let config = CeraConfig {
        in_channels: 2,      // e.g., Temp, Humidity
        latent_channels: 4,  // Compressed state
        aligned_channels: 2, // Invariant state
        num_levels: 10,      // Atmospheric levels
        output_size: 5,      // Prediction target
        epochs: 1,
        batch_size: 2,
        learning_rate: 0.01,
        lambda_pred: 1.0,
        lambda_emd: 0.1,
    };

    // 2. Initialize Model & Trainer
    let mut model = Cera::new(config).expect("Invalid config");
    let mut trainer = CeraTrainer::new(&mut model);

    // 3. Train on synthetic data (Batch Size * Num Levels, Channels)
    let inputs = DMatrix::<f32>::from_fn(20, 2, |_, _| OxidizeRng::default().r#gen());
    let targets = DMatrix::<f32>::from_fn(2, 5, |_, _| OxidizeRng::default().r#gen());
    let warm_inputs = DMatrix::<f32>::from_fn(20, 2, |_, _| OxidizeRng::default().r#gen());

    trainer.train(&inputs, &targets, &warm_inputs);
}

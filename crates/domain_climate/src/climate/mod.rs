//! # Climate Modeling: CERA Framework
//!
//! The `climate` module implements the **Climate-invariant Encoding through Representation Alignment (CERA)**
//! framework. This architecture is designed to improve the generalization of machine learning models
//! across different climate scenarios (e.g., pre-industrial vs. 4xCO2) by learning a latent space
//! that separates climate-specific information from physical dynamics.
//!
//! ## The Problem
//! Traditional ML models trained on one climate scenario often fail to generalize to others because
//! they overfit to the statistical properties (mean, variance) of the training climate.
//!
//! ## The Solution: CERA
//! CERA uses an Autoencoder to compress data into a latent space. It enforces two constraints:
//! 1. **Reconstruction**: The full latent space must reconstruct the input data (preserving physics).
//! 2. **Alignment (EMD)**: A subset of the latent channels ("Aligned Channels") is forced to have
//!    the same statistical distribution across different climates using Earth Mover's Distance (EMD).
//!
//! Only these *aligned channels* are fed into the Predictor, ensuring the prediction logic depends
//! on climate-invariant features.
//!
//! ## Architecture
//!
//! ```mermaid
//! graph TD
//!     subgraph Inputs
//!     IC[Input (Control Climate)]
//!     IW[Input (Warm Climate)]
//!     end
//!
//!     subgraph Autoencoder
//!     Enc[Encoder]
//!     Latent[Latent Space]
//!     Dec[Decoder]
//!     end
//!
//!     subgraph Latent_Structure
//!     Aligned[Aligned Channels]
//!     Unaligned[Unaligned Channels]
//!     end
//!
//!     subgraph Task
//!     Pred[Predictor]
//!     Out[Output Prediction]
//!     end
//!
//!     IC --> Enc
//!     IW --> Enc
//!     Enc --> Latent
//!     Latent --> Aligned
//!     Latent --> Unaligned
//!
//!     Latent --> Dec
//!     Dec --> Recon[Reconstruction]
//!
//!     Aligned --> Pred
//!     Pred --> Out
//!
//!     style Aligned fill:#aaffaa,stroke:#333,stroke-width:2px
//!     style Unaligned fill:#ffaaaa,stroke:#333,stroke-width:2px
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use domain_climate::climate::cera::Cera;
//! use domain_climate::climate::config::CeraConfig;
//! use domain_climate::climate::training::CeraTrainer;
//! use nalgebra::DMatrix;
//! use rand::Rng;
//!
//! // 1. Configure the model
//! let config = CeraConfig {
//!     in_channels: 2,         // e.g., Temperature, Humidity
//!     latent_channels: 4,     // Size of the compressed representation
//!     aligned_channels: 2,    // Channels forced to be climate-invariant
//!     num_levels: 10,         // Vertical levels in the atmosphere
//!     output_size: 5,         // Prediction target size
//!     epochs: 1,
//!     batch_size: 2,
//!     learning_rate: 0.01,
//!     lambda_pred: 1.0,       // Weight for prediction loss
//!     lambda_emd: 0.1,        // Weight for alignment loss
//! };
//!
//! // 2. Initialize the model
//! let mut model = Cera::new(config).expect("Invalid configuration");
//!
//! // 3. Create dummy data (Batch size * Num Levels, Channels)
//! // In reality, this would be your climate simulation data.
//! let inputs = DMatrix::from_fn(20, 2, |_, _| oxidize_core::rng::OxidizeRng::default().r#gen());
//! let targets = DMatrix::from_fn(2, 5, |_, _| oxidize_core::rng::OxidizeRng::default().r#gen());
//! let warm_inputs = DMatrix::from_fn(20, 2, |_, _| oxidize_core::rng::OxidizeRng::default().r#gen()); // Different climate
//!
//! // 4. Train (using the mock optimizer)
//! let mut trainer = CeraTrainer::new(&mut model);
//! trainer.train(&inputs, &targets, &warm_inputs);
//!
//! // 5. Predict
//! let prediction = model.predict(&inputs);
//! println!("Prediction shape: {:?}", prediction.shape());
//! ```

pub mod autoencoder;
pub mod cera;
pub mod config;
pub mod dataset;
pub mod loss;
pub mod predictor;
pub mod tensor_ops;
pub mod training;

// [cite:graph_parameters_rust]

use pure_math::theory_verification;
theory_verification!(
    module = "climate",
    epsilon = math_commons::registry::TOLERANCE_FAST,
    constants = {
        TEST = 1.0;
    },
    test = {}
);

# Domain Climate

Climate modeling, autoencoders, and the Climate-invariant Encoding through Representation Alignment (CERA) framework.

## 1. Overview & Architecture

The `domain_climate` crate implements numerical climate modeling algorithms and the **Climate-invariant Encoding through Representation Alignment (CERA)** framework.

Standard statistical models often fail to generalize across different climate regimes (e.g., pre-industrial vs. elevated CO2 scenarios) due to covariate shift in climatic variables. CERA addresses this challenge by training an autoencoder that isolates climate-invariant physical representations from regime-specific distribution artifacts.

### Module Structure

- **`cera`**: The core `Cera` architecture:
  - Autoencoder network compressing multi-level atmospheric variables into latent spaces.
  - Latent partition into aligned (climate-invariant) and unaligned feature channels.
- **`config`**: `CeraConfig` specifying model topology:
  - Input/output channel counts, vertical atmospheric pressure levels, and latent dimensions.
  - Hyperparameters for prediction loss weight (`lambda_pred`) and alignment loss weight (`lambda_emd`).
- **`dataset`**: Calibrated NASA GISTEMP v4 temperature anomalies (`1880..=2023`), Law Dome / Mauna Loa historical $CO_2$ series, IPCC AR6 carbon-cycle mass-balance projections, and seeded synthetic random-walk generators.
- **`training`**: `CeraTrainer` executing joint training loops:
  - Balances input reconstruction loss with Earth Mover's Distance (EMD) latent alignment penalties.
- **`loss`**: Loss computation routines including mean squared reconstruction error and statistical distribution divergence.
- **`predictor`**: Downstream prediction module operating strictly on aligned, climate-invariant features.
- **`tensor_ops`**: Linear algebra routines optimized for atmospheric column operations.

---

## 2. Developer Instructions

### Initializing and Predicting with the CERA Model

Configure a CERA network and evaluate climate-invariant predictions:

```rust
use domain_climate::climate::cera::Cera;
use domain_climate::climate::config::CeraConfig;
use nalgebra::DMatrix;

// Define CERA network architecture configuration
let config = CeraConfig {
    in_channels: 2,         // e.g. Temperature, Specific Humidity
    latent_channels: 4,     // Compressed latent dimensionality
    aligned_channels: 2,    // Subspace constrained to be climate-invariant
    num_levels: 5,          // Vertical atmospheric layers
    output_size: 3,         // Target prediction dimensions
    epochs: 1,
    batch_size: 2,
    learning_rate: 0.01,
    lambda_pred: 1.0,
    lambda_emd: 0.1,
};

// Instantiate the model
let model = Cera::new(config).expect("Valid CERA model configuration");

// Create synthetic atmospheric input data (batch_size * num_levels, in_channels)
let atmospheric_data = DMatrix::from_element(10, 2, 0.42);

// Generate prediction from invariant latent features
let prediction = model.predict(&atmospheric_data);
assert_eq!(prediction.ncols(), 3);
```

---

## 3. Ecosystem Integration

The CERA framework is documented in `docs/architecture/cera_framework.md` and integrated into `math_explorer` to support interactive climate simulation tabs in `math_explorer_gui`.

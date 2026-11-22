# ADR 0031: Calibrated Climate Reference Series and Shared Channel RNG Initialization

## Status
Accepted (extends [ADR 0013](0013-trainable-climate-autoencoder-backpropagation.md) and [ADR 0030](0030-remove-autoencoder-weight-update-shims.md))

## Context
Two defects in `crates/domain_climate` undermined numerical fidelity in interactive climate simulations:
1. **`ConvLayer::new` Per-Weight Reseeding**: In [`crates/domain_climate/src/climate/autoencoder.rs`](../../crates/domain_climate/src/climate/autoencoder.rs), `ConvLayer::new` instantiated `OxidizeRng::default()` inside the `DMatrix::from_fn` and `DVector::from_fn` closures for every scalar entry. Consequently, all channels and weights in a layer received identical initial values, collapsing multi-channel encoder, decoder, and predictor outputs to identical scalar trajectories.
2. **Uncalibrated Default Climate Series**: In [`crates/domain_climate/src/climate/dataset.rs`](../../crates/domain_climate/src/climate/dataset.rs), `get_temperature_anomalies` and `get_co2_projections` delegated directly to synthetic random-walk generators (`get_temperature_anomalies_with_rng` and `get_co2_projections_with_rng`) instead of returning calibrated climatological series aligned with the NASA GISTEMP v4 (1951-1980 baseline) and IPCC AR6 carbon-cycle mass-balance citations surfaced in the GUI.

## Decision
1. **Shared Deterministic `OxidizeRng` in `ConvLayer::new`**: Instantiate a single `OxidizeRng::default()` per `ConvLayer::new` call and sample Glorot/Xavier-scaled weights and biases sequentially across all kernel and bias entries so every channel starts with distinct, well-conditioned parameters while preserving cross-platform determinism.
2. **Calibrated Default Climate Series with Seeded Synthetic Fallbacks**:
   - Update `get_temperature_anomalies()` to return calibrated annual NASA GISTEMP v4 global land-ocean temperature anomalies (`1880..=2023`) referenced to the 1951-1980 baseline (`mean ≈ 0.00°C`).
   - Update `get_co2_projections(reduction_scenario)` to return Law Dome / Mauna Loa historical $CO_2$ observations (`1850..=2023`) coupled to an IPCC AR6 airborne-fraction / relaxation carbon-cycle mass-balance projection (`2023..=2100`, $C(t) = C_0 + \int_{t_0}^t (E(t') - S(t'))\,dt'$).
   - Retain `get_temperature_anomalies_with_rng` and `get_co2_projections_with_rng` for optional synthetic random-walk comparison in GUI tools, preserving existing public function signatures in `crates/unified_verification/api_baseline.json`.
3. **Honor Configured `learning_rate` in `Adam` Optimizer Updates**:
   - Update `Adam::update_matrix` and `Adam::update_vector` in [`crates/pure_math/src/pure_math/analysis/optimization/mod.rs`](../../crates/pure_math/src/pure_math/analysis/optimization/mod.rs) to pass `dt = 0.0` so `EvolutionEngine::step` uses `self.learning_rate` instead of overriding it with a hardcoded `0.01` step size.

## Consequences
- **Positive**: Multi-channel CERA autoencoders and predictors break symmetry at initialization, enabling distinct channel representations and reliable `cera_loss` reduction under `CeraTrainer::train`.
- **Positive**: Default temperature and $CO_2$ curves match their documented GISTEMP v4 1951-1980 baseline and IPCC AR6 mass-balance equations without introducing jitter on parameter sweeps.
- **Neutral**: Existing callers of `*_with_rng` retain deterministic synthetic random-walk generation behind explicit toggles.

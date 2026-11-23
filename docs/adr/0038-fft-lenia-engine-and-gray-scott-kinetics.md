# ADR 0038: Zero-Allocation 2D FFT Lenia Engine and Promoted Gray-Scott Kinetics

## Status
Accepted

## Context
Previously, `MorphogenesisTool` in `math_explorer_gui/src/tabs/morphogenesis/` coupled 2D pattern formation exclusively to `SchnakenbergKinetics` over 5-point `NeumannBoundary` stencils (`FiniteDifference2D`), while `GrayScottKinetics` existed only inside an integration test (`crates/domain_biology/tests/test_turing_regression.rs`) and `domain_biology` lacked non-local continuous cellular automata (`Lenia`). Simulating smooth concentric-ring `Lenia` kernels (`R` up to `32` cells, `65x65` stencil footprint) via direct spatial convolution requires $O(W H R^2)$ operations per step ($\approx 2.7 \times 10^8$ multiply-adds on a `256x256` torus), exceeding the `60 FPS` (`<= 8 ms`) cooperative WebAssembly frame budget.

## Decision
1. **Spectral 2D `LeniaEngine` (`domain_biology::biology::lenia`)**:
   - Introduced `KernelShells` (`R` in `4..=32`, `1..=4` concentric exponential bump shells normalized to unit $L^1$ spatial sum), `GrowthParams` ($G(u; \mu, \sigma) = 2\exp(-(u-\mu)^2/(2\sigma^2)) - 1$), `LeniaSpecies` (`Orbium`, `Geminium`, `Hydrogeminium`, `Scutium`) seeded deterministically through `OxidizeRng`, and `LeniaEngine`.
   - Pre-allocated all spatial fields, `Complex<f64>` frequency and transpose matrices, `rustfft` scratch buffers, and torus circular-mean trigonometric lookup tables at construction time, and pre-scaled $\hat{K}$ by $1/(W \cdot H)$ whenever kernel parameters change so `LeniaEngine::step` executes in $O(W H \log(W H))$ with **zero dynamic heap allocations** under `verified_engine::allocator::VerifiedAllocator`.
2. **Production `GrayScottKinetics` (`domain_biology::biology::morphogenesis::reaction`)**:
   - Promoted `GrayScottKinetics` (`new`, `mitosis`, `coral`, `ReactionKinetics<2>`, `ReactionModel`, `#[derive(Theory)]`) into `domain_biology::biology::morphogenesis` and wired a zero-allocation `MorphogenesisKineticsAdapter` into `MorphogenesisTool` (`Schnakenberg` Spots/Stripes and `Gray-Scott` Mitosis/Coral modes) while preserving backward compatibility with `PRESET_TURING_SPOTS`, `PRESET_TURING_STRIPES`, and `LegacyMorphoState`.
3. **Interactive `LeniaTool` ("Lenia Studio") (`math_explorer_gui::tabs::morphogenesis::lenia`)**:
   - Registered `LeniaTool` and `StarterPresetMetadata` cards inside `MorphogenesisTab` (`SimulationFramework`) with `AnimatedSlider` parameter sweeps, live mouse deposit/slice brushing, keyboard soliton stamping (`register_canvas_accessibility`, unlocked `Tab` traversal), `FiniteStateGuard` instability recovery, paired `egui_plot` $K(r)$/$G(u)$ and torus telemetry plots, `.png`/`.gif`/`.csv`/`.json` exports, and `< 4 KiB` URL genome presets (`URL_JSON_BOUNDS`).

## Consequences
- **Positive**: Reduces `256x256` Lenia circular convolution from $O(W H R^2)$ to $O(W H \log(W H))$ with zero heap allocations per step and zero new third-party dependencies (`rustfft` and `num-complex` already in workspace `Cargo.toml`).
- **Positive**: Enables side-by-side `SplitHorizontal` comparison of non-local spectral solitons (`LeniaTool`) against local PDE spot division (`MorphogenesisTool` in `Gray-Scott` mode).
- **Verification**: Validated via unit and integration tests in `crates/domain_biology` (`VerifiedAllocator` zero-allocation gate, shift-equivariance against direct circular convolution, `Orbium` translation and brush-cut healing), GUI/URL/export/accessibility contract suites in `math_explorer_gui`, and regenerated `crates/unified_verification/api_baseline.json`.

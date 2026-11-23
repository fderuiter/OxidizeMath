# ADR 0040: Zero-Allocation Spectral Lenia, Developmental Phyllotaxis, and Multi-Tool Biology Playground

## Status
Accepted (extends [ADR 0004](0004-unified-verification-dependency-audit.md) and [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md))

## Context
[`crates/domain_biology`](../../crates/domain_biology/README.md) previously implemented enzyme saturation (`MichaelisMenten`, `HillKinetics`), evolutionary replicator games (`HawkDovePopulation`), generic $N$-species reaction-diffusion (`ReactionDiffusionSystem`), and two-species Schnakenberg morphogenesis, while [`MorphogenesisTab`](../../math_explorer_gui/src/tabs/morphogenesis/mod.rs) exposed only a single Schnakenberg `MorphogenesisTool` and `GrayScottKinetics` remained private to [`crates/domain_biology/tests/test_turing_regression.rs`](../../crates/domain_biology/tests/test_turing_regression.rs). In addition, [`traceability.toml`](../../traceability.toml) mapped `biology` to a placeholder paper (`favorite_child.tex`) rather than a dedicated mathematical biology manuscript, and continuous spatial-temporal artificial life (Lenia) and botanical morphogenesis (Vogel golden-angle phyllotaxis and L-systems) were absent from the domain.

## Decision
1. **Promote `GrayScottKinetics` and Unify Reaction-Diffusion Switching**:
   - Promote `GrayScottKinetics` into [`crates/domain_biology/src/biology/morphogenesis/reaction.rs`](../../crates/domain_biology/src/biology/morphogenesis/reaction.rs) deriving `Theory` and implementing both `ReactionKinetics<2>` and `ReactionModel`, re-exported through `math_explorer::biology::morphogenesis`.
   - Upgrade [`MorphogenesisTool`](../../math_explorer_gui/src/tabs/morphogenesis/tool.rs) and [`MorphogenesisRunner`](../../math_explorer_gui/src/tabs/morphogenesis/runner.rs) to switch seamlessly between `SchnakenbergKinetics` (`TuringSystem`), `GrayScottKinetics` (`TuringSystem`), and `ReactionDiffusionSystem` (`ChemicalState` + `FiniteDifference2D`) while preserving backward-compatible state serialization.
2. **Zero-Allocation Spectral Lenia (`biology::lenia`)**:
   - Implement [`crates/domain_biology/src/biology/lenia/mod.rs`](../../crates/domain_biology/src/biology/lenia/mod.rs) using workspace `rustfft` and `num-complex` with preplanned 1D row/column transforms (`process_with_scratch`), preallocated transpose/spectral/scratch buffers, and precomputed toroidal trigonometric tables so `LeniaSystem::step` performs zero heap allocations under `VerifiedAllocator`.
   - Keep `rustfft` trait objects private to `LeniaSystem` so all public signatures remain reachable through the `math_explorer` facade per [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md).
3. **Developmental Phyllotaxis and Nutrient-Coupled L-Systems (`biology::phyllotaxis`)**:
   - Implement [`crates/domain_biology/src/biology/phyllotaxis/mod.rs`](../../crates/domain_biology/src/biology/phyllotaxis/mod.rs) combining Vogel's golden-angle spiral packing, Euclidean opposing-winding Fibonacci parastichy detection, and bounded double-buffered L-system 2D turtle derivation modulated by `MichaelisMenten` and `HillKinetics`.
4. **Multi-Tool `MorphogenesisTab` Registration and Academic Traceability**:
   - Register `MorphogenesisTool`, [`LeniaTool`](../../math_explorer_gui/src/tabs/morphogenesis/lenia_tool.rs), [`PhyllotaxisLSystemTool`](../../math_explorer_gui/src/tabs/morphogenesis/phyllotaxis_tool.rs), and [`EvolutionKineticsTool`](../../math_explorer_gui/src/tabs/morphogenesis/evolution_kinetics_tool.rs) under `domain: "biology"` with `StarterPresetMetadata` cards, `FiniteStateGuard` recovery, `AccessKit` keyboard-focusable canvas roles, and lossless URL fragment state round-trips.
   - Author [`papers/morphogenesis_and_lenia.tex`](../../papers/morphogenesis_and_lenia.tex) and map `biology = "morphogenesis_and_lenia.tex"` in [`traceability.toml`](../../traceability.toml) while retaining `favoritism = "favorite_child.tex"`.

## Consequences
- **Positive**: `MorphogenesisTab` automatically activates the multi-tool `SimulationFramework` side panel across four biology tools without adding external crate dependencies or increasing heap allocations inside verified step loops.
- **Positive**: Every `domain_biology` submodule (`neuroscience`, `kinetics`, `reaction_diffusion`, `morphogenesis`, `lenia`, `phyllotaxis`, `evolution`, `diffusion`) is now exercised by interactive GUI visualizations, documented in [`crates/domain_biology/README.md`](../../crates/domain_biology/README.md), and formally linked to [`papers/morphogenesis_and_lenia.tex`](../../papers/morphogenesis_and_lenia.tex).
- **Neutral**: Public additions in `domain_biology` require regenerating [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json) via `unified_verification regenerate-baseline`.

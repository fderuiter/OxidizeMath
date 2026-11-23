# ADR 0043: Developmental Phyllotaxis, Nutrient-Coupled L-Systems, and Multi-Tool Biology Playground

## Status
Accepted (extends [ADR 0004](0004-unified-verification-dependency-audit.md), [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md), and [ADR 0042](0042-fft-lenia-engine-and-gray-scott-kinetics.md))

## Context
Following [ADR 0042](0042-fft-lenia-engine-and-gray-scott-kinetics.md), [`crates/domain_biology`](../../crates/domain_biology/README.md) implemented enzyme saturation (`MichaelisMenten`, `HillKinetics`), evolutionary replicator games (`HawkDovePopulation`), generic $N$-species reaction-diffusion (`ReactionDiffusionSystem`), two-species Schnakenberg and Gray-Scott morphogenesis, and spectral 2D `LeniaEngine`, while [`MorphogenesisTab`](../../math_explorer_gui/src/tabs/morphogenesis/mod.rs) hosted `MorphogenesisTool` and `LeniaTool`. However, botanical morphogenesis (Vogel golden-angle phyllotaxis and nutrient-coupled L-systems) and interactive GUI exploration for evolutionary replicator and enzyme saturation kinetics remained absent, and [`traceability.toml`](../../traceability.toml) mapped `biology` to a placeholder paper (`favorite_child.tex`) rather than a dedicated mathematical biology manuscript.

## Decision
1. **Developmental Phyllotaxis and Nutrient-Coupled L-Systems (`biology::phyllotaxis`)**:
   - Implement [`crates/domain_biology/src/biology/phyllotaxis/mod.rs`](../../crates/domain_biology/src/biology/phyllotaxis/mod.rs) combining Vogel's golden-angle spiral packing, Euclidean opposing-winding Fibonacci parastichy detection, and bounded double-buffered L-system 2D turtle derivation modulated by `MichaelisMenten` and `HillKinetics` so `PhyllotaxisState::step` and `LSystemEngine::generate` execute with zero heap allocations under `VerifiedAllocator`.
2. **Multi-Tool `MorphogenesisTab` Expansion and Starter Presets**:
   - Register [`PhyllotaxisLSystemTool`](../../math_explorer_gui/src/tabs/morphogenesis/phyllotaxis_tool.rs) and [`EvolutionKineticsTool`](../../math_explorer_gui/src/tabs/morphogenesis/evolution_kinetics_tool.rs) alongside [`MorphogenesisTool`](../../math_explorer_gui/src/tabs/morphogenesis/tool.rs) and [`LeniaTool`](../../math_explorer_gui/src/tabs/morphogenesis/lenia/mod.rs) under `domain: "biology"` with `StarterPresetMetadata` cards (`gray_scott_coral`, `phyllotaxis_golden_sunflower`, `phyllotaxis_hill_shoot`, `evolution_hawk_dove_ess`), `FiniteStateGuard` recovery, `AccessKit` keyboard-focusable canvas roles, and lossless URL fragment state round-trips.
3. **Academic Traceability and API Baseline**:
   - Author [`papers/morphogenesis_and_lenia.tex`](../../papers/morphogenesis_and_lenia.tex) and map `biology = "morphogenesis_and_lenia.tex"` in [`traceability.toml`](../../traceability.toml) while retaining `favoritism = "favorite_child.tex"`.
   - Regenerate [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json) via `unified_verification regenerate-baseline`.

## Consequences
- **Positive**: `MorphogenesisTab` exposes a complete four-tool mathematical biology suite (`MorphogenesisTool`, `LeniaTool`, `PhyllotaxisLSystemTool`, and `EvolutionKineticsTool`) without duplicating `LeniaEngine` or `GrayScottKinetics` from [ADR 0042](0042-fft-lenia-engine-and-gray-scott-kinetics.md) and with zero heap allocations inside verified step loops.
- **Positive**: Every `domain_biology` submodule (`neuroscience`, `kinetics`, `reaction_diffusion`, `morphogenesis`, `lenia`, `phyllotaxis`, `evolution`, `diffusion`) is exercised by interactive GUI visualizations, documented in [`crates/domain_biology/README.md`](../../crates/domain_biology/README.md), and linked to [`papers/morphogenesis_and_lenia.tex`](../../papers/morphogenesis_and_lenia.tex).
- **Neutral**: Public additions in `domain_biology` require regenerating [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json).

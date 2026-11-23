# ADR 0047: Verified Zero-Allocation Botanical Morphogenesis Kernels and Multi-Scale Biology Studios

## Status
Accepted (extends [ADR 0013](0013-scientific-metadata-documentation-standard.md), [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md), and [ADR 0035](0035-fermi-spatial-games-and-ising-droplet-brushes.md))

## Context
[`crates/domain_biology/src/biology/mod.rs`](../../crates/domain_biology/src/biology/mod.rs) previously lacked botanical morphogenesis primitives (L-systems and phyllotaxis), while leaving `kinetics`, `reaction_diffusion`, and `evolution` unexposed in `math_explorer_gui`. In addition, the default [`ReactionModel::add_reaction_batch`](../../crates/domain_biology/src/biology/reaction_diffusion/traits.rs) allocated two `Vec<f64>` buffers per grid point on every explicit diffusion-reaction step, [`FermiGamePreset::HawkDove`](../../math_explorer_gui/src/tabs/solid_state/ising/preset.rs) hardcoded `[1.0, 0.0, 2.0, -1.0]` payoffs separately from [`HawkDovePopulation`](../../crates/domain_biology/src/biology/evolution.rs), and [`traceability.toml`](../../traceability.toml) mapped `biology` to `favorite_child.tex`.

## Decision
1. **Zero-Allocation Botanical Kernels and Batch Reaction Stepping (`domain_biology` & `math_explorer`)**:
   - Add [`crates/domain_biology/src/biology/lsystem/`](../../crates/domain_biology/src/biology/lsystem/mod.rs) providing `LSystemEngine`, `LSystemGrammar`, `ProductionRule`, and `BotanicalLSystemPreset` over double-buffered symbol and parameter tapes (`MAX_LSYSTEM_SYMBOLS = 65,536`) plus a stack-bounded 2D turtle interpreter (`MAX_BRANCH_STACK_DEPTH = 256`).
   - Add [`crates/domain_biology/src/biology/phyllotaxis/`](../../crates/domain_biology/src/biology/phyllotaxis/mod.rs) providing `VogelPhyllotaxisEngine`, `DivergenceAnglePreset`, and `ParastichyAnalysis` computing Vogel coordinates $(r_n = c\sqrt{n},\, \theta_n = n\alpha)$ up to `MAX_PRIMORDIA = 4,096` and detecting opposing Fibonacci $(F_k, F_{k+1})$ parastichy spiral arms in pre-allocated buffers.
   - Replace per-node `Vec<f64>` allocations in [`ReactionModel::add_reaction_batch`](../../crates/domain_biology/src/biology/reaction_diffusion/traits.rs) with stack-allocated `[0.0_f64; 16]` buffers (`STACK_REACTION_SPECIES_CAP = 16`), add cooperative [`HillSubstrateReaction`](../../crates/domain_biology/src/biology/reaction_diffusion/models.rs), extend [`HawkDovePopulation`](../../crates/domain_biology/src/biology/evolution.rs) with canonical payoff and ESS helpers, and re-export `lsystem` and `phyllotaxis` via `math_explorer::biology`.
2. **Botanical and Biochemical Interactive Studios (`math_explorer_gui`)**:
   - Register [`PhyllotaxisLSystemTool`](../../math_explorer_gui/src/tabs/morphogenesis/phyllotaxis_lsystem/mod.rs) inside `MorphogenesisTab` with AccessKit `Role::GraphicsDocument` focus navigation, WCAG semantic palette rendering, `(F_k, F_{k+1})` spiral overlays, and SVG/PNG/CSV/JSON exports.
   - Add auto-discovered [`BiochemicalTab`](../../math_explorer_gui/src/tabs/biochemical/mod.rs) (`url_slug = "biochemical"`, `academic_domain = "Biology"`) hosting [`BiochemicalDynamicsTool`](../../math_explorer_gui/src/tabs/biochemical/dynamics/mod.rs) (`MichaelisMenten`, `HillKinetics`, `ReactionDiffusionSystem`) and [`EvolutionaryBiologyTool`](../../math_explorer_gui/src/tabs/biochemical/evolution/mod.rs) (`HawkDovePopulation` replicator trajectories, $p^* = V/C$ ESS, and phase line), and delegate `FermiGamePreset::HawkDove` payoffs to `HawkDovePopulation::new(2.0, 4.0)`.
3. **Manuscript and Traceability Alignment**:
   - Author [`papers/computational_biology.tex`](../../papers/computational_biology.tex) and map `biology`, `biochemical`, and `computational_biology` to `computational_biology.tex` in [`traceability.toml`](../../traceability.toml).

## Consequences
- **Positive**: `VerifiedAllocator` integration tests verify zero dynamic heap allocations during `LSystemEngine::step_generation`, `LSystemEngine::interpret_turtle`, `VogelPhyllotaxisEngine::compute`, `VogelPhyllotaxisEngine::analyze_parastichies`, and `ReactionDiffusionSystem::step`.
- **Positive**: All public submodules of `domain_biology::biology` are interactively accessible across desktop and `wasm32-unknown-unknown` with `StarterPresetMetadata` cards, `URL_JSON_BOUNDS` stateless share links, and full paper traceability.
- **Neutral**: Full 3D mesh extrusion of botanical trees and stochastic context-sensitive L-system grammars remain deferred to preserve compile-time buffer caps and the `< 3 MiB` compressed WASM budget.

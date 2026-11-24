# ADR 20261123: Three-Tier Maturity Qualification Gates, SemVer Signature Locks, and Accessible Badge Surfaces

## Status

Accepted

## Context

OxidizeMath unifies hardened zero-allocation numerical solvers (such as Levenberg-Marquardt curve fitting and RK4/Dormand-Prince ODE integrators) alongside stochastic, representation-learning, and empirical research laboratories (such as `Cera` climate encoding and `favoritism` referee bias modeling) inside a single lockstep workspace (`[workspace.package].version = "0.4.0"`). Prior to this decision:

1. `scientific_metadata::theory::TheoryDescribable` and `#[derive(Theory)]` carried mathematical descriptions, LaTeX equations, citations, and parameter bounds, but lacked a formal maturity classification distinguishing mission-critical deterministic kernels from exploratory models.
2. `traceability.toml` mapped modules to LaTeX manuscripts (`[links]`) without recording per-crate or per-module qualification tiers or baseline SemVer locks.
3. `unified_verification` enforced uniform public API baseline checks across all crates without allowing fluid signature iteration in `Experimental` modules or locking `Certified` public signatures against unversioned breaking removals.
4. Rustdoc module headers, workspace/crate README matrices, `SimulationFramework` tool selectors, `Theory Context Portal` panels, and `TraceabilityTab` provided no visual or screen-reader signal of verification maturity.

## Decision

1. **`MaturityTier` Metadata Contract (`scientific_metadata` & `verified_engine_macros`)**:
   - Define `MaturityTier` (`#[default] Experimental`, `Validated`, `Certified`) in `scientific_metadata::theory` with `Ord`, `Display`, `FromStr`, `serde` serialization, badge labels (`[EXP]`/`[EXPERIMENTAL]`, `[VAL]`/`[VALIDATED]`, `[CERT]`/`[CERTIFIED]`), and WCAG screen-reader labels.
   - Add `fn maturity_tier(&self) -> MaturityTier { MaturityTier::Experimental }` to `TheoryDescribable` and optional `maturity = "Experimental" | "Validated" | "Certified"` support on `#[derive(Theory)]` while preserving 100% backwards compatibility with existing `#[verified]` and `#[theory(...)]` attributes.
   - Classify core deterministic solvers (`FitModel`, `OdeModel`, `OdeSolverTool`) as `MaturityTier::Certified` and research sandboxes (`Cera`, `CeraTool`, `FavoritismInputs`, `FavoritismTheory`) as `MaturityTier::Experimental`.

2. **Declarative Qualification Registry (`traceability.toml` & `oxidize_core`)**:
   - Extend `traceability.toml` with `[semver_lock] certified_baseline_version = "0.4.0"`, `[crates]` mapping all 18 mathematical workspace crates, and `[modules]` mapping every registered `[links]` module key to an explicit `MaturityTier`, enforcing monotonic containment (`module_tier <= crate_tier` and `item_tier <= crate_tier`).
   - Extend `TraceabilityReport` with serializable `crate_maturity_tiers`, `module_maturity_tiers`, `deterministic_contract_modules`, `zero_alloc_modules`, and `maturity_violations`.

3. **Single-Pass AST Qualification Gates (`oxidize_core::ast_visitor` & `xtask check-maturity`)**:
   - Extract tier evidence inside the existing `ParsedWorkspaceAstCache` single-parse batch (`crates/unified_verification/src/batch.rs`) without duplicate `syn::parse_file` passes:
     - `Certified`: requires non-vacuous `theory_verification!` deterministic contracts, `VerifiedAllocator` + `VerifiedEngine::run_verified` zero-allocation proofs, active `#[verified]` kernels, and zero stochastic RNG usage.
     - `Validated`: requires non-vacuous `theory_verification!` or golden regression fixtures, deterministic RNG seeding whenever RNG primitives appear, and zero unseeded entropy sources.
     - `Experimental`: permits exploratory APIs while still verifying basic compilation and absence of `todo!`/`unimplemented!` stubs.
   - Expose `cargo run -p xtask -- check-maturity` (`unified_verification check-maturity`) and execute it automatically inside `batch-static` (`lint-all`). Any attempt by `cera` or `favoritism` to claim `Certified` without deterministic and zero-allocation proofs fails the gate immediately.

4. **Tier-Aware SemVer Signature Locks (`unified_verification::api_drift`)**:
   - Resolve file-level maturity tiers from `traceability.toml` during `check-api-drift` and `regenerate-baseline`:
     - `Experimental` modules permit fluid public item additions, signature refinements, and removals without failing `check-api-drift`.
     - `Certified` modules enforce strict SemVer signature locks: removing or mutating a `Certified` public signature fails `check-api-drift` (`SemVer Lock Violation [Certified]`), and `regenerate-baseline` refuses to overwrite removed `Certified` signatures unless `[workspace.package].version` advances beyond `certified_baseline_version` or `--allow-breaking-certified` is explicitly supplied.

5. **Multi-Surface Badges & Accessibility (`math_explorer_gui` & Documentation)**:
   - Document tier criteria and crate/module maturity matrices across root `README.md`, `math_explorer/README.md`, crate `README.md` files, and module rustdoc headers.
   - Render `SemanticPalette` maturity badges meeting WCAG 2.2 AA contrast (`>= 4.5:1` across Dark, Light, and High-Contrast themes) with `AccessKit` `Role::Label` nodes and `AccessibleHoverText` live-region announcements across `SimulationFramework` tool selectors, `Theory Context Portal`, `StarterCard`/`ChallengeCard` headers, and `TraceabilityTab` on both native desktop and `wasm32-unknown-unknown` builds.

## Consequences

- Downstream consumers and interactive GUI users can immediately distinguish `Certified` deterministic zero-allocation solvers from `Validated` numerical models and `Experimental` research sandboxes across code, docs, and UI surfaces.
- `Experimental` modules can iterate on domain APIs without friction, whereas `Certified` public contracts are mechanically protected against accidental breaking drift under lockstep workspace versioning.
- All AST qualification checks Piggyback on `ParsedWorkspaceAstCache`, maintaining zero duplicate file parses in `lint-all` and CI pre-flight.

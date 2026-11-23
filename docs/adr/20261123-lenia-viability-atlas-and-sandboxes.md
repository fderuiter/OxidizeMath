# ADR 20261123: Downsampled 2D FFT Lenia Soliton Viability Atlas, Potential Histograms, and AST Rule Sandboxes

## Status
Accepted

## Context
Following [ADR 20261123: Zero-Allocation 2D FFT Lenia Engine and Promoted Gray-Scott Kinetics](20261123-fft-lenia-engine-and-gray-scott-kinetics.md), `LeniaTool` exposed unguided 1D `(mu, sigma)` sliders without a 2D `ParameterAtlasController` map, left `LeniaEngine::potential()` unread by the inspector plots, hardcoded Gaussian growth $G(u)$ and exponential bump kernels $K(r)$, and omitted `InteractiveTool` observables and challenges. Across a 400-point `(mu, sigma)` sweep around `Orbium`, roughly 70% of slider combinations drive solitons to extinction or trigger vacuum/domain explosion because localized survival requires aligning the growth peak $\mu$ with the non-local convolution range $U = A * K \in [0, 0.225]$ while keeping $G(0) \le 0$.

## Decision
1. **Stack-Buffered `32x32` Radix-2 2D FFT Viability Probe (`domain_biology::biology::lenia::fft_probe`) & `LeniaSpectralCartographer`**:
   - Added a zero-allocation `32x32` (`1024`-element `Complex<f64>` stack arrays) Cooley-Tukey radix-2 2D FFT stepper (`PROBE_STEPS = 24`) that short-circuits $G(0) > 0$ as `VacuumExplosion` and classifies `Extinction`, `LocalizedSoliton`, and `OvergrownBlob`.
   - Wired `LeniaSpectralCartographer` (`RegimeCartographer`) into `ParameterAtlasController` over `("mu", "sigma")` and `("radius", "mu")` axes, precomputing the downsampled seed state and invariant $\hat{K}$ spectrum per sweep, overlaying the analytical/bisected $G(0) = 0$ stability locus ($\mu = \sqrt{2 \ln 2}\,\sigma$ for Gaussian growth), and auto-reseeding washed-out grids (`total_mass < 0.25`) upon `AtlasTeleportEvent` into `Localized Soliton`.
2. **Live Neighborhood Potential $U = A * K$ Inspector Overlay**:
   - Synchronized `LeniaEngine::potential` eagerly upon seeding, brushing, and kernel updates (`compute_potential_in_place`, `max_potential`) and overlaid a 32-bin active-cell `U = A * K` histogram, shaded `[u_p05, max(U)]` band, and quantile markers onto the `Growth G(u)` plot and exported datasets.
3. **`CompiledExpr::eval_1d` Growth & Radial Kernel Sandboxes (`domain_biology::biology::lenia::kernel`)**:
   - Added `CompiledGrowthRule` (clamping `G(u)` to `[-1, 1]` with `-1.0` `NaN` fallback) and `CompiledKernelRule` (normalizing $K(r)$ on `[0, 1]` to unit $L^1$ torus mass via `write_normalized_torus_kernel`), backed by inline `[f64; 8]` parameter buffers so `LeniaEngine::step` and `evaluate_cell` perform zero heap allocations under `VerifiedAllocator`.
   - Embedded `MathExpressionInputBar` sandboxes for `G(u)` (`with_axes(&["u"])`) and `K(r)` (`with_axes(&["r"])`) with Chan polynomial and smooth-step presets, transient-edit isolation via `last_valid_compiled`, and `< 4 KiB` URL state serialization.
4. **Guided Soliton Stories & Regeneration/Fission Challenges**:
   - Implemented `observables`, `sync_callouts`, `apply_challenge`, `active_challenge_id`, and `evaluate_challenge` on `LeniaTool`, registering `"lenia_soliton_viability_tour"` (`PresetStoryMetadata`), `"lenia_geminium_fission"` (`GoalChallengeMetadata`), and the locked-kernel `"lenia_orbium_tail_regeneration"` (`ChallengeMetadata`).

## Consequences
- **Positive**: Makes narrow continuous-soliton viability islands and `U = A * K` growth-niche alignment directly visible and clickable within the `8 ms` progressive `AtlasRunner` slice budget with zero per-step or per-cell heap allocations.
- **Positive**: Unifies custom polynomial and smooth-step artificial-life rule authoring with `math_commons::ast` bytecode and federated `StarterGallery` stories and challenges.
- **Verification**: Validated via `VerifiedAllocator` zero-allocation tests in `domain_biology` and `math_explorer_gui`, URL state & locked-challenge tamper clamping tests, `unified_verification` static/AST/doc/traceability gates, and `wasm32` target checks.

# ADR 20261123: 2D Cut-and-Project Quasicrystal and Reciprocal Bragg Studio

## Status

Accepted (2026-11-23)

## Context

Previously, [`crates/domain_physics/src/physics/solid_state/mod.rs`](../../crates/domain_physics/src/physics/solid_state/mod.rs) relied on a placeholder `DUMMY = 1.0` verification anchor and modeled only periodic direct-space crystal cells (`lattice.rs`), 1D Bloch bands, phonons, screening, and Ising lattices. Higher-dimensional cut-and-project quasicrystals ($\mathbb{Z}^N \to E_\parallel \oplus E_\perp$), reciprocal-space Wigner-Seitz First Brillouin Zones across all five 2D Bravais nets, perpendicular phason flips, and zero-allocation 2D Bragg diffraction maps were absent from both `domain_physics` and `math_explorer_gui`.

## Decision

1. **Zero-Allocation 2D Cut-and-Project Hyperlattice Engine (`domain_physics::physics::solid_state::quasicrystal`):**
   - Added [`quasicrystal.rs`](../../crates/domain_physics/src/physics/solid_state/quasicrystal.rs) with submodules [`penrose_ammann.rs`](../../crates/domain_physics/src/physics/solid_state/quasicrystal/penrose_ammann.rs) and [`stampfli.rs`](../../crates/domain_physics/src/physics/solid_state/quasicrystal/stampfli.rs).
   - `QuasicrystalWorkspace::project_into` projects 5D Penrose ($N = 5$, $C_5 / D_{10}$, golden ratio $\varphi = \frac{1 + \sqrt{5}}{2} \approx 1.618034$ using de Bruijn's non-singular $\gamma_0 = 0.12$ decagonal cross-section), 4D Ammann-Beenker ($N = 4$, $D_8$, silver ratio $\delta_S = 1 + \sqrt{2} \approx 2.414214$ octagonal window), and 6D Stampfli-Gähler ($N = 6$, $D_{12}$, dodecagonal ratio $2 + \sqrt{3} \approx 3.732051$ over triangle, square, and hexagonal shield prototiles) into preallocated buffers with zero heap allocations.
   - Tracks perpendicular-space phason shifts $\mathbf{w} = (w_x, w_y) \in E_\perp$ against a reference $\mathbf{w}_0$ so prototiles entering or leaving the acceptance window are flagged via `phason_flipped`.

2. **2D Bravais Nets, First Brillouin Zones, and Pre-Planned `256x256` Bragg Diffraction (`domain_physics::physics::solid_state::reciprocal`):**
   - Added [`reciprocal.rs`](../../crates/domain_physics/src/physics/solid_state/reciprocal.rs) and [`diffraction.rs`](../../crates/domain_physics/src/physics/solid_state/reciprocal/diffraction.rs).
   - `BravaisLattice2D` and `Lattice2DParams` construct all five crystallographic 2D Bravais nets (`Oblique`, `Rectangular`, `CenteredRectangular`, `Square`, `Hexagonal`), exact reciprocal basis vectors satisfying $\mathbf{a}_i \cdot \mathbf{b}_j = 2\pi \delta_{ij}$, and stack-allocated Wigner-Seitz `BrillouinZone2D` polygons via half-plane clipping over bounded reciprocal shells.
   - `BraggDiffractionWorkspace` pre-plans row/column `rustfft` transforms once at construction and executes `256x256` Hann-tapered FFTs, sub-pixel parabolic Bragg peak refinement with exact kinematic structure factors $S(\mathbf{k}) = \frac{1}{N}\sum_j e^{-i\mathbf{k}\cdot\mathbf{r}_j}$, separable detector PSF smoothing, and rotational symmetry correlations $C_4$ through $C_{12}$ with zero dynamic heap allocations.
   - Replaced `DUMMY = 1.0` in [`crates/domain_physics/src/physics/solid_state/mod.rs`](../../crates/domain_physics/src/physics/solid_state/mod.rs) with golden, silver, Stampfli, reciprocal-orthogonality, and hexagonal Brillouin-zone area invariants.

3. **Interactive `QuasicrystalStudio` in `SolidStateTab` (`math_explorer_gui`):**
   - Added [`quasicrystal_studio.rs`](../../math_explorer_gui/src/tabs/solid_state/quasicrystal_studio.rs) and its submodules (`presets.rs`, `ui.rs`, `canvas.rs`, `tests.rs`, `wasm_tests.rs`), registered via `ToolMetadata` and four `StarterPresetMetadata` gallery entries without modifying [`crystal_viewer.rs`](../../math_explorer_gui/src/tabs/solid_state/crystal_viewer.rs) or [`band_structure.rs`](../../math_explorer_gui/src/tabs/solid_state/band_structure.rs).
   - Provides dual direct-space and reciprocal-space `256x256` canvases, `PlaybackTransport` phason drift, an interactive $E_\perp$ acceptance-window gizmo, AccessKit keyboard navigation (`Arrow` keys shift $\mathbf{w}$ while leaving `Tab` unlocked), `#[serde(deny_unknown_fields)]` state bounded by `URL_JSON_BOUNDS`, `FiniteStateGuard` recovery, and deferred `.svg`, `.png`, `.csv`, and `.json` exports.
   - Aligned with `CacheableOriginHandler` in [`scripts/playwright_smoke.py`](../../scripts/playwright_smoke.py) negotiating pre-compressed `.br` / `.gz` `.wasm` wire responses via `Accept-Encoding` (`Vary: Accept-Encoding`) and [`scripts/verify_wasm_budget.sh`](../../scripts/verify_wasm_budget.sh) enforcing the `<= 4 MB` (`4,194,304` bytes) compressed `.wasm` bundle budget across both Brotli and Gzip streams.

## Consequences

- Penrose, Ammann-Beenker, and Stampfli tile/geometry ratios match theoretical golden (`1.618034`), silver (`2.414214`), and dodecagonal (`3.732051`) constants well within `1%`.
- `256x256` diffraction maps satisfy $C_{10} \ge 0.99$ (Penrose), $C_8 \ge 0.99$ (Ammann-Beenker), and $C_{12} \ge 0.98$ (Stampfli), contrasting crystallographic 2-, 4-, and 6-fold Bravais symmetries.
- Live phason shifts and `256x256` 2D `rustfft` passes execute with zero dynamic heap allocations inside `VerifiedEngine::run_verified` on native and `wasm32-unknown-unknown` targets.

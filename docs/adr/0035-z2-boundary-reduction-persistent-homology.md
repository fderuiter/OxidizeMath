# ADR 0035: Exact $\mathbb{Z}_2$ Boundary Reduction for Persistent Homology and Client-Side TDA Workbench

## Status
Accepted (extends [ADR 0020](0020-precompute-distance-matrix-tda-filtration.md))

## Context
Three limitations prevented [`pure_math::statistics::tda`](../../crates/pure_math/src/pure_math/statistics/tda/mod.rs) and [`VietorisRipsTool`](../../math_explorer_gui/src/tabs/geometry_topology/vietoris_rips/mod.rs) from computing faithful persistent homology on nontrivial point clouds:
1. **Euler-Heuristic $\beta_1$ Over-Subtraction**: In [`homology.rs`](../../crates/pure_math/src/pure_math/statistics/tda/homology.rs), `betti_number_1` estimated 1-cycles via `(e - v + beta0).saturating_sub(t)`. Whenever a complex contained a $K_4$ clique (6 edges, 4 triangles, $\operatorname{rank}_{\mathbb{Z}_2}(\partial_2) = 3$), the redundant fourth face subtracted an extra degree of freedom and erased independent 1-cycles elsewhere in the complex.
2. **Aggregate LIFO Persistence Pairing**: In [`persistence.rs`](../../crates/pure_math/src/pure_math/statistics/tda/persistence.rs), `compute_persistence` stepped across caller-supplied `radii`, pushed anonymous birth timestamps onto a LIFO stack whenever $\Delta \beta_k > 0$, and popped the newest birth whenever $\Delta \beta_k < 0$. Nested or bridged loops swapped lifetimes (violating the Elder Rule), and simultaneous birth/death transitions within a step ($\Delta \beta_k = 0$) went undetected.
3. **20-Point 2D GUI Ceiling Without Local Import or Persistence Views**: `VietorisRipsTool` hard-coded 20 uniform random 2D points without file import, N-D ambient distance support, live $\beta_0$/$\beta_1$ readouts, persistence diagrams/barcodes, or interval exports.

## Decision
1. **Bounded 2D and $N$-D Point Clouds with Precomputed Distance Reuse**:
   - Define `MAX_TDA_POINTS = 250`, `MAX_TDA_DIMENSION = 32`, and `NdPointCloud` in [`core.rs`](../../crates/pure_math/src/pure_math/statistics/tda/core.rs), computing $\mathbb{R}^d$ Euclidean distance matrices once per cloud (preserving [ADR 0020](0020-precompute-distance-matrix-tda-filtration.md)) and projecting chosen coordinate axes to 2D for complex rendering.
   - Enforce `MAX_FILTRATION_TRIANGLES = 100_000` in [`complex.rs`](../../crates/pure_math/src/pure_math/statistics/tda/complex.rs) and [`persistence.rs`](../../crates/pure_math/src/pure_math/statistics/tda/persistence.rs) via streaming `select_nth_unstable_by` selection so dense 250-point filtrations remain responsive on desktop and `wasm32-unknown-unknown`.
2. **Zero-Allocation Bit-Packed $\mathbb{Z}_2$ Boundary Reduction**:
   - Introduce [`Z2BoundaryWorkspace`](../../crates/pure_math/src/pure_math/statistics/tda/boundary.rs) packing boundary columns into preallocated `u64` words.
   - Perform left-to-right $\mathbb{Z}_2$ column elimination (`XOR` over packed words) inside `Z2BoundaryWorkspace::reduce` with zero dynamic heap allocations so `VerifiedAllocator` post-initialization memory locks pass without exemptions.
   - Replace raw triangle subtraction in `betti_number_1` with `(e + beta0).saturating_sub(v) - rank_z2(partial_2)`.
3. **Exact Simplicial `(birth, death)` Pairing with Twist Clearing**:
   - Order 0-, 1-, and 2-simplices by `(diameter, dimension, vertices)`, reduce $\partial_2$ first, clear $\partial_2$ pivot columns in $\partial_1$ ($\partial_1 \circ \partial_2 = 0$), and reduce $\partial_1$ to pair exact $H_0$ and $H_1$ intervals in `compute_rips_persistence`, `compute_rips_persistence_nd`, and `compute_rips_persistence_from_distances`.
   - Map exact simplicial intervals onto caller `radii` in `compute_persistence` while preserving existing public signatures.
4. **Modular Client-Side Vietoris-Rips TDA Workbench**:
   - Split `vietoris_rips.rs` into [`vietoris_rips/{mod,io,plots,presets,tests}.rs`](../../math_explorer_gui/src/tabs/geometry_topology/vietoris_rips/mod.rs) ($\le 500$ lines per file).
   - Support zero-upload `.csv`, `.tsv`, and `.json` point-cloud import (`MAX_IMPORT_BYTES = 524_288`, UTF-8 BOM stripping, header/jagged-row/finiteness validation) and `.csv`/`.json` interval downloads (`dimension,birth,death,persistence,is_essential`) via [`web_io`](../../math_explorer_gui/src/web_io.rs).
   - Render synchronized dual `egui_plot` views (2D simplicial complex + $H_0$/$H_1$ persistence diagram or barcode with $y = x$ diagonal and active $\epsilon$ overlay), offscreen `.svg`/`.png` `ExportPrimitive` generation, and compact URL sharing (`GeneratorDescriptor` or $\le 64$ quantized 2D points under `4_096` bytes; full 250-point clouds in `2 MiB` `.json`/`.mws` states).

## Consequences
- **Mathematical Exactness**: Retains true 1-cycles in the presence of $K_4$ cliques, eliminates LIFO interval swapping on nested/bridged loops, and captures simultaneous transitions across coarse radius grids.
- **Memory Determinism & Safety**: `Z2BoundaryWorkspace::reduce` allocates zero heap bytes under `VerifiedEngine::run_verified`, while bounded triangle selection and strict file/URL limits prevent browser memory exhaustion.
- **Backward Compatibility**: Preserves all existing `pure_math::statistics::tda` signatures, 55-tool / 59-starter registration invariants, and the 32-tool `ExportCapabilities::plot()` figure export contract.

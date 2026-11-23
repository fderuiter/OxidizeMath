# ADR 0020: Precomputed Distance Matrix for TDA Filtration

## Status

Accepted

## Context
In `crates/pure_math/src/pure_math/statistics/tda/persistence.rs`, `compute_persistence` evaluates persistent homology across a sequence of filtration radius steps. Previously, each iteration invoked `vietoris_rips_complex(&cloud, radius)`, which called `cloud.distance_matrix()` inside `crates/pure_math/src/pure_math/statistics/tda/complex.rs`. For a point cloud with $N$ points and a filtration of $K$ steps, this caused $O(K \cdot N^2)$ redundant Euclidean distance recalculations and allocated $K$ identical distance matrices on the heap.

## Decision
We refactored Vietoris-Rips complex generation and persistent homology to precompute the invariant distance matrix once:
1. Extracted helper function `vietoris_rips_from_distances(dist: &[Vec<f64>], radius: f64) -> Result<SimplicialComplex, TdaError>` in `crates/pure_math/src/pure_math/statistics/tda/complex.rs` to construct simplicial complexes from precomputed pairwise distances.
2. Delegated public `vietoris_rips_complex` directly to `vietoris_rips_from_distances`.
3. In `crates/pure_math/src/pure_math/statistics/tda/persistence.rs`, validated radius slice preconditions upfront and computed `cloud.distance_matrix()` once prior to the filtration loop.
4. Passed the precomputed matrix reference `&dist` to `vietoris_rips_from_distances` for each radius in `compute_persistence`.
5. Preserved exact numerical equivalence and public API signatures (`vietoris_rips_complex` and `compute_persistence`).
6. Added regression tests verifying barcode parity and error handling for invalid/empty radii.

## Consequences
- **Performance**: Eliminates $K - 1$ redundant distance matrix allocations and $(K - 1) \cdot \frac{N(N - 1)}{2}$ distance evaluations across $K$ filtration steps.
- **Compatibility**: Preserved public function signatures and existing behavioral semantics.
- **Verification**: Verified with TDA unit tests and architectural consistency checks.

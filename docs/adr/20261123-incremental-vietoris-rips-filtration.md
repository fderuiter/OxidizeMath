# ADR 20261123: Incremental Vietoris-Rips Filtration, Bitset Simplex Discovery, and Distance Reuse

## Status

Accepted (extends [ADR 0020](0020-precompute-distance-matrix-tda-filtration.md), [ADR 0035](0035-z2-boundary-reduction-persistent-homology.md), and [ADR 0039](0039-metric-agnostic-3d-tda-exact-z2-homology-and-chaos-mashup.md))

## Context

Profiling `pure_math::statistics::tda` and `math_explorer_gui`'s `VietorisRipsTool` identified residual dynamic allocations and redundant passes across filtration sweeps and interactive slider updates:

1. **Simplex & Face Clones in `complex.rs`**: `SimplicialComplex::add_simplex` cloned every `Simplex` (including 0-simplices and already-present simplices) and invoked `Simplex::faces` on 0-simplices. Meanwhile, `vietoris_rips_from_distance_matrix` scanned $O(N^3)$ triples rather than forward-neighbor lists and pushed empty `HashSet` levels before popping them.
2. **Duplicate `UnionFind` Passes & `get_simplices` Clones in `homology.rs`**: `UnionFind::count_components` allocated a fresh `HashSet`, `betti_number_0` and `betti_number_1` cloned all simplices via `get_simplices`, and `betti_numbers` executed `betti_number_0` twice.
3. **Full-Cloud Enumeration & Per-Radius Workspace Rebuilds in `persistence.rs` and `workspace.rs`**: `compute_persistence_from_distances` enumerated and reduced simplices above `max(radii)` and continued $\mathbb{Z}_2$ column elimination after reaching the exact cycle-space rank ceiling $|E| - |V| + \beta_0$. Simultaneously, `TdaFiltrationWorkspace::evaluate_radius` rebuilt Union-Find, edges, $O(N^3)$ triangles, and $\partial_2$ boundary reduction from scratch at every radius step.
4. **GUI Distance Matrix & Simplex Clones in `VietorisRipsTool`**: `VietorisRipsTool::update_complex` cloned `distance_matrix` on every slider edit, lacked `cached_points` / `cached_points_3d` guards for direct point mutations, and cloned `Simplex` sets via `get_simplices` when refreshing plot geometry.

## Decision

1. **Direct Canonical Simplex Construction (`complex.rs`)**:
   - Insert 0-simplices directly in `SimplicialComplex::add_simplex` without cloning or calling `Simplex::faces`, and move newly inserted higher-dimensional simplices without cloning.
   - Construct forward-neighbor lists in `vietoris_rips_from_distance_matrix`, reserve `HashSet` capacities per active dimension, enumerate canonical triangles over forward neighbors, and push dimension levels only when non-empty.
2. **Single-Pass `UnionFind` & Borrowed Simplex Reads (`homology.rs`)**:
   - Promote `UnionFind` to `pub(crate)` with an `O(1)` component counter and zero-allocation `reset` / `reset_prefix` methods.
   - Borrow `&complex.simplices` directly in `betti_number_0` (with a zero-`HashMap` fast path for dense `0..n` vertex IDs and a sparse-ID fallback) and `betti_number_1`, sharing `beta0` in `betti_numbers` via `betti_1_from_counts` so `UnionFind` runs once.
3. **Bitset Simplex Filtration, Tetrahedron Domination Pruning, Rank-Ceiling Early Exit, and Incremental Streaming (`filtration.rs`, `boundary.rs`, `persistence.rs`, `workspace.rs`)**:
   - Add private `#[verified_engine::verified]` helpers in `tda/filtration.rs` using `Vec<Vec<u64>>` neighbor bitsets to intersect common forward neighbors, filter simplices up to `max(radii)` in `compute_persistence_from_distances`, prune 2-simplices whose boundary is spanned over $\mathbb{Z}_2$ by three strictly preceding faces of a 3-simplex, and terminate `Z2BoundaryWorkspace` column reduction as soon as $\partial_2$ reaches $|E| - |V| + \beta_0$ and $\partial_1$ reaches $|V| - \beta_0$.
   - Stream pre-sorted edges monotonically across `TdaFiltrationWorkspace::evaluate_radius` and `compute_persistence_into` with zero heap allocations under `VerifiedAllocator`, intersecting `u64` adjacency bitsets only when an edge closes a cycle, skipping dependent triangle reductions when `basis_rank == curr_cycles` or an earlier common apex exists, sorting unsorted `radii` in-place via preallocated scratch storage, and resetting in-place when `radius < prev_radius`.
4. **Cached Distance Reuse & Zero-Clone Plot Refresh (`vietoris_rips/{mod,filtration}.rs`)**:
   - Track `cached_points` and `cached_points_3d` on `VietorisRipsTool` so pairwise distances and persistence intervals recompute only when points change or clear, reuse `&self.distance_matrix` without cloning on slider edits, call `vietoris_rips_from_distances` when triangle truncation cannot occur, and populate 2D/3D plot caches by borrowing `&complex.simplices` directly into cleared buffers.

## Consequences

- **Performance & Zero-Allocation Compliance**: Release-mode benchmarks on a 50-step `N = 100` filtration drop from `6,397.5 ms` and `47,569,777` allocations to `2.61 ms` and `0` allocations (`2,453x` faster) in `TdaFiltrationWorkspace`, and `76.80 ms` and `335` allocations (`83.3x` faster) in `compute_persistence`. A 30-step `N = 20` `VietorisRipsTool` slider sweep drops from `1.834 ms` peak step latency and `244,798` allocations to `0.457 ms` peak (`4.21 ms` total across 30 steps), `15,269` allocations (`16.0x` fewer), and `0` distance matrix rebuilds.
- **Compatibility**: Preserves all public `pure_math::statistics::tda` signatures, exact $\mathbb{Z}_2$ Betti numbers (including isolated vertices where $|E| < |V|$ and $K_4$ cliques), barcode intervals, and error variants.

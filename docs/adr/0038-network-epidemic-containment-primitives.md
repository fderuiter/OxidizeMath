# ADR 0038: Network Epidemic Containment Primitives, Active Degree Moments, and Zero-Allocation Stepping

## Status
Accepted (extends [ADR 0021](0021-zero-allocation-grpo-advantage-normalization.md), [ADR 0022](0022-unify-preset-state-serialization.md), and [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md))

## Context
[`crates/domain_epidemiology/README.md`](../../crates/domain_epidemiology/README.md) documents real-time intervention efficacy analysis—including vaccination coverage and contact quarantine—across [`crates/domain_epidemiology/src/epidemiology/networks.rs`](../../crates/domain_epidemiology/src/epidemiology/networks.rs) and `math_explorer_gui`. Three limitations previously prevented interactive network containment experiments and verified zero-allocation stepping:
1. **Missing Vertex Immunization and Edge Quarantine Primitives**: `NodeState` modeled only `Susceptible`, `Infected`, and `Recovered` compartments without `NodeState::Vaccinated` or symmetric undirected edge quarantine on `NetworkEpidemicModel`.
2. **Unwired Residual Topology Analytics**: `NetworkEpidemicModel` did not expose active residual degree moments $(\langle k \rangle, \operatorname{Var}(k))$ accounting for immunized vertices and quarantined edges, leaving `heterogeneous_r0`, [`calculate_final_size`](../../crates/domain_epidemiology/src/epidemiology/analytics.rs), and [`probability_of_extinction`](../../crates/domain_epidemiology/src/epidemiology/stochastic.rs) disconnected from interactive graph containment in `math_explorer_gui`.
3. **Per-Step Heap Allocation in `step_with_rng`**: `NetworkEpidemicModel::step_with_rng` cloned `self.states` on every step (`let mut next_states = self.states.clone();`), violating NASA Power of 10 Rule 3 under [`VerifiedAllocator`](../../crates/verified_engine/src/allocator.rs) when `MEMORY_LOCK` is engaged.

## Decision
1. **Extend `NetworkEpidemicModel` with Containment Primitives and Canonical Edge Indexing**:
   - Add `NodeState::Vaccinated` alongside `InvalidNodeIndex`, `InvalidEdge`, and `InvalidIntervention` variants on `NetworkModelError` in [`crates/domain_epidemiology/src/epidemiology/networks.rs`](../../crates/domain_epidemiology/src/epidemiology/networks.rs).
   - Store canonical undirected `edges: Vec<[usize; 2]>` (`u < v`), `patient_zero: Option<usize>`, and parallel `edge_quarantined: Vec<Vec<bool>>` masks aligned with `adjacency` during `NetworkEpidemicModelBuilder::build` and `initialize_geometric_graph_with_rng`.
   - Provide zero-allocation mutators and queries `set_node_vaccinated`, `set_edge_quarantined` (updating both `(u, v)` and `(v, u)` symmetrically), `is_edge_quarantined`, `edges`, and `patient_zero`, rejecting out-of-range indices and vaccination of already-infected or recovered vertices.
2. **Preallocate `scratch_states` for Zero-Allocation Contagion Stepping**:
   - Replace `self.states.clone()` in `NetworkEpidemicModel::step_with_rng` with a preallocated `scratch_states: Vec<NodeState>` buffer copied via `copy_from_slice`, skipping quarantined contact edges and preserving `NodeState::Vaccinated` immunity without post-initialization heap allocations under `VerifiedEngine::run_verified`.
3. **Expose Residual Active Degree Moments and Budgeted GUI Controls**:
   - Add `active_degree`, `active_degree_moments`, and `effective_r0` on `NetworkEpidemicModel` so hub immunization and bridge quarantine update $(\langle k \rangle, \operatorname{Var}(k))$ and $R_0 = \frac{\beta}{\gamma}\frac{\langle k^2 \rangle - \langle k \rangle}{\langle k \rangle}$ in $O(|V| + |E|)$ with zero heap allocations.
   - Split the `NetworkPropagationTool` GUI module into focused submodules (`<= 500` lines/file, `<= 60` lines/function) pairing pointer/keyboard budgeted interventions, seeded `url_state` sharing, `StarterGallery` hub-vaccination and bridge-cut challenge presets, and guarded `calculate_final_size` / `probability_of_extinction` readouts.

```text
Containment invariants enforced by NetworkEpidemicModel and NetworkPropagationTool:
- Symmetric quarantine: is_edge_quarantined(u, v) == is_edge_quarantined(v, u)
- Zero heap allocation: set_node_vaccinated, set_edge_quarantined, active_degree_moments, and step_with_rng allocate 0 bytes under VerifiedAllocator
- Atomic state import: try_load_state rejects out-of-range indices, duplicates, patient-zero vaccination, and over-budget interventions prior to mutating self.model
```

## Consequences
- **Positive**: `NetworkEpidemicModel::step_with_rng`, `set_node_vaccinated`, `set_edge_quarantined`, and `active_degree_moments` execute with zero dynamic memory allocations under `VerifiedAllocator`.
- **Positive**: Users and educators can share deterministic seeded URLs and launch reproducible hub-vaccination and bridge-cut challenges with live $R_0$, extinction probability, and final outbreak size analytics.
- **Neutral**: Public API additions in `crates/domain_epidemiology/src/epidemiology/networks.rs` are recorded in [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json) and re-exported automatically through `math_explorer::epidemiology::networks`.

# ADR 0047: SharedWorkspaceBus Per-Pane Graph Coupling and Zero-Allocation Kuramoto Solver

## Status

Accepted (2026-11-22) (extends [ADR 0027](0027-prune-diagnostic-bus-channel-and-dead-helpers.md))

## Context

`MathExplorerApp` supports `SplitHorizontal`, `SplitVertical`, and `QuadGrid` multi-pane viewports, while ADR 0027 previously pruned an unused string-only diagnostic bus from `federated_registry`. Each viewport pane remained completely isolated, leaving `NetworkEpidemicModel::builder()` in `domain_epidemiology` unwired in GUI tools and forcing `NetworkPropagationTool` and `NetworkMetricsTool` to operate only on static or random local graphs. Researchers could not edit a graph in one pane and observe immediate topological consequences across contagion, network centrality, and nonlinear phase-oscillator synchronization models.

## Decision

1. **`SharedWorkspaceBus` on `MathExplorerApp`**:
   - Introduce `math_explorer_gui/src/app/workspace_bus.rs` containing two bounded, revision-counted, domain-agnostic channels:
     - `GraphTopologyChannel`: capped at `256` normalized 2D coordinates (`[-1.0, 1.0]`) and `4,096` finite undirected weighted edges with a monotonic `u64` revision counter.
     - `ScalarSignalChannel`: ring-buffered scalar `[t, value]` series capped at `1,024` finite samples per series.
   - Scope bus access per pane via explicit link toggles (`⛓ Link` / `🔗 Linked`) in `pane_chrome.rs` using an $O(1)$ RAII scope guard with zero mutexes and zero cross-test state leaks. Unlinked panes default to `false` and remain strictly isolated across swaps, rotations, and undo history.
2. **Cross-Domain Tool Coupling & `graph_editor.rs` Modularization**:
   - Split `graph_editor.rs` into focused submodules (`bus_sync.rs`, `interaction.rs`, `keyboard.rs`, `snapshot.rs`) under the 500-line ceiling and publish normalized coordinates and weighted edges onto `GraphTopologyChannel`.
   - Wire `NetworkPropagationTool` to `NetworkEpidemicModel::builder()`, live `heterogeneous_r0`, and click-to-seed / click-to-quarantine node transitions.
   - Wire `NetworkMetricsTool` to recompute centrality, clustering, and diameter strictly when `GraphTopologyChannel::revision()` advances.
3. **Zero-Allocation Kuramoto Phase-Oscillator Engine**:
   - Add `domain_physics::physics::chaos::kuramoto::KuramotoSystem` solving $\dot{\theta}_i = \omega_i + \frac{K}{N}\sum_j A_{ij}\sin(\theta_j - \theta_i)$ over compressed sparse row (CSR) weighted adjacency and preallocated RK4 scratch buffers (`k1..k4`, `scratch_thetas`), guaranteeing zero heap allocations inside `#[verified_engine::verified]` hot loops under `VerifiedAllocator`.
   - Add `KuramotoOscillatorTool` to `ChaosTab` streaming live order-parameter coherence $R(t) \in [0.0, 1.0]$ onto `ScalarSignalChannel`.
4. **`.mws` and URL Persistence**:
   - Persist `pane_bus_links` in `AppState` (`.mws` envelopes) and encode compact `&ws=<base64url>` multi-pane link descriptors in `url_state/codec.rs` validated under `URL_JSON_BOUNDS` (4,096 decoded bytes; 8,192 fragment bytes) while keeping single-pane and unlinked share URLs unchanged.

## Consequences

- Editing nodes, hubs, or bridge edges in a linked `GraphEditorTool` pane updates `NetworkPropagationTool`, `NetworkMetricsTool`, and `KuramotoOscillatorTool` within one frame.
- Unlinked multi-pane isolation (`tests_pane_isolation.rs`), feature-isolated builds (`--no-default-features`), zero-allocation `VerifiedAllocator` hot paths, `AstVisitor` non-doc attribute scoping for `#[embed_theory]`/`#[verified]` rustdoc examples, and WASM bundle limits remain intact.

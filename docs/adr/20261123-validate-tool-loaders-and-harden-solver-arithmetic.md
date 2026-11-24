# ADR 20261123: Validate Tool Loaders and Harden Solver Arithmetic Against 32-Bit Overflow

## Status

Accepted (extends [ADR 0022](0022-unify-preset-state-serialization.md))

## Context

1. **Unchecked Loader Domain Bounds & 32-Bit `wasm32` Traps**: `ReplicatorPresetConfig::validate` previously verified matrix squareness and symbolic parameters without checking `dt`, `time_horizon`, payoff magnitude, or population bounds, allowing crafted `#tab=` URLs and `.json` presets (`dt: 0.0` or `time_horizon / dt > MAX_ODE_STEPS`) to reach `ReplicatorDynamics::simulate_with_strategy`, where `(time_horizon / dt).ceil() as usize + 1` overflowed 32-bit `usize` on `wasm32-unknown-unknown` under `overflow-checks = true` and `panic = "abort"`.
2. **Graph Node Counter Saturation**: `SerializedGraphState::validate` previously accepted `next_node_id` values below existing node IDs or at `u32::MAX`, while `GraphEditorTool::click_add_node` and `GraphEditorTool::insert_node_from_keyboard` incremented `next_node_id += 1` without overflow checks.
3. **Unified Worker Constraint Bypass & Silent Neuroscience Divergence**: `UnifiedSimTool` checked parameter existence and finiteness in `persistence.rs` and `sculpt.rs` without enforcing declared `ParameterConstraint` `[min, max]` bounds outside active challenge extensions, and `HodgkinHuxleyUnified` relied on the default no-op `FiniteStateGuard` implementation (`is_finite_state() == true`), masking non-finite membrane voltage and gating states from worker fault recovery.

## Decision

1. **Evolutionary Game Preset & Solver Step Bounds**:
   - Enforce strategy count (`1..=16`), payoff finiteness (`[-10000, 10000]`), population proportions (`[0, 1]` with positive sum), `dt` (`[1e-6, 5.0]`), `time_horizon` (`(0, 10000]`), and `checked_ode_step_count` (`<= MAX_ODE_STEPS`) in `ReplicatorPresetConfig::validate` and `ReplicatorDynamicsTool::has_finite_parameters`.
   - Bound step counts in `ReplicatorDynamics::simulate_with_strategy` (`crates/domain_applied`) to `MAX_REPLICATOR_STEPS = 200_000` with `steps.saturating_add(1)` capacity allocation and non-positive/non-finite early return (`0` integration steps), preserving existing public signatures and `#[verified_engine::verified]` zero-allocation loop invariants.
2. **Graph Editor Counter Headroom & Checked Increments**:
   - Reject `next_node_id` values exceeding `(u32::MAX as usize) - 65_536` or `<= max_node_id` in `SerializedGraphState::validate` and `GraphEditorTool::check_finite_graph`.
   - Guard `next_node_id` increments in `GraphEditorTool::click_add_node` and `GraphEditorTool::insert_node_from_keyboard` with `checked_add(1)` bounded below `u32::MAX as usize` before mutating undo history or graph storage.
3. **Unified Worker `ParameterConstraint` Enforcement & `HodgkinHuxleyUnified` `FiniteStateGuard`**:
   - Validate all loaded parameters against `ParameterConstraint` `[min, max]` bounds across `UnifiedSimTool::validate_param_map` (`persistence.rs`), `UnifiedSimTool::parse_unified_state_entry` (`sculpt.rs`), and `UnifiedSimTool::has_finite_parameters` (`tool.rs`).
   - Implement `FiniteStateGuard::is_finite_state` and `FiniteStateGuard::diagnostic_report` on `HodgkinHuxleyUnified` across membrane potential `v`, gating variables `n, m, h`, time step `dt`, and conductances without masking `NaN` or `±Infinity`, activating `SimWorkerState::Faulted` recovery controls in `worker.rs`.

## Consequences

- **Positive**: Crafted `#tab=` links, `.json` presets, and `.mws` workspace sessions fail fast with error toasts and `PresetSerialization` warning telemetry before mutating active tool state or triggering `wasm32` arithmetic aborts, while keeping all existing JSON field names and starter presets intact.
- **Positive**: Direct library calls to `ReplicatorDynamics::simulate_with_strategy`, saturated `GraphEditorTool` node edits, and divergent `HodgkinHuxleyUnified` trajectories degrade safely with bounded allocations and structured worker fault recovery.

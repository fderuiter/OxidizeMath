# ADR 0019: Purge Dead Code Bypasses and Clear Verification Whitelist

## Status
Accepted

## Context
Four entries for dead code bypasses in `verification_whitelist.toml` suppressed compiler warnings for active methods and uncalled code across the workspace:
1. `AttractorPlotter::diverged` in `math_explorer_gui/src/tabs/chaos/attractors.rs` and `OdeSolverTool::diverged` in `math_explorer_gui/src/tabs/analysis/ode.rs` retained obsolete `#[allow(dead_code)]` attributes despite being actively invoked in simulation loops, UI controls, and test suites.
2. `SimpleState::new` in `math_explorer/examples/bench_rk4_alloc.rs` was uncalled because the benchmark constructs `SimpleState` via struct literals.
3. In `crates/oxidize_core/src/lib.rs`, `BooleanMaskSimulation` retained an unread `config` field, `BooleanMaskState` dimensions were not asserted, and `SimulationError::InitError` was unconstructed, requiring an `#[allow(dead_code)]` attribute on `mod tests`.

## Decision
1. Removed `#[allow(dead_code)]` from `AttractorPlotter::diverged` and `OdeSolverTool::diverged`.
2. Removed uncalled `SimpleState::new` and its `impl SimpleState` block from `math_explorer/examples/bench_rk4_alloc.rs`.
3. Removed unread `config` field from `BooleanMaskSimulation`, validated dimensions in `initialize` to construct `InitError` on zero dimensions, asserted state dimensions in `test_boolean_mask_simulation`, and removed `#[allow(dead_code)]` from `mod tests` in `crates/oxidize_core/src/lib.rs`.
4. Pruned all `[[dead_code]]` table entries from `verification_whitelist.toml`.

## Consequences
- Eliminates dead code bypasses and prevents compiler warning suppressions for active methods.
- Enforces strict dead-code verification across all crates with zero dead code whitelist entries.
- All public mathematical APIs, simulation logic, and UI behavior remain completely unchanged.

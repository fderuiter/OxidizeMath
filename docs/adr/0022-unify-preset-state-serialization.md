# 22. Unify Simulation Preset Dispatch on Validated Tool State Serialization

## Status
Accepted

## Context
`math_explorer_gui` previously maintained two divergent preset mechanisms on `InteractiveTool`:
1. `apply_preset_json`, whose default trait implementation unconditionally returned `Ok(())`, silently discarding starter preset configurations on 49 of the 52 registered tools while clearing error banners and presenting a successful launch state.
2. `export_preset_json` / `import_preset_json`, which wrapped `save_state` and an infallible `load_state(&mut self, &str)` without reporting schema mismatches, unknown fields, or out-of-range parameter values, and remained unimplemented (`None`) on 17 registered tools.

Additionally, `AttractorPlotter` reset parameters to defaults instead of parsing its preset JSON, `PotentialFlowTool` used incompatible schemas between gallery launch and file import/export, ~49 `STARTER_PRESET_JSON` constants contained drifted field names or unimplemented feature descriptions, and simulation toolbars rendered duplicate `🎨 Starter Gallery` buttons within a single view.

## Decision
1. Removed `InteractiveTool::apply_preset_json` and routed Starter Gallery card launches in `SimulationFramework` (`math_explorer_gui/src/framework/ui.rs`) through `InteractiveTool::import_preset_json`, synchronizing `saved_tool_states`, `baseline_tool_state`, and `last_valid_snapshot`, clearing stale plot/divergence state, and requesting repaint.
2. Added fallible `try_load_state(&mut self, state: &str) -> Result<(), String>` to `InteractiveTool` (`math_explorer_gui/src/framework/mod.rs`), delegating `load_state` and `import_preset_json` through `try_load_state` and enforcing `#[serde(deny_unknown_fields)]` on `PresetMetadata`, `PresetWrapper`, `UnifiedSimState`, `Camera3D`, `TraceSnapshotManager`, and all tool state structs.
3. Implemented `domain()`, `save_state()`, and validated `try_load_state()` across all 52 registered `InteractiveTool` implementations (including all 17 previously unserialized tools), unified `PotentialFlowTool` and `AttractorPlotter` on single validated state schemas, and aligned all 57 `StarterPresetMetadata` JSON payloads and descriptions with actual tool parameters and mathematical models.
4. Surfaced preset launch, import, and export failures consistently via `PresetDialogState::toast_error`, `DiagnosticCategory::PresetSerialization` warning telemetry on `federated_registry::global_registry()`, and screen-reader announcements across `framework/ui.rs`, `framework/export_ui.rs`, and `tabs/morphogenesis.rs`.
5. Consolidated Starter Gallery entry points to a single `🎨 Starter Gallery` button in `show_export_action_bar` (`framework/export_ui.rs`) for `SimulationFramework` tools (plus `MorphogenesisTab`'s standalone tab toolbar) and removed unreachable `show_quick_start_cards`.
6. Extracted gallery tests to `math_explorer_gui/src/starter_gallery/tests.rs` with contract tests covering all 57 starter cards, all 52 tool export/import round-trips, unknown-field and domain-mismatch rejection with telemetry/toast verification, and single-button toolbar invariants, and updated `docs/architecture/interactive_tool_walkthrough.md`.

## Consequences
- Every Starter Gallery card applies real, validated model parameters and updates baseline/saved tool state consistently.
- Portable JSON preset export and import work uniformly across all 52 `InteractiveTool` implementations and `MorphogenesisTab`, failing fast with actionable UI feedback and telemetry on malformed or mismatched payloads.
- Eliminates silent no-op preset dispatch, schema drift between gallery presets and file import/export, and duplicate gallery buttons in simulation toolbars.

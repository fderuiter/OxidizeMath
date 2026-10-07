# 8. Serde JSON Simulation Preset Import/Export Architecture

## Context
Math Explorer simulation tools previously hardcoded default parameter values inside enums and UI callbacks. Users had no standard mechanism to serialize simulation state to portable files or import saved parameter configurations across sessions or peers.

## Decision
1. Define dedicated configuration structs (`OdePresetConfig`, `MorphogenesisPresetConfig`, `ReplicatorPresetConfig`) deriving `serde::Serialize` and `serde::Deserialize`.
2. Define `InteractivePreset` trait providing uniform `to_json_string()` and `from_json_str()` methods.
3. Standardize exported JSON file structure using a metadata envelope containing `schema_version`, `title`, `domain`, and `timestamp`.
4. Validate schema and matrix constraints (e.g., square payoff matrix, matching strategy counts) on deserialization to ensure safe parameter restoration without application panics.
5. Provide non-blocking UI buttons and toast notifications for invalid file imports.

## Consequences
- Enables portable JSON preset state exchange across ODE, Morphogenesis, and Replicator Dynamics simulation tools.
- Guarantees parameter safety and graceful error handling on corrupted or mismatched preset files.

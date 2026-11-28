# ADR 20261127: Create Tier 3 `domain_engineering`, Promote Shared Math Primitives to `pure_math`, and Enforce Zero Cross-Domain Dependencies

## Status

Accepted

## Context

Prior to this decision, Tier 3 scientific domain crates violated peer isolation in three ways:

1. **Cross-Domain Cargo Coupling (`domain_physics -> domain_applied`, `domain_biology -> domain_applied`)**: `domain_physics` depended on `domain_applied` solely to import `KalmanFilter`/`KalmanModel` (for radar motion tracking) and `FitnessStrategy`/`MatrixPayoff` (for Ising spin-glass evolutionary dynamics), while `domain_biology` depended on `domain_applied` to import `ReplicatorDynamics`/`MatrixPayoff` and triangulated cortical mesh helpers (`Surface`, `cortical_thickness_batch`). Both `PhysicsError` and `BiologyError` wrapped `AppliedError`, and `__doctest_facade` modules forwarded `domain_applied` across sibling crates.
2. **Fragmented Engineering Solvers and Mismatched Verification Citations**: Millimeter-wave FMCW radar signal processing (`FmcwConfig`, `chirp_z_transform`, `Beamformer`, `MusicEstimator`, `EllipticalFilter`, `PhaseUnwrapper`, `BiQuadraticSurface`, `TrackingFilter`) lived inside `domain_physics::physics::medical::radar_gating` alongside clinical LINAC gating (`GatingLogic`, `BeamState`), whereas coupled Arrhenius/SEI battery aging (`battery_degradation`) and CNC/UART calculators (`applied::engineering::{cnc, uart}`) lived inside `domain_applied` with mismatched `theory_verification!` citations and constant-equality placeholders.
3. **Unchecked Peer Topology Drift**: `WorkspaceManifestModel::audit_tier_dependencies` prevented lower tiers from depending on higher tiers, but permitted `TierKind::Domains03` crates to add new `domain_* -> domain_*` sibling edges.

## Decision

1. **Promote Shared Mathematical Primitives to [`pure_math`](../../crates/03_domains/pure_math/README.md)**:
   - Move discrete-time linear-Gaussian filtering (`KalmanFilter`, `KalmanFilterBuilder`, `KalmanModel`, `KalmanSystem`, `KalmanError`) to `pure_math::pure_math::analysis::kalman`.
   - Move evolutionary game-theory traits and simplex replicator dynamics (`FitnessStrategy`, `MatrixPayoff`, `ReplicatorDynamics`, `GameTheoryError`) to `pure_math::pure_math::analysis::evolutionary`.
   - Move triangulated 3D surface meshes (`Surface`, `mean_vertex_height`) and CSR-indexed symmetric cortical thickness kernels (`FlatSpatialGrid`, `SpatialGrid`, `ThicknessWorkspace`, `cortical_thickness*`) to `pure_math::pure_math::differential_geometry`, retaining compatibility re-exports in `domain_applied`.

2. **Decouple `domain_physics` and `domain_biology` and Extract [`domain_engineering`](../../crates/03_domains/domain_engineering/README.md)**:
   - Remove `domain_applied` from `domain_physics` and `domain_biology` (`Cargo.toml`, `src/lib.rs`, and `src/error/umbrella.rs`), leaving zero `domain_* -> domain_*` dependency edges across all Tier 3 crates.
   - Establish `crates/03_domains/domain_engineering` (`Validated` maturity tier) consolidating `engineering::radar`, `engineering::battery_degradation`, `engineering::cnc`, and `engineering::uart` with `EngineeringDomainError`, `RadarError`, `BatteryError`, and `HardwareError`, while keeping clinical `GatingLogic`, `BeamState`, and `GatingError` in `domain_physics::physics::medical::radar_gating`.
   - Author [`docs/math_specs/domain_engineering.md`](../math_specs/domain_engineering.md) and replace placeholder `theory_verification!` blocks across `engineering`, `radar`, and `medical` with deterministic physical invariants (FMCW range-beat reciprocity, round-trip $\lambda / 4$ phase displacement, Taylor $V T^n = C$ scaling, `8N1` UART frame efficiency, and latency-compensated gating hysteresis).

3. **Preserve Facade Compatibility and Enforce Static Cross-Domain Isolation**:
   - Add the `engineering` feature to [`oxidizemath`](../../crates/04_bindings/oxidizemath/Cargo.toml), [`math_explorer`](../../crates/04_bindings/math_explorer/Cargo.toml), [`oxidize-cli`](../../apps/oxidize-cli/Cargo.toml), [`math_explorer_gui`](../../apps/math_explorer_gui/Cargo.toml), `xtask`, and CI `feature-matrix`, while re-exporting legacy `physics::medical::radar_gating::*`, `applied::battery_degradation`, and `applied::engineering` paths in `oxidizemath` when `engineering` is enabled.
   - Extend `WorkspaceManifestModel::audit_tier_dependencies` in `unified_verification` (`check-profiles` / `batch-static`) to reject any `TierKind::Domains03 -> domain_*` dependency across `[dependencies]`, `[build-dependencies]`, and `[dev-dependencies]`.

## Consequences

- **Positive**: Restores strict peer isolation across all Tier 3 `domain_*` crates so `domain_physics`, `domain_biology`, and `domain_engineering` compile and test independently against `pure_math` and Tier 0–2 engines.
- **Positive**: Unifies RF/mmWave radar, electrochemical battery aging, and hardware prognostics under `domain_engineering` with canonical 4-part specifications and physical `theory_verification!` invariants while keeping `RadarGatingTool`, `CapacityFadeTool`, and CLI examples working unchanged.
- **Trade-offs**: Workspace crate count increases from 31 to 32 crates (21 release crates), recorded across architecture guides, [`traceability.toml`](../../traceability.toml), [`api_baseline.json`](../../apps/unified_verification/api_baseline.json), and [`Verification_Certificate.json`](../../Verification_Certificate.json).

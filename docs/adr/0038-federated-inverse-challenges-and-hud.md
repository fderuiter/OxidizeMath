# ADR 0038: Federated Inverse Challenges, Unified Challenge HUD, and Stateless Challenge URLs

## Status

Accepted (extends [ADR 0022: Unify Preset and State Serialization](./0022-unify-preset-state-serialization.md))

## Context

[ADR 0022](./0022-unify-preset-state-serialization.md) unified `StarterPresetMetadata` and user-saved `.json` snapshots around `PresetMetadata` and `PresetWrapper<S>` with strict `#[serde(deny_unknown_fields)]` schema enforcement. While that contract eliminates schema drift for forward presets, every starter card loads a pre-solved forward parameter state without locked boundary conditions, quantitative target bounds, or live error scoring.

At the same time, verified inverse and clinical domain algorithms—including `BatemanModel`, `solve_ka`, and `SuperpositionModel` in `domain_applied::applied::pharmacokinetics`, `calculate_cost` and `update_weights` in `domain_physics::physics::medical::optimization`, rotating-frame `BlochSimulator` pulse sequences in `domain_physics::physics::mri`, and bounce-back momentum-exchange drag in `domain_physics::physics::fluid_dynamics::lattice_boltzmann`—were not exposed as interactive design problems in `math_explorer_gui`.

Adding inverse-problem metadata directly onto `PresetMetadata` or `PresetWrapper` would break `#[serde(deny_unknown_fields)]` compatibility across all existing forward starter presets and saved `.mws` / `.json` payloads.

## Decision

1. **Parallel Federated `ChallengeMetadata` Registry**:
   - Introduce `ChallengeMetadata` collected via `inventory::collect!(ChallengeMetadata)` beside `StarterPresetMetadata` in `math_explorer_gui::starter_gallery`.
   - Keep `PresetMetadata` and `PresetWrapper` completely unchanged (`#[serde(deny_unknown_fields)]`) so all forward presets load without modification.
   - Each `ChallengeMetadata` entry declares canonical `id`, `domain`, `tool_name`, human-readable locked inputs, tunable budget summary, target criteria, `baseline_state_generator: fn() -> String`, and `clamp_locked_state: fn(&str) -> Result<String, String>`.

2. **Unified Challenge HUD and `CalloutManager` Annotations**:
   - Extend `InteractiveTool` with optional challenge hooks (`apply_challenge`, `active_challenge_id`, `evaluate_challenge`) returning `ChallengeEvaluation` (`score: 0.0..=100.0`, `passed`, `residual_label`, `residual_value`, `detail`).
   - Render a theme-aware Challenge HUD in `SimulationFramework` using `SemanticPalette` tokens meeting WCAG 2.2 AA contrast (`>= 4.5:1` text, `>= 3.0:1` strokes across `Dark`, `Light`, and `HighContrast`) and emitting polite ARIA live announcements (`announce_polite`) on launch, baseline reset, and pass transitions.
   - Surface target thresholds and trajectory extrema directly on plots via `CalloutManager::set_auto_callout` and `CalloutManager::render_on_plot`.

3. **Stateless URL Challenge Hydration with Tamper Clamping**:
   - Extend `ShareUrlParams` in `url_state::codec` with `challenge: Option<String>` supporting `#tab=<slug>&tool=<id>&challenge=<id>&state=<base64url>` as well as baseline `#tab=<slug>&challenge=<id>` links within the existing `4,096`-byte JSON and `8,192`-byte fragment caps.
   - During URL hydration, invoke `(meta.clamp_locked_state)(...)` before applying state so tampered locked fields in shared URLs are deterministically clamped back to canonical `ChallengeMetadata` values.

4. **Zero-Heap-Allocation Preallocated Scoring**:
   - Preallocate trajectory arrays, `SuperpositionModel` dose-time schedules, `DVector<f64>` 64x64 dose grids, and scalar LBM bounce-back force accumulators outside hot loops so `VerifiedEngine::run_verified` executes with zero dynamic heap allocations under `VerifiedAllocator`.

## Consequences

- Learners can launch inverse challenges from a dedicated **Inverse Challenges** tab in `StarterGallery` or share stateless `#tab=...&challenge=<id>` links without server-side state.
- All forward `StarterPresetMetadata` entries and `PresetWrapper` files remain 100% backward-compatible with ADR 0022.
- Domain physics and pharmacokinetic solvers gain direct interactive coverage and zero-allocation regression tests.

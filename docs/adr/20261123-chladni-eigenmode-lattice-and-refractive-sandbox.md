# ADR 20261123: Chladni Eigenmode Lattice and 2D AST Refractive Sandbox

## Status

Accepted (extends [ADR 0035](0035-compiled-ast-bytecode-and-unified-formula-sandboxes.md), [ADR 0038](0038-federated-inverse-challenges-and-hud.md), and [ADR 0042](0042-2d-chladni-plate-and-wave-tank-solver.md))

## Context

1. **Detuned Nodal Forcing and Static Boundary Readouts**: `SquareChladniPlate` in `math_explorer_gui/src/tabs/analysis/chladni_wave_tank/presets.rs` seeded the `(1, 3)` cosine Chladni mode ($f_{1,3} = 0.3142\text{ Hz}$) while driving `(0.22, 0.50)` along its horizontal nodal line at `0.56 Hz`, eroding resonant standing-wave correlation over long integrations. Meanwhile, the sidebar printed `Clamped f(1,1)` regardless of active `BoundaryMode2D` or `CircularDrumhead` selection and lacked direct `(m, n)` normal-mode navigation.
2. **Peak Courant Under-Reporting for Sub-Unity Refractive Indices**: `Wave2DSolver::safe_substep_plan` in [`crates/pure_math/src/pure_math/analysis/pde/wave_2d.rs`](../../crates/pure_math/src/pure_math/analysis/pde/wave_2d.rs) scaled the CFL time-step limit by $\max s(x, y)$ when $n(x, y) < 1.0$ ($s = 1 / n > 1$), whereas `Wave2DSolver::effective_courant` omitted the peak relative wave-speed factor and under-reported the true peak Courant number.
3. **Locked Biconvex Lens and Missing AST Observables**: `ChladniWaveTankTool` hardcoded a fixed biconvex lens without 2D `CompiledExpr` refractive-index authoring (`n(x, y)`), and omitted `InteractiveTool::observables`, `sync_callouts`, and `StarterGallery` story/challenge packs.

## Decision

1. **Peak Wave-Speed Courant Scaling (`wave_2d.rs`, `wave_2d/masks.rs`)**:
   - Extract `masks::max_wave_speed_scale` (floored at `1.0`) and apply it in both `Wave2DSolver::safe_substep_plan` and `Wave2DSolver::effective_courant` so fast-wave metamaterials ($n < 1.0$) report the true peak Courant number (`0.333` at $n = 0.6$, $\Delta t = 0.40$, $\Delta x = 1.0$) with zero dynamic heap allocations under `VerifiedAllocator`.
2. **Resonant `(1, 3)` Forcing and Clickable `6x6` Eigenmode Lattice (`eigenmodes.rs`, `presets.rs`, `ui.rs`)**:
   - Update `SquareChladniPlate` and `SQUARE_CHLADNI_PRESET_JSON` to drive antinode `(0.35, 0.08)` at `0.3142 Hz`.
   - Introduce `chladni_wave_tank/eigenmodes.rs` with a clickable `6x6` `(m, n)` normal-mode lattice supporting `Neumann` cosine modes, `Dirichlet` sine modes, `Clamped` square plate modes (`analytic_clamped_square_omega` and `clamped_beam_eigenfunction`), `Sponge` modes, and `CircularDrumhead` Bessel $j_{m,n}$ modes, automatically steering the driver off nodal lines and clamps.
3. **Cold-Path 2D `CompiledExpr` Refractive Sandbox and Bounded Persistence (`refractive.rs`, `chladni_wave_tank.rs`)**:
   - Introduce `chladni_wave_tank/refractive.rs` backed by `MathExpressionInputBar::new("n(x, y)", ...).with_axes(&["x", "y"])` with Luneburg, Parabolic GRIN, and Biconvex presets, evaluating `CompiledExpr::eval_2d` strictly on cold-path mask rebuilds (`n` clamped to `0.6..=3.0` with `1.0` non-finite fallback) while retaining `last_valid_compiled` across transient syntax edits.
   - Persist `#[serde(default)]` `custom_clamps` (`<= 16` pins), `refractive_mode`, `refractive_expr` (`<= 256` bytes), and `refractive_params` on `ChladniWaveTankState` under `#[serde(deny_unknown_fields)]`.
4. **Live Wave/Sand Observables and Federated Story & Challenge Packs (`story_packs.rs`)**:
   - Expose `sand_nodal_energy` (`ChladniSandSystem::mean_normalized_energy`), `focal_gain`, `focal_fwhm`, `effective_courant`, `dominant_k`, `custom_pin_count`, and `driver_frequency` via `ChladniWaveTankTool::observables()`, wire `CalloutManager` in `sync_callouts`, and register `PresetStoryMetadata` and `GoalChallengeMetadata` packs in `starter_gallery/story_packs.rs`.

## Consequences

- **Positive**: `SquareChladniPlate` sustains strong `(1, 3)` modal correlation (`>= 0.60` at $t = 60\text{ s}$), Luneburg and GRIN `n(x, y)` formulas reach `>= 3.3x` downstream focal gain at $x = 92$, `effective_courant` accurately tracks sub-unity refractive indices with zero hot-loop allocations, and `#state=` share URLs round-trip custom nodal pins and metamaterial formulas within `URL_JSON_BOUNDS`.
- **Neutral**: Public item signatures in `crates/pure_math` remain unchanged in `crates/unified_verification/api_baseline.json`, and all new modules stay within the `500`-line file ceiling.

# ADR 20261127: Promote Standalone GUI Solvers into Domain Crates and Enforce Headless GUI Opt-Out Guards

## Status

Accepted

## Context

Following [ADR 20261123](20261123-unify-ast-metrics-and-ratcheted-debt-caps.md) and [ADR 20261125](20261125-metadata-driven-tier-migration.md), `apps/math_explorer_gui` still embedded standalone numerical solvers across 19 modules under `#![allow(unused)]` / `#[verified(opt_out = "gui_tool")]` exemptions:

1. **Unverified Standalone Solvers in GUI Tabs**: State estimation (`FastEkf`, `rk4_step_with_jacobian`, `FastCvKalman`), closed-loop UPO/ADME plants and Gaussian GRPO controllers, custom Lorenz/flow RK4 steppers, general-relativity 4D/wormhole/custom-lapse/Kerr radial geodesic integrators, duplicate Braess Wardrop and Smith dynamics solvers, duplicate Parrondo Markov/SSA engines, Bayes-Nash auction quadrature, allocating `DMatrix` MRI k-space reconstruction, Onsager-Yang 2D Ising phase classification, Gray-Scott linear dispersion analysis, and 1D tight-binding lattice scattering lived inside `apps/math_explorer_gui` instead of Tier 3 domain crates (`pure_math`, `domain_applied`, `domain_biology`, `domain_physics`).
2. **Facade Invisibility and Allocation Drift**: Because `apps/math_explorer_gui` is a binary application crate, external consumers importing `oxidizemath` or `math_explorer` could not access these solvers, and GUI-embedded routines such as masked k-space reconstruction allocated `DMatrix` buffers per call outside `VerifiedEngine::run_verified`.
3. **AST Opt-Out Suppression and Headless `gui_tool` Exemptions**: `oxidize_core::ast_visitor::AstVisitor` forced `is_verified = false` for every function inside an `opt_out` scope even when individual kernels carried explicit `#[verified]` / `#[verified_engine::verified]` attributes (such as `tabs/elliptic_curves/atlas.rs`), while `unified_verification::policy_audit` permitted headless pure-Rust modules without any UI imports (`eframe`, `egui`, `egui_plot`, `rfd`, `web_sys`, `crate::ui_theme`, `crate::framework`) to claim `opt_out = "gui_tool"`.

## Decision

1. **Promote Kalman, Closed-Loop Plants, and GRPO Controllers (`crates/03_domains/domain_applied`)**:
   - Promote `rk4_step_with_jacobian`, `FastEkf`, `FastCvKalman`, and closed-loop plant models (`LorenzKalmanModel`, `SeirKalmanModel`, `PkKalmanModel`) into `domain_applied::applied::algorithms::kalman`, rewiring `closed_loop_control`, `medical/radar_gating`, and `medical/imrt/motion.rs`.
   - Promote UPO plants (`applied::engineering::upo_control`), closed-loop two-compartment ADME plants (`applied::pharmacokinetics::closed_loop`), and zero-allocation Gaussian GRPO controllers (`applied::grpo::controller`).

2. **Unify Chaos Trajectories, General Relativity Geodesics, Game Theory, and Parrondo Engines (`pure_math`, `domain_applied`, `domain_physics`)**:
   - Rewire `vietoris_rips/chaos_bridge.rs` and `chaos/attractors/flow.rs` to step trajectories through `LorenzBuilder` and `RungeKutta4`.
   - Promote 4D Schwarzschild, Morris-Thorne wormhole, custom-lapse, and Kerr equatorial radial RK4 integrators into `domain_physics::physics::high_energy::general_relativity`, correcting Kerr `worldline_clock_dilation` to include radial coordinate motion `g_rr (dr/d(ct))^2`.
   - Promote general-graph Frank-Wolfe Wardrop equilibrium (`pure_math::pure_math::graph_theory::wardrop` with `WardropError` wired into `PureMathError`, `AppliedError`, and `oxidizemath::error::Error`) and unify 2-route/3-route Braess Wardrop + Smith dynamics in `domain_applied::applied::game_theory::equilibrium`.
   - Consolidate fixed-size stack Parrondo Markov solvers and continuous-time Gillespie SSA ensembles into `pure_math::pure_math::statistics::markov::parrondo`, and promote Bayes-Nash auction quadrature and `MechanismWorkspace` into `domain_applied::applied::game_theory::mechanism_design`.

3. **Zero-Allocation MRI K-Space Reconstruction and Physics/Biology Atlases (`domain_physics`, `domain_biology`)**:
   - Promote `KSpaceScannerWorkspace` and masked 2D FFT k-space reconstruction into `domain_physics::physics::mri::reconstruction` using pre-allocated flat slice buffers with zero heap allocations inside `VerifiedEngine::run_verified`.
   - Promote Onsager-Yang Ising atlas metrics (`domain_physics::physics::solid_state::ising_atlas`), 1D tight-binding lattice scattering (`domain_physics::physics::quantum::lattice_scattering`), and Gray-Scott Turing dispersion classification (`domain_biology::biology::morphogenesis::dispersion`).

4. **AST Verification Counting, Headless `gui_tool` Policy Enforcement, and Debt Ratchet (`oxidize_core`, `unified_verification`, `verification_whitelist.toml`)**:
   - Update `AstVisitor::record_fn_metrics` so explicit `#[verified]` / `#[verified_engine::verified]` annotations on functions inside an `opt_out` scope count toward `verified_funcs`, `verified_prod_funcs` / `verified_test_funcs`, `verified_asserts`, and `active_funcs` while leaving unannotated functions opted out.
   - Add `gui_opt_out` inspection to `unified_verification::policy_audit` (both full and staged runs) to fail closed when any production file under `apps/math_explorer_gui/src/` declares `opt_out = "gui_tool"` without referencing UI symbols, directing contributors to `pure_math` or `domain_*`.
   - Remove `opt_out = "gui_tool"` from all headless helper modules in `apps/math_explorer_gui/src/` and ratchet `verification_whitelist.toml` caps down (`"gui_tool"` `768 -> 674`, `"math_explorer_gui"` `997 -> 903`).

## Consequences

- All promoted solvers are re-exported through `oxidizemath` and `math_explorer`, carry runnable docstrings and `#[verified_engine::verified]` instrumentation, and pass zero-allocation `VerifiedEngine::run_verified` tests under `VerifiedAllocator`.
- `AstVisitor` accurately credits explicit `#[verified_engine::verified]` functions inside `opt_out = "gui_tool"` modules such as `tabs/elliptic_curves/atlas.rs`.
- `unified_verification` blocks regressions that embed new headless solvers under `opt_out = "gui_tool"` in `apps/math_explorer_gui/src/`.

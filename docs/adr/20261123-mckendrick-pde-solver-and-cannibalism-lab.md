# ADR 20261123: Zero-Allocation Upwind McKendrick-von Foerster PDE Solver and Cannibalism Lexis Lab

## Status

Accepted (2026-11-23)

## Context

Previously, [`crates/domain_applied/src/applied/cannibalism/mckendrick_von_foerster.rs`](../../crates/domain_applied/src/applied/cannibalism/mckendrick_von_foerster.rs) and [`crates/domain_applied/src/applied/cannibalism/juvenile_adult_dynamics.rs`](../../crates/domain_applied/src/applied/cannibalism/juvenile_adult_dynamics.rs) omitted the age-transport operator $\frac{\partial n}{\partial a}$ from [`papers/cannibalism.tex`](../../papers/cannibalism.tex) and hardcoded `birth_rate` to `100.0`. In addition, [`crates/domain_applied/src/applied/cannibalism/two_dimensional_ode.rs`](../../crates/domain_applied/src/applied/cannibalism/two_dimensional_ode.rs) subtracted an unscaled constant `- self.phi_n_c`, allowing `N(t)` to cross below zero, and `math_explorer_gui` exposed no interactive tab for the `cannibalism` domain.

## Decision

1. **Guarded Holling Predation and Upwind Transport Operators in `domain_applied::applied::cannibalism`:**
   - Preserved the 6-argument `CannibalismModel::new` constructor (`handling_time = 0.0`), added `CannibalismModel::with_handling_time`, and replaced unguarded constant subtraction with density-guarded Holling Type I/II predation $\frac{\phi_{NC} N_+ C_+}{1 + h N_+}$.
   - Clamped [`death_rate.rs`](../../crates/domain_applied/src/applied/cannibalism/death_rate.rs) to non-negative mortality and added `mckendrick_upwind_transport`, `characteristic_survival_step`, `renewal_birth_rate`, `juvenile_upwind_transport`, and `step_juvenile_adult_coupled` while keeping all 9 baseline public signatures intact.

2. **Zero-Allocation Characteristic-Shift PDE Solver (`AgeStructuredSolver`) and *Tribolium* LPA Map:**
   - Added [`age_structured.rs`](../../crates/domain_applied/src/applied/cannibalism/age_structured.rs) defining `AgeStructuredSolver`, `AgeStructuredConfig`, and `PredationRegime` (`Diekmann` and `Tribolium`), preallocating ping-pong cohort and kernel `Vec<f64>` buffers up to `MAX_GRID_SIZE` so `step()` and `reconfigure()` execute with zero dynamic heap allocations under `VerifiedAllocator`.
   - Added [`tribolium.rs`](../../crates/domain_applied/src/applied/cannibalism/tribolium.rs) defining stack-allocated `TriboliumLpaModel` for discrete stage-structured *Tribolium* 2-cycle and period-doubling bifurcation sweeps, alongside [`math_explorer/examples/cannibalism_demo.rs`](../../math_explorer/examples/cannibalism_demo.rs).

3. **Interactive `CannibalismTab` Lexis & Bifurcation Lab in `math_explorer_gui`:**
   - Added `math_explorer_gui/src/tabs/cannibalism/` (`// @explorer_feature = "applied"`, `#tab=cannibalism` URL slug, `"Applied Math"` academic domain) hosting `CannibalismTool` with a scrolling 2D Lexis `(t, a)` cohort-density heatmap (`flatten_2d_index` + WCAG high-contrast colormaps), instantaneous age-pyramid and adult trajectory plots, deferred bifurcation sweeps, and starter presets for Diekmann Hopf waves (`beta0 = 45.0`) and *Tribolium* callow 2-cycles (`c_pa = 0.7`).

## Consequences

- `CannibalismModel::derivative`, `AgeStructuredSolver::step`, and `TriboliumLpaModel::step` preserve positivity and execute with zero dynamic heap allocations inside `VerifiedEngine::run_verified`.
- Analytical Diekmann equilibria ($A^* = 3.6715$) agree with subcritical PDE steady states while supercritical fecundity ($\beta_0 = 45.0$) and callow predation ($c_{pa} = 0.7$) render interactive cohort waves and bifurcations across native and `wasm32-unknown-unknown` targets within the 3 MB WASM budget.

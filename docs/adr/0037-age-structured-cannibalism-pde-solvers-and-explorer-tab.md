# ADR 0037: Zero-Allocation Age-Structured Cannibalism PDE Solvers and Interactive Explorer Tab

## Status
Accepted (extends [ADR 0001](0001-dynamic-workspace-discovery.md), [ADR 0016](0016-standardize-crate-documentation-suite.md), and [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md))

## Context
[`papers/cannibalism.tex`](../../papers/cannibalism.tex) derives three coupled formulations of cannibalistic population dynamics:
1. **McKendrick-von Foerster Age-Structured Transport PDE** with Diekmann vulnerability $C(a)$ and Ricker functional response saturation $\Phi(c(t))$:
   $$\frac{\partial n}{\partial t} + \frac{\partial n}{\partial a} = -\mu(t, a, N)\,n(t, a), \qquad n(t, 0) = \int_0^{a_{\max}} \beta(a, N(t))\,n(t, a)\,da$$
2. **van den Bosch Juvenile-Adult Delay Dynamics** advecting juveniles across $[0, \alpha]$ under cannibalistic pressure $I_t + C(a)A(t)$ and coupling maturation flux $n(t, \alpha)$ into $\frac{dA}{dt} = n(t, \alpha) - f(I(t))A(t)$ to produce supercritical dwarf-and-giant Hopf cohort cycles.
3. **Lumped Two-Dimensional Cannibal-Normal ODE** (`CannibalismModel`).

Previously, [`crates/domain_applied/src/applied/cannibalism/`](../../crates/domain_applied/src/applied/cannibalism/mod.rs) exposed only point-wise scalar evaluators (`death_rate`, `mckendrick_von_foerster`, `birth_rate`, `juvenile_dynamics`, `adult_dynamics`) and an unstepped `CannibalismModel` struct, with no PDE time-stepping engine and no interactive GUI visualization in `math_explorer_gui`.

## Decision
1. **Zero-Allocation Characteristic Upwind PDE Solvers in `domain_applied`**:
   - Introduce `CannibalismKernel` in [`death_rate.rs`](../../crates/domain_applied/src/applied/cannibalism/death_rate.rs) encapsulating natural mortality $\nu(a)$, juvenile vulnerability $C(a)$, Ricker saturation $\Phi(c(t))$, and Ricker-damped fecundity $\beta(a, N)$, explicitly guarding $N(t) = 0$ and $C(a) = 0$ against division by zero.
   - Introduce `McKendrickSolver` in [`mckendrick_von_foerster.rs`](../../crates/domain_applied/src/applied/cannibalism/mckendrick_von_foerster.rs) pre-allocating `ages`, `density`, `next_density`, and `mortality` in `new(...)`, enforcing CFL stability ($\Delta t_{\text{sub}} \le \Delta a$) via automatic substepping, and advancing positivity-preserving exponential upwind steps (`#[verified_engine::verified]`, zero heap allocations in `step_upwind`).
   - Introduce `JuvenileAdultParams` and `JuvenileAdultSolver` in [`juvenile_adult_dynamics.rs`](../../crates/domain_applied/src/applied/cannibalism/juvenile_adult_dynamics.rs) coupling $[0, \alpha]$ juvenile advection with maturation flux $n(t, \alpha)$ and lifeboat energy survival $f(I(t))$, alongside `CannibalismModel::step_rk4` in [`two_dimensional_ode.rs`](../../crates/domain_applied/src/applied/cannibalism/two_dimensional_ode.rs) using `RungeKutta4`.
   - Preserve all five existing scalar functions and `CannibalismModel::new` intact for backward API compatibility.
2. **Auto-Discovered `PopulationStructureTab` in `math_explorer_gui`**:
   - Add [`math_explorer_gui/src/tabs/cannibalism/`](../../math_explorer_gui/src/tabs/cannibalism/mod.rs) annotated with `// @explorer_feature = "applied"`, returning `"cannibalism"` from `url_slug()` and `"Applied Math"` from `academic_domain()` without editing `tabs/mod.rs` or `app/mod.rs`.
   - Host `PopulationStructureTool` inside `SimulationFramework::new("cannibalism")` with `PlaybackTransport::in_thread`, `FiniteStateGuard` recovery remedies, `16..=512` `ToolStateBounds` age-grid validation, `PlatformInstant::now()` per-frame substep budgeting for `wasm32`, WCAG high-contrast Lexis `(a, t)` heatmaps, age pyramids, `(J, A)` Hopf phase portraits, 2D ODE trajectory comparisons, and three federated `StarterPresetMetadata` cards.

## Consequences
- **Positive**: `McKendrickSolver::step_upwind` and `JuvenileAdultSolver::step_upwind` execute with zero heap allocations (verified via `VerifiedAllocator`), preserve non-negativity unconditionally, and reproduce analytical characteristic translation, exponential mass decay, and supercritical Hopf cohort cycles.
- **Positive**: Desktop and WebAssembly users can interactively explore Lexis cohort ridges, age pyramids, Hopf phase portraits, and lumped ODE approximations with full URL state sharing (`#tab=cannibalism`) and CSV/JSON/PNG/SVG exports.
- **Neutral**: Expands the `math_explorer_gui` catalog to `22` tabs (`6` in `Applied Math`), `61` registered tools (`38` plot+dataset tools), and `74` starter presets.

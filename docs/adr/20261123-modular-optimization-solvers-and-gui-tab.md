# ADR 20261123: Modular Zero-Allocation Optimization Solvers, Domain Integrations, and Optimization Workbench Tab

## Status
Accepted (extends [ADR 0001](0001-dynamic-workspace-discovery.md), [ADR 0008](0008-finite-state-solver-guard.md), and [ADR 0037](0037-consolidate-optimizer-softmax-and-attention-kernels.md))

## Context
`crates/pure_math/src/pure_math/analysis/optimization.rs` previously combined `SGD`, `Adam`, and an un-solved `L1RegularizedLeastSquares` objective into a single file where `SGD::update_vector` and `SGD::update_matrix` hardcoded a `0.01` step size instead of honoring `self.learning_rate`. Furthermore, linear programming, active-set convex quadratic programming, box-constrained `L-BFGS-B`, augmented Lagrangian, derivative-free (`NelderMead`, `BrentMinimizer`), and multi-objective Pareto solvers were absent, leaving `LoraEnsemble`, `BoxSet`, medical IMRT inverse planning, and `math_explorer_gui` without constrained or multi-objective optimization engines.

## Decision
1. **Modular Preallocated Solver Suite (`pure_math::analysis::optimization` & `roots`)**:
   - Decompose [`crates/pure_math/src/pure_math/analysis/optimization/mod.rs`](../../crates/pure_math/src/pure_math/analysis/optimization/mod.rs) into focused submodules (`first_order`, `proximal`, `linear_programming`, `quadratic_programming`, `projected_gradient`, `lbfgsb`, `augmented_lagrangian`, `derivative_free`, `pareto`) while preserving all existing public re-exports.
   - Fix `SGD::update_vector` and `SGD::update_matrix` to scale gradients by `self.learning_rate`, add ISTA/FISTA proximal solvers with soft-thresholding to `L1RegularizedLeastSquares`, add Two-Phase `SimplexSolver` with Bland's anti-cycling rule and dual shadow prices, add `ActiveSetQpSolver`, `ProjectedGradientSolver`, `LbfgsBSolver`, `AugmentedLagrangianSolver`, `NelderMead`, `ParetoWorkspace` utilities, and 1D `BrentMinimizer` in [`crates/pure_math/src/pure_math/analysis/roots.rs`](../../crates/pure_math/src/pure_math/analysis/roots.rs), all verified zero-allocation under `VerifiedEngine::run_verified` via preallocated workspace structs.
2. **Domain Crate & Existing GUI Tool Integrations**:
   - Wire `L-BFGS-B`, `NelderMead`, and FISTA weight optimization into `LoraEnsemble` ([`crates/domain_applied/src/applied/lorahub/ensemble.rs`](../../crates/domain_applied/src/applied/lorahub/ensemble.rs)), `BoxBounds` projection and projected equilibria into `BoxSet` ([`crates/domain_applied/src/applied/game_theory/equilibrium.rs`](../../crates/domain_applied/src/applied/game_theory/equilibrium.rs)), and constrained `L-BFGS-B` IMRT beam-weight solving plus Tumor-vs-OAR Pareto trade-off frontier extraction into [`crates/domain_physics/src/physics/medical/optimization.rs`](../../crates/domain_physics/src/physics/medical/optimization.rs) and `DoseCalculationTool` ([`math_explorer_gui/src/tabs/medical/imrt_pareto.rs`](../../math_explorer_gui/src/tabs/medical/imrt_pareto.rs)).
   - Add projected/reflective interior box barriers (`[w_min, w_max] x [b_min, b_max]`) with 2D/3D overlays to `LossLandscapeTool` ([`math_explorer_gui/src/tabs/ai/loss_landscape/barriers.rs`](../../math_explorer_gui/src/tabs/ai/loss_landscape/barriers.rs)).
3. **Auto-Discovered `OptimizationTab` (`// @explorer_feature = "applied"`)**:
   - Introduce `OptimizationTab` in [`math_explorer_gui/src/tabs/optimization/mod.rs`](../../math_explorer_gui/src/tabs/optimization/mod.rs) hosting `LinearProgrammingTool`, `QuadraticProgrammingTool`, `ConstrainedContourTool`, and `ParetoFrontierTool`, and bind `optimization = "lorahub.tex"` in [`traceability.toml`](../../traceability.toml).

## Consequences
- **Positive**: Unifies first-order, proximal, linear, quadratic, nonlinear constrained, derivative-free, and multi-objective solvers under strict zero-heap-allocation hot loops across native and `wasm32-unknown-unknown` targets, with interactive visualization in `OptimizationTab`, `DoseCalculationTool`, and `LossLandscapeTool`.
- **Neutral**: Workspace GUI inventory contracts advance to 27 `ExplorerTab` modules, 80 `InteractiveTool` entries, and 113 `StarterPresetMetadata` cards.

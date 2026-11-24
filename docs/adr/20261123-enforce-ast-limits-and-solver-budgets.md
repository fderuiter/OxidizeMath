# ADR 20261123: Enforce AST Complexity Limits and Synchronous Solver Instruction Budgets

## Status
Accepted (extends [ADR 0016](0016-ode-multi-solver-comparative-overlays.md), [ADR 0017](0017-native-ast-expression-parser-and-live-input.md), [ADR 0026](0026-bounded-ast-height-and-iterative-traversal.md), [ADR 0035](0035-compiled-ast-bytecode-and-unified-formula-sandboxes.md), and [ADR 0036](0036-coupled-formula-banks-and-custom-regression-callbacks.md))

## Context
1. **Unbounded Lexical, AST Branch, and Bytecode Growth**: While [ADR 0026](0026-bounded-ast-height-and-iterative-traversal.md) bounded AST parse height to `MAX_PARSE_DEPTH = 64` and `egui_plot/src/latex.rs` capped LaTeX formulas at `MAX_LATEX_INPUT_BYTES = 2048` and `MAX_LATEX_TOKENS = 512`, `crates/math_commons/src/ast/lexer.rs` scanned arbitrarily long strings, `parser.rs` tracked recursion height without counting cumulative nodes across balanced binary/ternary branches, and `compiled.rs`, `latex.rs`, `codegen.rs`, and `glsl.rs` permitted unbounded instruction vectors and output string growth.
2. **Hydration Payload Amplifiers**: Compact `#state=` URLs (`MAX_DECODED_URL_STATE_BYTES = 4096`) and preset/tool JSON snapshots (`MAX_TOOL_STATE_BYTES = 2 MiB`) could pack hundreds of operators or repeated function calls into custom expressions, amplifying parse/compile and evaluation work during synchronous hydration.
3. **Multiplicative Solver & Contour Work**: Even with bounded formulas, combining multi-node custom bytecode with `MAX_ODE_STEPS = 200_000`, multi-solver Euler/Heun/RK4 comparison (`7` stages/step), 8 seeded phase-plane orbits, 5-run finite-difference sensitivity sweeps, or dense `ComplexMappingTool` conformal contour sampling (`2 * 100 * 161` samples) could trigger tens to hundreds of millions of synchronous VM instructions on the UI thread.

## Decision
1. **Explicit AST Input, Node, Instruction, and Output Caps (`crates/math_commons/src/ast/limits.rs`)**:
   - Enforced `MAX_EXPR_BYTES = 2048` ($O(1)$ pre-scan byte check) and `MAX_EXPR_TOKENS = 512` in `lexer.rs`, matching `egui_plot/src/latex.rs`.
   - Tracked cumulative node creation across all branches in `parser.rs` capped at `MAX_AST_NODES = 256` (`ParseError::NodeLimitExceeded`).
   - Enforced `MAX_COMPILED_INSTRUCTIONS = 256` in `CompiledExpr` (`EvalError::InstructionLimitExceeded`) with zero heap allocations on `exec_bytecode` hot paths, and bounded serialized output growth in `latex.rs`, `codegen.rs`, and `glsl.rs` (`MAX_LATEX_OUTPUT_BYTES = MAX_CODEGEN_OUTPUT_BYTES = MAX_GLSL_OUTPUT_BYTES = 16_384`).
2. **Sandbox Validation & Last-Valid Retention**:
   - Hardened `MathExpressionInputBar` (`math_input.rs`) and `CoupledExpressionBank` (`coupled.rs`, `MAX_COUPLED_BANK_INSTRUCTIONS = 384`) to enforce byte, node, and instruction caps, cap interactive single-line edits at `MAX_EXPR_BYTES`, bound preview LaTeX strings, and retain `last_valid_expr` / `last_valid_compiled` (`active_compiled`) across invalid or oversized edits.
   - Extended `ReplicatorPresetConfig::validate_symbolic` (`presets/configs.rs`) to validate symbolic payoff formulas against `parse_equation` and `CompiledExpr::compile`.
3. **Synchronous Instruction Budgets & Accessible Status Feedback**:
   - Introduced deterministic instruction budgets (`instructions * evaluations`) across `OdeSolverTool` primary + seeded orbits (`MAX_ODE_SYNC_INSTRUCTIONS = 2_000_000` in `ode/budget.rs`), aligned Euler/Heun/RK4 comparison trajectories (`MAX_ODE_COMPARISON_SYNC_INSTRUCTIONS = 1_500_000`), `SensitivityEngine` perturbation sweeps (`MAX_SENSITIVITY_SYNC_INSTRUCTIONS = 2_000_000` in `sensitivity.rs`), and `ComplexMappingTool` conformal grid contours (`MAX_CONFORMAL_SYNC_INSTRUCTIONS = 1_200_000` in `complex_mapping/cache.rs`).
   - Rejected invalid or over-budget custom formulas deterministically during `OdeSolverTool` and `ComplexMappingTool` state/URL hydration, and surfaced inline status notices paired with deduplicated WAI-ARIA polite/assertive live-region announcements (`accessibility.rs`).

## Consequences
- **Positive**: Eliminates parser exponential branch blow-ups, unbounded transpiler allocations, and synchronous UI-thread freezes from `4,096`-byte `#state=` URLs, `2 MiB` JSON states, and `200,000`-step multi-node custom ODE or conformal grid evaluations without introducing asynchronous worker complexity.
- **Positive**: Preserves bit-identical numerical trajectories across all built-in presets, starter gallery entries, and symbolic replicator payoff presets while keeping screen-reader users informed of validation errors and work-budget scaling.

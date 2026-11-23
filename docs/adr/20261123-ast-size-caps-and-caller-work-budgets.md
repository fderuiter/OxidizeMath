# ADR 20261123: Enforce AST Size Caps, Bounded LaTeX Rendering, and Caller Work Budgets

## Status
Accepted (extends [ADR 0026](0026-bounded-ast-height-and-iterative-traversal.md), [ADR 0035](0035-compiled-ast-bytecode-and-unified-formula-sandboxes.md), and [ADR 0037](0037-ast-arity-scope-validation-and-scaled-latex.md))

## Context
[`crates/math_commons/src/ast/lexer.rs`](../../crates/math_commons/src/ast/lexer.rs) and [`crates/math_commons/src/ast/parser.rs`](../../crates/math_commons/src/ast/parser.rs) previously bounded only recursion depth and tree height (`MAX_PARSE_DEPTH = 64`). Balanced binary formulas of modest height (for example height `10` with `2,047` AST nodes or height `19` with `1,048,575` AST nodes) bypassed the height guard, allocated exponential `Expr` trees and megabyte `Expr::to_latex` strings, and stalled interactive simulation loops in [`OdeSolverTool`](../../math_explorer_gui/src/tabs/analysis/ode.rs) and [`ComplexMappingTool`](../../math_explorer_gui/src/tabs/analysis/complex_mapping.rs) when hydrating `#state=` share URLs or `.json` presets.

## Decision
1. **Enforce Three-Tier Lexer and Parser Caps in `math_commons::ast`**:
   - Cap UTF-8 input length at `MAX_AST_INPUT_BYTES = 2_048` and non-EOF tokens at `MAX_AST_TOKENS = 1_024` inside [`tokenize`](../../crates/math_commons/src/ast/lexer.rs), mirroring `MAX_LATEX_INPUT_BYTES` in [`egui_plot/src/latex.rs`](../../egui_plot/src/latex.rs).
   - Cap total allocated `Expr` nodes per parsed formula at `MAX_AST_NODES = 256` inside [`Parser`](../../crates/math_commons/src/ast/parser.rs) (mirroring `MAX_MATH_BOX_NODES = 256`), decrementing a shared `remaining_nodes` budget across leaf, unary, function, and `wrap_binary` constructors and returning `ParseError::NodeLimitExceeded(256)` with plain-language diagnostics while preserving existing public AST signatures and height-`64` chain tests.
2. **Bound `Expr::to_latex` Expansion and Output Bytes**:
   - Bound node expansion (`256` `Eval` items) and output length (`MAX_LATEX_OUTPUT_BYTES = 4_096`) in [`crates/math_commons/src/ast/latex.rs`](../../crates/math_commons/src/ast/latex.rs), emitting `\dots` while draining pending closing delimiters so programmatic `Expr` trees remain syntactically balanced and bounded.
3. **Reject Invalid `custom_expr` Payloads and Cap Caller Work Budgets**:
   - Reject invalid or oversized `custom_expr` and `coupled_exprs` payloads during deserialization in [`math_explorer_gui/src/tabs/analysis/ode/budget.rs`](../../math_explorer_gui/src/tabs/analysis/ode/budget.rs) and [`math_explorer_gui/src/tabs/analysis/complex_mapping.rs`](../../math_explorer_gui/src/tabs/analysis/complex_mapping.rs).
   - Cap `steps * ast_nodes` in [`OdeSolverTool`](../../math_explorer_gui/src/tabs/analysis/ode/ui.rs) (`MAX_CUSTOM_ODE_WORK_BUDGET = 250_000` for interactive trajectories and `MAX_SENSITIVITY_ODE_WORK_BUDGET = 100_000` per sensitivity perturbation) and `grid_points * ast_nodes` in [`ComplexMappingTool`](../../math_explorer_gui/src/tabs/analysis/complex_mapping/ui.rs) (`MAX_COMPLEX_MAPPING_WORK_BUDGET = 400_000`).

## Consequences
- **Positive**: Balanced exponential AST formulas are rejected in $O(1)$–$O(N_{\text{cap}})$ time, `Expr::to_latex` never allocates unbounded preview strings, and custom ODE / conformal mapping share links and presets hydrate well within interactive `< 250` ms frame budgets.
- **Positive**: Zero heap allocations are preserved in `Expr::eval_2d`, `Expr::eval_complex_z`, and `CompiledExpr` evaluators under `VerifiedAllocator`, and `api_baseline.json` public signatures remain unchanged.
- **Neutral**: Formulas exceeding `256` AST nodes, `1,024` tokens, or `2,048` bytes surface plain-language syntax errors, and high-complexity valid formulas (`<= 256` nodes) automatically clamp interactive step size or contour grid density to stay within frame budgets.

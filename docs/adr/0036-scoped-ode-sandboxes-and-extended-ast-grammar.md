# ADR 0036: Scoped ODE Formula Sandboxes and Extended AST Delimiter/Power Grammar

## Status
Accepted (extends [ADR 0008](0008-finite-state-guard-solver-divergence.md), [ADR 0017](0017-native-ast-expression-parser-and-live-input.md), [ADR 0026](0026-bounded-ast-height-and-iterative-traversal.md), and [ADR 0035](0035-compiled-ast-bytecode-and-unified-formula-sandboxes.md))

## Context
1. **Unscoped Custom ODE Evaluation & False Step-0 Divergence**: `OdeSolverTool` constructed `MathExpressionInputBar::new` without coordinate axis scoping, cloned `Expr` and `HashMap<String, f64>` per trajectory run in `DynamicAstOde`, evaluated free parameters outside `{k, r, cap_k, K}` to `NaN`, marked transient syntax edits as step-0 solver divergences, rendered a duplicate hardcoded `k` slider alongside custom formulas, and omitted discovered custom parameters from `SensitivityEngine` and state persistence.
2. **Absolute-Value Pipe Delimiters & Trigonometric Power Shorthand**: While `Expr::to_latex` rendered `Func::Abs` as `\left|...\right|`, `math_commons::ast` rejected `|x|` vertical bars in `tokenize` and rejected Desmos-style unary function exponents such as `sin^2(t)` in `parse_ident_or_func`.

## Decision
1. **Pipe Absolute-Value Delimiters & Unary Function Exponents in `math_commons::ast`**: Add `TokenKind::Pipe` (`|`) in `crates/math_commons/src/ast/lexer.rs` and track `in_abs` in `crates/math_commons/src/ast/parser.rs` (resetting inside parenthesized groups and argument lists) so `|y|`, `2|y|`, `|x||y|`, and `||x| - 1|` build `Func::Abs` while unclosed `|y + 1` returns `ParseError::UnexpectedToken`. Extend `parse_ident_or_func` to accept `^` on unary functions (`func.arity() == 1`), parsing exponents at binding power `7` and wrapping `BinaryOp::Pow` within `MAX_PARSE_DEPTH` (`64`).
2. **Scoped Zero-Allocation `OdeSolverTool` Custom Sandboxes**: Configure `OdeSolverTool`'s `MathExpressionInputBar` with `.with_axes(&["t", "y"])` and seed `ParameterSliderBank` with `k = 1.0`, `r = 1.0`, and `cap_k = 10.0`. Borrow `&CompiledExpr` and `&[f64]` slices in `DynamicAstOde` for zero-allocation `CompiledExpr::eval_2d` stepping, retain `active_compiled()` across transient edits without false step-0 divergence, suppress the duplicate `k` slider in `OdePreset::Custom`, evaluate discovered custom parameters plus `y0` in `SensitivityEngine`, and persist `custom_params` with legacy `(param_k, param_r, param_cap_k)` fallback.

## Consequences
- **Positive**: Unifies all eight `MathExpressionInputBar` sandboxes on scoped `CompiledExpr` bytecode and `ParameterSliderBank` sliders while enabling `|x|` and `sin^2(x)` notation across every real and complex formula input without changing public API signatures in `api_baseline.json`.
- **Positive**: Eliminates per-run `Expr`/`HashMap` cloning in custom ODE integration and sensitivity sweeps while preserving `FiniteStateGuard` divergence recovery on genuine numerical blow-ups such as `1 / (y - 1)` at `y(0) = 1`.

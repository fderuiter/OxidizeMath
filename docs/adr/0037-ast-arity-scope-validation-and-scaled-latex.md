# ADR 0037: Parse-Time AST Arity Enforcement, Scoped Input Validation, and Scaled LaTeX Delimiter Rendering

## Status

Accepted

## Context

Three coupled gaps degraded the custom-equation and formula-reading experience across OxidizeMath:

1. **Garbled LaTeX Delimiters, Over-Dots, Ellipses, and Set Symbols**: `crates/math_commons/src/ast/latex.rs` wraps parenthesized binary expressions, `Func::Abs`, and function calls in `\left` and `\right` commands per ADR 0017 and ADR 0026, while seven `TheoryDescribable` descriptions across domains emit `\left`, `\right`, `\dot`, `\dots`, and `\mathbb`. `egui_plot/src/latex.rs` previously lacked handlers for these commands and rendered raw command words (`left`, `right`, `dot`, `dots`, `mathbb`) on both the 2D `MathBox` canvas and `LayoutJob` paths without scaling delimiters around tall fractions.
2. **Deferred Arity Errors and Unchecked Tool Variable Scopes**: `crates/math_commons/src/ast/parser.rs` previously permitted zero-argument and wrong-argument-count parenthesized function calls (such as `sin()` or `atan(y, t)`) to succeed during `parse` and fail only during evaluation or bytecode compilation. Meanwhile, `MathExpressionInputBar` lacked variable-scope policies to reject foreign coordinate identifiers (such as `sin(x)` in scalar `(t, y)` ODE mode or `z^2 + c` in conformal `w = f(z)` mapping).
3. **False Solver Divergence, Stale/Clamped Complex Readouts, and Logistic Parameter Metadata**: Entering an invalid custom ODE after a numerical blow-up could leave `Solver Divergence Warning` visible in `OdeSolverTool` despite ADR 0008 reserving recovery banners strictly for numerical blow-up. In `ComplexMappingTool`, invalid custom inputs evaluated stale fallback bytecode or rendered numeric inspection readouts instead of `Undefined`. Finally, custom logistic ODEs (`r * y * (1 - y / cap_k)` / `K`) lacked human-readable growth-rate and carrying-capacity slider labels/ranges and custom `param_r` / `param_cap_k` finiteness clamping in `ode/recovery.rs`.

## Decision

1. **Scaled `MathBox::Delimited` and `MathBox::Accent` Rendering in `egui_plot`**:
   - Extend `egui_plot::MathBox` with `Delimited` (`MathDelimiter`) and `Accent` (`MathAccent`), plus symbol mappings for `\dots`, `\ldots`, `\cdots`, `\vdots`, `\ddots`, and `\mathbb{...}`.
   - Match `\left...\right` delimiter pairs (and auto-upgrade matched plain `(...)` / `[...]` enclosing tall `Fraction` or `Sqrt` nodes) to paint height-scaled vector curves, brackets, pipes, and braces on the 2D canvas, vertically center `MathBox::Row` children along fraction axes, and emit clean Unicode glyphs and combining accents (`\u{0307}`, `\u{0308}`) in `LayoutJob` text.
2. **Parse-Time Function Arity Enforcement in `math_commons::ast`**:
   - Add `ParseError::WrongArgumentCount { func, expected, found, span }` to `crates/math_commons/src/ast/parser.rs` and enforce `args.len() == func.arity()` in `parse_parenthesized_func`, reporting plain-language 1-indexed errors across both CLI/facade and GUI callers while preserving variable implicit multiplication (ADR 0029), bounded depth `64` (ADR 0026), and zero-allocation evaluation (ADR 0035).
3. **Variable-Scope Policies and Parameter Slider Specs in `MathExpressionInputBar`**:
   - Add iterative `VariableScopePolicy` (`with_allowed_vars` for closed tool scopes such as `ComplexMappingTool` `[z, a]` and `WaveSimulatorTool` `[x, v0, width]`, and `with_disallowed_vars` for foreign coordinate guardrails in `OdeSolverTool` while preserving ADR 0035/0036 free-parameter discovery).
   - Extend `ParameterSliderBank` with `ParamSliderSpec` (`with_param_spec`) so `r` (`r (Growth Rate)`, `0.0..=5.0`) and `cap_k` / `K` (`K (Carrying Capacity)`, `1.0..=100.0`, default `10.0`) expose domain-labeled sliders and finite sensitivity metrics.
4. **Strict Separation of Expression Errors from Numerical Blow-Up and Undefined Complex Outputs**:
   - Guard `OdeSolverTool::recalculate_active` so expression syntax/scope errors never raise `diverged` or `Solver Divergence Warning`, and extend `ode/recovery.rs` finiteness checks and clamping to cover `param_r` and `param_cap_k` in `OdePreset::Custom`.
   - Evaluate `ComplexMappingTool::eval_z` via `math_input.compiled()` so invalid custom expressions omit mapped `w`-plane contours and display `Undefined` inspection metrics.

## Consequences

- **Positive**: All 20 AST functions, custom input presets, and audited `TheoryDescribable` equations render with scaled delimiters, over-dots, ellipses, and blackboard symbols without leaked LaTeX command text. Invalid function arities and out-of-scope variables are caught immediately inline with WCAG 2.2 AA error badges and deduplicated ARIA announcements, eliminating false ODE divergence banners and numeric `ComplexMappingTool` clamping.
- **Negative / Trade-offs**: `ParseError` and `MathBox` gain public enum variants, requiring `crates/unified_verification/api_baseline.json` regeneration to maintain API drift tracking.

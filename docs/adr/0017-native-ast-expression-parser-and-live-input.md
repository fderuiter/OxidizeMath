# ADR 0017: Native AST Expression Pratt Parser and Live Input Integration

## Context
In mathematical exploration and visualization tools, users frequently require dynamic equation inputs for ODE solving ($y' = f(t, y)$) and complex conformal mapping ($w = f(z)$) rather than relying solely on pre-compiled hardcoded equations. External expression parsing crates can introduce substantial dependencies or incompatible WASM build targets (`wasm32-unknown-unknown`). Additionally, a unified parser needs to support implicit multiplication (e.g., $2x$, $2(x+1)$, $k y \sin(t)$), equation prefix notation ($y' = \dots$, $w = \dots$), robust recursion depth bounds against stack exhaustion, and LaTeX formula generation for interactive formula previews.

## Decision
We implemented a zero-dependency Pratt-parser AST evaluator in `crates/math_commons/src/ast/` and integrated it with an interactive input bar and visualization tabs:
1. **Zero-Dependency Pratt Parser & Lexer**: Implemented in `crates/math_commons/src/ast/` with support for binary operators (`+`, `-`, `*`, `/`, `^`), unary operators (`+`, `-`), implicit multiplication, variable bindings, and standard transcendental functions (`sin`, `cos`, `tan`, `exp`, `ln`, `log10`, `sqrt`, `abs`, `sinh`, `cosh`, `tanh`, `asin`, `acos`, `atan`).
2. **Safety & Depth Bounds**: Enforced strict recursion depth bounds (`MAX_PARSE_DEPTH = 64`, `MAX_EVAL_DEPTH = 64`) to prevent stack overflow vulnerabilities from maliciously nested expressions.
3. **Prefix Normalization & LaTeX Generation**: Added stripping of common equation LHS patterns (`y' = ...`, `w = ...`, `f(t, y) = ...`) and AST-to-LaTeX serialization compatible with `egui_plot::render_math_label`.
4. **Optimized Specialized Evaluators**: Provided `eval_2d(t, y)` and `eval_complex_z(z)` execution routines tailored for high-frequency inner loops in ODE numerical steppers and 2D conformal grid evaluation.
5. **Interactive UI Component**: Created `MathExpressionInputBar` in `math_explorer_gui/src/framework/math_input.rs` with live syntax validation and LaTeX preview, wired into ODE Solver and Complex Mapping tabs.

## Consequences
- **Extensibility**: Users can define and simulate arbitrary mathematical expressions in real time without recompilation.
- **Portability & Safety**: Clean zero-dependency implementation maintains full WebAssembly compatibility and guards against unbounded recursion.
- **Traceability**: All architectural records and integrity verification checks pass cleanly.

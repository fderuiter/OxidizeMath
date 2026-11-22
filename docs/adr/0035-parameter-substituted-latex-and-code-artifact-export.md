# ADR 0035: Parameter-Substituted LaTeX and Multi-Target Code Artifact Export

- **Status:** Accepted (extends [ADR 0017](0017-native-ast-expression-parser-and-live-input.md), [ADR 0024](0024-deferred-modal-exports-and-plot-primitives.md), and [ADR 0026](0026-bounded-ast-height-and-iterative-traversal.md))
- **Date:** 2026-11-22

## Context

Previously, `math_explorer_gui` and `egui_plot` exported numerical snapshots (`.csv`), visual captures (`.svg`, `.png`), and session state presets (`.json`), while `Expr::to_latex` and `reflective_ui::render_formula` exposed only uncalibrated symbolic strings. Users calibrating mathematical models or curve fits could not directly export parameter-substituted LaTeX equations, standalone Rust 2024 evaluators, or `numpy`/`scipy` Python reproduction scripts without manually rewriting equations and risking transcription errors across operator precedence (`^` vs `.powf(...)` vs `**`), log-axis regression transforms, and unit scales.

At the same time, any code generation pipeline had to obey three architectural constraints:
1. **Bounded Iterative AST Traversal ([ADR 0026](0026-bounded-ast-height-and-iterative-traversal.md)):** AST formatting and transpilation must execute on an explicit heap frame stack bounded by `MAX_EVAL_DEPTH` (`64`) with zero call-stack recursion (NASA Power of 10 Rule 1).
2. **Zero Hot-Path Overhead & Bundle Budget:** Code and LaTeX string construction must execute strictly on cold preview, clipboard copy, and deferred file download events (`ExportQueue`), adding zero heap allocations to interactive simulation frames and zero third-party Cargo crates to keep the compressed WASM bundle under `3 MB` and `'self'` CSP intact.
3. **Accessible Deferred Export Lifecycle ([ADR 0024](0024-deferred-modal-exports-and-plot-primitives.md)):** Modal focus management must reuse `egui_plot::ModalLifecycle`, file saves must flow through `egui_plot::file_io::save_bytes` with explicit MIME mappings, and all copy/save actions or non-finite parameter guards must announce confirmations via `TwinSlotLiveRegion`.

## Decision

1. **Bounded Iterative AST Code Generators (`crates/math_commons/src/ast/codegen.rs` & `ast/latex.rs`):**
   - Add `Expr::to_latex_with_params(&self, params: &HashMap<String, f64>) -> Result<String, EvalError>`, `Expr::to_rust_expr`, `Expr::to_rust_fn`, `Expr::to_python_expr`, and `Expr::to_python_ode_rhs`.
   - Drive all three target emitters with explicit `Vec` frame stacks enforcing `depth <= MAX_EVAL_DEPTH` (`64`) without recursion.
   - Substitute finite scalar parameters from `params` (wrapping negative literals in parentheses where operator precedence requires), reject `NaN`/`±Inf` parameters with `EvalError::DomainError`, keep built-in constants (`pi`, `e`, `tau`) and bound independent/state variables (`t`, `y`, `x`, `z`) symbolic, emit `.powf(...)` and `f64` methods in Rust (prefixing unused function parameters with `_` so `rustc --edition 2024 -D warnings` passes cleanly), and emit `**` and `np.*` vectorizable calls in Python.

2. **`CodeArtifact`, `InteractiveTool`, and Deferred `ExportFormat` Extensions (`math_explorer_gui/src/framework/` & `egui_plot/src/file_io.rs`):**
   - Introduce `CodeArtifact` (`symbolic_latex`, `calibrated_latex`, `rust_source`, `python_source`) in `math_explorer_gui/src/framework/code_export.rs` and default `fn export_code_artifact(&self) -> Option<CodeArtifact> { None }` on `InteractiveTool`.
   - Extend `ExportCapabilities` with `code: bool` and `code_disabled_reason: &'static str` (`ExportCapabilities::plot_with_code()`), add `ExportFormat::{Tex, Rust, Python}` (`stem_suffix = "model"`) in `export_queue.rs`, and register `.tex` (`application/x-tex;charset=utf-8`), `.rs` (`text/x-rust;charset=utf-8`), and `.py` (`text/x-python;charset=utf-8`) in `egui_plot::file_io::infer_mime_type`.

3. **Accessible `Code & LaTeX` Modal Drawer and Inline Formula Actions:**
   - Implement `CodeExportModalState` (`math_explorer_gui/src/framework/code_export_modal.rs`) atop `egui_plot::ModalLifecycle` (`Escape` close, focus restoration, `LaTeX` / `Rust` / `Python` tabs, symbolic vs. calibrated LaTeX toggle, read-only monospace code viewer, and `Copy` / `Download` triggers routed through `TwinSlotLiveRegion`).
   - Wire inline `Copy LaTeX` / `Copy Calibrated LaTeX` and `Copy` / `Save .tex` controls into `MathExpressionInputBar` (`math_input.rs`) and `reflective_ui::render_formula_with_export`.

4. **Six Flagship Tool Templates and `FitResult` Analytical Codegen:**
   - Implement calibrated standalone Rust 2024 and `numpy`/`scipy` templates across `OdeSolverTool` (RK4 / `solve_ivp`), `ComplexMappingTool` (`Complex` arithmetic / `numpy` grid mapping), `RiemannIntegrationTool` (Riemann & Simpson quadrature / `scipy.integrate.quad`), `BeamProfilingTool` (Beer-Lambert attenuation), `CapacityFadeTool` (Arrhenius SEI + cyclic fade), and `SirTool` (RK4 compartmental epidemic dynamics).
   - Add `FitResult::to_latex`, `FitResult::to_rust_fn`, and `FitResult::to_python_fn` (`egui_plot/src/fit_export.rs`) supporting linear ($y(x)$), semilog ($10^{f(x)}$), and log-log ($10^{f(\log_{10} x)}$) coordinate regressions, alongside inline copy and `.tex`/`.rs`/`.py` save buttons in `egui_plot`'s fit badge overlay.

## Verification Commands

```bash
# AST iterative codegen unit tests and zero-allocation AST evaluator guard
cargo test -p math_commons --all-features

# FitResult linear/semilog/log-log codegen parity and rustc 2024 checks
cargo test -p egui_plot --lib --all-features -- fit_export --test-threads=1

# Flagship tool Rust 2024 compilation, Python execution, and modal/ARIA tests
cd math_explorer_gui && cargo test --bins --all-features -- code_export_tests --test-threads=1
```

## Consequences

- **Positive:** Researchers and students can copy or download calibrated LaTeX equations, zero-dependency Rust 2024 sources, and runnable Python (`numpy`/`scipy`) scripts directly from interactive simulations and curve-fit overlays with $10^{-9}$ numerical parity.
- **Positive:** All AST code generation runs iteratively in $O(\text{depth})$ heap frames bounded by `MAX_EVAL_DEPTH = 64`, adds zero simulation-loop allocations, introduces zero external Cargo dependencies, and preserves the `< 3 MB` WASM release budget.
- **Neutral:** Domain tools without closed-form or ODE bookmark templates continue returning `None` from `InteractiveTool::export_code_artifact`, keeping the `Code & LaTeX` button deterministically disabled with an explicit tooltip reason.

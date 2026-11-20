# ADR 0027: Remove Unread AST Accumulators from Verification Macros

## Status
Accepted (extends [ADR 0019](0019-purge-dead-code-bypasses.md) and [ADR 0022](0022-cleanup-verification-macros-dead-code.md))

## Context
`crates/verified_engine_macros` retained three unread AST accumulators that performed unnecessary heap allocations and syntax-tree traversals during procedural macro expansion:
1. **Unread `DeepAstVisitor::assertions` Counter**: `crates/verified_engine_macros/src/lib.rs` defined `assertions: usize` on `DeepAstVisitor` and incremented it inside `visit_macro` on every `assert!`, `assert_eq!`, `assert_ne!`, and `debug_assert!` invocation across `#[verified]` functions, even though `verified` never read `visitor.assertions` after assertion-density counting moved to `AssertVisitor` in `crates/oxidize_core/src/ast_visitor.rs`.
2. **Unread `RustOpVisitor::vars` Set**: `crates/verified_engine_macros/src/semantic_integrity.rs` populated `vars: HashSet<String>` inside `visit_expr_path` for every identifier path in `#[embed_theory]` functions without reading `visitor.vars`.
3. **Discarded `parse_latex_math` Variable Vector**: `crates/verified_engine_macros/src/latex_parser.rs` allocated a `String` for every alphabetic token into `vars: Vec<String>` and returned `(Vec<String>, Vec<MathOp>)`, while its sole caller `verify_semantic_integrity` immediately discarded `.0`.

## Decision
1. **Remove `assertions` from `DeepAstVisitor`**: Delete `assertions`, `visit_macro`, and its `#[allow(clippy::collapsible_if)]` attribute from `DeepAstVisitor` in `crates/verified_engine_macros/src/lib.rs`, preserving direct-recursion and 60-statement enforcement alongside runtime metric injection in `#[verified]`.
2. **Simplify `parse_latex_math` and `RustOpVisitor`**: Remove `vars` and the alphabetic variable-collection branch from `parse_latex_math` in `crates/verified_engine_macros/src/latex_parser.rs` so it returns `Vec<MathOp>` directly while keeping LaTeX command parsing (`\frac`, `\times`, `\cdot`). Delete `RustOpVisitor::vars` and `visit_expr_path` in `crates/verified_engine_macros/src/semantic_integrity.rs`.
3. **Add Unit Test Coverage**: Add unit tests in `crates/verified_engine_macros/src/latex_parser.rs` and `crates/verified_engine_macros/src/semantic_integrity.rs` covering ASCII and LaTeX operator extraction, structural command filtering, unary negation, exponential methods, and missing-operation diagnostics.

## Consequences
- **Positive**: Eliminates unused `String`, `Vec<String>`, and `HashSet<String>` allocations and redundant `visit_macro`/`visit_expr_path` AST hooks across all `#[verified]` and `#[embed_theory]` macro expansions, aligning `verified_engine_macros` with [ADR 0019](0019-purge-dead-code-bypasses.md) and [ADR 0022](0022-cleanup-verification-macros-dead-code.md).
- **Neutral**: Preserves all compile-time safety diagnostics (`#[verified]` recursion/statement limits and `#[embed_theory]` semantic operator checks), assertion traceability via `AssertVisitor` in `crates/oxidize_core/src/ast_visitor.rs`, and public macro signatures.

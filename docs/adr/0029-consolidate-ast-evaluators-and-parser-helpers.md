# ADR 0029: Consolidate AST Evaluators and Parser Helpers

## Status
Accepted (extends [ADR 0017](0017-native-ast-expression-parser-and-live-input.md) and [ADR 0023](0023-separate-metadata-ids-from-display-labels.md))

## Context
`crates/math_commons/src/ast/` accumulated structural duplication across its evaluation and parsing submodules:
1. **Duplicated Real and Complex AST Walkers and Dispatch Tables**: `crates/math_commons/src/ast/eval.rs` maintained four separate recursive `Expr` tree walkers (`eval_internal`, `eval_2d_internal`, `eval_complex_internal`, and `eval_complex_z_internal`), two 14-arm `Func` dispatch matches (`eval_func_f64` and `eval_func_complex`), two binary operator helpers (`eval_binop_f64` and `eval_binop_complex`), and two ASCII case-insensitive constant tables (`lookup_const_f64` and `lookup_const_complex`).
2. **Repeated `ParseError` Templates and Token Predicates**: `crates/math_commons/src/ast/parser.rs` repeated all five plain-language `ParseError` format strings across `#[error(...)]` attributes and `ParseError::user_message` (unlike `LexError` in `crates/math_commons/src/ast/lexer.rs`), and defined identical `matches!` predicates in `Parser::can_start_prefix` and `Parser::can_start_implicit_mul`.

## Decision
1. **Unify Real and Complex Evaluation via Private `EvalScalar` Trait**: Define a private `EvalScalar: num_complex::ComplexFloat<Real = f64>` trait in `crates/math_commons/src/ast/eval.rs` implemented for `f64` and `Complex64` encapsulating `NAN`, `from_real`, division-by-zero guards (`rhs == 0.0` for `f64` vs `rhs.norm_sqr() < 1e-12` for `Complex64`), exponentiation (`powf` vs real/complex `powf`/`powc`), and complex-only `i` constant resolution. Replace the duplicated walkers and lookup tables with generic `Expr::eval_checked`, `Expr::eval_fast`, `eval_unary`, `eval_binop`, `eval_func`, and `lookup_const` helpers.
2. **Single-Source `ParseError` Formatting and Token Atom Predicate**: Delegate `ParseError::user_message` in `crates/math_commons/src/ast/parser.rs` directly to `self.to_string()` so `#[error(...)]` attributes serve as the single template definition, and merge `can_start_prefix` and `can_start_implicit_mul` into `Parser::can_start_atom`.

## Consequences
- **Positive**: Eliminates duplicated `Expr` traversal logic, 14-arm transcendental dispatch tables, constant lookups, token predicates, and error message templates with a net code reduction across `crates/math_commons/src/ast/`.
- **Positive**: Preserves zero heap allocations across `eval`, `eval_2d`, `eval_complex`, and `eval_complex_z` under `VerifiedAllocator` and `VerifiedEngine::run_verified`.
- **Neutral**: All public AST types, function signatures in `crates/unified_verification/api_baseline.json`, numerical results, and 1-indexed error messages remain unchanged.

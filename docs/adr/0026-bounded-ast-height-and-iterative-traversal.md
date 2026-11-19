# ADR 0026: Bounded AST Subtree Height and Iterative Tree Traversal

## Status
Accepted (extends [ADR 0017](0017-native-ast-expression-parser-and-live-input.md))

## Context
[ADR 0017](0017-native-ast-expression-parser-and-live-input.md) introduced `math_commons::ast` with `MAX_PARSE_DEPTH = 64` and `MAX_EVAL_DEPTH = 64` to guard against deeply nested expressions. However, three stack-safety gaps remained:
1. **Left-Associative Loop Bypass**: `Parser::parse_expr` checked `depth > MAX_PARSE_DEPTH` only when entering recursive right-hand, unary, parenthesized, or function-argument sub-calls. Left-associative binary operator chains (e.g., `1 + 1 + ...`) and implicit multiplications (`x x x ...`) wrapped `lhs` inside an unbounded `loop` at constant call-stack `depth`, constructing left-skewed `Expr` trees of arbitrary height.
2. **Recursive LaTeX Formatting**: `Expr::to_latex` / `Expr::to_latex_prec` traversed left and right subtrees recursively, overflowing the thread stack when `MathExpressionInputBar::reparse` formatted a deep AST imported from presets or session state.
3. **Compiler-Generated Recursive `Drop`**: `Expr` owns `Box<Expr>` and `Vec<Expr>` children, causing compiler-generated `Drop` glue to recurse proportionally to tree height during deallocation.

## Decision
1. **Exact AST Subtree Height Enforcement in `Parser`**: Update `Parser::parse_expr`, `parse_prefix`, `parse_unary`, `parse_ident_or_func`, and `parse_parenthesized_func` in `crates/math_commons/src/ast/parser.rs` to return `(Expr, usize)` tracking the exact AST edge height alongside call-stack `depth`. Reject any binary, implicit-multiplication, unary, or function node whose resulting subtree height exceeds `MAX_PARSE_DEPTH` (`64`) with `Err(ParseError::RecursionLimitExceeded(MAX_PARSE_DEPTH))`.
2. **Iterative Worklist `Drop` for `Expr`**: Implement `Drop for Expr` in `crates/math_commons/src/ast/expr.rs` with a zero-allocation fast path for leaf variants (`Number`, `Variable`) and an explicit heap `Vec<Expr>` worklist (`take_children`) for compound nodes, eliminating recursive drop glue without `unsafe` code.
3. **Iterative Frame-Stack LaTeX Serialization**: Refactor `Expr::to_latex_prec` in `crates/math_commons/src/ast/latex.rs` to drive an explicit `Vec<LatexFrame<'_>>` stack appending into a single `String` buffer, removing recursion in `to_latex` while preserving identical LaTeX output across all operator precedences and built-in functions.

## Consequences
- **Positive**: Expressions exceeding height `64`—including left-associative chains, implicit multiplications, and mixed parenthesized/unary subtrees—are rejected deterministically with `ParseError::RecursionLimitExceeded(64)` across interactive typing, preset JSON imports, and session state loads.
- **Positive**: Programmatically constructed `Expr` trees of arbitrary depth format via `to_latex()` and deallocate via `Drop` in bounded stack space, aligning `crates/math_commons/src/ast/` with NASA Power of 10 Rule 1 (no recursion).
- **Neutral**: Valid expressions up to AST height and parse depth `64` behave identically with zero public API changes.

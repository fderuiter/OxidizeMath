# 13. Consolidate Binary Operator Macros into math_commons

## Status
Accepted

## Context
Previously, `crates/pure_math/src/math_types.rs` defined a private local `impl_binop!` macro implementing symmetric binary operator traits (`Add`, `Sub`, `Mul`, `Div`, `Rem`) for value and reference operand combinations. Meanwhile, `crates/math_commons/src/primitives.rs` defined an `impl_binop!` macro for bounded floating-point primitives. This resulted in duplicate macro definitions across crates, differing patterns, and unnecessary maintenance overhead.

## Decision
1. Extended `impl_binop!` in `crates/math_commons/src/primitives.rs` to support the multi-token pattern `($type:ident, $trt:ident, $mth:ident)`.
2. Fully qualified all standard operator trait paths (`std::ops::$trt`) in `math_commons::impl_binop!` to allow consumption across any crate without requiring explicit imports.
3. Removed the redundant local macro definition in `crates/pure_math/src/math_types.rs` and replaced invocations with `math_commons::impl_binop!(Integer, ...)` for `Add`, `Sub`, `Mul`, `Div`, and `Rem`.
4. Cleaned up unused imports in `crates/pure_math/src/math_types.rs` and kept the file size within the 500-line constraint.

## Consequences
- Single authoritative definition for binary operator generation across standard Rust numeric and wrapper types.
- Deduplication of macro logic between `pure_math` and `math_commons`.
- Guaranteed consistency in reference-pair implementations (`(T, T)`, `(T, &T)`, `(&T, T)`, and `(&T, &T)`).
- Zero runtime overhead and seamless compile-time trait implementations.

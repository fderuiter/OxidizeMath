# Verified Engine Macros

Static analysis procedural macros enforcing NASA Power of 10 rules and scientific theory metadata.

## 1. Overview & Architecture

The `verified_engine_macros` crate provides compile-time procedural macros that enforce architectural invariants and automate metadata generation across the OxidizeMath workspace.

By analyzing abstract syntax trees (AST) during compilation, these macros emit spanned plain-text `rustc` diagnostics for safety violations (such as direct recursion and excessive statement counts) before code is compiled into binaries. Furthermore, the derive and attribute macros generate reflection interfaces and link verified LaTeX equations without leaking internal crate dependencies to downstream callers.

### Module Structure

- **`#[verified]`**: Attribute macro for free functions, inherent methods, and trait methods (including bodyless trait signatures and `const fn`):
  - NASA Power of 10 Rule 1: Prohibits direct recursive calls (`opt_out = "reason"` skips verification; `allow_recursion = "reason"` permits same-name inherent delegation while keeping statement limits and runtime metrics active).
  - NASA Power of 10 Rule 4: Restricts function statement length to a maximum of 60 statements.
  - Injects runtime `ComplexityMetrics` instrumentation on non-`const` function bodies.
- **`#[derive(Theory)]`**: Derive macro implementing `TheoryDescribable` on structs with named fields:
  - Validates struct-level `#[theory(description = "...", phonetic = "...", citation = "...")]` and field-level `#[theory(min = ..., max = ..., step = ..., default = ...)]` attributes with full `syn::Error` accumulation.
  - Accepts finite integer and floating-point bounds (including negative literals), enforces `min <= max`, `step > 0.0`, and `min <= default <= max` at macro-expansion time, and spans primitive numeric trait checks to each field's type token.
- **`#[embed_theory]`**: Attribute macro dynamically resolving workspace `.tex` formulations from `CARGO_MANIFEST_DIR`, checking AST semantic integrity, and emitting unnamed `const _` file-change trackers.
- **`runtime_violation!`**: Procedural macro generating structured JSON fatal violation payloads for runtime allocator aborts.

---

## 2. Developer Instructions

### Deriving Scientific Theory Metadata

Apply `#[derive(Theory)]` through `verified_engine` or `math_explorer::verified_engine` to expose structured model parameters and academic citations:

```rust
use scientific_metadata::theory::TheoryDescribable;
use verified_engine_macros::Theory;

/// A simple gravitational acceleration model.
#[derive(Theory)]
#[theory(
    description = "Gravitational acceleration model",
    phonetic = "Gravitational acceleration model",
    citation = "Newton, I. (1687). Philosophiae Naturalis Principia Mathematica."
)]
pub struct GravityModel {
    #[theory(min = 0.0, max = 100.0, step = 0.1, default = 9.81)]
    pub g: f64,
    #[theory(min = 1, max = 64, step = 1)]
    pub substeps: usize,
}

let model = GravityModel { g: 9.81, substeps: 4 };
assert_eq!(model.theory_description(), "Gravitational acceleration model");
assert_eq!(
    model.theory_citation(),
    "Newton, I. (1687). Philosophiae Naturalis Principia Mathematica."
);
let params = model.theory_parameters();
assert_eq!(params["g"].max, 100.0);
assert_eq!(params["g"].default, Some(9.81));
assert_eq!(params["substeps"].max, 64.0);
assert_eq!(params["substeps"].default, None);
assert_eq!(model.g, 9.81);
assert_eq!(model.substeps, 4);
```

---

## 3. Ecosystem Integration

Procedural macros defined in this crate are re-exported through `verified_engine` and `math_explorer::verified_engine`, resolving `verified_engine` and `scientific_metadata` paths hygienically so domain and external crates do not need unused direct `scientific_metadata` dependencies.

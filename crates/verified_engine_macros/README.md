# Verified Engine Macros

Static analysis procedural macros enforcing NASA Power of 10 rules and scientific theory metadata.

## 1. Overview & Architecture

The `verified_engine_macros` crate provides compile-time procedural macros that enforce architectural invariants and automate metadata generation across the OxidizeMath workspace.

By analyzing abstract syntax trees (AST) during compilation, these macros enforce safety constraints (such as prohibiting direct recursion and restricting function statement lengths) before code is compiled into binaries. Furthermore, the derive macros automatically generate reflection interfaces required for scientific model inspection and dynamic parameter tuning.

### Module Structure

- **`#[verified]`**: Attribute macro that inspects function implementations to verify:
  - NASA Power of 10 Rule 1: Prohibits direct recursive calls (bypassable via explicit `opt_out`).
  - NASA Power of 10 Rule 4: Restricts function statement length to a maximum of 60 statements.
  - Injects runtime instrumentation for counting function calls and arithmetic operations.
- **`#[derive(Theory)]`**: Derive macro that automatically implements `TheoryDescribable` on simulation structures:
  - Extracts theory descriptions, phonetic screen-reader text, and academic literature citations from struct attributes.
  - Generates parameter constraints (`min`, `max`, `step`) and dynamic getters/setters from field annotations.
- **`#[embed_theory]`**: Attribute macro for embedding verified LaTeX mathematical formulations and checking semantic integrity.
- **`runtime_violation!`**: Procedural macro generating structured fatal violation payloads and terminating execution.

---

## 2. Developer Instructions

### Deriving Scientific Theory Metadata

Apply `#[derive(Theory)]` to expose structured model parameters and academic citations to the workspace:

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
    #[theory(min = 0.0, max = 100.0, step = 0.1)]
    pub g: f64,
}

let mut model = GravityModel { g: 9.81 };
assert_eq!(model.theory_description(), "Gravitational acceleration model");
assert_eq!(
    model.theory_citation(),
    "Newton, I. (1687). Philosophiae Naturalis Principia Mathematica."
);
assert_eq!(model.get_parameter("g"), Some(9.81));

model.set_parameter("g", 1.62);
assert_eq!(model.get_parameter("g"), Some(1.62));
```

---

## 3. Ecosystem Integration

Procedural macros defined in this crate are re-exported through `verified_engine` and utilized across all scientific domain crates to generate parameter reflection models for `math_explorer_gui`.

# 12. Bessel Y Stub Non-Panicking NaN Return

## Status
Accepted

## Context
The `bessel_y` special function in `crates/pure_math` previously invoked the `unimplemented!()` macro, which conflicted with the function doc comment indicating it returns NaN as a placeholder pending full analytical or series implementation without external dependencies. Calling `bessel_y` caused unexpected thread panics and process termination.

## Decision
1. Updated `bessel_y` in `crates/pure_math/src/pure_math/special_functions/bessel.rs` to return `f64::NAN` instead of panicking.
2. Preserved the public API signature (`pub fn bessel_y(_nu: f64, _x: f64) -> f64`) and verification attribute annotations.
3. Added unit tests in `crates/pure_math/tests/test_pde_special_funcs.rs` ensuring `bessel_y` reliably returns NaN without panic across standard and edge-case inputs (zeros, negative inputs, infinities, and NaNs).

## Consequences
- `bessel_y` conforms to its documented placeholder behavior and avoids unexpected runtime panics.
- Satisfies architectural decision record requirements for modifications to core logic crates.

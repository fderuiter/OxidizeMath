# `math_diagnostics`

Zero-allocation `#![no_std]` Tier 0 diagnostic traits and severity contracts for OxidizeMath engines and domain solvers.

## 1. Overview & Architecture

`math_diagnostics` provides the foundational structured error telemetry contract shared across `#![no_std]` numerical kernels (`engine_linalg`, `engine_diffeq`, `engine_pde`, `math_commons`), mathematical libraries (`pure_math`, `domain_*`), and `std` event buses (`diagnostics`, `federated_registry`):

- **`Severity`**: Operational classification (`Info`, `Warning`, `Error`, `Fatal`) implementing `Copy`, `Clone`, `Eq`, `Hash`, and `core::fmt::Display`.
- **`DiagnosticVisitor`**: Stack-only key-value visitor callback interface (`record_str`, `record_display`, `record_usize`, `record_u64`, `record_i64`, `record_f64`, `record_bool`) requiring neither `std` nor `alloc`.
- **`Diagnostic`**: Supertrait of `core::error::Error + Send + Sync + 'static` exposing `severity(&self)` and zero-allocation `visit_fields(&self, visitor)`. When the optional `std` feature is enabled by host or bus crates, `Diagnostic::metadata(&self)` materializes visited fields into a `HashMap<String, String>`.

```rust
use core::fmt;
use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct WorkspaceLimitError {
    pub required: usize,
    pub available: usize,
}

impl fmt::Display for WorkspaceLimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "workspace requires {} scalars, found {}",
            self.required, self.available
        )
    }
}

impl core::error::Error for WorkspaceLimitError {}

impl Diagnostic for WorkspaceLimitError {
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visitor.record_str("error_type", "WorkspaceLimitError");
        visitor.record_usize("required", self.required);
        visitor.record_usize("available", self.available);
    }
}

struct CountVisitor(usize);

impl DiagnosticVisitor for CountVisitor {
    fn record_str(&mut self, _key: &str, _value: &str) {
        self.0 += 1;
    }

    fn record_display(&mut self, _key: &str, _value: &dyn fmt::Display) {
        self.0 += 1;
    }
}

let err = WorkspaceLimitError { required: 16, available: 8 };
let mut visitor = CountVisitor(0);
err.visit_fields(&mut visitor);
assert_eq!(err.severity(), Severity::Error);
assert_eq!(visitor.0, 3);
```

## 2. Developer Instructions

Verify strict `#![no_std]` WebAssembly compilation and zero-allocation unit tests from the workspace root:

```bash
cargo check -p math_diagnostics --target wasm32-unknown-unknown --no-default-features
cargo test -p math_diagnostics --all-features
```

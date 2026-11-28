# OxidizeMath Standalone WebAssembly & TypeScript SDK (`oxidize_wasm`)

![License](https://img.shields.io/badge/license-AGPL-blue.svg)
![Language](https://img.shields.io/badge/language-Rust%20%2F%20TypeScript-orange.svg)
![Maturity](https://img.shields.io/badge/maturity-Validated-0969da.svg)

Standalone `#[wasm_bindgen]` binding crate packaging OxidizeMath's Tier 2 numerical engines (`engine_linalg`, `engine_diffeq`, `engine_pde`, and `math_commons::ast`) for external JavaScript, TypeScript, and Web Worker consumers without GUI (`eframe` / `egui`) dependencies.

> **Note:** For the full workspace overview, see the [Project Root README](../../../README.md).

## 1. Overview & Architecture

The `oxidize_wasm` crate (`@oxidizemath/wasm`) provides an ergonomic, zero-allocation WebAssembly SDK backed directly by OxidizeMath's `#![no_std]` Tier 2 mathematical kernels:

- **Zero-Allocation Pre-Allocated Workspaces**: `WasmLinalgWorkspace`, `WasmSdeWorkspace`, `WasmPdeWorkspace`, `WasmAstKernel`, and `WasmF64Buffer` allocate scratch storage once at construction (`try_new`) and execute repeated solves/steps in place with zero per-call heap allocations.
- **Zero-Copy Linear Memory Views**: Every handle exposes byte-offset pointers (`byte_offset()`, `matrix_ptr()`, `solution_ptr()`, `u_ptr()`, `v_ptr()`) and scalar lengths alongside `wasm_linear_memory()`, allowing JavaScript callers to wrap WASM linear memory directly via `new Float64Array(wasmLinearMemory().buffer, ptr, len)` without copying typed arrays across the boundary.
- **Structured Diagnostic Exceptions**: Fallible operations lower `math_diagnostics::Diagnostic` visitor metadata (`code`, `errorType`, `severity`, `fields`) into typed JavaScript `Error` exceptions (`WasmSdkError`) instead of opaque panics or aborts.
- **Compact Bundle Budget**: Enforced in CI at `<= 250 KB` (`256,000` bytes) Brotli-compressed `.wasm`.

## 2. Developer Instructions

### Building the Standalone WebAssembly Package

```bash
cargo build --target wasm32-unknown-unknown --release -p oxidize_wasm
wasm-bindgen --target web --typescript --out-dir target/wasm-sdk-pkg --out-name oxidize_wasm \
  target/wasm32-unknown-unknown/release/oxidize_wasm.wasm
bash scripts/verify_wasm_budget.sh --sdk target/wasm-sdk-pkg 256000
```

### Rust & WebAssembly Usage Example

```rust
use oxidize_wasm::{WasmAstKernel, WasmLinalgWorkspace, WasmPdeWorkspace, WasmSdeWorkspace};

let mut linalg = WasmLinalgWorkspace::try_new(4).expect("linalg workspace");
let a = [4.0, 1.0, 1.0, 3.0];
let b = [1.0, 2.0];
let mut x = [0.0; 2];
linalg.solve_cholesky(&a, &b, &mut x, 2).expect("SPD Cholesky solve");
assert!((4.0 * x[0] + x[1] - 1.0).abs() < 1e-12);
assert!((x[0] + 3.0 * x[1] - 2.0).abs() < 1e-12);

let mut sde = WasmSdeWorkspace::try_new(2, 2, 42).expect("SDE workspace");
let mut state = [1.0, -0.5];
sde.step_ou(0.0, 0.01, 1.5, 0.0, 0.2, &mut state).expect("OU step");
assert!(state[0].is_finite() && state[1].is_finite());

let mut pde = WasmPdeWorkspace::try_new_2d(8, 8, 1.0, 1.0).expect("2D PDE workspace");
pde.seed_schnakenberg_steady_state(0.1, 0.9, 0.02);
pde.step_schnakenberg_2d_in_place(0.005, 1.0, 18.0, 0.1, 0.9, 2)
    .expect("Schnakenberg step");
assert_eq!(pde.grid_len(), 64);

let mut ast = WasmAstKernel::try_compile_3d("sin(k * x) * cos(k * y) + cos(k * z)")
    .expect("compiled 3D AST");
ast.set_param("k", 1.5).expect("bound parameter k");
let wgsl = ast.to_compute_wgsl(32, 3.14159265).expect("compute WGSL");
assert!(wgsl.contains("@compute @workgroup_size(4, 4, 4)"));
```

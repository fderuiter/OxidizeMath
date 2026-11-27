# ADR 20261127: Standalone `oxidize_wasm` SDK, Off-Main-Thread Web Worker Runtime, and WebGPU `128^3` Volumetric Compute Foundry

## Status

Accepted

## Context

Following the `#![no_std]` Tier 0 and Tier 2 extraction milestones ([ADR 20261123](20261123-nostd-diffeq-and-pde-engines.md), [ADR 20261124](20261124-no-std-math-diagnostics-and-causal-error-wiring.md), and [ADR 20261126](20261126-extract-math-scalars-and-engine-opt.md)), three browser-delivery and compute bottlenecks remained:

1. **Monolithic WebAssembly Delivery**: External JavaScript and TypeScript developers could not import OxidizeMath's `#![no_std]` numerical engines (`engine_linalg`, `engine_diffeq`, `engine_pde`, `math_commons::ast`) without pulling in the full `eframe` / `egui_plot` GUI bundle (`apps/math_explorer_gui`).
2. **Main-Thread `wasm32` Stepping Contention**: On `wasm32-unknown-unknown`, `apps/math_explorer_gui/src/async_sim/` relied exclusively on cooperative single-threaded frame slicing (`wasm_scheduler`) because `std::thread::spawn` panics on `wasm32` and static hosts such as GitHub Pages do not serve `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` headers required for `SharedArrayBuffer`.
3. **2D-Only WGSL Transpilation and `48^3` Volumetric Ceiling**: `math_commons::ast::shader` only emitted 2D `(x, y)` fragment/helper WGSL, and `VolumetricFieldStudioTool` (`apps/math_explorer_gui/src/tabs/quantum/volumetric_studio/`) capped grids at `48^3` (`110,592` voxels) on a single CPU thread without `@compute` WGSL export or `128^3` (`2,097,152`-voxel) WebGPU compute execution.

## Decision

1. **Standalone Tier 4 [`oxidize_wasm`](../../crates/04_bindings/oxidize_wasm/README.md) WebAssembly SDK (`@oxidizemath/wasm`)**:
   - Create `crates/04_bindings/oxidize_wasm` (`crate-type = ["cdylib", "rlib"]`) wrapping `engine_linalg`, `engine_diffeq`, `engine_pde`, and `math_commons::ast` with zero GUI or Tier 3 domain dependencies.
   - Provide pre-allocated workspace handles (`WasmF64Buffer`, `WasmF32Buffer`, `WasmLinalgWorkspace`, `WasmSdeWorkspace`, `WasmPdeWorkspace`, `WasmAstKernel`) supporting both caller `Float32Array` / `Float64Array` slices and zero-copy `WebAssembly.Memory` byte-offset descriptors (`wasmLinearMemory()`, `byteOffset()`, `matrixPtr()`, `solutionPtr()`, `uPtr()`, `vPtr()`).
   - Lower `math_diagnostics::Diagnostic` visitor metadata (`code`, `errorType`, `severity`, `fields`) into structured JavaScript `Error` objects via `WasmSdkError` instead of panicking across the WASM boundary, and enforce a `<= 250 KB` (`256,000`-byte) Brotli `.wasm` budget in [`scripts/verify_wasm_budget.sh`](../../scripts/verify_wasm_budget.sh) and CI release artifacts (`oxidize-wasm-npm-sdk.tar.gz`).

2. **Dual-Path Off-Main-Thread Web Worker Bridge & Thin `web-sys` WebGPU `@compute` Dispatcher (`apps/math_explorer_gui/src/async_sim/{web_worker,webgpu}.rs`)**:
   - Probe `crossOriginIsolated` and `SharedArrayBuffer` availability at runtime on `wasm32`: select zero-copy `SharedArrayBuffer` double-buffered ring transport when cross-origin isolated, degrade gracefully to a dedicated `Blob` `web_sys::Worker` (`worker-src 'self' blob:`) exchanging transferable `ArrayBuffer` frames via `postMessage` on standard static hosts (including GitHub Pages without COOP/COEP headers), and retain cooperative `wasm_scheduler` fallback in headless environments without `Worker`.
   - Implement a thin `wasm-bindgen` / `web-sys` WebGPU `@compute` dispatcher (`GpuAdapter`, `GpuDevice`, `GpuComputePipeline`, and double-buffered `MAP_READ` staging `GpuBuffer`s) alongside `async_sim/wasm.rs` without adding `wgpu` or `naga` to the WASM bundle while keeping `eframe::WebRunner` on WebGL2.

3. **Bounded `@compute` WGSL Foundry and `128^3` WebGPU Volumetric Studio**:
   - Extend `math_commons::ast` (`shader.rs`, `compute_shader.rs`) and `framework/shader_foundry.rs` (`shader_foundry_compute.rs`) with bounded (`MAX_EVAL_DEPTH = 64`, `MAX_AST_NODES = 256`, `MAX_GLSL_OUTPUT_BYTES = 16_384`) `@compute @workgroup_size(...)` WGSL emitters across 3D volumetric fields, 2D PDE stencils (`MorphogenesisTool`), 1D quantum wavepackets (`WaveSimulatorTool`), ODE RK4 rollouts (`OdeSolverTool`), and Monte Carlo payoff ensembles (`MonteCarloWorkbenchTool`).
   - Add `ExportCapabilities::with_compute_shader()`, `ExportFormat::WgslCompute`, and `"Export Compute WGSL"` / `"Copy Compute WGSL"` toolbar actions to `SimulationFramework`.
   - Upgrade `VolumetricFieldStudioTool` with a `ComputeBackend` (`WebGpuCompute` vs `CpuZeroAlloc`) selector (`#[serde(default)]` for backward-compatible preset loading) and inline step-latency telemetry, scaling grid resolution up to `128^3` (`2,097,152` voxels) in `WebGpuCompute` mode and automatically clamping to `<= 48^3` CPU zero-alloc stepping while keeping `.wgsl` exports enabled when `navigator.gpu` is unavailable.

## Consequences

- **Positive**: External JS/TS consumers can install a self-contained `< 250 KB` Brotli-compressed WebAssembly numerical SDK with typed `Diagnostic` errors and zero-allocation workspaces.
- **Positive**: Browser volumetric stepping offloads heavy 3D sweeps from the UI thread on both COOP/COEP-isolated hosts (`SharedArrayBuffer`) and GitHub Pages (`Worker` transferable `ArrayBuffer`s), while `VolumetricFieldStudioTool` scales to `128^3` voxels and exports standalone `@compute` WGSL kernels.
- **Trade-offs**: Workspace member count increases from 30 to 31 crates, recorded in [`system_overview.md`](../architecture/system_overview.md), [`traceability.toml`](../../traceability.toml), [`verification_whitelist.toml`](../../verification_whitelist.toml), and [`api_baseline.json`](../../apps/unified_verification/api_baseline.json).

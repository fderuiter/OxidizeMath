# ADR 0035: Pre-Planned Zero-Allocation 2D MRI FFT Workspace and Analytical Phantom

## Status
Accepted (extends [ADR 0029](0029-separable-fft-mri-reconstruction.md))

## Context
In `crates/domain_physics/src/physics/mri/reconstruction.rs`, [ADR 0029](0029-separable-fft-mri-reconstruction.md) introduced transposed unit-stride 1D `rustfft` passes in `separable_transform`, reducing 2D MRI k-space simulation and reconstruction complexity to $O(R \cdot C \log(R \cdot C))$. However, calling `simulate_signal_2d` or `inverse_dft_2d` inside an interactive frame loop constructed a fresh `FftPlanner`, allocated a new `scratch` vector, and allocated two intermediate `DMatrix<Complex<f64>>` transpose matrices on every pass—violating NASA Power of 10 Rule 3 (`VerifiedAllocator` lock during `VerifiedEngine::run_verified`). Additionally, interactive 2D MRI studios require zero-allocation DC-centering shifts (`fftshift` / `ifftshift`) and reusable analytical Modified Shepp-Logan $(\rho, T_1, T_2)$ tissue phantoms even when `domain_biology` is disabled.

## Decision
1. **`Fft2dWorkspace` in `crates/domain_physics/src/physics/mri/reconstruction.rs`**:
   - Cache forward and inverse row/column `Arc<dyn Fft<f64>>` plans alongside a single pre-allocated `scratch: Vec<Complex<f64>>` buffer and a `cols x rows` `transposed: DMatrix<Complex<f64>>` matrix.
   - Provide `simulate_signal_2d_into`, `inverse_dft_2d_into`, `reconstruct_normalized_2d_into` ($1 / (R \cdot C)$ scaling), `forward_transposed_in_place`, and two-transpose fused `filter_kspace_2d_into` using an `8x8` cache-blocked slice transpose so $64 \times 64$ and $128 \times 128$ passes execute in `< 1 ms` with zero dynamic heap allocations inside `VerifiedEngine::run_verified`.
   - Delegate `separable_transform` to `Fft2dWorkspace` so `simulate_signal_2d` and `inverse_dft_2d` retain identical public signatures without code duplication.
2. **Zero-Allocation k-Space Centering and Spatial Bloch Grid Helpers**:
   - Add `fftshift_2d_into` and `ifftshift_2d_into` to map DC `(0, 0)` to `(R/2, C/2)` and back across arbitrary even, odd, and rectangular dimensions without allocation.
   - Add `BlochSimulator::advance_isochromat_grid` in `crates/domain_physics/src/physics/mri/bloch.rs` to evolve multi-isochromat voxel spin ensembles under precomputed $(E_1, E_2)$ relaxation and rotating-frame $\Delta B_0$ precession with zero dynamic allocation.
3. **Analytical Modified Shepp-Logan Phantom (`crates/domain_physics/src/physics/mri/phantom.rs`)**:
   - Provide `MODIFIED_SHEPP_LOGAN_ELLIPSES`, `sample_shepp_logan_tissue`, and `fill_shepp_logan_grid` to populate pre-allocated $(\rho, T_1, T_2)$ tissue slices with zero heap allocations.

## Consequences
- **Positive**: Eliminates per-pass planner, scratch, and transpose allocations during 2D MRI k-space simulation and reconstruction, enabling verified zero-allocation worker ticks under `1 ms` on native desktop and `wasm32`.
- **Positive**: Gives GUI and CLI callers a domain-native analytical $(\rho, T_1, T_2)$ phantom when compiled with `--no-default-features --features physics`.
- **Verification**: Validated under `VerifiedEngine::run_verified` in `crates/domain_physics/tests/measure_mri.rs` and `math_explorer_gui` MRI studio tests, with public signatures recorded in `crates/unified_verification/api_baseline.json`.

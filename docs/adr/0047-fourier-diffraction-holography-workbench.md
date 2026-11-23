# ADR 0047: 2D Helmholtz Angular-Spectrum Diffraction and Gerchberg-Saxton Phase-Holography Workbench

## Status

Accepted — 2026-11-23

## Context

`domain_physics::physics::optics` previously exposed only two scalar sensing helpers (`lambertian_intensity` and `lws_range_resolution`) alongside a placeholder `DUMMY = 1.0` invariant in `theory_verification!`, with no interactive `OpticsTab` in `math_explorer_gui`. Researchers and students exploring wave optics inside OxidizeMath lacked:

1. Exact monochromatic scalar Helmholtz wavefield propagation capable of reproducing near-field Arago-Poisson bright spots, Rayleigh-Talbot periodic self-imaging carpets, thin-lens focusing, and Zernike wavefront aberrations without per-frame heap allocations.
2. Iterative Gerchberg-Saxton spatial light modulator (SLM) phase-hologram synthesis $e^{i\phi(x, y)}$ demonstrating elimination of conjugate twin-image symmetry below the real-scalar $1/\sqrt{2} \approx 0.707$ amplitude floor.

## Decision

1. **Zero-Allocation Wave-Optics Kernels (`crates/domain_physics/src/physics/optics/`)**:
   - Preserve `sensing::lambertian_intensity` and `sensing::lws_range_resolution` unchanged for API compatibility while replacing `DUMMY = 1.0` in [`mod.rs`](../../crates/domain_physics/src/physics/optics/mod.rs) with exact Fresnel-zone and Rayleigh-Talbot invariants.
   - Introduce [`diffraction.rs`](../../crates/domain_physics/src/physics/optics/diffraction.rs) and [`apertures.rs`](../../crates/domain_physics/src/physics/optics/apertures.rs) housing `AngularSpectrumPropagator` (with private `rustfft` plans and pre-allocated row, column, transpose, and scratch buffers), evanescent mode filtering ($k_x^2 + k_y^2 \le k^2$), fast 1D-reduced axial $(x, z)$ carpets, and complex aperture/lens/Zernike generators.
   - Introduce [`holography.rs`](../../crates/domain_physics/src/physics/optics/holography.rs) housing `GerchbergSaxtonSolver` seeded deterministically via `oxidize_core::rng::OxidizeRng`, supporting both Fourier-lens and lensless angular-spectrum regimes plus calibrated 8-bit `[0..=255]` SLM phase quantization.

2. **Lazy Interactive Workbench (`math_explorer_gui/src/tabs/optics/`)**:
   - Register `OpticsTab` (`// @explorer_feature = "physics"`, academic domain `"Physics"`, URL slug `"optics"`) and `FourierOpticsWorkbenchTool` (`domain = "optics"`) with four Starter Gallery cards, `ToolStateBounds` validation, `SemanticPalette` / high-contrast colormaps, URL state roundtrip, and `.png`, `.svg`, `.csv`, and `.json` exports.
   - Defer FFT planning and wavefield buffer allocation inside `RefCell<Option<WorkbenchCache>>` so `Default::default()` performs zero heavy work during application startup.

## Consequences

- **Positive**: `VerifiedEngine::run_verified` confirms zero dynamic heap allocations across propagation, axial carpets, and Gerchberg-Saxton iterations, Parseval optical power conservation within `1.0e-12`, Arago bright-spot formation ($I > 1.0$ at $N_F = 2.0$), machine-precision Rayleigh-Talbot revivals, and `64x64` phase-hologram convergence (`target_nrmse < 0.32` within 50 iterations).
- **Trade-offs**: Restricts scope to scalar monochromatic Helmholtz diffraction and 2D SLM phase retrieval without adding new Cargo dependencies or 3D vector Maxwell solvers, keeping WASM bundle size comfortably inside the `< 3 MB` budget.

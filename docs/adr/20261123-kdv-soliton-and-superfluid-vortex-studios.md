# ADR 20261123: Zero-Allocation 1D KdV / Cubic NLS Soliton and 2D Gross–Pitaevskii Superfluid Vortex Studios

## Status

Accepted (2026-11-23)

## Context

`OxidizeMath` previously offered linear 1D/2D Schrödinger wave-packet studios (`wave_sim`) and 2D Störmer–Verlet hyperbolic/biharmonic wave tanks (`chladni_wave_tank`), but lacked interactive nonlinear dispersive wave and topological quantum fluid studios:

1. In mathematical analysis (`crates/pure_math` / `AnalysisTab`), users could not contrast nonlinear Korteweg–de Vries ($u_t + 6 u u_x + u_{xxx} = 0$) and 1D cubic Nonlinear Schrödinger ($i\psi_t + \psi_{xx} + 2\sigma|\psi|^2\psi = 0$) soliton recovery against pure third-order linear Airy dispersion ($u_t + u_{xxx} = 0$), nor measure inverse-scattering 2-soliton overtaking phase shifts ($\Delta_1, \Delta_2$) alongside the conserved $(M, P, H)$ hierarchy.
2. In quantum mechanics (`crates/domain_physics` / `QuantumTab`), users could not simulate nonlinear 2D Gross–Pitaevskii Bose–Einstein condensates ($i\partial_t\psi = (-\frac{1}{2}\nabla^2 + V + g|\psi|^2)\psi$), imaginary-time ground-state relaxation, or quantized $\pm 2\pi$ topological vortex cores.

Both solvers must obey NASA Power of 10 Rule 3 (zero dynamic heap allocations inside solver hot loops under `VerifiedAllocator`), stay within the `3 MB` compressed `wasm32-unknown-unknown` bundle budget, and satisfy WCAG 2.2 AA contrast and AccessKit keyboard navigation contracts.

## Decision

1. **1D Pseudospectral KdV, Airy, and Cubic NLS Solver (`crates/pure_math/src/pure_math/analysis/pde/kdv.rs`)**:
   - Pre-plan forward and inverse `rustfft` transforms (`N <= 512`) and pre-allocate all Integrating-Factor RK4 (Lawson IFRK4), Orszag $2/3$-rule dealiasing, and Strang split-step work buffers at construction.
   - Integrate linear stiff dispersion $u_t + u_{xxx} = 0$ analytically in Fourier space via unitary phase factors $e^{i k^3 \Delta t}$, enabling an instant real-time toggle disabling nonlinear steepening ($6 u u_x$ / $2\sigma|\psi|^2\psi$) with zero re-allocation.
   - Compute conserved mass $M$, momentum $P$, and Hamiltonian $H$ via spectral Parseval derivatives, track pulse peaks via 3-point parabolic sub-cell interpolation, and expose exact analytical 2-soliton phase shifts $\Delta_{\text{fast}}, \Delta_{\text{slow}}$.
2. **2D Strang Split-Step Gross–Pitaevskii & Plaquette Vortex Solver (`crates/domain_physics/src/physics/quantum/gross_pitaevskii.rs`)**:
   - Pre-plan 1D row and column `rustfft` transforms (`<= 128 x 128`) over flat `flatten_2d_index` buffers with pre-allocated transpose and scratch arrays.
   - Advance real-time unitary dynamics (conserving $M = \iint |\psi|^2\,dx\,dy$ within `< 10^-12`) and Wick-rotated imaginary-time cooling ($\tau = i t$ with in-place norm renormalization) across isotropic/anisotropic harmonic, central-obstacle, and periodic-box traps.
   - Detect quantized topological vortices in-place by summing principal wrapped phase differences $\Delta\theta \in (-\pi, \pi]$ counter-clockwise around each interior $2\times 2$ bulk plaquette to classify $+1$ ($+2\pi$) and $-1$ ($-2\pi$) winding charges.
3. **Interactive Studios (`SolitonCollisionTool` in `AnalysisTab` & `SuperfluidVortexTool` in `QuantumTab`)**:
   - Register four `StarterPresetMetadata` cards per studio (`8` total), enforce `#[serde(deny_unknown_fields)]` + `ToolStateBounds` + `FiniteStateGuard`, render WCAG-compliant `SemanticPalette` / `colormap` density and cyclic $[-\pi, \pi]$ phase rasters, expose AccessKit `GraphicsDocument` keyboard interaction without trapping `Tab`, and support `.csv`, `.json`, `.png`, and `.svg` exports.
   - Explicitly exclude 3D vortex-tube filament solvers, rotating-frame gauge-field advection, and full numerical Zakharov–Shabat eigenvalue solvers to keep desktop and WASM frame budgets lightweight.
   - Negotiate pre-compressed `.br`/`.gz` assets via `Accept-Encoding` in `CacheableOriginHandler` within [`scripts/playwright_smoke.py`](../../scripts/playwright_smoke.py) so warm-navigation HTTP cache verification reflects production compressed wire sizes (`~2.6 MB` Brotli) within Chromium's incognito `MemBackendImpl` single-entry cache cap (`6.25 MiB`).

## Consequences

- **Positive**: `VerifiedEngine::run_verified` confirms zero heap allocations across KdV IFRK4, 1D NLS, 2D GPE Strang steps, invariant integration, and $2\times 2$ plaquette vortex detection; 512-point 2-soliton collisions preserve $(M, P, H)$ within `< 10^-5` and match exact analytical phase shifts (`+0.5493`, `-1.0986`) within `< 5e-3`.
- **Negative / Trade-offs**: Pre-allocated work vectors per solver instance occupy ~80 KB (`KdvSolver1D` at `N = 512`) and ~1.3 MB (`GrossPitaevskiiSolver2D` at `128 x 128`), bounded strictly by `MAX_KDV_POINTS = 512` and `MAX_GPE_DIM = 128`.

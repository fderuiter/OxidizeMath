# ADR 0047: Zero-Allocation Geometric Optics, Caustics, and Fraunhofer Diffraction Engine

## Status

Accepted

## Context

`domain_physics::physics::optics` previously exposed only two scalar sensing formulas (`sensing::lambertian_intensity` and `sensing::lws_range_resolution`) with a placeholder `DUMMY = 1.0` verification constant and `optics = "graph_parameters_rust.tex"` in `traceability.toml`. Delivering an interactive geometric and physical optics laboratory across desktop and `wasm32-unknown-unknown` requires:

1. **Zero dynamic heap allocations** inside interactive ray-fan integration, caustic envelope extraction, and Fraunhofer diffraction profile sampling under `VerifiedAllocator::lock_allocations()` (NASA Power of 10 Rule 3).
2. **Reuse of existing verified math kernels** (`pure_math::analysis::ode::{ArrayState, OdeSystem, RungeKutta4}` for graded-index eikonal integration, `pure_math::special_functions::bessel::bessel_j` for circular Airy patterns, and `math_commons::math_kernel::colormap` for WCAG-compliant rendering) with zero new Cargo dependencies and `< 3 MiB` compressed WASM bundle size.
3. **Auto-discovered GUI plugin integration** (`Optics & Caustics` tab under academic domain `Physics`, URL slug `optics`) without manual edits to `math_explorer_gui/src/app/mod.rs`.

## Decision

1. **Submodules in `crates/domain_physics/src/physics/optics/`**:
   - `refraction.rs`: Exact 2D vector Snell refraction, critical-angle total internal reflection, $s$/$p$-polarized Fresnel amplitude and power coefficients ($R_s + T_s = 1$, $R_p + T_p = 1$, Brewster extinction $R_p(\theta_B) = 0$), three-term Cauchy glass dispersion, and ray-circle / ray-segment intersection primitives.
   - `ray_tracing/{mod.rs, surfaces.rs}`: Canonical eikonal ODE system $\dot{\mathbf{r}} = \mathbf{p}$, $\dot{\mathbf{p}} = \frac{1}{2}\nabla(n^2)$ integrated via `RungeKutta4<ArrayState<4>>` across Luneburg, Maxwell fisheye, regularized Eaton, and parabolic GRIN media alongside discrete multi-surface lenses, circular coffee-cup mirrors, and dispersive prisms into pre-allocated `RayFanWorkspace` slices (`trace_ray_fan`).
   - `caustics.rs`: Adjacent-ray focal envelope intersection (`extract_caustic_envelope`), closed-form nephroid reference loci, and pre-allocated transverse detector intensity histograms (`CausticEnvelopeBuffer`).
   - `diffraction.rs`: Closed-form single-slit, $N$-slit grating, and circular Airy intensity evaluators (`sample_fraunhofer_profile`) into pre-allocated `FraunhoferProfileBuffer` slices with analytical critical-point markers and Hankel asymptotic stabilization for $|x| > 20$.
2. **`OpticsTab` in `math_explorer_gui/src/tabs/optics/`**:
   - Auto-discovered via `// @explorer_feature = "physics"`, hosting `CausticsRayBenchTool` and `FraunhoferDiffractionTool` with four `StarterPresetMetadata` cards (`Coffee-Cup Nephroid`, `Luneburg Metamaterial`, `Dispersive Prism`, `Fraunhofer Grating`), `FiniteStateGuard`, `ToolStateBounds`, `AccessKit` `GraphicsDocument` keyboard navigation, and `SVG`, `PNG`, `CSV`, and `JSON` exports.
3. **Traceability & Manuscript**:
   - Authored `papers/geometric_optics_and_caustics.tex` and updated `traceability.toml` (`optics = "geometric_optics_and_caustics.tex"`).

## Consequences

- **Positive**: Verified Brewster extinction (`< 1e-24`), Fresnel energy conservation (`< 1e-12`), coffee-cup nephroid cusp recovery (`< 1e-3` of $R/2$), Luneburg and Maxwell fisheye focusing, Airy Bessel zeros (`< 1e-11`), and 100 consecutive ray-fan, caustic, and Fraunhofer steps with zero heap allocations.
- **Trade-offs**: Full 3D vector Maxwell / FDTD volumetric solvers and Monte Carlo global illumination remain intentionally out of scope to preserve deterministic 60 FPS interactive execution and the WASM bundle budget.

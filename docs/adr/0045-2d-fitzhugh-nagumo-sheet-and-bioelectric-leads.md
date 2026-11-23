# ADR 0045: 2D FitzHugh-Nagumo Excitable Sheet, 1D Hodgkin-Huxley Cable, and 3-Lead Bioelectric Dipole Integration

Date: 2026-11-21

## Status

Accepted

## Context

`domain_biology::biology::neuroscience` previously modeled only 0D single-patch Hodgkin-Huxley neurons (`HodgkinHuxleyModel`) without spatial diffusion, while `domain_physics::physics::medical` covered radiation therapy planning and respiratory motion management without extracellular volume-conductor dipole integration. In `math_explorer_gui`, `NeuroscienceTab` lacked tactile spatial electrophysiology brushes (`InteractiveTool::on_brush`), spiral-wave arrhythmia scenarios, conduction-block scar ablation, and multi-lead extracellular ECG/LFP readouts.

Real-time WebAssembly simulation slices must stay strictly within the cooperative `8 ms` frame budget (`WASM_FRAME_SLICE_BUDGET`) and execute zero heap allocations inside numerical step loops under `VerifiedAllocator`.

## Decision

1. **2D FitzHugh-Nagumo Kinetics and Excitable Sheet (`domain_biology`)**:
   - Implement `FitzHughNagumoKinetics` (`ReactionKinetics<2>`, `ReactionModel`, `Theory`) in `crates/domain_biology/src/biology/morphogenesis/reaction.rs` with damped Newton-Raphson `resting_state()`.
   - Add `ExcitableSheet2D` (`96x96` default) in `crates/domain_biology/src/biology/neuroscience/sheet.rs` coupling `TuringSystem<2, FitzHughNagumoKinetics, FiniteDifference2D, FusedEulerSolver>` (`[D_u, 0.0]`) with pre-allocated `prev_u`, `scar_mask`, `stimulus`, `clamp_mask`, and `clamp_voltage` buffers and exact zero-flux Neumann scar-face cancellation.
2. **1D Diffusive Hodgkin-Huxley Cable (`domain_biology`)**:
   - Add `HodgkinHuxleyCable` (`96` compartments default) in `crates/domain_biology/src/biology/neuroscience/cable.rs` coupling `HodgkinHuxleyModel` with `FiniteDifference1D` (`SpatialDiffusion<1>`) and zero-allocation compartment buffers.
3. **3-Lead Extracellular Dipole Integration (`domain_physics`)**:
   - Add `crates/domain_physics/src/physics/medical/bioelectric.rs` exposing `DipoleLeadArray`, `LeadPosition`, `LeadReadoutMode`, and shared `BioelectricPresetKind` / `BioelectricPresetSpec` evaluating $\Phi_k \propto \sum (\nabla V \cdot \mathbf{d}_k) / \|\mathbf{d}_k\|^3$ across pre-allocated spatial kernel buffers with zero heap allocations per step.
4. **Interactive GUI & Feature-Gated Cross-Domain Coupling (`math_explorer_gui`)**:
   - Expose `ExcitableSheetTool` in `math_explorer_gui/src/tabs/neuroscience/excitable_sheet/` implementing `InteractiveTool::on_brush`, `on_click`, and `on_keyboard`, `FiniteStateGuard`, `ToolStateBounds`, AccessKit canvas focus, `ExportCapabilities::plot()`, and `export_raster_image` (perceptual and WCAG high-contrast colormaps), gating `domain_physics::physics::medical::bioelectric` behind `#[cfg(feature = "physics")]`.
   - Expose `BioelectricLeadTool` in `math_explorer_gui/src/tabs/medical/bioelectric_leads.rs` sharing the clinical electrophysiology starter presets across `NeuroscienceTab` and `MedicalTab`.

## Consequences

- **Positive**: Unlocks interactive wavefront collision annihilation, S1-S2 spiral arrhythmia induction, conduction-block scar rotor pinning, catheter ablation to tissue boundaries, 1D ionic vs 2D phenomenological comparison, and 3-lead Einthoven pseudo-ECG / LFP dipole mapping with zero per-step heap allocations and `< 1 ms` native / `< 8 ms` WebAssembly step times.
- **Guardrails**: Preserves isolated domain feature compilation (`biology` vs `physics`), `<= 500`-line files, and `<= 60`-line functions; public API baseline (`api_baseline.json`) is regenerated to track the new exports.

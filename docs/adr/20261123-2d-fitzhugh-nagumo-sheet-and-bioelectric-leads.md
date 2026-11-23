# ADR 20261123: 2D FitzHugh-Nagumo Excitable Sheet, Virtual Electrodes, and 3-Lead Dipole Integration

## Status

Accepted — Date: 2026-11-23

## Context

`domain_biology::biology::neuroscience` previously modeled single-cell electrophysiology (`HodgkinHuxleyNeuron`) and 0D phase-coupled networks (`KuramotoNetwork`), while `domain_biology::biology::morphogenesis` provided generic 2D reaction-diffusion operators (`TuringSystem`, `FiniteDifference2D`, `FusedEulerSolver`) only for morphogenetic Turing kinetics (`SchnakenbergKinetics`, `GrayScottKinetics`). Users had no way to explore spatial action-potential propagation, refractory wavefront collision annihilation, S1-S2 spiral-wave arrhythmias, zero-flux conduction-block scars, catheter isthmus ablation, or how cellular transmembrane voltage gradients ($\nabla V$) generate clinical extracellular 3-lead ECG and local field potential (LFP) waveforms.

## Decision

1. **Modular Trait Reuse in `domain_biology`**:
   - Implement `FitzHughNagumoKinetics` (`ReactionKinetics<2>` and `ReactionModel`) in `crates/domain_biology/src/biology/morphogenesis/reaction.rs` ($\partial_t u = u - u^3/3 - w + I_{\text{ext}}$, $\partial_t w = \varepsilon(u + a - bw)$) with damped Newton-Raphson `resting_state()`.
   - Build `ExcitableSheet2D` (`96x96` default) on top of `TuringSystem<2, FitzHughNagumoKinetics, FiniteDifference2D, FusedEulerSolver>` with `[D_u, 0.0]` diffusion, pre-allocated electrode current/clamp probe masks, and post-step zero-flux Neumann flux cancellation across `scar_mask` cells so `step(dt)` performs zero heap allocations.
   - Build `HodgkinHuxleyCable` (`96` compartments) coupling `HodgkinHuxleyModel` with `FiniteDifference1D` for direct 1D ionic-channel vs. 2D FitzHugh-Nagumo comparison.

2. **Pre-Allocated 3-Lead Dipole Integrator in `domain_physics`**:
   - Add `crates/domain_physics/src/physics/medical/bioelectric.rs` (`DipoleLeadArray`, `LeadPosition`, `LeadReadoutMode`, `BioelectricPresetKind`) pre-computing spatial dipole weights $(w_{x,k}, w_{y,k}) = \kappa (x - x_k, y - y_k) / \|\mathbf{d}_k\|^3$ so `evaluate_sheet` and `evaluate_cable_1d` integrate unipolar LFP and bipolar Einthoven pseudo-ECG (`[Lead I, Lead II, Lead III]`) with zero heap allocations.

3. **Interactive Cross-Domain Workspaces in `math_explorer_gui`**:
   - Register `ExcitableSheetTool` in `NeuroscienceTab` and `BioelectricLeadTool` in `MedicalTab`, sharing 4 clinical presets (`Dual-Edge Wavefront Collision`, `S1-S2 Spiral Arrhythmia`, `Scar-Pinned Rotor & Ablation`, `Whole-Sheet Defibrillation`), cooperative `6.5 ms` frame slice budgeting (`< 8 ms` Wasm slice budget), `FiniteStateGuard` rollback, `focused_canvas_id` shortcut guarding, `ExportCapabilities::plot()` tabular + upscaled `384x384` `ColorImage` raster exports, and AccessKit `GraphicsDocument` live summaries.

## Consequences

- Reuses existing `ReactionKinetics<2>`, `FiniteDifference2D`, `FiniteDifference1D`, `FusedEulerSolver`, `HodgkinHuxleyModel`, and `math_kernel::colormap` without duplicating PDE stencils or colormap kernels.
- Both `96x96` PDE stepping and 3-lead dipole integration pass `VerifiedAllocator` zero-allocation gates and stay well under the `8 ms` WebAssembly frame slice budget.

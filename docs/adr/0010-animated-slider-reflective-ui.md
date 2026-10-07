# Architecture Decision Record 0010: Encapsulated AnimatedSlider Widget in Reflective UI

## Context
Parameter controls rendered via `reflective_ui.rs` in Math Explorer previously used static `egui::Slider` controls. Users had no automated way to sweep parameter values across theoretical bounds or observe parameter trajectories in real-time simulations without manual mouse dragging.

## Decision
We implemented a reusable `AnimatedSlider` widget in `math_explorer_gui/src/reflective_ui.rs` and integrated it across reflective UI parameter rendering and simulation tools:
1. Created `AnimatedSlider<'a>` and `AnimatedSliderState` stored in `egui::Id` temporary memory.
2. Coupled an `egui::Button` toggle (`▶` / `⏸`) with `egui::Slider` to control playback animation loops.
3. Implemented ping-pong parameter value sweeping frame-by-frame using `dt`, boundary direction reversal (`direction = -1.0` / `1.0`), parameter clamping within `ParameterConstraint` limits, and `ui.ctx().request_repaint()`.
4. Handled widget focus loss to automatically pause playback when sliders lose interaction focus.
5. Updated `render_theory_parameter` and `render_all_theory_parameters` to render parameters using `AnimatedSlider`.
6. Replaced static parameter controls in `bifurcation.rs` and `ode.rs` with `AnimatedSlider`.
7. Added unit tests verifying play toggling, boundary clamping, direction reversal, and focus loss handling.

## Consequences
- **Positive:** Parameters across reflective UI and simulation tabs can be animated continuously, triggering smooth real-time visualization updates.
- **Positive:** Interactive manual dragging is preserved when paused or active.
- **Positive:** Parameter bound overshoots are clamped gracefully without numerical instability.
- **Positive:** Passes architectural verification and pre-flight pipelines.

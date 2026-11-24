#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Parameter sensitivity, wide-range tornado, and Monte Carlo uncertainty panel UI adapter.
//!
//! Re-exports the core [`SensitivityEngine`] from `math_commons::sensitivity` and renders
//! theme-aware elasticity cards, tornado excursion intervals, Monte Carlo quantile summaries,
//! and synchronous instruction-budget scaling notices.

use eframe::egui;
#[cfg(test)]
use std::collections::HashMap;

pub use math_commons::sensitivity::{
    canonical_f64_bits, MonteCarloSummary, ParameterUncertainty, SensitivityEngine,
    SensitivityMetric, SplitMix64, TornadoBounds, TornadoInterval, DEFAULT_MONTE_CARLO_SAMPLES,
    DEFAULT_MONTE_CARLO_SEED, HIGH_SENSITIVITY_THRESHOLD, MAX_SENSITIVITY_BATCH_EVALS,
    MAX_SENSITIVITY_SYNC_INSTRUCTIONS, MIN_SENSITIVITY_STEPS,
};

/// Returns `true` when parameters are mid-gesture (pointer held down or local `AnimatedSlider` sweep active),
/// matching continuous-edit coalescing in `WorkspaceSessionManager` while scoping sweeps to the current pass/pane.
#[must_use]
pub fn is_transient_parameter_gesture(ctx: &egui::Context, sweep_active: bool) -> bool {
    sweep_active || ctx.input(|i| i.pointer.any_down())
}

#[path = "sensitivity_ui.rs"]
mod sensitivity_ui;
pub use sensitivity_ui::{render_labeled_sensitivity_panel_ui, render_sensitivity_panel_ui};

#[cfg(test)]
#[path = "sensitivity_tests.rs"]
pub(crate) mod tests;

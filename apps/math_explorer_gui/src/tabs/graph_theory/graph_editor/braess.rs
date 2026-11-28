//! Braess's Paradox Congestion Routing Lab backed by `pure_math::graph_theory::wardrop`.
//!
//! Re-exports [`AffineLatency`], [`FrankWolfeScratch`], [`WardropError`], [`WardropSummary`], and
//! [`solve_wardrop_in_place`] from the verified graph-theory domain crate while hosting UI overlay
//! state for [`super::GraphEditorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::CalloutManager;
use egui_plot::PlotPoint;
pub use math_explorer::pure_math::graph_theory::wardrop::{
    solve_wardrop_in_place, AffineLatency, FrankWolfeScratch, WardropError, WardropSummary,
};

const INITIAL_EDGE_CAP: usize = 128;
const DEMAND_CURVE_SAMPLES: usize = 21;

/// User hypothesis choice before toggling the zero-cost `u -> v` shortcut.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ShortcutPrediction {
    /// User predicts adding a zero-cost road speeds up commuters.
    SpeedsUpCommuters,
    /// User predicts commuter travel time is unchanged.
    NoChange,
    /// User predicts selfish routing triggers a slower Nash equilibrium trap.
    SlowsAllCommutersNashTrap,
}

/// Serializable congestion routing configuration for `GraphEditorTool`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BraessCongestionConfig {
    /// Whether congestion routing and Wardrop equilibrium overlays are active.
    pub enabled: bool,
    /// Origin node ID `s`.
    pub source_node: usize,
    /// Destination node ID `t`.
    pub sink_node: usize,
    /// Total commuter demand `D` routed from `s` to `t`.
    pub total_demand: f64,
    /// Whether the `u -> v` (`1 -> 2`) zero-cost shortcut is enabled.
    pub shortcut_enabled: bool,
    /// Optional predict-before-toggle hypothesis selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prediction: Option<ShortcutPrediction>,
}

impl Default for BraessCongestionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            source_node: 0,
            sink_node: 3,
            total_demand: 1.0,
            shortcut_enabled: false,
            prediction: None,
        }
    }
}

impl BraessCongestionConfig {
    /// Returns true if congestion configuration matches its inactive default.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// Interactive Braess's paradox congestion lab hosted inside [`super::GraphEditorTool`].
pub struct BraessCongestionLab {
    /// Active congestion parameters (`s`, `t`, demand `D`, shortcut toggle, hypothesis).
    pub config: BraessCongestionConfig,
    /// Per-edge affine latency coefficients aligned with `GraphEditorTool::edges`.
    pub edge_latencies: Vec<AffineLatency>,
    /// Default slope `a` used when adding new edges in the graph editor.
    pub new_edge_a: f64,
    /// Default base cost `b` used when adding new edges in the graph editor.
    pub new_edge_b: f64,
    /// Callout overlay manager marking the `1.50 -> 2.00` Nash trap.
    pub callout_manager: CalloutManager,
    /// Preallocated solver scratch workspace.
    pub scratch: FrankWolfeScratch,
    /// Equilibrium edge flows $f_e^*$.
    pub equilibrium_flows: Vec<f64>,
    /// Equilibrium edge latencies $l_e(f_e^*)$.
    pub equilibrium_latencies: Vec<f64>,
    /// Latest Wardrop solve result.
    pub status: Result<WardropSummary, WardropError>,
    /// Counterfactual latency without the `u -> v` (`1 -> 2`) shortcut when present.
    pub without_shortcut_latency: Option<f64>,
    pub(super) active_curve: Vec<PlotPoint>,
    pub(super) no_shortcut_curve: Vec<PlotPoint>,
}

impl Default for BraessCongestionLab {
    fn default() -> Self {
        Self {
            config: BraessCongestionConfig::default(),
            edge_latencies: Vec::new(),
            new_edge_a: 0.0,
            new_edge_b: 1.0,
            callout_manager: CalloutManager::new(),
            scratch: FrankWolfeScratch::new(),
            equilibrium_flows: Vec::with_capacity(INITIAL_EDGE_CAP),
            equilibrium_latencies: Vec::with_capacity(INITIAL_EDGE_CAP),
            status: Err(WardropError::MissingTerminal),
            without_shortcut_latency: None,
            active_curve: Vec::with_capacity(DEMAND_CURVE_SAMPLES),
            no_shortcut_curve: Vec::with_capacity(DEMAND_CURVE_SAMPLES),
        }
    }
}

#[path = "braess_ui.rs"]
mod braess_ui;
pub use braess_ui::BraessUiAction;

#[cfg(test)]
#[path = "braess_tests.rs"]
mod tests;

// [cite:dijkstra]

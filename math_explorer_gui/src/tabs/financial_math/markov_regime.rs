//! Continuous-Time Markov Chain (CTMC) and Hidden Markov Model (HMM) regime-switching workbench (`MarkovRegimeTool`).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::markov::hmm::{BaumWelchReport, BaumWelchWorkspace};

#[path = "markov_regime/engine.rs"]
mod engine;
#[path = "markov_regime/io.rs"]
pub(crate) mod io;
#[path = "markov_regime/ui.rs"]
mod ui;

#[cfg(test)]
#[path = "markov_regime/tests.rs"]
mod tests;

pub(crate) const MAX_CTMC_STATES: usize = 5;
pub(crate) const MAX_HMM_STATES: usize = 4;
pub(crate) const MAX_HMM_SYMBOLS: usize = 6;
pub(crate) const MAX_IMPORTED_OBSERVATIONS: usize = 4_096;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RegimeWorkbenchMode {
    ContinuousTimeCtmc,
    HiddenMarkovHmm,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MarkovRegimeState {
    pub horizon_t: f64,
    pub mode: RegimeWorkbenchMode,
    pub ctmc_states: usize,
    pub ctmc_source_state: usize,
    pub ctmc_rates: Vec<Vec<f64>>,
    pub hmm_states: usize,
    pub hmm_symbols: usize,
    pub hmm_initial: Vec<f64>,
    pub hmm_transitions: Vec<Vec<f64>>,
    pub hmm_emissions: Vec<Vec<f64>>,
    pub observations: Vec<usize>,
    pub bw_max_iters: usize,
    pub bw_tolerance: f64,
    pub rng_seed: u64,
}

pub struct MarkovRegimeTool {
    pub(crate) horizon_t: f64,
    pub(crate) mode: RegimeWorkbenchMode,
    pub(crate) ctmc_states: usize,
    pub(crate) ctmc_source_state: usize,
    pub(crate) ctmc_rates: Vec<Vec<f64>>,
    pub(crate) hmm_states: usize,
    pub(crate) hmm_symbols: usize,
    pub(crate) hmm_initial: Vec<f64>,
    pub(crate) hmm_transitions: Vec<Vec<f64>>,
    pub(crate) hmm_emissions: Vec<Vec<f64>>,
    pub(crate) observations: Vec<usize>,
    pub(crate) bw_max_iters: usize,
    pub(crate) bw_tolerance: f64,
    pub(crate) rng_seed: u64,
    pub(crate) state_rev: u64,

    // Preallocated Baum-Welch workspace
    pub(crate) bw_workspace: BaumWelchWorkspace<f64>,
    pub(crate) last_bw_report: Option<BaumWelchReport<f64>>,

    // Cached CTMC outputs
    pub(crate) ctmc_prob_series: Vec<(String, Vec<PlotPoint>)>,
    pub(crate) ctmc_gillespie_points: Vec<PlotPoint>,
    pub(crate) ctmc_absorption_points: Vec<PlotPoint>,
    pub(crate) ctmc_steady_state: Option<Vec<f64>>,
    pub(crate) ctmc_absorption_times: Vec<(usize, f64)>,

    // Cached HMM outputs
    pub(crate) hmm_posterior_series: Vec<(String, Vec<PlotPoint>)>,
    pub(crate) viterbi_points: Vec<PlotPoint>,
    pub(crate) viterbi_states: Vec<usize>,
    pub(crate) calc_error: Option<String>,
    pub(crate) io_state: io::MarkovIoState,
}

impl Default for MarkovRegimeTool {
    fn default() -> Self {
        let obs = vec![2, 2, 1, 2, 0, 0, 0, 1, 0, 0, 2, 2, 2, 1, 2, 2];
        let mut tool = Self {
            horizon_t: 6.0,
            mode: RegimeWorkbenchMode::ContinuousTimeCtmc,
            ctmc_states: 3,
            ctmc_source_state: 0,
            ctmc_rates: vec![
                vec![0.0, 0.55, 0.15],
                vec![0.40, 0.0, 0.35],
                vec![0.20, 0.65, 0.0],
            ],
            hmm_states: 2,
            hmm_symbols: 3,
            hmm_initial: vec![0.6, 0.4],
            hmm_transitions: vec![vec![0.82, 0.18], vec![0.22, 0.78]],
            hmm_emissions: vec![vec![0.60, 0.25, 0.15], vec![0.10, 0.25, 0.65]],
            observations: obs.clone(),
            bw_max_iters: 30,
            bw_tolerance: 1e-6,
            rng_seed: 2026,
            state_rev: 1,
            bw_workspace: BaumWelchWorkspace::new(2, 3, obs.len().max(64)),
            last_bw_report: None,
            ctmc_prob_series: Vec::new(),
            ctmc_gillespie_points: Vec::new(),
            ctmc_absorption_points: Vec::new(),
            ctmc_steady_state: None,
            ctmc_absorption_times: Vec::new(),
            hmm_posterior_series: Vec::new(),
            viterbi_points: Vec::new(),
            viterbi_states: Vec::new(),
            calc_error: None,
            io_state: io::MarkovIoState::default(),
        };
        tool.recalculate();
        tool
    }
}

impl MarkovRegimeTool {
    pub(crate) fn bump_revision(&mut self) {
        self.state_rev = self.state_rev.wrapping_add(1).max(1);
    }

    pub(crate) fn to_state(&self) -> MarkovRegimeState {
        MarkovRegimeState {
            horizon_t: self.horizon_t,
            mode: self.mode,
            ctmc_states: self.ctmc_states,
            ctmc_source_state: self.ctmc_source_state,
            ctmc_rates: self.ctmc_rates.clone(),
            hmm_states: self.hmm_states,
            hmm_symbols: self.hmm_symbols,
            hmm_initial: self.hmm_initial.clone(),
            hmm_transitions: self.hmm_transitions.clone(),
            hmm_emissions: self.hmm_emissions.clone(),
            observations: self.observations.clone(),
            bw_max_iters: self.bw_max_iters,
            bw_tolerance: self.bw_tolerance,
            rng_seed: self.rng_seed,
        }
    }
}

impl InteractiveTool for MarkovRegimeTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Markov Regime Switching"
    }

    fn domain(&self) -> &'static str {
        "financial_math"
    }

    fn state_revision(&self) -> u64 {
        self.state_rev
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn has_finite_parameters(&self) -> bool {
        Self::validate_state(&self.to_state()).is_ok()
    }

    fn clamp_parameters(&mut self) -> bool {
        let next_t = if self.horizon_t.is_finite() {
            self.horizon_t.clamp(0.2, 100.0)
        } else {
            6.0
        };
        let next_tol = if self.bw_tolerance.is_finite() {
            self.bw_tolerance.clamp(1e-10, 1e-1)
        } else {
            1e-6
        };
        let next_iters = self.bw_max_iters.clamp(1, 200);
        let changed = (next_t - self.horizon_t).abs() > f64::EPSILON
            || (next_tol - self.bw_tolerance).abs() > f64::EPSILON
            || next_iters != self.bw_max_iters;
        self.horizon_t = next_t;
        self.bw_tolerance = next_tol;
        self.bw_max_iters = next_iters;
        if changed {
            self.recalculate();
        }
        changed
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.to_state()).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: MarkovRegimeState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Markov regime state: {e}"))?;
        self.apply_state(parsed)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.render_workbench_ui(ui);
        });
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "MarkovRegimeTool",
        display_name: "Markov Regime Switching",
        domain: "financial_math",
        tags: &[
            "markov",
            "ctmc",
            "hidden markov",
            "baum welch",
            "viterbi",
            "regime switching",
        ],
        build: || Box::new(MarkovRegimeTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for MarkovRegimeTool {
    fn theory_description(&self) -> String {
        match self.mode {
            RegimeWorkbenchMode::ContinuousTimeCtmc => format!(
                "Continuous-time Markov generator $P(t) = \\exp(G t)$ across {} states over horizon $T = {:.2}$ with stationary balance $\\pi G = 0$.",
                self.ctmc_states, self.horizon_t
            ),
            RegimeWorkbenchMode::HiddenMarkovHmm => format!(
                "Hidden Markov regime inference across {} latent states and {} symbols via scaled Baum-Welch EM and Viterbi decoding.",
                self.hmm_states, self.hmm_symbols
            ),
        }
    }

    fn phonetic_description(&self) -> String {
        "Continuous time Markov generator exponential and Hidden Markov Baum Welch regime switching theory"
            .into()
    }

    fn theory_citation(&self) -> String {
        "A tutorial on hidden Markov models and selected applications in speech recognition (Rabiner, 1989)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Regime Switching".into(),
            "Models latent bull/bear or hot/cold state transitions in continuous and discrete time"
                .into(),
        );
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Bull-Bear Hidden Regime Decoder",
    "domain": "financial_math",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "horizon_t": 8.0,
    "mode": "HiddenMarkovHmm",
    "ctmc_states": 3,
    "ctmc_source_state": 0,
    "ctmc_rates": [
      [0.0, 0.50, 0.15],
      [0.35, 0.0, 0.40],
      [0.25, 0.55, 0.0]
    ],
    "hmm_states": 2,
    "hmm_symbols": 3,
    "hmm_initial": [0.55, 0.45],
    "hmm_transitions": [
      [0.85, 0.15],
      [0.20, 0.80]
    ],
    "hmm_emissions": [
      [0.65, 0.25, 0.10],
      [0.10, 0.25, 0.65]
    ],
    "observations": [2, 2, 2, 1, 2, 0, 0, 0, 0, 1, 0, 2, 2, 2],
    "bw_max_iters": 25,
    "bw_tolerance": 0.000001,
    "rng_seed": 31415
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "markov_regime_starter",
        title: "Bull-Bear Hidden Regime Decoder",
        description: "CTMC exp(Gt) & HMM Baum-Welch / Viterbi: Infer latent market and momentum regimes from discrete observation logs.",
        category: "Risk Management",
        icon: "🔀",
        domain: "financial_math",
        tool_name: Some("MarkovRegimeTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

// [cite:clinical_trials_statistics]

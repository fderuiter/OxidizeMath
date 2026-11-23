//! Ornstein-Uhlenbeck CSV calibration, confidence-band solver, and snippet workbench (`OuProcessTool`).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::ou_process::MomentumType;

#[path = "ou_process/engine.rs"]
mod engine;
#[path = "ou_process/io.rs"]
pub(crate) mod io;
#[path = "ou_process/ui.rs"]
mod ui;

#[cfg(test)]
#[path = "ou_process/tests.rs"]
mod tests;

pub(crate) const MAX_OU_STEPS: usize = 500;
pub(crate) const MAX_OU_PATHS: usize = 256;
pub(crate) const MAX_IMPORTED_CSV_POINTS: usize = 4_096;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OuProcessState {
    pub mu: f64,
    pub theta: f64,
    pub sigma: f64,
    pub x0: f64,
    pub dt: f64,
    pub sample_dt: f64,
    pub threshold: f64,
    pub n_steps: usize,
    pub n_paths: usize,
    pub rng_seed: u64,
    pub observed_series: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct OuBandRow {
    pub step: usize,
    pub time: f64,
    pub mean: f64,
    pub p05: f64,
    pub p25: f64,
    pub p75: f64,
    pub p95: f64,
}

pub struct OuProcessTool {
    pub(crate) mu: f64,
    pub(crate) theta: f64,
    pub(crate) sigma: f64,
    pub(crate) x0: f64,
    pub(crate) dt: f64,
    pub(crate) sample_dt: f64,
    pub(crate) threshold: f64,
    pub(crate) n_steps: usize,
    pub(crate) n_paths: usize,
    pub(crate) rng_seed: u64,
    pub(crate) observed_series: Vec<f64>,
    pub(crate) state_rev: u64,

    // Preallocated simulation scratch buffers
    pub(crate) traj_scratch: Vec<f64>,
    pub(crate) col_scratch: Vec<f64>,

    // Cached plot series & summary diagnostics
    pub(crate) mean_points: Vec<PlotPoint>,
    pub(crate) p05_points: Vec<PlotPoint>,
    pub(crate) p25_points: Vec<PlotPoint>,
    pub(crate) p75_points: Vec<PlotPoint>,
    pub(crate) p95_points: Vec<PlotPoint>,
    pub(crate) sample_path_points: Vec<PlotPoint>,
    pub(crate) observed_points: Vec<PlotPoint>,
    pub(crate) eq_line_points: [PlotPoint; 2],
    pub(crate) band_rows: Vec<OuBandRow>,
    pub(crate) momentum_class: MomentumType,
    pub(crate) exceedance_prob: f64,
    pub(crate) half_life: f64,
    pub(crate) calc_error: Option<String>,
    pub(crate) io_state: io::OuIoState,
}

impl Default for OuProcessTool {
    fn default() -> Self {
        let obs = vec![0.64, 0.61, 0.58, 0.59, 0.55, 0.53, 0.52, 0.50, 0.51, 0.49];
        let mut tool = Self {
            mu: 0.50,
            theta: 1.35,
            sigma: 0.12,
            x0: 0.64,
            dt: 0.05,
            sample_dt: 0.25,
            threshold: 0.55,
            n_steps: 80,
            n_paths: 96,
            rng_seed: 2026,
            observed_series: obs,
            state_rev: 1,
            traj_scratch: vec![0.0; MAX_OU_PATHS * (MAX_OU_STEPS + 1)],
            col_scratch: Vec::with_capacity(MAX_OU_PATHS),
            mean_points: Vec::new(),
            p05_points: Vec::new(),
            p25_points: Vec::new(),
            p75_points: Vec::new(),
            p95_points: Vec::new(),
            sample_path_points: Vec::new(),
            observed_points: Vec::new(),
            eq_line_points: [PlotPoint::new(0.0, 0.50), PlotPoint::new(4.0, 0.50)],
            band_rows: Vec::new(),
            momentum_class: MomentumType::Normal,
            exceedance_prob: 0.0,
            half_life: 0.0,
            calc_error: None,
            io_state: io::OuIoState::default(),
        };
        tool.recalculate();
        tool
    }
}

impl OuProcessTool {
    pub(crate) fn bump_revision(&mut self) {
        self.state_rev = self.state_rev.wrapping_add(1).max(1);
    }

    pub(crate) fn to_state(&self) -> OuProcessState {
        OuProcessState {
            mu: self.mu,
            theta: self.theta,
            sigma: self.sigma,
            x0: self.x0,
            dt: self.dt,
            sample_dt: self.sample_dt,
            threshold: self.threshold,
            n_steps: self.n_steps,
            n_paths: self.n_paths,
            rng_seed: self.rng_seed,
            observed_series: self.observed_series.clone(),
        }
    }
}

impl InteractiveTool for OuProcessTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Ornstein-Uhlenbeck Calibration"
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
        let next_mu = if self.mu.is_finite() {
            self.mu.clamp(-1_000.0, 1_000.0)
        } else {
            0.5
        };
        let next_theta = if self.theta.is_finite() {
            self.theta.clamp(0.01, 50.0)
        } else {
            1.0
        };
        let next_sigma = if self.sigma.is_finite() {
            self.sigma.clamp(0.0, 50.0)
        } else {
            0.15
        };
        let next_dt = if self.dt.is_finite() {
            self.dt.clamp(0.001, 2.0)
        } else {
            0.05
        };
        let next_steps = self.n_steps.clamp(10, MAX_OU_STEPS);
        let next_paths = self.n_paths.clamp(8, MAX_OU_PATHS);
        let changed = (next_mu - self.mu).abs() > f64::EPSILON
            || (next_theta - self.theta).abs() > f64::EPSILON
            || (next_sigma - self.sigma).abs() > f64::EPSILON
            || (next_dt - self.dt).abs() > f64::EPSILON
            || next_steps != self.n_steps
            || next_paths != self.n_paths;
        (self.mu, self.theta, self.sigma, self.dt) = (next_mu, next_theta, next_sigma, next_dt);
        (self.n_steps, self.n_paths) = (next_steps, next_paths);
        if changed {
            self.recalculate();
        }
        changed
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.to_state()).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: OuProcessState =
            serde_json::from_str(state).map_err(|e| format!("Invalid OU process state: {e}"))?;
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
        id: "OuProcessTool",
        display_name: "Ornstein-Uhlenbeck Calibration",
        domain: "financial_math",
        tags: &[
            "ornstein uhlenbeck",
            "mean reversion",
            "euler maruyama",
            "sde",
            "confidence bands",
            "calibration",
        ],
        build: || Box::new(OuProcessTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for OuProcessTool {
    fn theory_description(&self) -> String {
        format!(
            "Ornstein-Uhlenbeck mean-reverting SDE $dX_t = \\theta(\\mu - X_t)\\,dt + \\sigma\\,dW_t$ calibrated with $\\mu = {:.3}$, $\\theta = {:.3}$, $\\sigma = {:.3}$, and Euler-Maruyama step $\\Delta t = {:.3}$.",
            self.mu, self.theta, self.sigma, self.dt
        )
    }

    fn phonetic_description(&self) -> String {
        "Ornstein Uhlenbeck stochastic differential equation calibration and confidence band theory"
            .into()
    }

    fn theory_citation(&self) -> String {
        "On the Theory of the Brownian Motion (Uhlenbeck & Ornstein, 1930)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Mean Reversion".into(),
            "Pulls transient streaks exponentially back toward long-term equilibrium mu at rate theta".into(),
        );
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Sticky Heat-Check Mean Reversion",
    "domain": "financial_math",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "mu": 0.46,
    "theta": 0.42,
    "sigma": 0.16,
    "x0": 0.68,
    "dt": 0.05,
    "sample_dt": 0.25,
    "threshold": 0.52,
    "n_steps": 100,
    "n_paths": 120,
    "rng_seed": 777,
    "observed_series": [0.68, 0.66, 0.63, 0.65, 0.60, 0.58, 0.55, 0.54]
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "ou_process_starter",
        title: "Sticky Heat-Check Mean Reversion",
        description: "dXt = theta(mu - Xt)dt + sigma dWt: Calibrate OU parameters from CSV series and inspect 5%-95% Euler-Maruyama bands.",
        category: "Risk Management",
        icon: "📈",
        domain: "financial_math",
        tool_name: Some("OuProcessTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

// [cite:clinical_trials_statistics]

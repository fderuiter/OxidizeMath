//! Interactive Clinical Trial Sample Size & Statistical Power Curve Calculator.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod export;
pub mod ui;

#[cfg(test)]
mod tests;

use super::clinical_io::ClinicalIoState;
use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use eframe::egui;
use math_explorer::applied::clinical_trials::sample_size::{
    calculate_sample_size_means, calculate_sample_size_proportions,
};

/// Standard statistical power milestones tabulated in protocol sizing cards.
pub const POWER_MILESTONES: [f64; 8] = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99];
const POWER_SWEEP_STEPS: usize = 50;

/// Hypothesis test family used for sample size and power curve calculation.
#[derive(Debug, PartialEq, Eq, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum CalculationMode {
    /// Two-sample continuous outcome comparison (`calculate_sample_size_means`).
    Means,
    /// Two-sample binary proportion comparison (`calculate_sample_size_proportions`).
    Proportions,
}

impl CalculationMode {
    /// Human-readable mode label for protocol sizing reports.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Means => "Compare Means (Two-Sample Normal / t-Test)",
            Self::Proportions => "Compare Proportions (Two-Sample Pooled Z / Chi-Square)",
        }
    }
}

/// One row of the protocol sample-size sizing schedule at a target power `1 - beta`.
#[derive(Clone, Copy, Debug)]
pub struct ProtocolSizingRow {
    /// Target statistical power `1 - beta`.
    pub power: f64,
    /// Required subjects per arm `N`.
    pub n_per_arm: usize,
    /// Total 1:1 two-arm enrollment `2 * N`.
    pub total_n: usize,
}

/// Interactive two-arm sample size calculator with power-vs-`N` curve and protocol table exports.
pub struct SampleSizeCalculatorTool {
    /// Active hypothesis-testing calculation mode (`Means` vs `Proportions`).
    pub mode: CalculationMode,
    /// Two-sided Type I error rate `alpha`.
    pub alpha: f64,
    /// Target statistical power `1 - beta`.
    pub power: f64,
    /// Minimum clinically important mean difference `delta` (`Means` mode).
    pub delta: f64,
    /// Assumed common standard deviation `sigma` (`Means` mode).
    pub sigma: f64,
    /// Control arm event proportion `p1` (`Proportions` mode).
    pub p1: f64,
    /// Treatment arm event proportion `p2` (`Proportions` mode).
    pub p2: f64,
    /// Required per-arm sample size at `(alpha, power)` or domain validation error.
    pub result: Result<usize, String>,
    /// Swept `[n_per_arm, power]` curve across `1 - beta in [0.50, 0.99]`.
    pub(crate) power_curve: Vec<[f64; 2]>,
    /// Evaluated protocol sizing milestones across [`POWER_MILESTONES`].
    pub(crate) protocol_rows: Vec<ProtocolSizingRow>,
    /// Non-blocking save/status helper for `.csv` and `.md` protocol sizing cards.
    pub(crate) io_state: ClinicalIoState,
}

impl Default for SampleSizeCalculatorTool {
    fn default() -> Self {
        let mut tool = Self {
            mode: CalculationMode::Means,
            alpha: 0.05,
            power: 0.80,
            delta: 0.5,
            sigma: 1.0,
            p1: 0.5,
            p2: 0.6,
            result: Ok(0),
            power_curve: Vec::with_capacity(POWER_SWEEP_STEPS),
            protocol_rows: Vec::with_capacity(POWER_MILESTONES.len()),
            io_state: ClinicalIoState::default(),
        };
        tool.recalculate();
        tool
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SampleSizeState {
    mode: CalculationMode,
    alpha: f64,
    power: f64,
    delta: f64,
    sigma: f64,
    p1: f64,
    p2: f64,
}

impl SampleSizeCalculatorTool {
    /// Evaluates per-arm sample size at a given statistical power `p` under active parameters.
    pub fn evaluate_at_power(&self, p: f64) -> Result<usize, String> {
        match self.mode {
            CalculationMode::Means => {
                calculate_sample_size_means(self.alpha, p, self.delta, self.sigma)
                    .map_err(|e| e.to_string())
            }
            CalculationMode::Proportions => {
                calculate_sample_size_proportions(self.alpha, p, self.p1, self.p2)
                    .map_err(|e| e.to_string())
            }
        }
    }

    /// Recomputes required `N`, the `[N, 1 - beta]` power curve, and protocol sizing milestones.
    pub fn recalculate(&mut self) {
        self.result = self.evaluate_at_power(self.power);
        self.power_curve.clear();
        self.protocol_rows.clear();

        if self.result.is_err() {
            return;
        }

        for idx in 0..POWER_SWEEP_STEPS {
            let frac = idx as f64 / (POWER_SWEEP_STEPS - 1) as f64;
            let p = 0.50 + frac * (0.99 - 0.50);
            if let Ok(n) = self.evaluate_at_power(p) {
                self.power_curve.push([n as f64, p]);
            }
        }

        for &p in &POWER_MILESTONES {
            if let Ok(n) = self.evaluate_at_power(p) {
                self.protocol_rows.push(ProtocolSizingRow {
                    power: p,
                    n_per_arm: n,
                    total_n: n.saturating_mul(2),
                });
            }
        }
    }
}

impl InteractiveTool for SampleSizeCalculatorTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Sample Size Calculator"
    }

    fn domain(&self) -> &'static str {
        "clinical_trials"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        export::collect_export_datasets(self)
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::DEFAULT
    }

    fn has_finite_parameters(&self) -> bool {
        [
            self.alpha, self.power, self.delta, self.sigma, self.p1, self.p2,
        ]
        .iter()
        .all(|v| v.is_finite())
    }

    fn clamp_parameters(&mut self) -> bool {
        let cf = |v: f64, lo: f64, hi: f64, fb: f64| {
            if v.is_finite() {
                v.clamp(lo, hi)
            } else {
                fb
            }
        };
        self.alpha = cf(self.alpha, 0.001, 0.20, 0.05);
        self.power = cf(self.power, 0.50, 0.999, 0.80);
        self.delta = cf(self.delta, 0.01, 5.0, 0.50);
        self.sigma = cf(self.sigma, 0.10, 10.0, 1.0);
        self.p1 = cf(self.p1, 0.01, 0.99, 0.50);
        self.p2 = cf(self.p2, 0.01, 0.99, 0.60);
        if self.mode == CalculationMode::Proportions && (self.p1 - self.p2).abs() < 1e-6 {
            self.p2 = (self.p1 + 0.10).clamp(0.02, 0.99);
        }
        self.recalculate();
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&SampleSizeState {
            mode: self.mode,
            alpha: self.alpha,
            power: self.power,
            delta: self.delta,
            sigma: self.sigma,
            p1: self.p1,
            p2: self.p2,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let s: SampleSizeState =
            serde_json::from_str(state).map_err(|e| format!("Invalid sample size state: {e}"))?;
        if ![s.alpha, s.power, s.delta, s.sigma, s.p1, s.p2]
            .iter()
            .all(|v| v.is_finite())
            || !(0.001..=0.20).contains(&s.alpha)
            || !(0.5..=0.999).contains(&s.power)
            || !(0.01..=5.0).contains(&s.delta)
            || !(0.1..=10.0).contains(&s.sigma)
            || !(0.01..=0.99).contains(&s.p1)
            || !(0.01..=0.99).contains(&s.p2)
        {
            return Err("Sample size parameters out of bounds".into());
        }
        let result = match s.mode {
            CalculationMode::Means => {
                calculate_sample_size_means(s.alpha, s.power, s.delta, s.sigma)
                    .map_err(|e| e.to_string())?
            }
            CalculationMode::Proportions => {
                calculate_sample_size_proportions(s.alpha, s.power, s.p1, s.p2)
                    .map_err(|e| e.to_string())?
            }
        };
        self.mode = s.mode;
        self.alpha = s.alpha;
        self.power = s.power;
        self.delta = s.delta;
        self.sigma = s.sigma;
        self.p1 = s.p1;
        self.p2 = s.p2;
        self.result = Ok(result);
        self.recalculate();
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        ui::show_sample_size_inside(self, ui);
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "SampleSizeCalculatorTool",
        display_name: "Sample Size Calculator",
        domain: "clinical_trials",
        tags: &["sample_size", "power", "protocol", "clinical_trials"],
        build: || Box::new(SampleSizeCalculatorTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for SampleSizeCalculatorTool {
    fn theory_description(&self) -> String {
        match self.mode {
            CalculationMode::Means => format!(
                "Two-sample hypothesis test sample size for means: $N = \\frac{{2 (Z_{{\\alpha/2}} + Z_\\beta)^2 \\sigma^2}}{{\\delta^2}}$ with significance $\\alpha = {:.3}$, power $1-\\beta = {:.2}$, effect size $\\delta = {:.2}$, $\\sigma = {:.2}$.",
                self.alpha, self.power, self.delta, self.sigma
            ),
            CalculationMode::Proportions => format!(
                "Two-sample hypothesis test sample size for proportions: $N = \\frac{{(Z_{{\\alpha/2}} \\sqrt{{2\\bar{{p}}(1-\\bar{{p}})}} + Z_\\beta \\sqrt{{p_1(1-p_1) + p_2(1-p_2)}})^2}}{{(p_1 - p_2)^2}}$ with $\\alpha = {:.3}$, power $1-\\beta = {:.2}$, $p_1 = {:.2}$, $p_2 = {:.2}$.",
                self.alpha, self.power, self.p1, self.p2
            ),
        }
    }

    fn phonetic_description(&self) -> String {
        "Clinical trial sample size and statistical power theory".into()
    }

    fn theory_citation(&self) -> String {
        "Sample Size Calculations in Clinical Research (Chow et al., 2017)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Means".into(),
            "Normal approximation for difference between two independent continuous means".into(),
        );
        map.insert(
            "Proportions".into(),
            "Normal approximation with pooled variance for binary outcome proportions".into(),
        );
        map
    }
}

/// Starter preset JSON payload for statistical power and sample size design.
pub const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Statistical Power & Sample Size",
    "domain": "clinical_trials",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "mode": "Means",
    "alpha": 0.05,
    "power": 0.85,
    "delta": 0.5,
    "sigma": 1.0,
    "p1": 0.5,
    "p2": 0.65
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "sample_size_starter",
        title: "Statistical Power & Sample Size",
        description: "N = 2(z_α + z_β)²σ² / δ²: Neyman-Pearson hypothesis test sizing for minimum detectable effect.",
        category: "Trial Design",
        icon: "👥",
        domain: "clinical_trials",
        tool_name: Some("SampleSizeCalculatorTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

// [cite:clinical_trials_statistics]

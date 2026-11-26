//! Standalone calibration and post-hoc recalibration workbench for probabilistic AI forecasts.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

/// Interactive `egui_plot` views for reliability diagrams, Murphy bars, `ROC`/`PR`, and `CRPS`.
pub mod plots;
/// Starter gallery presets for [`CalibrationWorkbenchTool`].
pub mod presets;
/// Serializable configuration, synthetic cohort generation, and evaluation snapshots.
pub mod state;
#[cfg(test)]
mod tests;
/// Side-panel controls and live metric comparison table.
pub mod ui;

use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use scientific_metadata::theory::TheoryDescribable;
use state::{evaluate_workbench, CalibrationWorkbenchConfig, WorkbenchEvaluation};

/// Standalone interactive workbench for proper scoring rules, reliability curves,
/// Murphy decompositions, `ROC`/`PR` sweeps, `CRPS`, and post-hoc recalibration.
pub struct CalibrationWorkbenchTool {
    pub(crate) config: CalibrationWorkbenchConfig,
    pub(crate) evaluation: WorkbenchEvaluation,
    revision: u64,
}

impl Default for CalibrationWorkbenchTool {
    fn default() -> Self {
        let config = CalibrationWorkbenchConfig::default();
        let evaluation = evaluate_workbench(&config);
        Self {
            config,
            evaluation,
            revision: 1,
        }
    }
}

impl CalibrationWorkbenchTool {
    /// Recomputes raw and recalibrated metrics, reliability curves, threshold sweeps, and `CRPS` series.
    pub fn recompute(&mut self) {
        self.evaluation = evaluate_workbench(&self.config);
        self.revision = self.revision.wrapping_add(1);
    }
}

impl InteractiveTool for CalibrationWorkbenchTool {
    fn name(&self) -> &'static str {
        "Calibration Workbench"
    }

    fn domain(&self) -> &'static str {
        "ai"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn state_revision(&self) -> u64 {
        self.revision
    }

    fn has_finite_parameters(&self) -> bool {
        self.config.logit_scale.is_finite()
            && self.config.logit_bias.is_finite()
            && self.config.crps_spread.is_finite()
    }

    fn clamp_parameters(&mut self) -> bool {
        let clamped = self.config.clamp_in_place();
        if clamped {
            self.recompute();
        }
        clamped
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("raw_brier", self.evaluation.raw_metrics.brier),
            ("cal_brier", self.evaluation.cal_metrics.brier),
            ("raw_ece", self.evaluation.raw_metrics.ece),
            ("cal_ece", self.evaluation.cal_metrics.ece),
            ("raw_log_loss", self.evaluation.raw_metrics.log_loss),
            ("cal_log_loss", self.evaluation.cal_metrics.log_loss),
            ("roc_auc", self.evaluation.cal_metrics.roc_auc),
            ("pr_auc", self.evaluation.cal_metrics.pr_auc_step),
            ("mean_crps", self.evaluation.mean_cal_crps),
        ]
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.config).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: CalibrationWorkbenchConfig = serde_json::from_str(state)
            .map_err(|e| format!("Invalid calibration workbench state: {e}"))?;
        parsed.validate()?;
        self.config = parsed;
        self.recompute();
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        if ui::draw_controls_panel(ui, &mut self.config, &self.evaluation) {
            self.recompute();
        }
        plots::draw_workbench_plot(ui, &self.config, &self.evaluation);
    }
}

impl TheoryDescribable for CalibrationWorkbenchTool {
    fn theory_description(&self) -> String {
        format!(
            "Supervised probabilistic forecast calibration across {} bins with logit scale $s = {:.2}$ and bias $b = {:.2}$: evaluates quadratic Brier score with Murphy partition $BS = REL - RES + UNC$, symmetric log-loss, Gaussian and empirical CRPS, ROC and PR threshold sweeps, and post-hoc Platt $\\sigma(a z + b)$, temperature $\\sigma(z / T)$, and monotonic PAVA isotonic recalibration.",
            self.config.num_bins, self.config.logit_scale, self.config.logit_bias
        )
    }

    fn phonetic_description(&self) -> String {
        "Probabilistic calibration proper scoring rules and post-hoc recalibration workbench".into()
    }

    fn theory_citation(&self) -> String {
        "On Calibration of Modern Neural Networks (Guo et al., 2017); Strictly Proper Scoring Rules, Prediction, and Estimation (Gneiting & Raftery, 2007); A New Vector Partition of the Probability Score (Murphy, 1973)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "CalibrationWorkbenchTool",
        display_name: "Calibration Workbench",
        domain: "ai",
        tags: &["calibration", "brier", "ece", "roc", "crps"],
        build: || Box::new(CalibrationWorkbenchTool::default()),
    }
}

// [cite:clinical_trials_statistics]
// [cite:self_calibration_paper]

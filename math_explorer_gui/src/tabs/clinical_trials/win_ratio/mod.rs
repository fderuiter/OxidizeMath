//! Interactive Composite Win Ratio Stress Tester tool (`WinRatioStressTesterTool`).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod cohort_io;
mod export_view;
pub mod protocol_export;
mod ui;

#[cfg(test)]
mod tests;

use std::sync::mpsc::Receiver;
use std::sync::OnceLock;

use eframe::egui;
use egui_plot::ExportPrimitive;
use math_explorer::applied::win_ratio::censoring::TierCount;
use math_explorer::applied::win_ratio::power_sweep::{
    run_power_sweep_into, run_weibull_power_sweep, PowerSweepConfig, PowerSweepReport,
    PowerSweepWorkspace, MAX_ARM_SIZE, MAX_SWEEP_REPLICATES, MAX_SWEEP_STEPS,
};
use math_explorer::applied::win_ratio::simulation::{win_ratio_parameter, SimulationParams};
use math_explorer::applied::win_ratio::u_statistic::UnmatchedWinRatioSummary;

use self::cohort_io::{evaluate_loaded_cohort, LoadedCohortAnalysis};
use self::protocol_export::{format_protocol_csv, format_protocol_json, ProtocolExportContext};
use crate::framework::{ExportCapabilities, InteractiveTool};

/// Active chart tab inside [`WinRatioStressTesterTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum ChartMode {
    /// Empirical Power-by-`N` and tie-fraction curve across sample sizes.
    #[default]
    PowerCurve,
    /// Hierarchical Tier 1 (fatal) vs Tier 2 (non-fatal) win/loss/tie waterfall.
    TierWaterfall,
    /// Uploaded local two-arm Tier-1 Kaplan-Meier curves.
    CohortSurvival,
}

/// Compact bookmarkable URL and preset configuration (intentionally omits uploaded patient rows).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WinRatioStressTesterState {
    pub lambda1: f64,
    pub lambda2: f64,
    pub alpha: f64,
    pub theta: f64,
    pub censor_horizon: f64,
    pub dropout_rate: f64,
    pub sig_alpha: f64,
    pub n_min: usize,
    pub n_max: usize,
    pub n_steps: usize,
    pub replicates: usize,
    pub seed: u64,
    pub chart_mode: ChartMode,
}

/// Interactive browser & desktop stress tester for hierarchical composite win ratios.
pub struct WinRatioStressTesterTool {
    pub(crate) params: SimulationParams,
    pub(crate) sweep_cfg: PowerSweepConfig,
    pub(crate) seed: u64,
    pub(crate) chart_mode: ChartMode,
    pub(crate) apply_horizon_to_cohort: bool,
    workspace: Option<PowerSweepWorkspace>,
    cached_report: OnceLock<PowerSweepReport>,
    pub(crate) uploaded_cohort: Option<LoadedCohortAnalysis>,
    pub(crate) status_banner: Option<(String, bool)>,
    pub(crate) open_cohort_rx: Option<Receiver<Result<(String, String), String>>>,
    pub(crate) save_protocol_rx: Option<Receiver<Result<String, String>>>,
}

impl Default for WinRatioStressTesterTool {
    fn default() -> Self {
        Self {
            params: SimulationParams::new(0.14, 0.32, 0.72, 0.65),
            sweep_cfg: PowerSweepConfig::default(),
            seed: 2026,
            chart_mode: ChartMode::PowerCurve,
            apply_horizon_to_cohort: false,
            workspace: None,
            cached_report: OnceLock::new(),
            uploaded_cohort: None,
            status_banner: None,
            open_cohort_rx: None,
            save_protocol_rx: None,
        }
    }
}

impl WinRatioStressTesterTool {
    #[must_use]
    pub fn report(&self) -> &PowerSweepReport {
        self.cached_report.get_or_init(|| {
            run_weibull_power_sweep(&self.params, &self.sweep_cfg, self.seed)
                .expect("validated WinRatioStressTesterTool parameters must produce sweep report")
        })
    }

    pub(crate) fn invalidate_sweep_cache(&mut self) {
        self.cached_report = OnceLock::new();
    }

    pub(crate) fn recompute_parametric_sweep(&mut self) {
        let ws = self
            .workspace
            .get_or_insert_with(PowerSweepWorkspace::default);
        if let Ok((count, representative_summary, representative_tiers)) =
            run_power_sweep_into(&self.params, &self.sweep_cfg, self.seed, ws)
        {
            let next_report = PowerSweepReport {
                theoretical_uncensored_wr: win_ratio_parameter(&self.params),
                representative_summary,
                representative_tiers,
                points: ws.points_buf[..count].to_vec(),
            };
            self.cached_report = OnceLock::from(next_report);
        }
    }

    pub(crate) fn refresh_uploaded_cohort_if_needed(&mut self) {
        let Some(existing) = self.uploaded_cohort.take() else {
            return;
        };
        let horizon = self
            .apply_horizon_to_cohort
            .then_some(self.sweep_cfg.censor_horizon);
        match evaluate_loaded_cohort(
            existing.source_name.clone(),
            existing.treatment.clone(),
            existing.control.clone(),
            self.sweep_cfg.sig_alpha,
            horizon,
        ) {
            Ok(updated) => self.uploaded_cohort = Some(updated),
            Err(err) => {
                self.uploaded_cohort = Some(existing);
                self.status_banner = Some((err, true));
            }
        }
    }

    #[must_use]
    pub fn active_summary_and_tiers(&self) -> (UnmatchedWinRatioSummary, [TierCount; 2]) {
        if let Some(cohort) = &self.uploaded_cohort {
            (cohort.summary, cohort.tiers)
        } else {
            let rep = self.report();
            (rep.representative_summary, rep.representative_tiers)
        }
    }

    #[must_use]
    pub fn build_protocol_csv(&self) -> String {
        let rep = self.report();
        let (summary, tiers) = self.active_summary_and_tiers();
        format_protocol_csv(&ProtocolExportContext {
            params: &self.params,
            cfg: &self.sweep_cfg,
            seed: self.seed,
            theoretical_wr: rep.theoretical_uncensored_wr,
            active_summary: &summary,
            active_tiers: &tiers,
            cohort: self.uploaded_cohort.as_ref(),
            sweep_points: &rep.points,
        })
    }

    #[must_use]
    pub fn build_protocol_json(&self) -> String {
        let rep = self.report();
        let (summary, tiers) = self.active_summary_and_tiers();
        format_protocol_json(&ProtocolExportContext {
            params: &self.params,
            cfg: &self.sweep_cfg,
            seed: self.seed,
            theoretical_wr: rep.theoretical_uncensored_wr,
            active_summary: &summary,
            active_tiers: &tiers,
            cohort: self.uploaded_cohort.as_ref(),
            sweep_points: &rep.points,
        })
    }
}

impl InteractiveTool for WinRatioStressTesterTool {
    fn name(&self) -> &'static str {
        "Composite Win Ratio Stress Tester"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn domain(&self) -> &'static str {
        "clinical_trials"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.build_export_datasets()
    }

    fn export_plot_primitives(&self) -> Vec<ExportPrimitive> {
        self.build_export_primitives()
    }

    fn has_finite_parameters(&self) -> bool {
        [
            self.params.lambda1,
            self.params.lambda2,
            self.params.alpha,
            self.params.theta,
            self.sweep_cfg.censor_horizon,
            self.sweep_cfg.dropout_rate,
            self.sweep_cfg.sig_alpha,
        ]
        .iter()
        .all(|v| v.is_finite())
    }

    fn clamp_parameters(&mut self) -> bool {
        self.params.lambda1 = self.params.lambda1.clamp(0.02, 1.0);
        self.params.lambda2 = self.params.lambda2.clamp(0.02, 1.5);
        self.params.alpha = self.params.alpha.clamp(0.20, 1.0);
        self.params.theta = self.params.theta.clamp(0.25, 1.5);
        self.sweep_cfg.censor_horizon = self.sweep_cfg.censor_horizon.clamp(0.5, 20.0);
        self.sweep_cfg.dropout_rate = self.sweep_cfg.dropout_rate.clamp(0.0, 0.50);
        self.sweep_cfg.sig_alpha = self.sweep_cfg.sig_alpha.clamp(0.01, 0.10);
        self.sweep_cfg.n_min = self.sweep_cfg.n_min.clamp(10, MAX_ARM_SIZE);
        self.sweep_cfg.n_max = self
            .sweep_cfg
            .n_max
            .clamp(self.sweep_cfg.n_min, MAX_ARM_SIZE);
        self.sweep_cfg.n_steps = self.sweep_cfg.n_steps.clamp(2, MAX_SWEEP_STEPS);
        self.sweep_cfg.replicates = self.sweep_cfg.replicates.clamp(20, MAX_SWEEP_REPLICATES);
        self.invalidate_sweep_cache();
        true
    }

    fn save_state(&self) -> Option<String> {
        let persisted_mode = if self.chart_mode == ChartMode::CohortSurvival {
            ChartMode::PowerCurve
        } else {
            self.chart_mode
        };
        serde_json::to_string(&WinRatioStressTesterState {
            lambda1: self.params.lambda1,
            lambda2: self.params.lambda2,
            alpha: self.params.alpha,
            theta: self.params.theta,
            censor_horizon: self.sweep_cfg.censor_horizon,
            dropout_rate: self.sweep_cfg.dropout_rate,
            sig_alpha: self.sweep_cfg.sig_alpha,
            n_min: self.sweep_cfg.n_min,
            n_max: self.sweep_cfg.n_max,
            n_steps: self.sweep_cfg.n_steps,
            replicates: self.sweep_cfg.replicates,
            seed: self.seed,
            chart_mode: persisted_mode,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: WinRatioStressTesterState =
            serde_json::from_str(state).map_err(|e| format!("Invalid win ratio state: {e}"))?;
        let candidate_params = SimulationParams::new(s.lambda1, s.lambda2, s.alpha, s.theta);
        candidate_params.validate().map_err(|e| e.to_string())?;
        if !(0.01..=2.0).contains(&s.lambda1)
            || !(0.01..=3.0).contains(&s.lambda2)
            || !(0.10..=1.0).contains(&s.alpha)
            || !(0.10..=3.0).contains(&s.theta)
            || !(0.1..=100.0).contains(&s.censor_horizon)
            || !(0.0..=2.0).contains(&s.dropout_rate)
            || !(0.001..=0.25).contains(&s.sig_alpha)
        {
            return Err("Win ratio parameters out of interactive bounds".to_string());
        }
        let candidate_cfg = PowerSweepConfig {
            n_min: s.n_min,
            n_max: s.n_max,
            n_steps: s.n_steps,
            replicates: s.replicates,
            censor_horizon: s.censor_horizon,
            dropout_rate: s.dropout_rate,
            sig_alpha: s.sig_alpha,
        };
        candidate_cfg.validate().map_err(|e| e.to_string())?;

        self.params = candidate_params;
        self.sweep_cfg = candidate_cfg;
        self.seed = s.seed;
        self.chart_mode = if s.chart_mode == ChartMode::CohortSurvival {
            ChartMode::PowerCurve
        } else {
            s.chart_mode
        };
        self.invalidate_sweep_cache();
        self.refresh_uploaded_cohort_if_needed();
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.poll_async_io(ui.ctx());
        let panel_id = crate::framework::scoped_panel_id(ui, "win_ratio_stress_side_panel");
        crate::framework::PaneSidePanel::left(panel_id)
            .show_inside(ui, |ui| self.render_side_controls(ui));
        egui::CentralPanel::default().show_inside(ui, |ui| self.render_central_view(ui));
    }
}

// [cite:win_ratio]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "WinRatioStressTesterTool",
        display_name: "Composite Win Ratio Stress Tester",
        domain: "clinical_trials",
        tags: &["win-ratio", "composite-endpoints", "censoring", "power", "u-statistics"],
        build: || Box::new(WinRatioStressTesterTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for WinRatioStressTesterTool {
    fn theory_description(&self) -> String {
        format!(
            "Unmatched hierarchical composite win ratio $\\widehat{{WR}} = N_W / N_L$ under right censoring with Pocock/Bebu-Lachin two-sample U-statistic variance and bivariate Weibull survival $S(t, x) = \\exp(-(\\lambda_1 t + \\lambda_2 x)^\\alpha)$, $\\theta = {:.2}$, follow-up horizon $C = {:.1}$.",
            self.params.theta, self.sweep_cfg.censor_horizon
        )
    }

    fn phonetic_description(&self) -> String {
        "Hierarchical composite win ratio two-sample U-statistic and bivariate Weibull censoring sensitivity".into()
    }

    fn theory_citation(&self) -> String {
        "The win ratio: a new approach to the analysis of composite endpoints in clinical trials based on clinical priorities (Pocock et al., 2012; Bebu & Lachin, 2016)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "U-Statistic Inference".into(),
            "Projecting pairwise win/loss kernels onto subject marginals calibrates two-sided confidence intervals and Type I error across O(n1*n2) dependent pairs.".into(),
        );
        map.insert(
            "Right-Censored Waterfall".into(),
            "Indeterminate censored pairs at Tier 1 (fatal) pass cleanly to Tier 2 (non-fatal hospitalization) rather than misclassifying follow-up time as survival.".into(),
        );
        map
    }
}

pub(crate) const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Composite Win Ratio Stress Tester",
    "domain": "clinical_trials",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "lambda1": 0.18,
    "lambda2": 0.40,
    "alpha": 0.68,
    "theta": 0.58,
    "censor_horizon": 4.5,
    "dropout_rate": 0.08,
    "sig_alpha": 0.05,
    "n_min": 40,
    "n_max": 160,
    "n_steps": 5,
    "replicates": 60,
    "seed": 4242,
    "chart_mode": "TierWaterfall"
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "win_ratio_stress_tester_starter",
        title: "Composite Win Ratio Stress Tester",
        description: "WR = N_W / N_L: Hierarchical fatal/non-fatal U-statistic CIs, censoring waterfalls, and Weibull power sweeps.",
        category: "Composite Endpoints",
        icon: "🏆",
        domain: "clinical_trials",
        tool_name: Some("WinRatioStressTesterTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

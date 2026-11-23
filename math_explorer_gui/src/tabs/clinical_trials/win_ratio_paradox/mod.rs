//! Standalone `WinRatioParadoxLab` workbench in `ClinicalTrialsTab` exposing composite-endpoint
//! severity reversals between pooled time-to-first-event Hazard Ratios and prioritized Win Ratio.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod challenge;
pub mod model;
pub mod plots;
pub mod ui;

#[cfg(test)]
mod tests;

use crate::framework::{
    CalloutManager, ChallengeEvaluation, ExportCapabilities, InteractiveTool, ToolStateBounds,
};
use eframe::egui;
use math_explorer::applied::win_ratio::pair_comparison::{HigherIsBetter, WinRatioAnalysis};
use math_explorer::applied::win_ratio::WinRatioScratch;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Grid resolution along each axis of the 2D `(mortality_reduction, early_hosp_rate)` regime map.
pub const REGIME_GRID_DIM: usize = 24;
/// Maximum patients per arm supported without reallocating scratch buffers.
pub const MAX_SUBJECTS_PER_ARM: usize = 40;
/// Canonical base RNG seed per ADR-0035.
pub const BASE_SEED: u64 = 0x5749_4E52_4154_494F;

/// Serializable configuration for [`WinRatioParadoxLab`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WinRatioParadoxConfig {
    /// Patients per trial arm (`10..=40`).
    pub sample_size_per_arm: usize,
    /// Administrative follow-up censoring horizon in months (`12.0..=60.0`).
    pub horizon_months: f64,
    /// Control-arm fatal event rate over the follow-up horizon (`0.20..=0.85`).
    pub control_mortality_rate: f64,
    /// Relative fatal mortality reduction in the Treatment arm (`0.0..=0.90`).
    pub mortality_reduction: f64,
    /// Fraction of Treatment patients experiencing early non-fatal hospitalization (`0.0..=0.95`).
    pub early_hosp_rate: f64,
    /// Mean onset time (months) of early non-fatal Treatment hospitalizations (`1.0..=24.0`).
    pub early_hosp_mean_months: f64,
    /// Toggle 1:1 `MatchedPairing` vs all-pairs `UnmatchedPairing`.
    pub matched_pairing: bool,
    /// Deterministic `OxidizeRng` stream counter (`0` = canonical analytical cohort).
    pub seed_counter: u64,
}

impl Default for WinRatioParadoxConfig {
    fn default() -> Self {
        Self {
            sample_size_per_arm: 20,
            horizon_months: 36.0,
            control_mortality_rate: 0.60,
            mortality_reduction: 5.0 / 6.0,
            early_hosp_rate: 0.90,
            early_hosp_mean_months: 4.487_285,
            matched_pairing: false,
            seed_counter: 0,
        }
    }
}

/// Computed statistical metrics contrasting pooled first-event survival with prioritized Win Ratio.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WinRatioParadoxMetrics {
    /// Pooled composite time-to-first-event Hazard Ratio (`> 1.0` penalizes Treatment).
    pub pooled_hazard_ratio: f64,
    /// Unmatched (or active-mode) prioritized Win Ratio (`> 1.0` favors Treatment).
    pub win_ratio: f64,
    /// 95% confidence interval for the Win Ratio.
    pub win_ratio_ci: (f64, f64),
    /// Two-sided asymptotic p-value for $H_0: \text{WR} = 1$.
    pub win_ratio_p_value: f64,
    /// 1:1 matched-pair Win Ratio for side-by-side sensitivity comparison.
    pub matched_win_ratio: f64,
    /// Sample Win Ratio via `WinLossCounts` cross-check.
    pub sample_win_ratio: f64,
    /// Independent two-sample t-statistic on fatal survival duration (`Treatment - Control`).
    pub t_statistic: f64,
    /// Two-sided p-value from `TTestIndependent`.
    pub t_p_value: f64,
    /// Total prioritized wins.
    pub wins: i32,
    /// Total prioritized losses.
    pub losses: i32,
    /// Total tied pairs across both tiers.
    pub ties: i32,
    /// Tier-1 (fatal survival) wins.
    pub tier1_wins: i32,
    /// Tier-1 (fatal survival) losses.
    pub tier1_losses: i32,
    /// Tier-2 (hospitalization-free time) wins.
    pub tier2_wins: i32,
    /// Tier-2 (hospitalization-free time) losses.
    pub tier2_losses: i32,
    /// True when pooled `HR > 1.0` penalizes treatment while prioritized `WR > 1.0` favors treatment.
    pub is_severity_reversal: bool,
}

/// Standalone clinical paradox workbench contrasting composite time-to-first-event HR vs Win Ratio.
pub struct WinRatioParadoxLab {
    /// Current user-editable cohort parameters.
    pub config: WinRatioParadoxConfig,
    /// Latest computed statistical metrics.
    pub metrics: WinRatioParadoxMetrics,
    /// Preallocated hierarchical `WinRatioAnalysis` (`[HigherIsBetter, HigherIsBetter]`).
    pub analysis: WinRatioAnalysis<f64>,
    /// Preallocated zero-allocation workspace for subject outcomes and 2D regime grid.
    pub scratch: WinRatioScratch,
    /// Kaplan-Meier step points for Treatment composite time-to-first-event.
    pub km_comp_treat: Vec<[f64; 2]>,
    /// Kaplan-Meier step points for Control composite time-to-first-event.
    pub km_comp_ctrl: Vec<[f64; 2]>,
    /// Kaplan-Meier step points for Treatment fatal survival.
    pub km_fatal_treat: Vec<[f64; 2]>,
    /// Kaplan-Meier step points for Control fatal survival.
    pub km_fatal_ctrl: Vec<[f64; 2]>,
    /// Cached RGBA pixels for the 2D `(mortality_reduction, early_hosp_rate)` regime map.
    pub regime_pixels: Vec<egui::Color32>,
    /// Uploaded GPU texture handle for the 2D regime map.
    pub regime_texture: Option<egui::TextureHandle>,
    /// Whether the regime texture needs re-uploading on next frame.
    pub texture_dirty: bool,
    /// Plot callout overlay manager.
    pub callout_manager: CalloutManager,
    /// Active inverse challenge identifier, if any.
    pub active_challenge: Option<&'static str>,
}

impl Default for WinRatioParadoxLab {
    fn default() -> Self {
        let analysis = WinRatioAnalysis::new()
            .add_strategy(Box::new(HigherIsBetter))
            .add_strategy(Box::new(HigherIsBetter));
        let mut lab = Self {
            config: WinRatioParadoxConfig::default(),
            metrics: WinRatioParadoxMetrics::default(),
            analysis,
            scratch: WinRatioScratch::new(MAX_SUBJECTS_PER_ARM, REGIME_GRID_DIM, REGIME_GRID_DIM),
            km_comp_treat: Vec::with_capacity(96),
            km_comp_ctrl: Vec::with_capacity(96),
            km_fatal_treat: Vec::with_capacity(96),
            km_fatal_ctrl: Vec::with_capacity(96),
            regime_pixels: vec![egui::Color32::BLACK; REGIME_GRID_DIM * REGIME_GRID_DIM],
            regime_texture: None,
            texture_dirty: true,
            callout_manager: CalloutManager::new(),
            active_challenge: None,
        };
        lab.recalculate();
        lab
    }
}

impl WinRatioParadoxLab {
    fn compute_revision(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.config.sample_size_per_arm.hash(&mut h);
        self.config.horizon_months.to_bits().hash(&mut h);
        self.config.control_mortality_rate.to_bits().hash(&mut h);
        self.config.mortality_reduction.to_bits().hash(&mut h);
        self.config.early_hosp_rate.to_bits().hash(&mut h);
        self.config.early_hosp_mean_months.to_bits().hash(&mut h);
        self.config.matched_pairing.hash(&mut h);
        self.config.seed_counter.hash(&mut h);
        self.active_challenge.hash(&mut h);
        h.finish()
    }
}

impl InteractiveTool for WinRatioParadoxLab {
    fn name(&self) -> &'static str {
        "Win Ratio Paradox Lab"
    }

    fn domain(&self) -> &'static str {
        "clinical_trials"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        vec![
            ("Treatment Composite KM".into(), self.km_comp_treat.clone()),
            ("Control Composite KM".into(), self.km_comp_ctrl.clone()),
            ("Treatment Fatal KM".into(), self.km_fatal_treat.clone()),
            ("Control Fatal KM".into(), self.km_fatal_ctrl.clone()),
        ]
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::DEFAULT
    }

    fn state_revision(&self) -> u64 {
        self.compute_revision()
    }

    fn has_finite_parameters(&self) -> bool {
        let c = &self.config;
        c.horizon_months.is_finite()
            && c.control_mortality_rate.is_finite()
            && c.mortality_reduction.is_finite()
            && c.early_hosp_rate.is_finite()
            && c.early_hosp_mean_months.is_finite()
    }

    fn clamp_parameters(&mut self) -> bool {
        let before = self.config.clone();
        self.clamp_config_in_place();
        if self.config != before {
            self.recalculate();
            true
        } else {
            false
        }
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.config).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: WinRatioParadoxConfig =
            serde_json::from_str(state).map_err(|e| format!("Invalid WinRatio state: {e}"))?;
        self.config = parsed;
        self.clamp_config_in_place();
        self.recalculate();
        Ok(())
    }

    fn apply_challenge(&mut self, challenge_id: Option<&str>) -> Result<(), String> {
        self.set_challenge_mode(challenge_id)
    }

    fn active_challenge_id(&self) -> Option<&'static str> {
        self.active_challenge
    }

    fn evaluate_challenge(&self) -> Option<ChallengeEvaluation> {
        self.eval_active_challenge()
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        ui::show_win_ratio_paradox_lab(self, ui);
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

impl scientific_metadata::theory::TheoryDescribable for WinRatioParadoxLab {
    fn theory_description(&self) -> String {
        format!(
            "Prioritized composite-endpoint Win Ratio $\\text{{WR}} = N_W / N_L = {:.4}$ ($p = {:.2e}$) vs pooled time-to-first-event Hazard Ratio $\\text{{HR}} = {:.4}$ and fatal survival $t = {:.4}$: early non-fatal hospitalizations invert pooled time-to-first-event hazard while hierarchical death-first adjudication recovers true clinical benefit.",
            self.metrics.win_ratio,
            self.metrics.win_ratio_p_value,
            self.metrics.pooled_hazard_ratio,
            self.metrics.t_statistic
        )
    }

    fn phonetic_description(&self) -> String {
        "Prioritized hierarchical Win Ratio versus time-to-first-event composite survival reversal"
            .into()
    }

    fn theory_citation(&self) -> String {
        "Pocock, S. J., Ariti, C. A., Collier, T. J., & Wang, D. (2012). The win ratio: a new approach to the analysis of composite endpoints in clinical trials based on clinical priorities. European Heart Journal, 33(2), 176-182.".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Prioritized Win Ratio".into(),
            "All-pairs hierarchical comparison adjudicating fatal events before non-fatal hospitalizations".into(),
        );
        map.insert(
            "Time-to-First-Event Trap".into(),
            "Pooled composite hazard ratio treats early minor hospitalizations identically to death".into(),
        );
        map
    }
}

/// Starter preset JSON payload reproducing `HR = 3.8623`, `WR = 1.6667` (`p = 4.20e-7`), and `t = 5.0746`.
pub const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Composite Endpoint Severity Reversal",
    "domain": "clinical_trials",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "sample_size_per_arm": 20,
    "horizon_months": 36.0,
    "control_mortality_rate": 0.6,
    "mortality_reduction": 0.8333333333333334,
    "early_hosp_rate": 0.9,
    "early_hosp_mean_months": 4.487285,
    "matched_pairing": false,
    "seed_counter": 0
  }
}"#;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "WinRatioParadoxLab",
        display_name: "Win Ratio Paradox Lab",
        domain: "clinical_trials",
        tags: &["win ratio", "survival", "hazard ratio", "composite endpoint", "paradox"],
        build: || Box::new(WinRatioParadoxLab::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "win_ratio_composite_reversal",
        title: "Composite Endpoint Severity Reversal",
        description: "WR = N_W / N_L = 1.67 vs Pooled HR = 3.86: Prioritized death-before-hospitalization Win Ratio rescues a life-saving therapy penalized by early non-fatal hospitalizations.",
        category: "Clinical Trials",
        icon: "⚖",
        domain: "clinical_trials",
        tool_name: Some("WinRatioParadoxLab"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

// [cite:win_ratio]

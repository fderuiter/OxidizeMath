//! Dedicated `WinRatioTool` workbench for prioritized composite clinical endpoints,
//! Clenshaw-Curtis `ProbabilityWinRatioContext` Weibull `PR(c)` curves, `calculate_bmi`,
//! and secondary continuous `TTestIndependent` analysis.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod cohort;
pub mod engine;
pub mod export;
pub mod ui;

#[cfg(test)]
mod tests;

use super::clinical_io::ClinicalIoState;
use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use cohort::{
    ComparatorKind, PairingMode, ParsedCompositeCohort, WinRatioPlotMode, DEFAULT_COHORT_CSV,
    DEFAULT_PRC_SAMPLES, MAX_HORIZON_C, MAX_PRC_SAMPLES, MAX_WEIBULL_ALPHA, MAX_WEIBULL_LAMBDA,
    MAX_WEIBULL_THETA, MIN_HORIZON_C, MIN_PRC_SAMPLES, MIN_WEIBULL_ALPHA, MIN_WEIBULL_LAMBDA,
    MIN_WEIBULL_THETA,
};
use eframe::egui;
use engine::WinRatioEvaluation;
use math_explorer::applied::clinical_trials::types::GroupData;
use math_explorer::applied::win_ratio::pair_comparison::{OutcomeComparator, WinRatioAnalysis};

/// Interactive workbench for hierarchical composite Win Ratio and Bivariate Weibull `PR(c)` analysis.
pub struct WinRatioTool {
    /// Raw CSV or TSV cohort text (`Arm, Tier1, Tier2[, Tier3 | WeightKg, HeightM]`).
    pub cohort_text: String,
    /// Pairing mode (`Unmatched` all-pairs vs `Matched` 1:1 pairs).
    pub pairing_mode: PairingMode,
    /// Comparator strategy for Tier 1 (e.g., fatal event survival time).
    pub tier1_kind: ComparatorKind,
    /// Comparator strategy for Tier 2 (e.g., non-fatal hospitalization count).
    pub tier2_kind: ComparatorKind,
    /// Comparator strategy for Tier 3 (e.g., continuous biomarker or BMI).
    pub tier3_kind: ComparatorKind,
    /// Whether 5-column rows derive Tier 3 BMI via `calculate_bmi(weight_kg, height_m)`.
    pub derive_bmi_tier3: bool,
    /// Endpoint tier index (`0..=2`) tested by `TTestIndependent`.
    pub ttest_tier_index: usize,
    /// Two-sided significance level `alpha` for `TTestIndependent`.
    pub ttest_alpha: f64,
    /// Reference Treatment weight (`kg`) for interactive `calculate_bmi`.
    pub treat_weight_kg: f64,
    /// Reference Treatment height (`m`) for interactive `calculate_bmi`.
    pub treat_height_m: f64,
    /// Reference Control weight (`kg`) for interactive `calculate_bmi`.
    pub ctrl_weight_kg: f64,
    /// Reference Control height (`m`) for interactive `calculate_bmi`.
    pub ctrl_height_m: f64,
    /// Bivariate Weibull fatal event scale `lambda1`.
    pub weibull_lambda1: f64,
    /// Bivariate Weibull non-fatal event scale `lambda2`.
    pub weibull_lambda2: f64,
    /// Bivariate Weibull event correlation parameter `alpha in (0, 1]`.
    pub weibull_alpha: f64,
    /// Bivariate Weibull treatment hazard multiplier `theta > 0`.
    pub weibull_theta: f64,
    /// Maximum follow-up censoring horizon `c_max` for `PR(c)`.
    pub horizon_c: f64,
    /// Number of Clenshaw-Curtis evaluation points along `[c_min, c_max]`.
    pub prc_samples: usize,
    /// Selected plot mode (`WeibullPrc` vs `EmpiricalHierarchy`).
    pub plot_mode: WinRatioPlotMode,

    pub(crate) parsed_cohort: ParsedCompositeCohort,
    pub(crate) compiled_analysis: Option<WinRatioAnalysis<f64>>,
    pub(crate) compiled_tier_comparators: Vec<Box<dyn OutcomeComparator<f64>>>,
    pub(crate) ttest_groups: Option<(GroupData, GroupData)>,
    pub(crate) prc_points: Vec<[f64; 2]>,
    pub(crate) win_prob_points: Vec<[f64; 2]>,
    pub(crate) loss_prob_points: Vec<[f64; 2]>,
    pub(crate) surv_treat_points: Vec<[f64; 2]>,
    pub(crate) surv_ctrl_points: Vec<[f64; 2]>,
    pub(crate) eval: WinRatioEvaluation,
    pub(crate) error_message: Option<String>,
    pub(crate) io_state: ClinicalIoState,
}

impl Default for WinRatioTool {
    fn default() -> Self {
        let mut tool = Self {
            cohort_text: DEFAULT_COHORT_CSV.to_string(),
            pairing_mode: PairingMode::Unmatched,
            tier1_kind: ComparatorKind::HigherIsBetter,
            tier2_kind: ComparatorKind::LowerIsBetter,
            tier3_kind: ComparatorKind::Threshold { threshold: 5.0 },
            derive_bmi_tier3: false,
            ttest_tier_index: 2,
            ttest_alpha: 0.05,
            treat_weight_kg: 74.0,
            treat_height_m: 1.75,
            ctrl_weight_kg: 82.0,
            ctrl_height_m: 1.74,
            weibull_lambda1: 0.10,
            weibull_lambda2: 0.20,
            weibull_alpha: 0.80,
            weibull_theta: 0.70,
            horizon_c: 12.0,
            prc_samples: DEFAULT_PRC_SAMPLES,
            plot_mode: WinRatioPlotMode::WeibullPrc,
            parsed_cohort: ParsedCompositeCohort::default(),
            compiled_analysis: None,
            compiled_tier_comparators: Vec::with_capacity(3),
            ttest_groups: None,
            prc_points: Vec::with_capacity(MAX_PRC_SAMPLES),
            win_prob_points: Vec::with_capacity(MAX_PRC_SAMPLES),
            loss_prob_points: Vec::with_capacity(MAX_PRC_SAMPLES),
            surv_treat_points: Vec::with_capacity(MAX_PRC_SAMPLES),
            surv_ctrl_points: Vec::with_capacity(MAX_PRC_SAMPLES),
            eval: WinRatioEvaluation::default(),
            error_message: None,
            io_state: ClinicalIoState::default(),
        };
        tool.recalculate();
        tool
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WinRatioState {
    weibull_theta: f64,
    weibull_lambda1: f64,
    weibull_lambda2: f64,
    weibull_alpha: f64,
    horizon_c: f64,
    prc_samples: usize,
    pairing_mode: PairingMode,
    tier1_kind: ComparatorKind,
    tier2_kind: ComparatorKind,
    tier3_kind: ComparatorKind,
    derive_bmi_tier3: bool,
    ttest_tier_index: usize,
    ttest_alpha: f64,
    treat_weight_kg: f64,
    treat_height_m: f64,
    ctrl_weight_kg: f64,
    ctrl_height_m: f64,
    plot_mode: WinRatioPlotMode,
    cohort_text: String,
}

impl WinRatioTool {
    /// Re-parses the cohort, prepares buffers, and runs the zero-allocation evaluation pass.
    pub fn recalculate(&mut self) {
        self.prepare_workspace();
        self.evaluate_hot_path();
    }

    /// Polls non-blocking file import and report save channels.
    pub fn poll_io(&mut self, ctx: &egui::Context) {
        self.io_state.poll_save("Win Ratio report");
        if let Some(outcome) = self.io_state.poll_open() {
            match outcome {
                Ok((filename, content)) => match self.import_cohort_str(&content) {
                    Ok((n_t, n_c)) => {
                        self.io_state.record_info(format!(
                            "Imported {filename} (Treatment N={n_t}, Control N={n_c})"
                        ));
                    }
                    Err(err) => {
                        self.io_state
                            .record_error(format!("Cohort import rejected ({filename}): {err}"));
                    }
                },
                Err(err) => self.io_state.record_error(err),
            }
            ctx.request_repaint();
        }
    }

    /// Validates and applies an imported CSV or TSV cohort text atomically.
    pub fn import_cohort_str(&mut self, raw: &str) -> Result<(usize, usize), String> {
        let parsed = cohort::parse_composite_cohort(raw, self.derive_bmi_tier3)?;
        if self.pairing_mode == PairingMode::Matched
            && parsed.treatment.len() != parsed.control.len()
        {
            return Err(format!(
                "Matched pairing requires equal arm sizes (Treatment N={}, Control N={})",
                parsed.treatment.len(),
                parsed.control.len()
            ));
        }
        let counts = (parsed.treatment.len(), parsed.control.len());
        self.cohort_text = crate::web_io::strip_utf8_bom(raw).to_string();
        self.recalculate();
        Ok(counts)
    }

    fn to_state(&self) -> WinRatioState {
        WinRatioState {
            weibull_theta: self.weibull_theta,
            weibull_lambda1: self.weibull_lambda1,
            weibull_lambda2: self.weibull_lambda2,
            weibull_alpha: self.weibull_alpha,
            horizon_c: self.horizon_c,
            prc_samples: self.prc_samples,
            pairing_mode: self.pairing_mode,
            tier1_kind: self.tier1_kind,
            tier2_kind: self.tier2_kind,
            tier3_kind: self.tier3_kind,
            derive_bmi_tier3: self.derive_bmi_tier3,
            ttest_tier_index: self.ttest_tier_index,
            ttest_alpha: self.ttest_alpha,
            treat_weight_kg: self.treat_weight_kg,
            treat_height_m: self.treat_height_m,
            ctrl_weight_kg: self.ctrl_weight_kg,
            ctrl_height_m: self.ctrl_height_m,
            plot_mode: self.plot_mode,
            cohort_text: self.cohort_text.clone(),
        }
    }

    fn validate_state(s: &WinRatioState) -> Result<(), String> {
        let floats = [
            s.weibull_theta,
            s.weibull_lambda1,
            s.weibull_lambda2,
            s.weibull_alpha,
            s.horizon_c,
            s.ttest_alpha,
            s.treat_weight_kg,
            s.treat_height_m,
            s.ctrl_weight_kg,
            s.ctrl_height_m,
        ];
        if !floats.iter().all(|v| v.is_finite())
            || !s.tier1_kind.is_valid()
            || !s.tier2_kind.is_valid()
            || !s.tier3_kind.is_valid()
            || !(MIN_WEIBULL_LAMBDA..=MAX_WEIBULL_LAMBDA).contains(&s.weibull_lambda1)
            || !(MIN_WEIBULL_LAMBDA..=MAX_WEIBULL_LAMBDA).contains(&s.weibull_lambda2)
            || !(MIN_WEIBULL_ALPHA..=MAX_WEIBULL_ALPHA).contains(&s.weibull_alpha)
            || !(MIN_WEIBULL_THETA..=MAX_WEIBULL_THETA).contains(&s.weibull_theta)
            || !(MIN_HORIZON_C..=MAX_HORIZON_C).contains(&s.horizon_c)
            || !(MIN_PRC_SAMPLES..=MAX_PRC_SAMPLES).contains(&s.prc_samples)
            || !(0.001..=0.20).contains(&s.ttest_alpha)
            || s.ttest_tier_index > 2
            || !(20.0..=300.0).contains(&s.treat_weight_kg)
            || !(1.0..=2.5).contains(&s.treat_height_m)
            || !(20.0..=300.0).contains(&s.ctrl_weight_kg)
            || !(1.0..=2.5).contains(&s.ctrl_height_m)
        {
            return Err("WinRatioTool parameters out of valid clinical bounds".into());
        }
        Ok(())
    }
}

impl InteractiveTool for WinRatioTool {
    fn name(&self) -> &'static str {
        "Composite Win Ratio"
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
        export::collect_export_datasets(self)
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::DEFAULT
    }

    fn has_finite_parameters(&self) -> bool {
        [
            self.weibull_lambda1,
            self.weibull_lambda2,
            self.weibull_alpha,
            self.weibull_theta,
            self.horizon_c,
            self.ttest_alpha,
            self.treat_weight_kg,
            self.treat_height_m,
            self.ctrl_weight_kg,
            self.ctrl_height_m,
        ]
        .iter()
        .all(|v| v.is_finite())
            && self.tier1_kind.is_valid()
            && self.tier2_kind.is_valid()
            && self.tier3_kind.is_valid()
    }

    fn clamp_parameters(&mut self) -> bool {
        let cf =
            |v: f64, lo: f64, hi: f64, fb: f64| if v.is_finite() { v.clamp(lo, hi) } else { fb };
        self.weibull_lambda1 = cf(
            self.weibull_lambda1,
            MIN_WEIBULL_LAMBDA,
            MAX_WEIBULL_LAMBDA,
            0.10,
        );
        self.weibull_lambda2 = cf(
            self.weibull_lambda2,
            MIN_WEIBULL_LAMBDA,
            MAX_WEIBULL_LAMBDA,
            0.20,
        );
        self.weibull_alpha = cf(
            self.weibull_alpha,
            MIN_WEIBULL_ALPHA,
            MAX_WEIBULL_ALPHA,
            0.80,
        );
        self.weibull_theta = cf(
            self.weibull_theta,
            MIN_WEIBULL_THETA,
            MAX_WEIBULL_THETA,
            0.70,
        );
        self.horizon_c = cf(self.horizon_c, MIN_HORIZON_C, MAX_HORIZON_C, 12.0);
        self.ttest_alpha = cf(self.ttest_alpha, 0.001, 0.20, 0.05);
        self.treat_weight_kg = cf(self.treat_weight_kg, 20.0, 300.0, 74.0);
        self.treat_height_m = cf(self.treat_height_m, 1.0, 2.5, 1.75);
        self.ctrl_weight_kg = cf(self.ctrl_weight_kg, 20.0, 300.0, 82.0);
        self.ctrl_height_m = cf(self.ctrl_height_m, 1.0, 2.5, 1.74);
        self.prc_samples = self.prc_samples.clamp(MIN_PRC_SAMPLES, MAX_PRC_SAMPLES);
        self.recalculate();
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.to_state()).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let s: WinRatioState =
            serde_json::from_str(state).map_err(|e| format!("Invalid WinRatioTool state: {e}"))?;
        Self::validate_state(&s)?;
        let mut candidate = Self {
            cohort_text: s.cohort_text,
            pairing_mode: s.pairing_mode,
            tier1_kind: s.tier1_kind,
            tier2_kind: s.tier2_kind,
            tier3_kind: s.tier3_kind,
            derive_bmi_tier3: s.derive_bmi_tier3,
            ttest_tier_index: s.ttest_tier_index,
            ttest_alpha: s.ttest_alpha,
            treat_weight_kg: s.treat_weight_kg,
            treat_height_m: s.treat_height_m,
            ctrl_weight_kg: s.ctrl_weight_kg,
            ctrl_height_m: s.ctrl_height_m,
            weibull_lambda1: s.weibull_lambda1,
            weibull_lambda2: s.weibull_lambda2,
            weibull_alpha: s.weibull_alpha,
            weibull_theta: s.weibull_theta,
            horizon_c: s.horizon_c,
            prc_samples: s.prc_samples,
            plot_mode: s.plot_mode,
            ..Self::default()
        };
        candidate.recalculate();
        if let Some(err) = candidate.error_message {
            return Err(err);
        }
        *self = candidate;
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        ui::show_win_ratio_inside(self, ui);
    }
}

impl scientific_metadata::theory::TheoryDescribable for WinRatioTool {
    fn theory_description(&self) -> String {
        format!(
            "Prioritized composite endpoint Win Ratio $R = N_W / N_L$ across {} pairs and Clenshaw-Curtis probability win ratio $PR(c) = W(c)/L(c)$ under bivariate Weibull survival $S_j(t,x) = \\exp[-(\\theta_j \\lambda_1 t + \\theta_j \\lambda_2 x)^\\alpha]$ ($PR_W = \\theta^{{-\\alpha}} = {:.3}$).",
            self.eval.total_pairs, self.eval.weibull_pr_w
        )
    }

    fn phonetic_description(&self) -> String {
        "Prioritized composite endpoint Win Ratio and bivariate Weibull probability win ratio theory"
            .into()
    }

    fn theory_citation(&self) -> String {
        "The win ratio: a new approach to the analysis of composite endpoints in clinical trials based on clinical priorities (Pocock et al., 2012)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Sample Win Ratio".into(),
            "Hierarchical pairwise evaluation across fatal, hospitalization, and continuous tiers"
                .into(),
        );
        map.insert(
            "Probability Win Ratio PR(c)".into(),
            "Clenshaw-Curtis numerical integration of bivariate Weibull survival models".into(),
        );
        map
    }
}

pub use cohort::{COMPOSITE_STARTER_PRESET_JSON, WEIBULL_PRC_STARTER_PRESET_JSON};

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "WinRatioTool",
        display_name: "Composite Win Ratio",
        domain: "clinical_trials",
        tags: &[
            "win ratio",
            "composite endpoint",
            "weibull",
            "pocock",
            "ttest",
            "bmi",
            "rct",
        ],
        build: || Box::new(WinRatioTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "win_ratio_composite_starter",
        title: "Composite Endpoint Win Ratio",
        description: "R = N_w / N_l: Prioritized fatal, hospitalization, and threshold biomarker pairwise evaluation with 95% CIs.",
        category: "Composite Endpoints",
        icon: "🏆",
        domain: "clinical_trials",
        tool_name: Some("WinRatioTool"),
        preset_generator: || COMPOSITE_STARTER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "win_ratio_weibull_prc_starter",
        title: "Bivariate Weibull PR(c) Integral",
        description: "PR(c) = W(c) / L(c): Clenshaw-Curtis probability win ratio over bivariate Weibull survival horizons.",
        category: "Composite Endpoints",
        icon: "📈",
        domain: "clinical_trials",
        tool_name: Some("WinRatioTool"),
        preset_generator: || WEIBULL_PRC_STARTER_PRESET_JSON,
    }
}

// [cite:win_ratio]
// [cite:clinical_trials_statistics]

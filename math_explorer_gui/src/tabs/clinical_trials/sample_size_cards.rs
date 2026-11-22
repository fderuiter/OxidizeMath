//! Interactive `TTestIndependent`, `ChiSquare2x2`, and `calculate_risk_metrics`
//! hypothesis and 2x2 contingency analysis cards for [`super::sample_size::SampleSizeCalculatorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use math_explorer::applied::clinical_trials::analysis::{calculate_risk_metrics, RiskMetrics};
use math_explorer::applied::clinical_trials::hypothesis_testing::{
    ChiSquare2x2, HypothesisTest, TTestIndependent, TestResult,
};
use math_explorer::applied::clinical_trials::types::{ContingencyTable, GroupData};

const DEFAULT_TREATMENT_SAMPLES: &str = "12.4, 14.1, 13.8, 15.2, 14.6, 13.9";
const DEFAULT_CONTROL_SAMPLES: &str = "10.2, 11.5, 10.8, 9.9, 11.1, 10.6";

/// Optional persisted state for the continuous t-test and 2x2 contingency cards.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HypothesisCardsState {
    /// Comma-separated continuous values for the Treatment sample.
    pub treatment_samples: String,
    /// Comma-separated continuous values for the Control sample.
    pub control_samples: String,
    /// 2x2 cell `a`: Treatment events.
    pub treatment_event: u32,
    /// 2x2 cell `b`: Treatment non-events.
    pub treatment_no_event: u32,
    /// 2x2 cell `c`: Control events.
    pub control_event: u32,
    /// 2x2 cell `d`: Control non-events.
    pub control_no_event: u32,
}

impl Default for HypothesisCardsState {
    fn default() -> Self {
        Self {
            treatment_samples: DEFAULT_TREATMENT_SAMPLES.to_string(),
            control_samples: DEFAULT_CONTROL_SAMPLES.to_string(),
            treatment_event: 18,
            treatment_no_event: 82,
            control_event: 34,
            control_no_event: 66,
        }
    }
}

impl HypothesisCardsState {
    /// Returns `true` when all card inputs match defaults so legacy presets omit `cards`.
    #[must_use]
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Validates that sample text lengths and contingency counts fall within safe bounds.
    pub fn validate(&self) -> Result<(), String> {
        if self.treatment_samples.len() > 8_192 || self.control_samples.len() > 8_192 {
            return Err("Sample vector text exceeds 8 KiB bound".into());
        }
        let _ = parse_group_samples(&self.treatment_samples)?;
        let _ = parse_group_samples(&self.control_samples)?;
        let _ = ContingencyTable::new(
            self.treatment_event,
            self.treatment_no_event,
            self.control_event,
            self.control_no_event,
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }
}

/// Cached evaluation outputs for the hypothesis testing and risk metric cards.
pub struct HypothesisCardsEval {
    /// Independent two-sample t-test output (`TTestIndependent`).
    pub t_test: Result<TestResult, String>,
    /// Pearson 2x2 chi-square output (`ChiSquare2x2`).
    pub chi_square: Result<TestResult, String>,
    /// Relative Risk and Odds Ratio output (`calculate_risk_metrics`).
    pub risk_metrics: Result<RiskMetrics, String>,
}

fn parse_group_samples(raw: &str) -> Result<GroupData, String> {
    let mut values = Vec::new();
    for token in raw.split([',', '\n', '\t']) {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            continue;
        }
        if values.len() >= math_commons::registry::MAX_ITERATIONS {
            return Err(format!(
                "Sample vector exceeds {} values",
                math_commons::registry::MAX_ITERATIONS
            ));
        }
        let v = trimmed
            .parse::<f64>()
            .map_err(|_| format!("Invalid sample number '{trimmed}'"))?;
        values.push(v);
    }
    GroupData::new(values).map_err(|e| e.to_string())
}

/// Evaluates `TTestIndependent`, `ChiSquare2x2`, and `calculate_risk_metrics` at level `alpha`.
#[must_use]
pub fn evaluate_hypothesis_cards(state: &HypothesisCardsState, alpha: f64) -> HypothesisCardsEval {
    let t_test = parse_group_samples(&state.treatment_samples)
        .and_then(|g1| parse_group_samples(&state.control_samples).map(|g2| (g1, g2)))
        .and_then(|(g1, g2)| {
            TTestIndependent::new(alpha)
                .test((&g1, &g2))
                .map_err(|e| e.to_string())
        });

    let table_res = ContingencyTable::new(
        state.treatment_event,
        state.treatment_no_event,
        state.control_event,
        state.control_no_event,
    )
    .map_err(|e| e.to_string());

    let chi_square = table_res.clone().and_then(|tbl| {
        ChiSquare2x2::new(alpha)
            .test(&tbl)
            .map_err(|e| e.to_string())
    });

    let risk_metrics =
        table_res.and_then(|tbl| calculate_risk_metrics(&tbl, alpha).map_err(|e| e.to_string()));

    HypothesisCardsEval {
        t_test,
        chi_square,
        risk_metrics,
    }
}

/// Renders the interactive `TTestIndependent` card, returning `true` when inputs change.
pub fn render_t_test_card(
    ui: &mut egui::Ui,
    cards: &mut HypothesisCardsState,
    eval: &HypothesisCardsEval,
) -> bool {
    let mut changed = false;
    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    ui.group(|ui| {
        ui.strong("Independent Two-Sample t-Test (TTestIndependent)");
        ui.label("Treatment Values:");
        changed |= ui
            .add(egui::TextEdit::singleline(&mut cards.treatment_samples).desired_width(f32::INFINITY))
            .changed();
        ui.label("Control Values:");
        changed |= ui
            .add(egui::TextEdit::singleline(&mut cards.control_samples).desired_width(f32::INFINITY))
            .changed();
        match &eval.t_test {
            Ok(res) => {
                render_copyable_metric(ui, "t-Statistic", &format!("{:.4}", res.statistic));
                render_copyable_metric(ui, "Two-Tailed p-value", &format!("{:.4}", res.p_value));
                if let Some((lo, hi)) = res.confidence_interval {
                    render_copyable_metric(ui, "Mean Diff CI", &format!("[{lo:.3}, {hi:.3}]"));
                }
                let sig_text = if res.is_significant {
                    "Significant (p < α)"
                } else {
                    "Not Significant (p ≥ α)"
                };
                let sig_color = if res.is_significant {
                    palette.success
                } else {
                    palette.muted_text
                };
                ui.colored_label(sig_color, sig_text);
            }
            Err(err) => {
                ui.colored_label(palette.error, err);
            }
        }
    });
    changed
}

/// Renders the 2x2 contingency `ChiSquare2x2` and `calculate_risk_metrics` card.
pub fn render_contingency_and_risk_card(
    ui: &mut egui::Ui,
    cards: &mut HypothesisCardsState,
    eval: &HypothesisCardsEval,
) -> bool {
    let mut changed = false;
    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    ui.group(|ui| {
        ui.strong("2×2 Contingency: ChiSquare2x2 & Risk Metrics (RR / OR)");
        egui::Grid::new("contingency_2x2_grid")
            .num_columns(3)
            .show(ui, |ui| {
                ui.label("Arm");
                ui.label("Event");
                ui.label("No Event");
                ui.end_row();
                ui.label("Treatment");
                changed |= ui
                    .add(egui::DragValue::new(&mut cards.treatment_event).range(0..=100_000))
                    .changed();
                changed |= ui
                    .add(egui::DragValue::new(&mut cards.treatment_no_event).range(0..=100_000))
                    .changed();
                ui.end_row();
                ui.label("Control");
                changed |= ui
                    .add(egui::DragValue::new(&mut cards.control_event).range(0..=100_000))
                    .changed();
                changed |= ui
                    .add(egui::DragValue::new(&mut cards.control_no_event).range(0..=100_000))
                    .changed();
                ui.end_row();
            });
        match &eval.chi_square {
            Ok(chi) => {
                render_copyable_metric(ui, "Chi-Square (χ²)", &format!("{:.4}", chi.statistic));
                render_copyable_metric(ui, "Chi-Square p-value", &format!("{:.4}", chi.p_value));
            }
            Err(err) => {
                ui.colored_label(palette.warning, err);
            }
        }
        match &eval.risk_metrics {
            Ok(rm) => {
                render_copyable_metric(
                    ui,
                    "Relative Risk (RR)",
                    &format!("{:.3} [{:.3}, {:.3}]", rm.relative_risk, rm.rr_ci.0, rm.rr_ci.1),
                );
                render_copyable_metric(
                    ui,
                    "Odds Ratio (OR)",
                    &format!("{:.3} [{:.3}, {:.3}]", rm.odds_ratio, rm.or_ci.0, rm.or_ci.1),
                );
            }
            Err(err) => {
                ui.colored_label(palette.warning, err);
            }
        }
    });
    changed
}

// [cite:clinical_trials_statistics]

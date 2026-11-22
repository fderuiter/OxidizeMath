//! Interactive Win Ratio Workbench for hierarchical composite endpoints and
//! bivariate Weibull survival simulations.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::AccessibleHoverText;
use crate::framework::{bounded_drag_value, ExportCapabilities, InteractiveTool, ToolStateBounds};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;

use super::win_ratio_io::{
    evaluate_cohort, evaluate_weibull_simulator, format_waterfall_csv, format_waterfall_json,
    parse_win_ratio_cohort, CohortEvaluation, FileDialogChannels, PairingMode, TierComparatorKind,
    TierSpec, WeibullEvaluation, WeibullSimConfig, WinRatioView, MAX_WIN_RATIO_TIERS,
};
use super::win_ratio_weibull::{
    render_waterfall_bar_plot, render_weibull_controls_grid, render_weibull_curves_plot,
};

const DEFAULT_COHORT_TEXT: &str = "arm,survival_days,hosp_count,kccq_change\n\
Treatment,980,0,14.5\n\
Treatment,920,1,11.0\n\
Treatment,860,1,8.5\n\
Treatment,740,2,4.0\n\
Treatment,610,1,9.0\n\
Treatment,540,2,2.5\n\
Control,890,2,3.0\n\
Control,710,3,-2.0\n\
Control,620,2,1.5\n\
Control,480,3,-5.0\n\
Control,390,4,-8.0\n\
Control,310,3,-6.5";

fn default_tier_specs() -> Vec<TierSpec> {
    vec![
        TierSpec::new("Survival Days", TierComparatorKind::HigherIsBetter, 0.0),
        TierSpec::new("Hospitalizations", TierComparatorKind::LowerIsBetter, 0.0),
        TierSpec::new("QoL Score Delta", TierComparatorKind::Threshold, 5.0),
        TierSpec::new("Biomarker Score", TierComparatorKind::HigherIsBetter, 0.0),
    ]
}

/// Interactive workbench for prioritized Win Ratio cohort analysis and Weibull power curves.
pub struct WinRatioWorkbenchTool {
    cohort_text: String,
    pairing_mode: PairingMode,
    active_view: WinRatioView,
    tier_specs: Vec<TierSpec>,
    weibull_cfg: WeibullSimConfig,
    cohort_eval: Option<CohortEvaluation>,
    weibull_eval: Option<WeibullEvaluation>,
    error_message: Option<String>,
    io_channels: FileDialogChannels,
}

impl Default for WinRatioWorkbenchTool {
    fn default() -> Self {
        let mut tool = Self {
            cohort_text: DEFAULT_COHORT_TEXT.to_string(),
            pairing_mode: PairingMode::Unmatched,
            active_view: WinRatioView::CohortWaterfall,
            tier_specs: default_tier_specs(),
            weibull_cfg: WeibullSimConfig::default(),
            cohort_eval: None,
            weibull_eval: None,
            error_message: None,
            io_channels: FileDialogChannels::default(),
        };
        tool.recalculate();
        tool
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WinRatioState {
    cohort_text: String,
    pairing_mode: PairingMode,
    active_view: WinRatioView,
    tier_specs: Vec<TierSpec>,
    weibull_cfg: WeibullSimConfig,
}

impl WinRatioWorkbenchTool {
    pub(crate) fn recalculate(&mut self) {
        self.error_message = None;
        match parse_win_ratio_cohort(&self.cohort_text)
            .and_then(|cohort| evaluate_cohort(&cohort, &self.tier_specs, self.pairing_mode))
        {
            Ok(eval) => self.cohort_eval = Some(eval),
            Err(err) => {
                self.cohort_eval = None;
                self.error_message = Some(err);
            }
        }
        match evaluate_weibull_simulator(&self.weibull_cfg) {
            Ok(sim) => self.weibull_eval = Some(sim),
            Err(err) => {
                self.weibull_eval = None;
                if self.error_message.is_none() {
                    self.error_message = Some(err);
                }
            }
        }
    }

    fn render_side_controls(&mut self, ui: &mut egui::Ui) {
        if let Some((_filename, content)) = self.io_channels.poll() {
            self.cohort_text = content;
            self.recalculate();
        }
        egui::ScrollArea::vertical().show(ui, |ui| {
            self.render_view_and_io_controls(ui);
            ui.separator();
            if self.active_view == WinRatioView::CohortWaterfall {
                self.render_cohort_controls(ui);
            } else if render_weibull_controls_grid(ui, &mut self.weibull_cfg) {
                self.recalculate();
            }
            self.render_status_and_metrics(ui);
        });
    }

    fn render_view_and_io_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Win Ratio Workbench");
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(
                &mut self.active_view,
                WinRatioView::CohortWaterfall,
                "Cohort Waterfall",
            );
            ui.radio_value(
                &mut self.active_view,
                WinRatioView::WeibullSimulator,
                "Weibull Simulator",
            );
        });
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("📂 Load CSV/TSV")
                .accessible_hover_text("Load local two-arm CSV or TSV cohort without network upload")
                .clicked()
            {
                self.io_channels.trigger_open_cohort();
            }
            let has_eval = self.cohort_eval.is_some();
            if ui
                .add_enabled(has_eval, egui::Button::new("💾 Save CSV"))
                .accessible_hover_text("Download tier waterfall summary as CSV")
                .clicked()
            {
                if let Some(eval) = &self.cohort_eval {
                    let csv = format_waterfall_csv(eval, &self.tier_specs);
                    self.io_channels
                        .trigger_save_csv("win_ratio_waterfall.csv", csv);
                }
            }
            if ui
                .add_enabled(has_eval, egui::Button::new("💾 Save JSON"))
                .accessible_hover_text("Download tier waterfall summary and CI as JSON")
                .clicked()
            {
                if let Some(eval) = &self.cohort_eval {
                    let json = format_waterfall_json(eval, &self.tier_specs);
                    self.io_channels
                        .trigger_save_json("win_ratio_waterfall.json", json);
                }
            }
        });
    }

    fn render_cohort_controls(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Pairing:");
            changed |= ui
                .radio_value(&mut self.pairing_mode, PairingMode::Unmatched, "Unmatched")
                .changed();
            changed |= ui
                .radio_value(&mut self.pairing_mode, PairingMode::Matched, "Matched 1:1")
                .changed();
        });
        let active_n = self
            .cohort_eval
            .as_ref()
            .map_or(3, |e| e.tiers.len())
            .min(self.tier_specs.len());
        ui.label(format!("Endpoint Hierarchy ({active_n} active tiers):"));
        for (idx, spec) in self.tier_specs.iter_mut().take(active_n).enumerate() {
            changed |= Self::render_single_tier_editor(ui, idx, spec);
        }
        ui.label("Cohort Data (arm, tier1, tier2, ...):");
        let edit = egui::TextEdit::multiline(&mut self.cohort_text)
            .desired_width(f32::INFINITY)
            .desired_rows(8);
        if ui.add(edit).changed() || changed {
            self.recalculate();
        }
    }

    fn render_single_tier_editor(ui: &mut egui::Ui, idx: usize, spec: &mut TierSpec) -> bool {
        let mut changed = false;
        ui.push_id(idx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(format!("T{}:", idx + 1));
                changed |= ui
                    .add(egui::TextEdit::singleline(&mut spec.label).desired_width(110.0))
                    .changed();
                egui::ComboBox::from_id_salt("cmp_kind")
                    .selected_text(spec.kind.label())
                    .show_ui(ui, |ui| {
                        for kind in [
                            TierComparatorKind::HigherIsBetter,
                            TierComparatorKind::LowerIsBetter,
                            TierComparatorKind::Threshold,
                        ] {
                            changed |= ui
                                .selectable_value(&mut spec.kind, kind, kind.label())
                                .changed();
                        }
                    });
                if spec.kind == TierComparatorKind::Threshold {
                    changed |= ui
                        .add(
                            bounded_drag_value(&mut spec.threshold, 0.0..=10_000.0, 0.5)
                                .prefix("τ="),
                        )
                        .changed();
                }
            });
        });
        changed
    }

    fn render_status_and_metrics(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        if let Some((is_err, msg)) = &self.io_channels.status {
            ui.colored_label(if *is_err { palette.error } else { palette.info }, msg);
        }
        if let Some(err) = &self.error_message {
            ui.colored_label(palette.error, err);
        }
        ui.separator();
        if let Some(eval) = &self.cohort_eval {
            let wlc = eval.win_loss_counts;
            render_copyable_metric(
                ui,
                "Subjects (T / C)",
                &format!("{} / {}", eval.n_treatment, eval.n_control),
            );
            render_copyable_metric(
                ui,
                "Wins / Losses / Ties",
                &format!("{} / {} / {}", eval.total_wins, eval.total_losses, eval.total_ties),
            );
            render_copyable_metric(
                ui,
                "Tier 1/2+ (Nb,Nd / Na,Nc)",
                &format!("{},{} / {},{}", wlc.n_b, wlc.n_d, wlc.n_a, wlc.n_c),
            );
            render_copyable_metric(ui, "Sample WR (R)", &format!("{:.3}", eval.sample_wr));
            render_copyable_metric(ui, "Win Proportion", &format!("{:.3}", eval.win_proportion));
            if let Some(stats) = eval.stats {
                render_copyable_metric(
                    ui,
                    "95% CI",
                    &format!("[{:.3}, {:.3}]", stats.ci_low, stats.ci_high),
                );
                render_copyable_metric(ui, "Two-Sided p-value", &format!("{:.4}", stats.p_value));
            }
        }
        if let Some(sim) = &self.weibull_eval {
            render_copyable_metric(ui, "Uncensored PR_W", &format!("{:.3}", sim.uncensored_wr));
            render_copyable_metric(
                ui,
                "W(c) / L(c)",
                &format!("{:.3} / {:.3}", sim.win_prob, sim.loss_prob),
            );
            render_copyable_metric(ui, "Horizon PR(c)", &format!("{:.3}", sim.prob_wr));
        }
    }

    fn render_central_plot(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        if self.active_view == WinRatioView::CohortWaterfall {
            render_waterfall_bar_plot(ui, self.cohort_eval.as_ref(), &self.tier_specs, palette);
        } else {
            render_weibull_curves_plot(ui, self.weibull_eval.as_ref(), palette);
        }
    }
}

impl InteractiveTool for WinRatioWorkbenchTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Win Ratio Workbench"
    }
    fn domain(&self) -> &'static str {
        "clinical_trials"
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        if self.active_view == WinRatioView::WeibullSimulator {
            if let Some(sim) = &self.weibull_eval {
                return vec![
                    ("Control Survival S0(t)".into(), sim.s0_curve.clone()),
                    ("Treatment Survival S1(t)".into(), sim.s1_curve.clone()),
                    ("Probability Win Ratio PR(t)".into(), sim.pr_curve.clone()),
                ];
            }
        }
        let Some(eval) = &self.cohort_eval else {
            return Vec::new();
        };
        let map_pts = |f: fn(&math_explorer::applied::win_ratio::pair_comparison::TierOutcomeCounts) -> usize| {
            eval.tiers
                .iter()
                .enumerate()
                .map(|(i, c)| [(i + 1) as f64, f(c) as f64])
                .collect()
        };
        vec![
            ("Treatment Wins".into(), map_pts(|c| c.wins)),
            ("Treatment Losses".into(), map_pts(|c| c.losses)),
            ("Unresolved Ties".into(), map_pts(|c| c.ties)),
        ]
    }

    fn has_finite_parameters(&self) -> bool {
        self.weibull_cfg.is_valid() && self.tier_specs.iter().all(TierSpec::is_valid)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&WinRatioState {
            cohort_text: self.cohort_text.clone(),
            pairing_mode: self.pairing_mode,
            active_view: self.active_view,
            tier_specs: self.tier_specs.clone(),
            weibull_cfg: self.weibull_cfg,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: WinRatioState =
            serde_json::from_str(state).map_err(|e| format!("Invalid win ratio state: {e}"))?;
        if parsed.tier_specs.is_empty()
            || parsed.tier_specs.len() > MAX_WIN_RATIO_TIERS
            || !parsed.tier_specs.iter().all(TierSpec::is_valid)
            || !parsed.weibull_cfg.is_valid()
        {
            return Err("Win ratio configuration out of valid bounds".into());
        }
        let cohort = parse_win_ratio_cohort(&parsed.cohort_text)?;
        let cohort_eval = evaluate_cohort(&cohort, &parsed.tier_specs, parsed.pairing_mode)?;
        let weibull_eval = evaluate_weibull_simulator(&parsed.weibull_cfg)?;

        self.cohort_text = parsed.cohort_text;
        self.pairing_mode = parsed.pairing_mode;
        self.active_view = parsed.active_view;
        self.tier_specs = parsed.tier_specs;
        self.weibull_cfg = parsed.weibull_cfg;
        self.cohort_eval = Some(cohort_eval);
        self.weibull_eval = Some(weibull_eval);
        self.error_message = None;
        Ok(())
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::DEFAULT
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "win_ratio_side_panel");
        crate::framework::PaneSidePanel::left(panel_id)
            .min_width(310.0)
            .show_inside(ui, |ui| self.render_side_controls(ui));
        egui::CentralPanel::default().show_inside(ui, |ui| self.render_central_plot(ui));
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "WinRatioWorkbenchTool",
        display_name: "Win Ratio Workbench",
        domain: "clinical_trials",
        tags: &["win_ratio", "composite_endpoints", "survival", "biostatistics"],
        build: || Box::new(WinRatioWorkbenchTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for WinRatioWorkbenchTool {
    fn theory_description(&self) -> String {
        format!(
            "Prioritized composite-endpoint Win Ratio $R_W = N_W / N_L$ across {} active tiers with asymptotic 95% confidence intervals and bivariate Weibull joint survival $S(t, x) = \\exp(-(\\lambda_1 t + \\lambda_2 x)^\\alpha)$.",
            self.cohort_eval.as_ref().map_or(3, |e| e.tiers.len())
        )
    }
    fn phonetic_description(&self) -> String {
        "Hierarchical win ratio composite endpoint and bivariate Weibull survival theory".into()
    }
    fn theory_citation(&self) -> String {
        "The Win Ratio: A New Approach to the Analysis of Composite Endpoints in Clinical Trials Based on Clinical Priorities (Pocock et al., 2012)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "CohortWaterfall".into(),
            "All-pairs or 1:1 matched hierarchical pairwise win-loss-tie waterfall evaluation".into(),
        );
        map.insert(
            "WeibullSimulator".into(),
            "Analytical bivariate Weibull survival model and Clenshaw-Curtis probability win-ratio integration".into(),
        );
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Composite Endpoint Win Ratio",
    "domain": "clinical_trials",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "cohort_text": "arm,survival_days,hosp_count,kccq_change\nTreatment,980,0,14.5\nTreatment,910,1,10.5\nTreatment,845,1,8.0\nTreatment,700,2,4.5\nControl,860,2,2.0\nControl,640,3,-3.0\nControl,510,2,0.5\nControl,380,4,-7.5",
    "pairing_mode": "Unmatched",
    "active_view": "CohortWaterfall",
    "tier_specs": [
      {"label": "Survival Days", "kind": "HigherIsBetter", "threshold": 0.0},
      {"label": "Hospitalizations", "kind": "LowerIsBetter", "threshold": 0.0},
      {"label": "QoL Score Delta", "kind": "Threshold", "threshold": 5.0}
    ],
    "weibull_cfg": {"lambda1": 0.08, "lambda2": 0.18, "alpha": 1.25, "theta": 0.65, "horizon_c": 8.0}
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "win_ratio_workbench_starter",
        title: "Composite Endpoint Win Ratio",
        description: "R_W = N_W / N_L: Prioritized death, hospitalization, and quality-of-life waterfall with 95% CI.",
        category: "Biostatistics",
        icon: "🏆",
        domain: "clinical_trials",
        tool_name: Some("WinRatioWorkbenchTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

// [cite:win_ratio]

#[cfg(test)]
#[path = "win_ratio_tests.rs"]
mod tests;

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    evaluate_kelly_sizing, plot_line, plot_vline, render_sports_central_panel,
    render_sports_left_panel, render_sports_plot, slider_f64,
};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::framework::{ExportCapabilities, InteractiveTool, ToolMetadata};
use eframe::egui;
use math_explorer::sports_quant::{
    Count, PoissonRate, UnitInterval, ZeroInflation, ZipDistribution, ZipParams,
};
use serde::{Deserialize, Serialize};

/// Bounded URL/preset state for `ZipCountPropsTool`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZipPropsState {
    pub zero_inflation_pi: f64,
    pub active_rate_lambda: f64,
    pub prop_line_half: usize,
    pub offered_over_odds: f64,
    pub bankroll: f64,
}

impl Default for ZipPropsState {
    fn default() -> Self {
        Self {
            zero_inflation_pi: 0.28,
            active_rate_lambda: 2.60,
            prop_line_half: 2,
            offered_over_odds: 2.15,
            bankroll: 1000.0,
        }
    }
}

impl ZipPropsState {
    pub(super) fn validate(&self) -> Result<(), String> {
        let vals = [
            self.zero_inflation_pi,
            self.active_rate_lambda,
            self.offered_over_odds,
            self.bankroll,
        ];
        if !vals.iter().all(|v| v.is_finite()) {
            return Err("ZIP prop parameters must be finite".into());
        }
        let ok = (0.0..=0.90).contains(&self.zero_inflation_pi)
            && (0.20..=10.0).contains(&self.active_rate_lambda)
            && self.prop_line_half <= 10
            && (1.05..=25.0).contains(&self.offered_over_odds)
            && (10.0..=1_000_000.0).contains(&self.bankroll);
        if !ok {
            return Err("ZIP prop parameters out of bounds".into());
        }
        Ok(())
    }
}

/// Interactive Zero-Inflated Poisson (ZIP) player prop count distribution and mean-median analyzer.
pub struct ZipCountPropsTool {
    pub(super) state: ZipPropsState,
    revision: u64,
    zip_over: UnitInterval,
    poisson_over: f64,
    zip_mean: f64,
    zip_median: usize,
    dispersion_ratio: f64,
    curve_zip_pmf: Vec<[f64; 2]>,
    curve_pois_pmf: Vec<[f64; 2]>,
}

impl Default for ZipCountPropsTool {
    fn default() -> Self {
        let mut tool = Self {
            state: ZipPropsState::default(),
            revision: 1,
            zip_over: UnitInterval::new_clamped(0.35),
            poisson_over: 0.35,
            zip_mean: 1.87,
            zip_median: 2,
            dispersion_ratio: 1.7,
            curve_zip_pmf: Vec::new(),
            curve_pois_pmf: Vec::new(),
        };
        tool.recompute();
        tool
    }
}

impl ZipCountPropsTool {
    pub(super) fn recompute(&mut self) {
        let pi: ZeroInflation = UnitInterval::new_clamped(self.state.zero_inflation_pi).into();
        let lam = PoissonRate::new(self.state.active_rate_lambda)
            .unwrap_or_else(|_| PoissonRate::new(2.0).unwrap());
        let zip = ZipDistribution::new(ZipParams::new(pi, lam));
        self.zip_mean = (1.0 - pi.value()) * lam.value();
        self.dispersion_ratio = 1.0 + pi.value() * lam.value();
        self.zip_over = zip.over_probability_unit(Count::new(self.state.prop_line_half as u32));

        let equi_rate = PoissonRate::new(self.zip_mean.max(0.05)).unwrap();
        let equi_pois = ZipDistribution::new(ZipParams::new(
            ZeroInflation::from(UnitInterval::new_clamped(0.0)),
            equi_rate,
        ));
        self.poisson_over = equi_pois
            .over_probability_unit(Count::new(self.state.prop_line_half as u32))
            .value();

        self.curve_zip_pmf.clear();
        self.curve_pois_pmf.clear();
        let mut cum = 0.0;
        let mut med = 0usize;
        let mut found_med = false;
        for k in 0..=14u32 {
            let p_zip = zip.pmf_unit(Count::new(k)).value();
            let p_eq = equi_pois.pmf_unit(Count::new(k)).value();
            cum += p_zip;
            if !found_med && cum >= 0.5 {
                med = k as usize;
                found_med = true;
            }
            self.curve_zip_pmf.push([k as f64, p_zip]);
            self.curve_pois_pmf.push([k as f64, p_eq]);
        }
        self.zip_median = med;
        self.revision = self.revision.wrapping_add(1);
    }

    fn render_controls(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        changed |= slider_f64(
            ui,
            &mut self.state.zero_inflation_pi,
            0.0..=0.85,
            "Zero-Inflation π",
        )
        .accessible_hover_text("Structural zero probability (bench/injury/low-usage game script)")
        .changed();
        changed |= slider_f64(
            ui,
            &mut self.state.active_rate_lambda,
            0.30..=8.0,
            "Active Rate λ",
        )
        .accessible_hover_text("Conditional Poisson scoring/stat rate when player is active")
        .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.prop_line_half, 0..=8).text("Prop Line k (+0.5)"),
            )
            .changed();
        ui.separator();
        changed |= slider_f64(
            ui,
            &mut self.state.offered_over_odds,
            1.10..=12.0,
            "Offered Over Odds",
        )
        .changed();
        changed |= ui
            .add(
                egui::DragValue::new(&mut self.state.bankroll)
                    .speed(25.0)
                    .range(10.0..=1_000_000.0)
                    .prefix("Bankroll: $"),
            )
            .changed();
        if changed {
            self.recompute();
        }
    }

    fn render_summary(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let line_val = (self.state.prop_line_half as f64) + 0.5;
        let (ev, full_pct, half_stake) = evaluate_kelly_sizing(
            self.zip_over,
            self.state.offered_over_odds,
            self.state.bankroll,
        );
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(
                palette.series[0],
                format!(
                    "Mean: {:.2} vs Median: {} | Dispersion Var/Mean: {:.2}x",
                    self.zip_mean, self.zip_median, self.dispersion_ratio
                ),
            );
            ui.separator();
            ui.colored_label(
                palette.accent,
                format!(
                    "Over {:.1}: ZIP {:.1}% vs Equi-Mean Poisson {:.1}%",
                    line_val,
                    self.zip_over.value() * 100.0,
                    self.poisson_over * 100.0
                ),
            );
            ui.separator();
            ui.colored_label(
                if ev > 0.0 {
                    palette.success
                } else {
                    palette.muted_text
                },
                format!(
                    "Over EV: ${:+.3} | Full Kelly: {:.1}% (${:.2} Half)",
                    ev, full_pct, half_stake
                ),
            );
        });
    }

    fn render_plot(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let line_val = (self.state.prop_line_half as f64) + 0.5;
        render_sports_plot(
            ui,
            "zip_count_props_plot",
            "Discrete Player Stat Count k",
            "Probability Mass P(Y = k)",
            &mut |plot_ui| {
                plot_ui.vline(plot_vline(
                    "Over/Under Prop Line",
                    line_val,
                    palette.accent,
                    6.0,
                ));
                plot_ui.line(plot_line(
                    "Zero-Inflated Poisson PMF",
                    &self.curve_zip_pmf,
                    palette.series[0],
                    2.5,
                    None,
                ));
                plot_ui.line(plot_line(
                    "Equi-Mean Standard Poisson PMF",
                    &self.curve_pois_pmf,
                    palette.series[1],
                    2.0,
                    Some(7.0),
                ));
            },
        );
    }
}

impl InteractiveTool for ZipCountPropsTool {
    fn name(&self) -> &'static str {
        "ZIP Count Props"
    }
    fn domain(&self) -> &'static str {
        "sports_analytics"
    }
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn state_revision(&self) -> u64 {
        self.revision
    }
    fn has_finite_parameters(&self) -> bool {
        self.state.validate().is_ok()
    }
    fn clamp_parameters(&mut self) -> bool {
        let d = ZipPropsState::default();
        self.state.zero_inflation_pi =
            super::clamp_finite(self.state.zero_inflation_pi, 0.0, 0.90, d.zero_inflation_pi);
        self.state.active_rate_lambda = super::clamp_finite(
            self.state.active_rate_lambda,
            0.20,
            10.0,
            d.active_rate_lambda,
        );
        self.state.prop_line_half = self.state.prop_line_half.min(10);
        self.state.offered_over_odds = super::clamp_finite(
            self.state.offered_over_odds,
            1.05,
            25.0,
            d.offered_over_odds,
        );
        self.state.bankroll =
            super::clamp_finite(self.state.bankroll, 10.0, 1_000_000.0, d.bankroll);
        self.recompute();
        true
    }
    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }
    fn try_load_state(&mut self, json: &str) -> Result<(), String> {
        self.state_bounds().validate_json(json)?;
        let parsed: ZipPropsState = serde_json::from_value(super::parse_state_value(json)?)
            .map_err(|e| format!("Invalid ZipPropsState: {e}"))?;
        parsed.validate()?;
        self.state = parsed;
        self.recompute();
        Ok(())
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        vec![
            ("ZIP Prop PMF".into(), self.curve_zip_pmf.clone()),
            (
                "Equi-Mean Standard Poisson PMF".into(),
                self.curve_pois_pmf.clone(),
            ),
        ]
    }
    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        render_sports_left_panel(parent_ui, "zip_props_side", "ZIP Count Props", &mut |ui| {
            self.render_controls(ui);
        });
        render_sports_central_panel(parent_ui, &mut |ui, palette| {
            self.render_summary(ui, palette);
            ui.separator();
            self.render_plot(ui, palette);
        });
    }
}

impl scientific_metadata::theory::TheoryDescribable for ZipCountPropsTool {
    fn theory_description(&self) -> String {
        "Zero-Inflated Poisson (ZIP) player prop model: mixes structural zero probability $\\pi \\in [0,1]$ with active Poisson rate $\\lambda > 0$ so $P(Y=0) = \\pi + (1-\\pi)e^{-\\lambda}$, producing overdispersion $(1 + \\pi \\lambda) > 1$ and separating mean from median on half-integer Over/Under lines.".into()
    }
    fn phonetic_description(&self) -> String {
        "Zero inflated Poisson discrete player prop distribution and mean median gap analyzer"
            .into()
    }
    fn theory_citation(&self) -> String {
        "Zero-Inflated Poisson Regression, with an Application to Defects in Manufacturing (Lambert, 1992); A New Interpretation of Information Rate (Kelly, 1956)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    ToolMetadata {
        id: "ZipCountPropsTool",
        display_name: "ZIP Count Props",
        domain: "sports_analytics",
        tags: &["zip", "poisson", "player_props", "overdispersion", "kelly"],
        build: || Box::new(ZipCountPropsTool::default()),
    }
}

// [cite:sports_quantitative_modeling]

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{BetSizeCalculatorTool, KellyStrategyRow};
use crate::accessibility::{announce_live_assertive_status, announce_live_status};
use crate::web_io::{spawn_save_as_file_dialog, SaveAsDialogConfig};
use eframe::egui;
use egui_plot::export::{copy_text_to_clipboard, escape_csv_field};
use egui_plot::PlotPoint;
use std::sync::mpsc::channel;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SizingExportKind {
    Csv,
    Json,
    Markdown,
}

const CSV_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "CSV Sizing Table (*.csv)",
    extensions: &["csv"],
    fallback_filename: "kelly_sizing_table.csv",
    write_error_prefix: "Failed to save Kelly sizing CSV",
};

const JSON_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "JSON Sizing Schedule (*.json)",
    extensions: &["json"],
    fallback_filename: "kelly_sizing_schedule.json",
    write_error_prefix: "Failed to save Kelly sizing JSON",
};

const MD_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "Markdown Sizing Table (*.md)",
    extensions: &["md"],
    fallback_filename: "kelly_sizing_table.md",
    write_error_prefix: "Failed to save Kelly sizing Markdown",
};

pub(crate) fn format_kelly_rows_csv(rows: &[KellyStrategyRow]) -> String {
    let mut out = String::from(
        "Strategy,Fraction_Pct,Bet_Amount_USD,Expected_Growth_Rate,Expected_Value_Per_Unit\n",
    );
    for row in rows {
        let label = escape_csv_field(&row.label);
        let growth = if row.growth_rate.is_finite() {
            format!("{:.6}", row.growth_rate)
        } else {
            String::from("-inf")
        };
        out.push_str(&format!(
            "{label},{:.4},{:.2},{growth},{:.6}\n",
            row.fraction * 100.0,
            row.bet_amount,
            row.ev_per_unit
        ));
    }
    out
}

pub(crate) fn format_kelly_rows_markdown(rows: &[KellyStrategyRow]) -> String {
    let mut out = String::from(
        "| Strategy | Fraction (%) | Bet Amount ($) | Growth Rate g(f) | EV / $1 |\n| :--- | ---: | ---: | ---: | ---: |\n",
    );
    for row in rows {
        let clean_label = row.label.replace('|', "\\|").replace(['\r', '\n'], " ");
        let growth = if row.growth_rate.is_finite() {
            format!("{:.6}", row.growth_rate)
        } else {
            String::from("-inf")
        };
        out.push_str(&format!(
            "| {clean_label} | {:.2}% | ${:.2} | {growth} | ${:.4} |\n",
            row.fraction * 100.0,
            row.bet_amount,
            row.ev_per_unit
        ));
    }
    out
}

impl BetSizeCalculatorTool {
    pub(super) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        if self.growth_curve_points.is_empty() {
            return Vec::new();
        }
        let to_vec = |pts: &[PlotPoint]| pts.iter().map(|p| [p.x, p.y]).collect();
        let mut out = vec![
            ("Break-Even (g = 0)".into(), to_vec(&self.zero_line_points)),
            (
                "Expected Growth Rate g(f)".into(),
                to_vec(&self.growth_curve_points),
            ),
        ];
        if !self.quarter_marker.is_empty() {
            out.push((
                "Quarter Kelly (0.25 f*)".into(),
                to_vec(&self.quarter_marker),
            ));
        }
        if !self.half_marker.is_empty() {
            out.push(("Half Kelly (0.50 f*)".into(), to_vec(&self.half_marker)));
        }
        if !self.full_marker.is_empty() {
            out.push(("Full Kelly (1.00 f*)".into(), to_vec(&self.full_marker)));
        }
        if !self.ruin_marker.is_empty() {
            out.push((
                "Double Kelly (2.00 f* Ruin)".into(),
                to_vec(&self.ruin_marker),
            ));
        }
        out
    }

    pub fn export_sizing_csv(&self) -> String {
        format_kelly_rows_csv(&self.strategy_rows)
    }

    pub fn export_sizing_json(&self) -> Result<String, String> {
        let strategies: Vec<serde_json::Value> = self
            .strategy_rows
            .iter()
            .map(|row| {
                serde_json::json!({
                    "strategy": row.label,
                    "fraction": row.fraction,
                    "fraction_pct": row.fraction * 100.0,
                    "bet_amount_usd": row.bet_amount,
                    "expected_growth_rate": row.growth_rate.is_finite().then_some(row.growth_rate),
                    "expected_value_per_unit": row.ev_per_unit,
                })
            })
            .collect();
        let doc = serde_json::json!({
            "win_probability_pct": self.probability_input,
            "decimal_odds": self.decimal_odds,
            "american_odds": self.american_odds,
            "bankroll_usd": self.bankroll,
            "expected_value_per_unit": self.ev_per_unit,
            "strategies": strategies,
        });
        super::super::format_pretty_json(&doc)
    }

    pub fn export_sizing_markdown(&self) -> String {
        format_kelly_rows_markdown(&self.strategy_rows)
    }

    pub(super) fn poll_save_channel(&mut self, ctx: &egui::Context) {
        let outcome = self.save_rx.as_ref().and_then(|rx| rx.try_recv().ok());
        if let Some(res) = outcome {
            self.save_rx = None;
            match res {
                Ok(msg) => {
                    announce_live_status(ctx, &msg);
                    self.io_status = Some((msg, false));
                }
                Err(err) => {
                    announce_live_assertive_status(ctx, &err);
                    self.io_status = Some((err, true));
                }
            }
        }
    }

    pub(super) fn copy_sizing_to_clipboard(&mut self, ctx: &egui::Context, kind: SizingExportKind) {
        let (payload, label) = match kind {
            SizingExportKind::Csv => (self.export_sizing_csv(), "CSV Kelly sizing table"),
            SizingExportKind::Json => match self.export_sizing_json() {
                Ok(json) => (json, "JSON Kelly sizing schedule"),
                Err(err) => {
                    announce_live_assertive_status(ctx, &err);
                    self.io_status = Some((err, true));
                    return;
                }
            },
            SizingExportKind::Markdown => {
                (self.export_sizing_markdown(), "Markdown Kelly sizing table")
            }
        };
        match copy_text_to_clipboard(&payload) {
            Ok(()) => {
                let msg = format!("Copied {label} to clipboard");
                announce_live_status(ctx, &msg);
                self.io_status = Some((msg, false));
            }
            Err(e) => {
                let err = format!("Clipboard copy failed: {e}");
                announce_live_assertive_status(ctx, &err);
                self.io_status = Some((err, true));
            }
        }
    }

    pub(super) fn save_sizing_file(&mut self, kind: SizingExportKind) {
        let (tx, rx) = channel();
        self.save_rx = Some(rx);
        match kind {
            SizingExportKind::Csv => spawn_save_as_file_dialog(
                tx,
                CSV_SAVE_CONFIG,
                "kelly_sizing_table.csv",
                self.export_sizing_csv(),
                |path, _| format!("Saved Kelly sizing CSV to {}", path.display()),
            ),
            SizingExportKind::Json => match self.export_sizing_json() {
                Ok(json) => spawn_save_as_file_dialog(
                    tx,
                    JSON_SAVE_CONFIG,
                    "kelly_sizing_schedule.json",
                    json,
                    |path, _| format!("Saved Kelly sizing JSON to {}", path.display()),
                ),
                Err(err) => {
                    self.save_rx = None;
                    self.io_status = Some((err, true));
                }
            },
            SizingExportKind::Markdown => spawn_save_as_file_dialog(
                tx,
                MD_SAVE_CONFIG,
                "kelly_sizing_table.md",
                self.export_sizing_markdown(),
                |path, _| format!("Saved Kelly sizing Markdown to {}", path.display()),
            ),
        }
    }
}

impl scientific_metadata::theory::TheoryDescribable for BetSizeCalculatorTool {
    fn theory_description(&self) -> String {
        let net_b = self.resolve_active_odds().map_or_else(
            |_| (self.decimal_odds - 1.0).max(0.01),
            |o| o.net_profit_multiplier(),
        );
        format!(
            "Kelly criterion optimal wagering fraction $f^* = \\frac{{b p - q}}{{b}}$ and logarithmic growth rate $g(f) = p \\ln(1 + b f) + q \\ln(1 - f)$ where $p = {:.1}\\%$, $q = 1 - p$, and net odds $b = {:.2}$.",
            self.probability_input, net_b
        )
    }
    fn phonetic_description(&self) -> String {
        "Kelly criterion optimal bet sizing and expected logarithmic growth rate theory".into()
    }
    fn theory_citation(&self) -> String {
        "A New Interpretation of Information Rate (Kelly, 1956)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Expected Growth Rate g(f)".into(),
            "Asymptotic logarithmic wealth growth per wager across stake fraction f".into(),
        );
        map
    }
}

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{BetSizeCalculatorTool, KellyStrategyRow};
use crate::accessibility::{announce_live_assertive_status, announce_live_status};
use crate::web_io::{spawn_save_as_file_dialog, SaveAsDialogConfig};
use eframe::egui;
use egui_plot::export::{copy_text_to_clipboard, escape_csv_field};
use std::sync::mpsc::channel;

const CSV_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "CSV Sizing Table (*.csv)",
    extensions: &["csv"],
    fallback_filename: "kelly_sizing_table.csv",
    write_error_prefix: "Failed to save Kelly sizing CSV",
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
    pub fn export_sizing_csv(&self) -> String {
        format_kelly_rows_csv(&self.strategy_rows)
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

    pub(super) fn copy_sizing_to_clipboard(&mut self, ctx: &egui::Context, markdown: bool) {
        let (payload, label) = if markdown {
            (self.export_sizing_markdown(), "Markdown Kelly sizing table")
        } else {
            (self.export_sizing_csv(), "CSV Kelly sizing table")
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

    pub(super) fn save_sizing_file(&mut self, markdown: bool) {
        let (tx, rx) = channel();
        self.save_rx = Some(rx);
        if markdown {
            spawn_save_as_file_dialog(
                tx,
                MD_SAVE_CONFIG,
                "kelly_sizing_table.md",
                self.export_sizing_markdown(),
                |path, _| format!("Saved Kelly sizing Markdown to {}", path.display()),
            );
        } else {
            spawn_save_as_file_dialog(
                tx,
                CSV_SAVE_CONFIG,
                "kelly_sizing_table.csv",
                self.export_sizing_csv(),
                |path, _| format!("Saved Kelly sizing CSV to {}", path.display()),
            );
        }
    }
}

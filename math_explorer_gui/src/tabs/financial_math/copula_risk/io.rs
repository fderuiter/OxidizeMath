//! Fair-odds CSV/JSON pricing sheet exports and code snippets for [`super::CopulaRiskTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::engine::decimal_to_american_odds;
use super::CopulaRiskTool;
use crate::accessibility::{announce_live_assertive_status, announce_live_status};
use crate::web_io::{spawn_save_as_file_dialog, SaveAsDialogConfig};
use eframe::egui;
use egui_plot::export::{copy_text_to_clipboard, escape_csv_field};
use std::sync::mpsc::{channel, Receiver};

const CSV_PRICING_SAVE: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "CSV Fair-Odds Pricing Sheet (*.csv)",
    extensions: &["csv"],
    fallback_filename: "copula_fair_odds_sheet.csv",
    write_error_prefix: "Failed to save copula CSV pricing sheet",
};

const JSON_PRICING_SAVE: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "JSON Fair-Odds Pricing Sheet (*.json)",
    extensions: &["json"],
    fallback_filename: "copula_fair_odds_sheet.json",
    write_error_prefix: "Failed to save copula JSON pricing sheet",
};

#[derive(Default)]
pub(crate) struct CopulaIoState {
    save_rx: Option<Receiver<Result<String, String>>>,
    pub status: Option<(String, bool)>,
}

impl CopulaRiskTool {
    pub fn export_pricing_csv(&self) -> String {
        let mut out = String::from(
            "Section,Item,Label,Probability,Naive_Probability,Fair_Decimal_Odds,Offered_Decimal_Odds,Fair_American_Odds\n",
        );
        let vig_factor = 1.0 / (1.0 + self.book_vig_pct / 100.0);
        for (idx, (label, &p)) in self
            .leg_labels
            .iter()
            .zip(self.marginals.iter())
            .enumerate()
        {
            let safe_label = escape_csv_field(label);
            let fair_dec = if p > 1e-6 { 1.0 / p } else { 999_999.0 };
            let off_dec = (fair_dec * vig_factor).max(1.01);
            let amer = decimal_to_american_odds(fair_dec);
            out.push_str(&format!(
                "Marginal,Leg {},{safe_label},{p:.6},{p:.6},{fair_dec:.4},{off_dec:.4},{amer:+.1}\n",
                idx + 1
            ));
        }
        if let Some(jp) = self.joint_probability {
            let fair_dec = if jp > 1e-6 { 1.0 / jp } else { 999_999.0 };
            let off_dec = (fair_dec * vig_factor).max(1.01);
            let amer = decimal_to_american_odds(fair_dec);
            out.push_str(&format!(
                "ActiveParlay,{}-Leg SGP,Correlated Parlay,{jp:.6},{:.6},{fair_dec:.4},{off_dec:.4},{amer:+.1}\n",
                self.num_legs, self.naive_probability
            ));
        }
        for row in &self.pricing_rows {
            out.push_str(&format!(
                "Sweep,rho={:+.2},Equicorrelated Sweep,{:.6},{:.6},{:.4},{:.4},{:+.1}\n",
                row.rho,
                row.joint_prob,
                row.naive_prob,
                row.fair_decimal_odds,
                row.offered_decimal_odds,
                row.fair_american_odds
            ));
        }
        out
    }

    pub fn export_pricing_json(&self) -> Result<String, String> {
        let vig_factor = 1.0 / (1.0 + self.book_vig_pct / 100.0);
        let active_fair = self
            .joint_probability
            .map(|jp| if jp > 1e-6 { 1.0 / jp } else { 999_999.0 });
        let active_offered = active_fair.map(|d| (d * vig_factor).max(1.01));
        let legs: Vec<serde_json::Value> = self
            .leg_labels
            .iter()
            .zip(self.marginals.iter())
            .enumerate()
            .map(|(i, (lbl, &p))| {
                let dec = if p > 1e-6 { 1.0 / p } else { 999_999.0 };
                serde_json::json!({
                    "leg_index": i + 1,
                    "label": lbl,
                    "marginal_probability": p,
                    "fair_decimal_odds": dec,
                    "fair_american_odds": decimal_to_american_odds(dec),
                })
            })
            .collect();
        let sweep: Vec<serde_json::Value> = self
            .pricing_rows
            .iter()
            .map(|r| {
                serde_json::json!({
                    "equicorrelation_rho": r.rho,
                    "joint_probability": r.joint_prob,
                    "naive_probability": r.naive_prob,
                    "fair_decimal_odds": r.fair_decimal_odds,
                    "offered_decimal_odds": r.offered_decimal_odds,
                    "fair_american_odds": r.fair_american_odds,
                })
            })
            .collect();
        let doc = serde_json::json!({
            "num_legs": self.num_legs,
            "book_vig_pct": self.book_vig_pct,
            "active_joint_probability": self.joint_probability,
            "naive_independent_probability": self.naive_probability,
            "active_fair_decimal_odds": active_fair,
            "active_offered_decimal_odds": active_offered,
            "legs": legs,
            "correlation_sweep_schedule": sweep,
        });
        super::super::format_pretty_json(&doc)
    }

    pub fn rust_snippet(&self) -> String {
        let probs = self
            .marginals
            .iter()
            .map(|p| format!("Probability::new({p:.4}).unwrap()"))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "use math_explorer::pure_math::statistics::copula::{{Correlation, CorrelationMatrix, Probability, sgp_joint_probability}};\nlet corr = CorrelationMatrix::equicorrelated({}, Correlation::new({:.4}).unwrap()).unwrap();\nlet joint = sgp_joint_probability(&[{probs}], &corr).unwrap();",
            self.num_legs, self.equicorrelation
        )
    }

    pub(crate) fn poll_io(&mut self, ctx: &egui::Context) {
        let outcome = self
            .io_state
            .save_rx
            .as_ref()
            .and_then(|rx| rx.try_recv().ok());
        if let Some(res) = outcome {
            self.io_state.save_rx = None;
            match res {
                Ok(msg) => {
                    announce_live_status(ctx, &msg);
                    self.io_state.status = Some((msg, false));
                }
                Err(err) => {
                    announce_live_assertive_status(ctx, &err);
                    self.io_state.status = Some((err, true));
                }
            }
        }
    }

    pub(crate) fn copy_payload(&mut self, ctx: &egui::Context, label: &str, payload: String) {
        match copy_text_to_clipboard(&payload) {
            Ok(()) => {
                let msg = format!("Copied {label} to clipboard");
                announce_live_status(ctx, &msg);
                self.io_state.status = Some((msg, false));
            }
            Err(e) => {
                let err = format!("Clipboard copy failed: {e}");
                announce_live_assertive_status(ctx, &err);
                self.io_state.status = Some((err, true));
            }
        }
    }

    pub(crate) fn trigger_save_csv(&mut self) {
        let (tx, rx) = channel();
        self.io_state.save_rx = Some(rx);
        spawn_save_as_file_dialog(
            tx,
            CSV_PRICING_SAVE,
            "copula_fair_odds_sheet.csv",
            self.export_pricing_csv(),
            |p, _| format!("Saved copula CSV pricing sheet to {}", p.display()),
        );
    }

    pub(crate) fn trigger_save_json(&mut self) {
        match self.export_pricing_json() {
            Ok(payload) => {
                let (tx, rx) = channel();
                self.io_state.save_rx = Some(rx);
                spawn_save_as_file_dialog(
                    tx,
                    JSON_PRICING_SAVE,
                    "copula_fair_odds_sheet.json",
                    payload,
                    |p, _| format!("Saved copula JSON pricing sheet to {}", p.display()),
                );
            }
            Err(err) => self.io_state.status = Some((err, true)),
        }
    }
}

// [cite:clinical_trials_statistics]

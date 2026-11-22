#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{CorrelatedRiskScenarioTool, RiskAnalysisMode};
use crate::accessibility::{announce_live_assertive_status, announce_live_status};
use crate::web_io::{spawn_save_as_file_dialog, SaveAsDialogConfig};
use eframe::egui;
use egui_plot::export::{copy_text_to_clipboard, escape_csv_field};
use std::sync::mpsc::channel;

const CSV_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "CSV Risk Scenario (*.csv)",
    extensions: &["csv"],
    fallback_filename: "correlated_risk_scenario.csv",
    write_error_prefix: "Failed to save risk scenario CSV",
};

const MD_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "Markdown Risk Scenario (*.md)",
    extensions: &["md"],
    fallback_filename: "correlated_risk_scenario.md",
    write_error_prefix: "Failed to save risk scenario Markdown",
};

fn sanitize_markdown_cell(text: &str) -> String {
    text.replace('|', "\\|").replace(['\r', '\n'], " ")
}

impl CorrelatedRiskScenarioTool {
    pub fn export_scenario_csv(&self) -> String {
        let safe_label = escape_csv_field(&self.scenario_label);
        match self.mode {
            RiskAnalysisMode::CopulaCorrelation => {
                let mut out = String::from(
                    "Scenario,Correlation_Rho,Copula_Joint_Prob,Naive_Indep_Prob,Edge_Delta\n",
                );
                for row in &self.copula_rows {
                    out.push_str(&format!(
                        "{safe_label},{:.4},{:.6},{:.6},{:+.6}\n",
                        row.rho, row.joint_prob, row.naive_prob, row.edge_delta
                    ));
                }
                out
            }
            RiskAnalysisMode::OrnsteinUhlenbeck => {
                let mut out =
                    String::from("Scenario,Seed,Time_t,Ensemble_Mean_Xt,Equilibrium_Mu\n");
                for pt in &self.ou_mean_path {
                    out.push_str(&format!(
                        "{safe_label},{},{:.4},{:.6},{:.6}\n",
                        self.seed, pt.x, pt.y, self.ou_mu
                    ));
                }
                out
            }
        }
    }

    pub fn export_scenario_markdown(&self) -> String {
        let label = sanitize_markdown_cell(&self.scenario_label);
        let mut out = format!("### Correlated Risk Scenario: {label}\n\n");
        out.push_str(&format!(
            "- **Copula SGP**: $P(A) = {:.2}$, $P(B) = {:.2}$, $\\rho = {:+.2}$ -> Joint $P(A \\cap B) = {:.4}$ (Naive ${:.4}$, $\\Delta = {:+.4}$)\n",
            self.prob_a,
            self.prob_b,
            self.rho,
            self.active_sgp_prob,
            self.naive_prob,
            self.active_sgp_prob - self.naive_prob
        ));
        if let Some(st) = self.ou_stats {
            out.push_str(&format!(
                "- **Ornstein-Uhlenbeck** (`seed={}`, {} paths, $T={:.2}$): $\\mu={:.2}, \\theta={:.2}, \\sigma={:.2}, X_0={:.2}$ -> Terminal Mean ${:.4} \\pm {:.4}$ (5%–95%: $[{:.4}, {:.4}]$)\n\n",
                self.seed,
                self.ou_paths,
                st.horizon,
                self.ou_mu,
                self.ou_theta,
                self.ou_sigma,
                self.ou_x0,
                st.mean_final,
                st.std_final,
                st.p05_final,
                st.p95_final
            ));
        }
        out.push_str("| Rho | Copula Joint P | Naive P(A)*P(B) | Edge Delta |\n| ---: | ---: | ---: | ---: |\n");
        for row in self.copula_rows.iter().step_by(5) {
            out.push_str(&format!(
                "| {:+.3} | {:.4} | {:.4} | {:+.4} |\n",
                row.rho, row.joint_prob, row.naive_prob, row.edge_delta
            ));
        }
        out
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

    pub(super) fn copy_scenario_to_clipboard(&mut self, ctx: &egui::Context, markdown: bool) {
        let (payload, kind) = if markdown {
            (self.export_scenario_markdown(), "Markdown risk scenario")
        } else {
            (self.export_scenario_csv(), "CSV risk scenario")
        };
        match copy_text_to_clipboard(&payload) {
            Ok(()) => {
                let msg = format!("Copied {kind} to clipboard");
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

    pub(super) fn save_scenario_file(&mut self, markdown: bool) {
        let (tx, rx) = channel();
        self.save_rx = Some(rx);
        if markdown {
            spawn_save_as_file_dialog(
                tx,
                MD_SAVE_CONFIG,
                "correlated_risk_scenario.md",
                self.export_scenario_markdown(),
                |path, _| format!("Saved risk scenario Markdown to {}", path.display()),
            );
        } else {
            spawn_save_as_file_dialog(
                tx,
                CSV_SAVE_CONFIG,
                "correlated_risk_scenario.csv",
                self.export_scenario_csv(),
                |path, _| format!("Saved risk scenario CSV to {}", path.display()),
            );
        }
    }
}

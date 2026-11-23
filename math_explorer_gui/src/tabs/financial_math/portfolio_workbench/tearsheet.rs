#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::PortfolioCopulaWorkbenchTool;
use crate::accessibility::{announce_live_assertive_status, announce_live_status};
use crate::web_io::{spawn_save_as_file_dialog, SaveAsDialogConfig};
use eframe::egui;
use egui_plot::export::{copy_text_to_clipboard, escape_csv_field};
use std::sync::mpsc::channel;

const SAVE_CSV_TEARSHEET_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "CSV Risk Tear-Sheet (*.csv)",
    extensions: &["csv"],
    fallback_filename: "portfolio_risk_tearsheet.csv",
    write_error_prefix: "Failed to save CSV risk tear-sheet",
};

const SAVE_JSON_TEARSHEET_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "JSON Risk Tear-Sheet (*.json)",
    extensions: &["json"],
    fallback_filename: "portfolio_risk_tearsheet.json",
    write_error_prefix: "Failed to save JSON risk tear-sheet",
};

impl PortfolioCopulaWorkbenchTool {
    pub fn export_tearsheet_csv(&self) -> String {
        let mut out = String::from("Section,Item,Metric,Value\n");
        let push_row = |buf: &mut String, sec: &str, item: &str, metric: &str, val: f64| {
            buf.push_str(&format!(
                "{},{},{},{:.6}\n",
                escape_csv_field(sec),
                escape_csv_field(item),
                escape_csv_field(metric),
                val
            ));
        };
        self.append_kelly_and_var_csv(&mut out, &push_row);
        self.append_copula_ou_bsm_csv(&mut out, &push_row);
        out
    }

    fn append_kelly_and_var_csv(
        &self,
        out: &mut String,
        push_row: &dyn Fn(&mut String, &str, &str, &str, f64),
    ) {
        if let Some(ref sol) = self.kelly_cache.cov_kelly {
            for (idx, &w) in sol.weights.iter().enumerate() {
                let name = self
                    .state
                    .assets
                    .get(idx)
                    .map_or("Asset", |a| a.label.as_str());
                push_row(out, "Kelly_Weights", name, "Covariance_Kelly_Weight", w);
            }
            push_row(
                out,
                "Kelly_Summary",
                "Covariance_Kelly",
                "Gross_Leverage",
                sol.metrics.gross_leverage,
            );
            push_row(
                out,
                "Kelly_Summary",
                "Covariance_Kelly",
                "Excess_Log_Growth",
                sol.metrics.excess_log_growth,
            );
        }
        if let Some(ref lo) = self.kelly_cache.long_only_capped {
            for (idx, &w) in lo.weights.iter().enumerate() {
                let name = self
                    .state
                    .assets
                    .get(idx)
                    .map_or("Asset", |a| a.label.as_str());
                push_row(out, "Kelly_Weights", name, "Long_Only_Weight", w);
            }
        }
        for (label, tail_opt) in [
            ("Simulated_Cov_Kelly", self.kelly_cache.cov_sim_tail),
            ("Parametric_Cov_Kelly", self.kelly_cache.cov_param_tail),
            ("Empirical_Imported", self.kelly_cache.empirical_tail),
        ] {
            if let Some(t) = tail_opt {
                push_row(out, "Tail_Risk", label, "VaR_95_USD", t.var_95);
                push_row(out, "Tail_Risk", label, "CVaR_95_USD", t.cvar_95);
                push_row(out, "Tail_Risk", label, "VaR_99_USD", t.var_99);
                push_row(out, "Tail_Risk", label, "CVaR_99_USD", t.cvar_99);
            }
        }
    }

    fn append_copula_ou_bsm_csv(
        &self,
        out: &mut String,
        push_row: &dyn Fn(&mut String, &str, &str, &str, f64),
    ) {
        push_row(
            out,
            "Gaussian_Copula",
            "Joint_SGP",
            "Copula_Probability",
            self.copula_cache.active_sgp_prob,
        );
        push_row(
            out,
            "Gaussian_Copula",
            "Tail_Crash",
            "Copula_Crash_Probability",
            self.copula_cache.active_crash_prob,
        );
        push_row(
            out,
            "Gaussian_Copula",
            "Tail_Crash",
            "Crash_Amplification_Multiplier",
            self.copula_cache.crash_multiplier,
        );
        if let Some(ref st) = self.ou_cache.forecast_stats {
            push_row(
                out,
                "OU_Forecast",
                "Terminal",
                "Quantile_05",
                st.percentile_05,
            );
            push_row(out, "OU_Forecast", "Terminal", "Median_50", st.median_final);
            push_row(
                out,
                "OU_Forecast",
                "Terminal",
                "Quantile_95",
                st.percentile_95,
            );
            push_row(
                out,
                "OU_Forecast",
                "Parameters",
                "Half_Life",
                self.ou_cache.half_life,
            );
        }
        if let Some(ref pair) = self.bsm_cache.pair {
            for (kind, g) in [("European_Call", &pair.call), ("European_Put", &pair.put)] {
                push_row(out, "Black_Scholes", kind, "Price", g.price);
                push_row(out, "Black_Scholes", kind, "Delta", g.delta);
                push_row(out, "Black_Scholes", kind, "Gamma", g.gamma);
                push_row(out, "Black_Scholes", kind, "Vega", g.vega);
                push_row(out, "Black_Scholes", kind, "Theta", g.theta);
                push_row(out, "Black_Scholes", kind, "Rho", g.rho);
            }
        }
    }

    fn kelly_weights_json(&self) -> Vec<serde_json::Value> {
        self.state
            .assets
            .iter()
            .enumerate()
            .map(|(i, a)| {
                serde_json::json!({
                    "asset": a.label,
                    "mu": a.mu,
                    "sigma": a.sigma,
                    "covariance_kelly_weight": self.kelly_cache.cov_kelly.as_ref().and_then(|s| s.weights.get(i)).copied(),
                    "long_only_weight": self.kelly_cache.long_only_capped.as_ref().and_then(|s| s.weights.get(i)).copied(),
                    "independent_kelly_weight": self.kelly_cache.indep_kelly.as_ref().and_then(|s| s.weights.get(i)).copied(),
                })
            })
            .collect()
    }

    pub fn export_tearsheet_json(&self) -> String {
        let fmt_tail = |t: Option<math_explorer::pure_math::statistics::kelly::TailRiskMetrics>| {
            t.map(|m| {
                serde_json::json!({
                    "var_95": m.var_95,
                    "cvar_95": m.cvar_95,
                    "var_99": m.var_99,
                    "cvar_99": m.cvar_99,
                })
            })
        };
        let doc = serde_json::json!({
            "bankroll_usd": self.state.bankroll,
            "risk_free_rate": self.state.risk_free_rate,
            "horizon_years": self.state.horizon,
            "kelly_weights": self.kelly_weights_json(),
            "kelly_summary": {
                "covariance_gross_leverage": self.kelly_cache.cov_kelly.as_ref().map(|s| s.metrics.gross_leverage),
                "covariance_excess_log_growth": self.kelly_cache.cov_kelly.as_ref().map(|s| s.metrics.excess_log_growth),
                "independent_gross_leverage": self.kelly_cache.indep_kelly.as_ref().map(|s| s.metrics.gross_leverage),
                "independent_excess_log_growth": self.kelly_cache.indep_kelly.as_ref().map(|s| s.metrics.excess_log_growth),
            },
            "tail_risk_usd": {
                "simulated_covariance_kelly": fmt_tail(self.kelly_cache.cov_sim_tail),
                "parametric_covariance_kelly": fmt_tail(self.kelly_cache.cov_param_tail),
                "empirical_imported": fmt_tail(self.kelly_cache.empirical_tail),
            },
            "gaussian_copula": {
                "legs": self.state.copula_legs,
                "rho": self.state.rho,
                "sgp_joint_prob": self.copula_cache.active_sgp_prob,
                "crash_prob": self.copula_cache.active_crash_prob,
                "crash_amplification_vs_naive": self.copula_cache.crash_multiplier,
            },
            "ou_analyzer": self.ou_cache.forecast_stats.as_ref().map(|st| serde_json::json!({
                "mu": self.state.ou_mu,
                "theta": self.state.ou_theta,
                "sigma": self.state.ou_sigma,
                "half_life": self.ou_cache.half_life,
                "quantile_05": st.percentile_05,
                "median_50": st.median_final,
                "quantile_95": st.percentile_95,
                "mean_final": st.mean_final,
            })),
            "black_scholes_greeks": self.bsm_cache.pair.as_ref().map(|p| serde_json::json!({
                "spot": self.state.bsm_spot,
                "strike": self.state.bsm_strike,
                "call": {
                    "price": p.call.price, "delta": p.call.delta, "gamma": p.call.gamma,
                    "vega": p.call.vega, "theta": p.call.theta, "rho": p.call.rho,
                },
                "put": {
                    "price": p.put.price, "delta": p.put.delta, "gamma": p.put.gamma,
                    "vega": p.put.vega, "theta": p.put.theta, "rho": p.put.rho,
                },
            })),
        });
        serde_json::to_string_pretty(&doc).unwrap_or_else(|_| String::from("{}"))
    }

    pub(super) fn trigger_save_tearsheet_dialog(&mut self, as_json: bool) {
        let (tx, rx) = channel();
        self.save_rx = Some(rx);
        if as_json {
            spawn_save_as_file_dialog(
                tx,
                SAVE_JSON_TEARSHEET_CONFIG,
                "portfolio_risk_tearsheet.json",
                self.export_tearsheet_json(),
                |path, _| format!("Saved JSON risk tear-sheet to {}", path.display()),
            );
        } else {
            spawn_save_as_file_dialog(
                tx,
                SAVE_CSV_TEARSHEET_CONFIG,
                "portfolio_risk_tearsheet.csv",
                self.export_tearsheet_csv(),
                |path, _| format!("Saved CSV risk tear-sheet to {}", path.display()),
            );
        }
    }

    pub(super) fn copy_tearsheet_to_clipboard(&mut self, ctx: &egui::Context, as_json: bool) {
        let (payload, label) = if as_json {
            (self.export_tearsheet_json(), "JSON risk tear-sheet")
        } else {
            (self.export_tearsheet_csv(), "CSV risk tear-sheet")
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
}

// [cite:clinical_trials_statistics]

//! Async CSV import, forecast exports, URL-size checks, and calibrated code snippets for [`super::OuProcessTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::OuProcessTool;
use crate::accessibility::{announce_live_assertive_status, announce_live_status};
use crate::framework::InteractiveTool;
use crate::url_state::MAX_DECODED_URL_STATE_BYTES;
use crate::web_io::{
    spawn_open_file_dialog, spawn_save_as_file_dialog, OpenDialogConfig, SaveAsDialogConfig,
};
use eframe::egui;
use egui_plot::export::copy_text_to_clipboard;
use std::sync::mpsc::{channel, Receiver};

const OU_CSV_OPEN_CONFIG: OpenDialogConfig = OpenDialogConfig {
    filter_name: "Time Series CSV (*.csv, *.txt)",
    extensions: &["csv", "txt"],
    accept: ".csv,.txt,text/csv,text/plain",
    read_error_prefix: "Failed to read OU series CSV",
};

const OU_CSV_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "OU Confidence Bands CSV (*.csv)",
    extensions: &["csv"],
    fallback_filename: "ou_confidence_bands.csv",
    write_error_prefix: "Failed to save OU CSV",
};

const OU_JSON_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "OU Calibration & Bands JSON (*.json)",
    extensions: &["json"],
    fallback_filename: "ou_calibration_bands.json",
    write_error_prefix: "Failed to save OU JSON",
};

#[derive(Default)]
pub(crate) struct OuIoState {
    open_rx: Option<Receiver<Result<(String, String), String>>>,
    save_rx: Option<Receiver<Result<String, String>>>,
    pub status: Option<(String, bool)>,
}

impl OuProcessTool {
    pub fn url_share_warning(&self) -> Option<String> {
        let len = self.save_state().map(|s| s.len()).unwrap_or(0);
        if len > MAX_DECODED_URL_STATE_BYTES {
            Some(format!(
                "Imported series state ({len} bytes) exceeds browser URL share cap ({MAX_DECODED_URL_STATE_BYTES} bytes). Export JSON/CSV to share the full dataset."
            ))
        } else {
            None
        }
    }

    pub fn rust_snippet(&self) -> String {
        format!(
            "use math_explorer::pure_math::statistics::ou_process::{{EulerMaruyama, OuParams, TimeStep}};\nuse math_explorer::rng::OxidizeRng;\n\nlet params = OuParams::from_values({:.6}, {:.6}, {:.6}).unwrap();\nlet solver = EulerMaruyama::new(params, TimeStep::new({:.6}).unwrap());\nlet mut rng = OxidizeRng::new({});\nlet mut path = vec![0.0; {}];\nsolver.simulate_into({:.6}, &mut path, &mut rng);",
            self.mu,
            self.theta,
            self.sigma,
            self.dt,
            self.rng_seed,
            self.n_steps + 1,
            self.x0
        )
    }

    pub fn python_snippet(&self) -> String {
        format!(
            "import numpy as np\n\nmu, theta, sigma, dt = {:.6}, {:.6}, {:.6}, {:.6}\nx = np.empty({})\nx[0] = {:.6}\nrng = np.random.default_rng({})\nfor t in range(1, len(x)):\n    x[t] = x[t - 1] + theta * (mu - x[t - 1]) * dt + sigma * np.sqrt(dt) * rng.standard_normal()",
            self.mu,
            self.theta,
            self.sigma,
            self.dt,
            self.n_steps + 1,
            self.x0,
            self.rng_seed
        )
    }

    pub fn latex_snippet(&self) -> String {
        format!(
            "dX_t = {:.4}\\,({:.4} - X_t)\\,dt + {:.4}\\,dW_t, \\quad X_0 = {:.4}, \\quad t_{{1/2}} = {:.4}",
            self.theta, self.mu, self.sigma, self.x0, self.half_life
        )
    }

    pub fn export_bands_csv(&self) -> String {
        let mut out = String::from("Step,Time,Mean,P05,P25,P75,P95\n");
        for r in &self.band_rows {
            out.push_str(&format!(
                "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                r.step, r.time, r.mean, r.p05, r.p25, r.p75, r.p95
            ));
        }
        out
    }

    pub fn export_bands_json(&self) -> Result<String, String> {
        let bands: Vec<serde_json::Value> = self
            .band_rows
            .iter()
            .map(|r| {
                serde_json::json!({
                    "step": r.step,
                    "time": r.time,
                    "mean": r.mean,
                    "p05": r.p05,
                    "p25": r.p25,
                    "p75": r.p75,
                    "p95": r.p95,
                })
            })
            .collect();
        let doc = serde_json::json!({
            "mu": self.mu,
            "theta": self.theta,
            "sigma": self.sigma,
            "half_life": self.half_life,
            "momentum_classification": self.momentum_class.description(),
            "threshold": self.threshold,
            "terminal_exceedance_probability": self.exceedance_prob,
            "dt": self.dt,
            "n_steps": self.n_steps,
            "n_paths": self.n_paths,
            "observed_count": self.observed_series.len(),
            "bands": bands,
        });
        super::super::format_pretty_json(&doc)
    }

    pub(crate) fn poll_io(&mut self, ctx: &egui::Context) {
        let opened = self
            .io_state
            .open_rx
            .as_ref()
            .and_then(|rx| rx.try_recv().ok());
        if let Some(res) = opened {
            self.io_state.open_rx = None;
            match res {
                Ok((name, raw)) => match self.import_csv_series(&raw) {
                    Ok(n) => {
                        let msg = format!(
                            "Calibrated OU parameters from {n} observations in {name} (μ={:.3}, θ={:.3}, σ={:.3})",
                            self.mu, self.theta, self.sigma
                        );
                        announce_live_status(ctx, &msg);
                        self.io_state.status = Some((msg, false));
                    }
                    Err(err) => {
                        let msg = format!("CSV import rejected ({name}): {err}");
                        announce_live_assertive_status(ctx, &msg);
                        self.io_state.status = Some((msg, true));
                    }
                },
                Err(err) => {
                    announce_live_assertive_status(ctx, &err);
                    self.io_state.status = Some((err, true));
                }
            }
            ctx.request_repaint();
        }

        let saved = self
            .io_state
            .save_rx
            .as_ref()
            .and_then(|rx| rx.try_recv().ok());
        if let Some(res) = saved {
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

    pub(crate) fn trigger_open_csv(&mut self) {
        let (tx, rx) = channel();
        self.io_state.open_rx = Some(rx);
        spawn_open_file_dialog(tx, OU_CSV_OPEN_CONFIG, |p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("series.csv")
                .to_string()
        });
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
            OU_CSV_SAVE_CONFIG,
            "ou_confidence_bands.csv",
            self.export_bands_csv(),
            |p, _| format!("Saved OU CSV bands to {}", p.display()),
        );
    }

    pub(crate) fn trigger_save_json(&mut self) {
        match self.export_bands_json() {
            Ok(payload) => {
                let (tx, rx) = channel();
                self.io_state.save_rx = Some(rx);
                spawn_save_as_file_dialog(
                    tx,
                    OU_JSON_SAVE_CONFIG,
                    "ou_calibration_bands.json",
                    payload,
                    |p, _| format!("Saved OU JSON calibration to {}", p.display()),
                );
            }
            Err(err) => self.io_state.status = Some((err, true)),
        }
    }
}

// [cite:clinical_trials_statistics]

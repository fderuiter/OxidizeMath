//! Observation CSV import, regime timeline CSV/JSON exports, URL warnings, and snippets for [`super::MarkovRegimeTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{MarkovRegimeTool, RegimeWorkbenchMode};
use crate::accessibility::{announce_live_assertive_status, announce_live_status};
use crate::framework::InteractiveTool;
use crate::url_state::MAX_DECODED_URL_STATE_BYTES;
use crate::web_io::{
    spawn_open_file_dialog, spawn_save_as_file_dialog, OpenDialogConfig, SaveAsDialogConfig,
};
use eframe::egui;
use egui_plot::export::copy_text_to_clipboard;
use std::sync::mpsc::{channel, Receiver};

const OBS_CSV_OPEN_CONFIG: OpenDialogConfig = OpenDialogConfig {
    filter_name: "HMM Observation Log (*.csv, *.txt)",
    extensions: &["csv", "txt"],
    accept: ".csv,.txt,text/csv,text/plain",
    read_error_prefix: "Failed to read observation CSV log",
};

const REGIME_CSV_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "Markov Regime Schedule (*.csv)",
    extensions: &["csv"],
    fallback_filename: "markov_regime_export.csv",
    write_error_prefix: "Failed to save Markov regime CSV",
};

const REGIME_JSON_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "Markov Regime Summary (*.json)",
    extensions: &["json"],
    fallback_filename: "markov_regime_export.json",
    write_error_prefix: "Failed to save Markov regime JSON",
};

#[derive(Default)]
pub(crate) struct MarkovIoState {
    open_rx: Option<Receiver<Result<(String, String), String>>>,
    save_rx: Option<Receiver<Result<String, String>>>,
    pub status: Option<(String, bool)>,
}

impl MarkovRegimeTool {
    pub fn url_share_warning(&self) -> Option<String> {
        let len = self.save_state().map(|s| s.len()).unwrap_or(0);
        if len > MAX_DECODED_URL_STATE_BYTES {
            Some(format!(
                "Observation log state ({len} bytes) exceeds browser URL share limit ({MAX_DECODED_URL_STATE_BYTES} bytes). Use Save JSON/CSV to share full logs."
            ))
        } else {
            None
        }
    }

    pub fn export_regime_csv(&self) -> String {
        match self.mode {
            RegimeWorkbenchMode::ContinuousTimeCtmc => {
                let mut out = String::from("Time");
                for s in 0..self.ctmc_states {
                    out.push_str(&format!(",P_State_{s}"));
                }
                out.push('\n');
                let len = self
                    .ctmc_prob_series
                    .first()
                    .map(|(_, v)| v.len())
                    .unwrap_or(0);
                for row_idx in 0..len {
                    let t = self.ctmc_prob_series[0].1[row_idx].x;
                    out.push_str(&format!("{t:.6}"));
                    for (_, curve) in &self.ctmc_prob_series {
                        out.push_str(&format!(",{:.6}", curve[row_idx].y));
                    }
                    out.push('\n');
                }
                out
            }
            RegimeWorkbenchMode::HiddenMarkovHmm => {
                let mut out = String::from("Step,Observation,Viterbi_State");
                for s in 0..self.hmm_states {
                    out.push_str(&format!(",Posterior_State_{s}"));
                }
                out.push('\n');
                for (t, &obs) in self.observations.iter().enumerate() {
                    let vit = self.viterbi_states.get(t).copied().unwrap_or(0);
                    out.push_str(&format!("{t},{obs},{vit}"));
                    for (_, series) in &self.hmm_posterior_series {
                        let p = series.get(t).map(|pt| pt.y).unwrap_or(0.0);
                        out.push_str(&format!(",{p:.6}"));
                    }
                    out.push('\n');
                }
                out
            }
        }
    }

    pub fn export_regime_json(&self) -> Result<String, String> {
        let mode_str = match self.mode {
            RegimeWorkbenchMode::ContinuousTimeCtmc => "ContinuousTimeCtmc",
            RegimeWorkbenchMode::HiddenMarkovHmm => "HiddenMarkovHmm",
        };
        let timeline: Vec<serde_json::Value> = self
            .observations
            .iter()
            .enumerate()
            .map(|(t, &obs)| {
                let posteriors: Vec<f64> = self
                    .hmm_posterior_series
                    .iter()
                    .map(|(_, pts)| pts.get(t).map(|p| p.y).unwrap_or(0.0))
                    .collect();
                serde_json::json!({
                    "step": t,
                    "observation": obs,
                    "viterbi_state": self.viterbi_states.get(t).copied().unwrap_or(0),
                    "posteriors": posteriors,
                })
            })
            .collect();

        let doc = serde_json::json!({
            "active_mode": mode_str,
            "horizon_t": self.horizon_t,
            "ctmc_steady_state": self.ctmc_steady_state,
            "ctmc_absorption_times": self.ctmc_absorption_times,
            "hmm_initial": self.hmm_initial,
            "hmm_transitions": self.hmm_transitions,
            "hmm_emissions": self.hmm_emissions,
            "baum_welch_iterations": self.last_bw_report.as_ref().map(|r| r.iterations),
            "baum_welch_final_log_likelihood": self.last_bw_report.as_ref().map(|r| r.final_log_likelihood),
            "hmm_timeline": timeline,
        });
        super::super::format_pretty_json(&doc)
    }

    pub fn rust_snippet(&self) -> String {
        match self.mode {
            RegimeWorkbenchMode::ContinuousTimeCtmc => format!(
                "use math_explorer::pure_math::statistics::markov::ctmc::ContinuousMarkovChain;\n// Evaluate P(t) = exp(G * {:.3}) across {} states\nlet p_t = chain.transition_probabilities({:.3}).unwrap();",
                self.horizon_t, self.ctmc_states, self.horizon_t
            ),
            RegimeWorkbenchMode::HiddenMarkovHmm => format!(
                "use math_explorer::pure_math::statistics::markov::hmm::HiddenMarkovModel;\nlet report = hmm.baum_welch(&observations, {}, {:.1e}).unwrap();\nlet path = hmm.viterbi(&observations).unwrap();",
                self.bw_max_iters, self.bw_tolerance
            ),
        }
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
                Ok((name, raw)) => match self.import_observations_csv(&raw) {
                    Ok(count) => {
                        let msg = format!("Imported {count} HMM observations from {name}");
                        announce_live_status(ctx, &msg);
                        self.io_state.status = Some((msg, false));
                    }
                    Err(err) => {
                        let msg = format!("Observation CSV rejected ({name}): {err}");
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

    pub(crate) fn trigger_open_observations(&mut self) {
        let (tx, rx) = channel();
        self.io_state.open_rx = Some(rx);
        spawn_open_file_dialog(tx, OBS_CSV_OPEN_CONFIG, |p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("observations.csv")
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
            REGIME_CSV_SAVE_CONFIG,
            "markov_regime_export.csv",
            self.export_regime_csv(),
            |p, _| format!("Saved Markov regime CSV to {}", p.display()),
        );
    }

    pub(crate) fn trigger_save_json(&mut self) {
        match self.export_regime_json() {
            Ok(payload) => {
                let (tx, rx) = channel();
                self.io_state.save_rx = Some(rx);
                spawn_save_as_file_dialog(
                    tx,
                    REGIME_JSON_SAVE_CONFIG,
                    "markov_regime_export.json",
                    payload,
                    |p, _| format!("Saved Markov regime JSON to {}", p.display()),
                );
            }
            Err(err) => self.io_state.status = Some((err, true)),
        }
    }
}

// [cite:clinical_trials_statistics]

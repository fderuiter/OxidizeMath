//! Solver evaluation, async dialog polling, `InteractiveTool` integration, and starter presets
//! for [`super::MarkovSequenceWorkbenchTool`].
// [cite:clinical_trials_statistics]

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    CtmcComputed, DtmcComputed, HmmComputed, MarkovScenario, MarkovSequenceWorkbenchTool,
    basketball_epv_dtmc_config, birth_death_ctmc_config, gamblers_ruin_dtmc_config,
    hot_hand_hmm_config, market_regime_hmm_config, redundant_server_ctmc_config,
};
use crate::presets::{
    DtmcStateKind, MAX_OBSERVATION_SEQUENCE_LEN, MarkovWorkbenchMode, MarkovWorkbenchPresetConfig,
};
use crate::web_io::{
    OpenDialogConfig, PlatformInstant, SaveAsDialogConfig, spawn_open_file_dialog,
    spawn_save_as_file_dialog,
};
use eframe::egui;
use math_explorer::nalgebra::{DMatrix, DVector};
use math_explorer::pure_math::statistics::markov::{
    BaumWelchConfig, BaumWelchReport, ContinuousMarkovChain, HiddenMarkovModel, MarkovChain,
    StateType,
};
use oxidize_core::rng::OxidizeRng;
use std::sync::mpsc::{Receiver, channel};

const WORKBENCH_OPEN_CONFIG: OpenDialogConfig = OpenDialogConfig {
    filter_name: "Markov Sequence, Matrix, or Preset (.csv, .json)",
    extensions: &["csv", "json"],
    accept: ".csv,.json,text/csv,application/json",
    read_error_prefix: "Failed to read Markov file",
};

const CSV_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "CSV Spreadsheet (.csv)",
    extensions: &["csv"],
    fallback_filename: "markov_export.csv",
    write_error_prefix: "Failed to write CSV export",
};

const JSON_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "JSON Document (.json)",
    extensions: &["json"],
    fallback_filename: "markov_export.json",
    write_error_prefix: "Failed to write JSON export",
};

#[derive(Default)]
pub(crate) struct WorkbenchIoState {
    open_rx: Option<Receiver<Result<(String, String), String>>>,
    save_rx: Option<Receiver<Result<String, String>>>,
    pub status_message: Option<(bool, String)>,
}

impl WorkbenchIoState {
    pub(crate) fn trigger_import_dialog(&mut self) {
        let (tx, rx) = channel();
        self.open_rx = Some(rx);
        spawn_open_file_dialog(tx, WORKBENCH_OPEN_CONFIG, |p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("markov_input")
                .to_string()
        });
    }

    pub(crate) fn trigger_save_csv(&mut self, default_filename: &str, content: String) {
        let (tx, rx) = channel();
        self.save_rx = Some(rx);
        spawn_save_as_file_dialog(tx, CSV_SAVE_CONFIG, default_filename, content, |p, fb| {
            p.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(fb)
                .to_string()
        });
    }

    pub(crate) fn trigger_save_json(
        &mut self,
        default_filename: &str,
        content: Result<String, String>,
    ) {
        match content {
            Ok(payload) => {
                let (tx, rx) = channel();
                self.save_rx = Some(rx);
                spawn_save_as_file_dialog(
                    tx,
                    JSON_SAVE_CONFIG,
                    default_filename,
                    payload,
                    |p, fb| {
                        p.file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or(fb)
                            .to_string()
                    },
                );
            }
            Err(err) => self.status_message = Some((true, err)),
        }
    }
}

pub(crate) fn rows_to_dmatrix(rows: &[Vec<f64>]) -> DMatrix<f64> {
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, Vec::len);
    let mut flat = Vec::with_capacity(nrows * ncols);
    for r in rows {
        flat.extend_from_slice(r);
    }
    DMatrix::from_row_slice(nrows, ncols, &flat)
}

pub(crate) fn dmatrix_to_rows(m: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..m.nrows())
        .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect())
        .collect()
}

impl MarkovSequenceWorkbenchTool {
    pub(crate) fn poll_io(&mut self, ctx: &egui::Context) {
        if let Some(Ok(outcome)) = self.io_state.open_rx.as_ref().map(Receiver::try_recv) {
            self.io_state.open_rx = None;
            let status = match outcome {
                Ok((name, raw)) => match self.import_file_str(&raw) {
                    Ok(summary) => (false, format!("Imported {name}: {summary}")),
                    Err(err) => (true, format!("Import rejected ({name}): {err}")),
                },
                Err(err) => (true, err),
            };
            self.io_state.status_message = Some(status);
            ctx.request_repaint();
        }
        if let Some(Ok(outcome)) = self.io_state.save_rx.as_ref().map(Receiver::try_recv) {
            self.io_state.save_rx = None;
            self.io_state.status_message = Some(match outcome {
                Ok(saved) => (false, format!("Exported Markov artifact: {saved}")),
                Err(err) => (true, err),
            });
            ctx.request_repaint();
        }
    }

    pub(crate) fn build_hmm(&self) -> Result<HiddenMarkovModel<f64>, String> {
        let initial = DVector::from_vec(self.config.hmm.initial.clone());
        let transitions = rows_to_dmatrix(&self.config.hmm.transitions);
        let emissions = rows_to_dmatrix(&self.config.hmm.emissions);
        HiddenMarkovModel::new(initial, transitions, emissions).map_err(|e| e.to_string())
    }

    pub(crate) fn build_ctmc(&self) -> Result<ContinuousMarkovChain<f64>, String> {
        let g = rows_to_dmatrix(&self.config.ctmc.generator);
        ContinuousMarkovChain::new(g).map_err(|e| e.to_string())
    }

    pub(crate) fn build_dtmc(&self) -> Result<MarkovChain<f64>, String> {
        let p = rows_to_dmatrix(&self.config.dtmc.transition_matrix);
        let kinds = self
            .config
            .dtmc
            .state_types
            .iter()
            .map(|k| match k {
                DtmcStateKind::Transient => StateType::Transient,
                DtmcStateKind::Absorbing => StateType::Absorbing,
            })
            .collect();
        MarkovChain::new(p, kinds).map_err(|e| e.to_string())
    }

    fn compute_hmm_outputs(&mut self) -> Result<HmmComputed, String> {
        let hmm = self.build_hmm()?;
        let obs = &self.config.hmm.observations;
        self.vit_ws
            .ensure_capacity(hmm.num_states(), hmm.num_observations(), obs.len());
        let mut viterbi_path = vec![0_usize; obs.len()];
        hmm.viterbi_into(obs, &mut self.vit_ws, &mut viterbi_path)
            .map_err(|e| e.to_string())?;
        let gamma = hmm
            .posterior_probabilities(obs)
            .map_err(|e| e.to_string())?;
        let posteriors = (0..hmm.num_states())
            .map(|s| (0..obs.len()).map(|t| gamma[(s, t)]).collect())
            .collect();
        let filter_last = hmm
            .filter(obs)
            .map_err(|e| e.to_string())?
            .iter()
            .copied()
            .collect();
        let log_likelihood = hmm.log_likelihood(obs).map_err(|e| e.to_string())?;
        let forward_likelihood = hmm.forward(obs).map_err(|e| e.to_string())?;
        Ok(HmmComputed {
            viterbi_path,
            posteriors,
            filter_last,
            log_likelihood,
            forward_likelihood,
            bw_history: self.hmm_out.bw_history.clone(),
            last_bw_report: self.hmm_out.last_bw_report,
        })
    }

    fn compute_ctmc_outputs(&self) -> Result<CtmcComputed, String> {
        let chain = self.build_ctmc()?;
        let n = chain.num_states();
        let src = self.config.ctmc.source_state.min(n - 1);
        let samples = 64_usize;
        let mut sweep_times = Vec::with_capacity(samples);
        let mut sweep_probs = vec![Vec::with_capacity(samples); n];
        for step in 0..samples {
            let t = self.config.ctmc.horizon_time * (step as f64 / (samples - 1) as f64);
            let p_t = chain
                .transition_probabilities(t)
                .map_err(|e| e.to_string())?;
            sweep_times.push(t);
            for state_j in 0..n {
                sweep_probs[state_j].push(p_t[(src, state_j)]);
            }
        }
        let eval_mat = chain
            .transition_probabilities(self.config.ctmc.eval_time)
            .map_err(|e| e.to_string())?;
        let steady_state = chain.steady_state().map(|v| v.iter().copied().collect());
        let mttf = if self.config.ctmc.transient_states.is_empty() {
            None
        } else {
            Some(
                chain
                    .expected_absorption_times(&self.config.ctmc.transient_states)
                    .map_err(|e| e.to_string())?
                    .iter()
                    .copied()
                    .collect(),
            )
        };
        Ok(CtmcComputed {
            sweep_times,
            sweep_probs,
            eval_matrix: dmatrix_to_rows(&eval_mat),
            steady_state,
            mttf,
        })
    }

    fn compute_dtmc_outputs(&self) -> Result<DtmcComputed, String> {
        let chain = self.build_dtmc()?;
        let n_mat = chain.fundamental_matrix().map_err(|e| e.to_string())?;
        let b_mat = chain
            .absorption_probabilities()
            .map_err(|e| e.to_string())?;
        let steps = chain
            .expected_absorption_times()
            .map_err(|e| e.to_string())?;
        let rewards = DVector::from_vec(self.config.dtmc.absorbing_rewards.clone());
        let epv = chain
            .expected_possession_value(&rewards)
            .map_err(|e| e.to_string())?;
        let (mut transient_indices, mut absorbing_indices) = (Vec::new(), Vec::new());
        for (i, kind) in self.config.dtmc.state_types.iter().enumerate() {
            match kind {
                DtmcStateKind::Transient => transient_indices.push(i),
                DtmcStateKind::Absorbing => absorbing_indices.push(i),
            }
        }
        Ok(DtmcComputed {
            transient_indices,
            absorbing_indices,
            fundamental: dmatrix_to_rows(&n_mat),
            absorption_probs: dmatrix_to_rows(&b_mat),
            expected_steps: steps.iter().copied().collect(),
            epv: epv.iter().copied().collect(),
        })
    }

    pub(crate) fn recompute_all(&mut self) -> Result<(), String> {
        let start = PlatformInstant::now();
        let hmm_out = self.compute_hmm_outputs()?;
        let ctmc_out = self.compute_ctmc_outputs()?;
        let dtmc_out = self.compute_dtmc_outputs()?;
        self.hmm_out = hmm_out;
        self.ctmc_out = ctmc_out;
        self.dtmc_out = dtmc_out;
        self.last_compute_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    /// Runs zero-allocation Baum-Welch EM over the active HMM observation sequence.
    pub fn run_baum_welch(&mut self) -> Result<BaumWelchReport<f64>, String> {
        let mut hmm = self.build_hmm()?;
        let bw_cfg = BaumWelchConfig::new(
            self.config.hmm.bw_max_iterations,
            self.config.hmm.bw_tolerance,
            self.config.hmm.bw_dirichlet_alpha,
        )
        .map_err(|e| e.to_string())?;
        let obs_len = self.config.hmm.observations.len();
        self.bw_ws
            .ensure_capacity(hmm.num_states(), hmm.num_observations(), obs_len);
        let mut history = vec![0.0_f64; bw_cfg.max_iterations];
        let start = PlatformInstant::now();
        let report = hmm
            .fit_baum_welch_into(
                &self.config.hmm.observations,
                &bw_cfg,
                &mut self.bw_ws,
                &mut history,
            )
            .map_err(|e| e.to_string())?;
        history.truncate(report.iterations);
        self.config.hmm.initial = hmm.initial().iter().copied().collect();
        self.config.hmm.transitions = dmatrix_to_rows(hmm.transitions());
        self.config.hmm.emissions = dmatrix_to_rows(hmm.emissions());
        self.hmm_out.bw_history = history;
        self.hmm_out.last_bw_report = Some(report);
        self.recompute_all()?;
        self.last_compute_ms += start.elapsed().as_secs_f64() * 1000.0;
        self.state_rev = self.state_rev.wrapping_add(1);
        Ok(report)
    }

    /// Generates a deterministic synthetic HMM observation sequence using `OxidizeRng`.
    pub fn simulate_hmm_sequence(&mut self) -> Result<(), String> {
        let hmm = self.build_hmm()?;
        let len = self
            .config
            .hmm
            .sim_length
            .clamp(1, MAX_OBSERVATION_SEQUENCE_LEN);
        let mut rng = OxidizeRng::new(self.config.rng_seed);
        let (_, obs) = hmm.generate(len, &mut rng).map_err(|e| e.to_string())?;
        self.config.hmm.observations = obs;
        self.hmm_out.bw_history.clear();
        self.hmm_out.last_bw_report = None;
        self.recompute_all()?;
        self.state_rev = self.state_rev.wrapping_add(1);
        Ok(())
    }

    /// Loads one of the built-in reference scenarios from `markov_demo.rs`.
    pub fn load_scenario(&mut self, scenario: MarkovScenario) -> Result<(), String> {
        let mut next = self.config.clone();
        match scenario {
            MarkovScenario::MarketRegimeHmm => {
                next.mode = MarkovWorkbenchMode::Hmm;
                next.hmm = market_regime_hmm_config();
            }
            MarkovScenario::HotHandHmm => {
                next.mode = MarkovWorkbenchMode::Hmm;
                next.hmm = hot_hand_hmm_config();
            }
            MarkovScenario::RedundantServerCtmc => {
                next.mode = MarkovWorkbenchMode::Ctmc;
                next.ctmc = redundant_server_ctmc_config();
            }
            MarkovScenario::BirthDeathQueueCtmc => {
                next.mode = MarkovWorkbenchMode::Ctmc;
                next.ctmc = birth_death_ctmc_config();
            }
            MarkovScenario::BasketballEpvDtmc => {
                next.mode = MarkovWorkbenchMode::AbsorbingDtmc;
                next.dtmc = basketball_epv_dtmc_config();
            }
            MarkovScenario::GamblersRuinDtmc => {
                next.mode = MarkovWorkbenchMode::AbsorbingDtmc;
                next.dtmc = gamblers_ruin_dtmc_config();
            }
        }
        self.selected_scenario = scenario;
        self.apply_config(next)
    }

    pub(crate) fn apply_config(
        &mut self,
        config: MarkovWorkbenchPresetConfig,
    ) -> Result<(), String> {
        config.validate()?;
        let prev = self.config.clone();
        let prev_bw = (self.hmm_out.bw_history.clone(), self.hmm_out.last_bw_report);
        self.config = config;
        self.hmm_out.bw_history.clear();
        self.hmm_out.last_bw_report = None;
        if let Err(err) = self.recompute_all() {
            self.config = prev;
            (self.hmm_out.bw_history, self.hmm_out.last_bw_report) = prev_bw;
            return Err(err);
        }
        self.state_rev = self.state_rev.wrapping_add(1);
        Ok(())
    }
}

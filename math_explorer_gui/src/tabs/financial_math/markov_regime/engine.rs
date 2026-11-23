//! CTMC generator exponential / Gillespie solver and HMM Baum-Welch / Viterbi / posterior engine.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    MarkovRegimeState, MarkovRegimeTool, MAX_CTMC_STATES, MAX_HMM_STATES, MAX_HMM_SYMBOLS,
    MAX_IMPORTED_OBSERVATIONS,
};
use crate::web_io::strip_utf8_bom;
use egui_plot::PlotPoint;
use math_explorer::nalgebra::{DMatrix, DVector};
use math_explorer::pure_math::statistics::markov::ctmc::ContinuousMarkovChain;
use math_explorer::pure_math::statistics::markov::hmm::{BaumWelchReport, HiddenMarkovModel};
use oxidize_core::rng::OxidizeRng;

#[inline(never)]
fn normalize_prob_slice(vals: &[f64]) -> Vec<f64> {
    let clamped: Vec<f64> = vals
        .iter()
        .map(|&v| if v.is_finite() { v.max(0.0) } else { 0.0 })
        .collect();
    let sum: f64 = clamped.iter().sum();
    if sum > 1e-12 {
        clamped.into_iter().map(|v| v / sum).collect()
    } else {
        let u = 1.0 / clamped.len().max(1) as f64;
        vec![u; clamped.len()]
    }
}

impl MarkovRegimeTool {
    pub(crate) fn resize_ctmc(&mut self, n: usize) {
        let n = n.clamp(2, MAX_CTMC_STATES);
        self.ctmc_states = n;
        self.ctmc_rates.resize_with(n, || vec![0.25; n]);
        for (i, row) in self.ctmc_rates.iter_mut().enumerate() {
            row.resize(n, 0.25);
            row[i] = 0.0;
        }
        if self.ctmc_source_state >= n {
            self.ctmc_source_state = 0;
        }
    }

    pub(crate) fn resize_hmm(&mut self, states: usize, symbols: usize) {
        let s = states.clamp(2, MAX_HMM_STATES);
        let m = symbols.clamp(2, MAX_HMM_SYMBOLS);
        self.hmm_states = s;
        self.hmm_symbols = m;
        self.hmm_initial.resize(s, 1.0 / s as f64);
        self.hmm_initial = normalize_prob_slice(&self.hmm_initial);
        self.hmm_transitions
            .resize_with(s, || vec![1.0 / s as f64; s]);
        for row in &mut self.hmm_transitions {
            row.resize(s, 1.0 / s as f64);
            *row = normalize_prob_slice(row);
        }
        self.hmm_emissions
            .resize_with(s, || vec![1.0 / m as f64; m]);
        for row in &mut self.hmm_emissions {
            row.resize(m, 1.0 / m as f64);
            *row = normalize_prob_slice(row);
        }
        for obs in &mut self.observations {
            if *obs >= m {
                *obs %= m;
            }
        }
        self.bw_workspace
            .ensure_capacity(s, m, self.observations.len().max(64));
    }

    pub(crate) fn validate_state(s: &MarkovRegimeState) -> Result<(), String> {
        let dims_ok = (2..=MAX_CTMC_STATES).contains(&s.ctmc_states)
            && s.ctmc_source_state < s.ctmc_states
            && s.ctmc_rates.len() == s.ctmc_states
            && s.ctmc_rates.iter().all(|r| r.len() == s.ctmc_states)
            && (2..=MAX_HMM_STATES).contains(&s.hmm_states)
            && (2..=MAX_HMM_SYMBOLS).contains(&s.hmm_symbols)
            && s.hmm_initial.len() == s.hmm_states
            && s.hmm_transitions.len() == s.hmm_states
            && s.hmm_transitions.iter().all(|r| r.len() == s.hmm_states)
            && s.hmm_emissions.len() == s.hmm_states
            && s.hmm_emissions.iter().all(|r| r.len() == s.hmm_symbols)
            && !s.observations.is_empty()
            && s.observations.len() <= MAX_IMPORTED_OBSERVATIONS
            && s.observations.iter().all(|&o| o < s.hmm_symbols);
        if !dims_ok {
            return Err("Markov regime state dimensions or observations out of bounds".into());
        }
        let nums_ok = s.horizon_t.is_finite()
            && (0.2..=100.0).contains(&s.horizon_t)
            && s.bw_tolerance.is_finite()
            && (1e-10..=1e-1).contains(&s.bw_tolerance)
            && (1..=200).contains(&s.bw_max_iters)
            && s.ctmc_rates
                .iter()
                .flatten()
                .all(|&v| v.is_finite() && (0.0..=50.0).contains(&v))
            && s.hmm_initial.iter().all(|v| v.is_finite() && *v >= 0.0)
            && s.hmm_transitions
                .iter()
                .flatten()
                .all(|v| v.is_finite() && *v >= 0.0)
            && s.hmm_emissions
                .iter()
                .flatten()
                .all(|v| v.is_finite() && *v >= 0.0);
        if !nums_ok {
            return Err("Markov regime numeric parameters out of valid bounds".into());
        }
        Ok(())
    }

    pub(crate) fn apply_state(&mut self, s: MarkovRegimeState) -> Result<(), String> {
        Self::validate_state(&s)?;
        self.horizon_t = s.horizon_t;
        self.mode = s.mode;
        self.ctmc_states = s.ctmc_states;
        self.ctmc_source_state = s.ctmc_source_state;
        self.ctmc_rates = s.ctmc_rates;
        self.hmm_states = s.hmm_states;
        self.hmm_symbols = s.hmm_symbols;
        self.hmm_initial = normalize_prob_slice(&s.hmm_initial);
        self.hmm_transitions = s
            .hmm_transitions
            .iter()
            .map(|r| normalize_prob_slice(r))
            .collect();
        self.hmm_emissions = s
            .hmm_emissions
            .iter()
            .map(|r| normalize_prob_slice(r))
            .collect();
        self.observations = s.observations;
        self.bw_max_iters = s.bw_max_iters;
        self.bw_tolerance = s.bw_tolerance;
        self.rng_seed = s.rng_seed;
        self.bw_workspace.ensure_capacity(
            self.hmm_states,
            self.hmm_symbols,
            self.observations.len(),
        );
        self.bump_revision();
        self.recalculate();
        Ok(())
    }

    pub fn import_observations_csv(&mut self, raw: &str) -> Result<usize, String> {
        let body = strip_utf8_bom(raw);
        let mut parsed_obs = Vec::new();
        let mut saw_data = false;

        for (line_idx, line) in body.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            for token in trimmed.split([',', ';', '\t', ' ']).map(str::trim) {
                if token.is_empty() {
                    continue;
                }
                match token.parse::<usize>() {
                    Ok(sym) => {
                        saw_data = true;
                        if sym >= self.hmm_symbols {
                            return Err(format!(
                                "Observation symbol {sym} out of bounds for {} symbols",
                                self.hmm_symbols
                            ));
                        }
                        if parsed_obs.len() >= MAX_IMPORTED_OBSERVATIONS {
                            return Err(format!(
                                "Observation log exceeds maximum {} entries",
                                MAX_IMPORTED_OBSERVATIONS
                            ));
                        }
                        parsed_obs.push(sym);
                    }
                    Err(_) if !saw_data && line_idx == 0 => break,
                    Err(_) => {
                        return Err(format!(
                            "Invalid integer observation token '{}' on line {}",
                            token,
                            line_idx + 1
                        ));
                    }
                }
            }
        }
        if parsed_obs.is_empty() {
            return Err("CSV log contained no valid observation symbols".into());
        }
        self.observations = parsed_obs;
        self.bw_workspace.ensure_capacity(
            self.hmm_states,
            self.hmm_symbols,
            self.observations.len(),
        );
        self.bump_revision();
        self.recalculate();
        Ok(self.observations.len())
    }

    #[inline(never)]
    pub(crate) fn build_hmm_model(&self) -> Result<HiddenMarkovModel<f64>, String> {
        let init = DVector::from_vec(normalize_prob_slice(&self.hmm_initial));
        let mut trans_flat = Vec::with_capacity(self.hmm_states * self.hmm_states);
        for row in &self.hmm_transitions {
            trans_flat.extend(normalize_prob_slice(row));
        }
        let mut emit_flat = Vec::with_capacity(self.hmm_states * self.hmm_symbols);
        for row in &self.hmm_emissions {
            emit_flat.extend(normalize_prob_slice(row));
        }
        let trans = DMatrix::from_row_slice(self.hmm_states, self.hmm_states, &trans_flat);
        let emit = DMatrix::from_row_slice(self.hmm_states, self.hmm_symbols, &emit_flat);
        HiddenMarkovModel::new(init, trans, emit).map_err(|e| format!("Invalid HMM: {e}"))
    }

    pub fn run_baum_welch_fit(&mut self) -> Result<BaumWelchReport<f64>, String> {
        let mut hmm = self.build_hmm_model()?;
        self.bw_workspace.ensure_capacity(
            self.hmm_states,
            self.hmm_symbols,
            self.observations.len(),
        );
        let report = hmm
            .baum_welch_with_workspace(
                &self.observations,
                self.bw_max_iters,
                self.bw_tolerance,
                &mut self.bw_workspace,
            )
            .map_err(|e| format!("Baum-Welch failed: {e}"))?;

        self.hmm_initial = hmm.initial().iter().copied().collect();
        for i in 0..self.hmm_states {
            self.hmm_transitions[i] = hmm.transitions().row(i).iter().copied().collect();
            self.hmm_emissions[i] = hmm.emissions().row(i).iter().copied().collect();
        }
        self.last_bw_report = Some(report);
        self.bump_revision();
        self.recalculate();
        Ok(report)
    }

    fn sample_ctmc_gillespie(&mut self, chain: &ContinuousMarkovChain<f64>, src: usize, n: usize) {
        let mut rng = OxidizeRng::new(self.rng_seed);
        let Ok(jumps) = chain.simulate_trajectory(src, self.horizon_t, &mut rng) else {
            return;
        };
        let denom = (n.saturating_sub(1)).max(1) as f64;
        for win in jumps.windows(2) {
            let y = win[0].1 as f64 / denom;
            self.ctmc_gillespie_points.push(PlotPoint::new(win[0].0, y));
            self.ctmc_gillespie_points.push(PlotPoint::new(win[1].0, y));
        }
        if let Some(&(last_t, last_s)) = jumps.last() {
            let y = last_s as f64 / denom;
            self.ctmc_gillespie_points.push(PlotPoint::new(last_t, y));
            self.ctmc_gillespie_points
                .push(PlotPoint::new(self.horizon_t, y));
        }
    }

    #[inline(never)]
    fn rebuild_ctmc(&mut self) -> Result<(), String> {
        self.ctmc_prob_series.clear();
        self.ctmc_gillespie_points.clear();
        self.ctmc_absorption_points.clear();
        self.ctmc_absorption_times.clear();
        let n = self.ctmc_states;
        let gen = DMatrix::from_fn(n, n, |i, j| {
            if i == j {
                -self.ctmc_rates[i]
                    .iter()
                    .enumerate()
                    .filter(|&(col, _)| col != i)
                    .map(|(_, &r)| r.max(0.0))
                    .sum::<f64>()
            } else {
                self.ctmc_rates[i][j].max(0.0)
            }
        });

        let chain = ContinuousMarkovChain::new(gen).map_err(|e| format!("CTMC error: {e}"))?;
        self.ctmc_steady_state = chain.steady_state().map(|v| v.iter().copied().collect());

        let transient: Vec<usize> = (0..n)
            .filter(|&i| -chain.generator()[(i, i)] > 1e-9)
            .collect();
        let denom = (n.saturating_sub(1)).max(1) as f64;
        if !transient.is_empty() && transient.len() < n {
            if let Ok(abs_vec) = chain.expected_absorption_times(&transient) {
                self.ctmc_absorption_times =
                    transient.into_iter().zip(abs_vec.iter().copied()).collect();
                for &(st, mean_t) in &self.ctmc_absorption_times {
                    if mean_t.is_finite() && mean_t >= 0.0 {
                        self.ctmc_absorption_points
                            .push(PlotPoint::new(mean_t, st as f64 / denom));
                    }
                }
            }
        }

        let steps = 60_usize;
        let mut curves: Vec<Vec<PlotPoint>> = vec![Vec::with_capacity(steps + 1); n];
        let src = self.ctmc_source_state.min(n - 1);
        for k in 0..=steps {
            let t = self.horizon_t * (k as f64 / steps as f64);
            let p_t = chain
                .transition_probabilities(t)
                .map_err(|e| format!("exp(Gt) error: {e}"))?;
            for (dst, curve) in curves.iter_mut().enumerate() {
                curve.push(PlotPoint::new(t, p_t[(src, dst)]));
            }
        }
        for (dst, pts) in curves.into_iter().enumerate() {
            self.ctmc_prob_series
                .push((format!("P(X_t = S{} | X_0 = S{})", dst, src), pts));
        }

        self.sample_ctmc_gillespie(&chain, src, n);
        Ok(())
    }

    #[inline(never)]
    fn rebuild_hmm(&mut self) -> Result<(), String> {
        self.hmm_posterior_series.clear();
        self.viterbi_points.clear();
        self.viterbi_states.clear();

        let hmm = self.build_hmm_model()?;
        let gamma = hmm
            .posterior_probabilities(&self.observations)
            .map_err(|e| format!("Posterior error: {e}"))?;
        let viterbi = hmm
            .viterbi(&self.observations)
            .map_err(|e| format!("Viterbi error: {e}"))?;

        for state_idx in 0..self.hmm_states {
            let mut pts = Vec::with_capacity(self.observations.len());
            for t in 0..self.observations.len() {
                pts.push(PlotPoint::new(t as f64, gamma[(state_idx, t)]));
            }
            self.hmm_posterior_series
                .push((format!("Posterior γ_t(State {state_idx})"), pts));
        }
        let denom = (self.hmm_states.saturating_sub(1)).max(1) as f64;
        for (t, &st) in viterbi.iter().enumerate() {
            self.viterbi_points
                .push(PlotPoint::new(t as f64, st as f64 / denom));
        }
        self.viterbi_states = viterbi;
        Ok(())
    }

    pub fn recalculate(&mut self) {
        self.calc_error = None;
        if let Err(e) = self.rebuild_ctmc() {
            self.calc_error = Some(e);
            return;
        }
        if let Err(e) = self.rebuild_hmm() {
            self.calc_error = Some(e);
        }
    }
}

// [cite:clinical_trials_statistics]

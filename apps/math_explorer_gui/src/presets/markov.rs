//! Preset configuration models and validation routines for `MarkovSequenceWorkbenchTool`.
// [cite:clinical_trials_statistics]

use serde::{Deserialize, Serialize};

/// Maximum number of states permitted in interactive Markov presets.
pub const MAX_MARKOV_STATES: usize = 16;

/// Maximum number of observable symbols permitted in interactive HMM presets.
pub const MAX_HMM_SYMBOLS: usize = 16;

/// Maximum observation sequence length permitted in interactive HMM presets (`ToolStateBounds`).
pub const MAX_OBSERVATION_SEQUENCE_LEN: usize = 16_384;

/// Maximum Baum-Welch EM iterations permitted in interactive presets.
pub const MAX_BAUM_WELCH_ITERATIONS: usize = 500;

/// Active solver mode in `MarkovSequenceWorkbenchTool`.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MarkovWorkbenchMode {
    /// Hidden Markov Model regime decoding and Baum-Welch EM fitting.
    #[default]
    Hmm,
    /// Continuous-Time Markov Chain reliability, `exp(Gt)`, steady-state, and MTTF.
    Ctmc,
    /// Absorbing Discrete-Time Markov Chain fundamental matrix `N`, absorption, and EPV.
    AbsorbingDtmc,
}

impl MarkovWorkbenchMode {
    /// Human-readable label for mode tabs.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Hmm => "HMM Regimes & Baum-Welch",
            Self::Ctmc => "CTMC Reliability",
            Self::AbsorbingDtmc => "Absorbing DTMC Valuation",
        }
    }
}

/// State classification serialized inside [`AbsorbingDtmcWorkbenchConfig`].
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum DtmcStateKind {
    /// Transient state that eventually transitions out.
    Transient,
    /// Absorbing terminal state (`P[i, i] = 1`).
    Absorbing,
}

/// Hidden Markov Model sub-configuration for `MarkovSequenceWorkbenchTool`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HmmWorkbenchConfig {
    /// Display labels for each hidden state (`1..=MAX_MARKOV_STATES`).
    pub state_names: Vec<String>,
    /// Display labels for each observable symbol (`1..=MAX_HMM_SYMBOLS`).
    pub symbol_names: Vec<String>,
    /// Initial state distribution `\pi` summing to `1.0`.
    pub initial: Vec<f64>,
    /// Row-stochastic transition matrix `A` (`N x N`).
    pub transitions: Vec<Vec<f64>>,
    /// Row-stochastic emission matrix `B` (`N x M`).
    pub emissions: Vec<Vec<f64>>,
    /// Observed symbol sequence (`1..=MAX_OBSERVATION_SEQUENCE_LEN`).
    pub observations: Vec<usize>,
    /// Baum-Welch maximum EM iteration cap (`1..=MAX_BAUM_WELCH_ITERATIONS`).
    pub bw_max_iterations: usize,
    /// Baum-Welch log-likelihood convergence tolerance (`>= 0`).
    pub bw_tolerance: f64,
    /// Symmetric Dirichlet pseudocount smoothing (`>= 0`).
    pub bw_dirichlet_alpha: f64,
    /// Sequence length used when simulating synthetic trajectories.
    pub sim_length: usize,
}

/// Continuous-Time Markov Chain sub-configuration for `MarkovSequenceWorkbenchTool`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CtmcWorkbenchConfig {
    /// Display labels for each continuous-time state (`2..=MAX_MARKOV_STATES`).
    pub state_names: Vec<String>,
    /// Generator rate matrix `G` (`N x N`, rows sum to `0`, off-diagonals `>= 0`).
    pub generator: Vec<Vec<f64>>,
    /// Optional subset of transient state indices used for MTTF (`expected_absorption_times`).
    pub transient_states: Vec<usize>,
    /// Initial state index for `P_{i,j}(t) = [exp(Gt)]_{i,j}` sweep curves.
    pub source_state: usize,
    /// Upper horizon `T` for the `P(t)` sweep plot (`> 0`).
    pub horizon_time: f64,
    /// Point-in-time evaluation horizon `t_eval` (`>= 0`) for tabular `P(t_eval)` reports.
    pub eval_time: f64,
}

/// Absorbing Discrete-Time Markov Chain sub-configuration for `MarkovSequenceWorkbenchTool`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AbsorbingDtmcWorkbenchConfig {
    /// Display labels for each state (`2..=MAX_MARKOV_STATES`).
    pub state_names: Vec<String>,
    /// Row-stochastic transition matrix `P` (`N x N`).
    pub transition_matrix: Vec<Vec<f64>>,
    /// Transient vs. absorbing classification for each state.
    pub state_types: Vec<DtmcStateKind>,
    /// Terminal payoff/reward assigned to each absorbing state (in state index order).
    pub absorbing_rewards: Vec<f64>,
}

/// Full serialized preset configuration for `MarkovSequenceWorkbenchTool`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MarkovWorkbenchPresetConfig {
    /// Active workbench mode.
    pub mode: MarkovWorkbenchMode,
    /// Deterministic seed for `OxidizeRng`.
    pub rng_seed: u64,
    /// HMM regime inference and Baum-Welch configuration.
    pub hmm: HmmWorkbenchConfig,
    /// CTMC reliability and generator configuration.
    pub ctmc: CtmcWorkbenchConfig,
    /// Absorbing DTMC and EPV configuration.
    pub dtmc: AbsorbingDtmcWorkbenchConfig,
}

fn validate_labels(labels: &[String], min: usize, max: usize, kind: &str) -> Result<(), String> {
    if labels.len() < min || labels.len() > max {
        return Err(format!(
            "{kind} count must be in [{min}, {max}] (got {})",
            labels.len()
        ));
    }
    for (idx, name) in labels.iter().enumerate() {
        if name.trim().is_empty() || name.len() > 64 {
            return Err(format!(
                "{kind} label at index {idx} must be non-empty and <= 64 chars"
            ));
        }
    }
    Ok(())
}

fn validate_stochastic_rows(
    rows: &[Vec<f64>],
    expected_rows: usize,
    expected_cols: usize,
    label: &str,
) -> Result<(), String> {
    if rows.len() != expected_rows {
        return Err(format!(
            "{label} row count ({}) must equal {expected_rows}",
            rows.len()
        ));
    }
    for (r_idx, row) in rows.iter().enumerate() {
        if row.len() != expected_cols {
            return Err(format!(
                "{label} row {r_idx} length ({}) must equal {expected_cols}",
                row.len()
            ));
        }
        let mut sum = 0.0;
        for &p in row {
            if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                return Err(format!(
                    "{label} contains invalid probability {p} in row {r_idx}"
                ));
            }
            sum += p;
        }
        if (sum - 1.0).abs() > 1e-6 {
            return Err(format!(
                "{label} row {r_idx} sums to {sum:.6} instead of 1.0"
            ));
        }
    }
    Ok(())
}

fn validate_hmm(hmm: &HmmWorkbenchConfig) -> Result<(), String> {
    validate_labels(&hmm.state_names, 1, MAX_MARKOV_STATES, "HMM state")?;
    validate_labels(&hmm.symbol_names, 1, MAX_HMM_SYMBOLS, "HMM symbol")?;
    let (n, m) = (hmm.state_names.len(), hmm.symbol_names.len());
    validate_stochastic_rows(std::slice::from_ref(&hmm.initial), 1, n, "HMM initial")?;
    validate_stochastic_rows(&hmm.transitions, n, n, "HMM transitions")?;
    validate_stochastic_rows(&hmm.emissions, n, m, "HMM emissions")?;
    if hmm.observations.is_empty() || hmm.observations.len() > MAX_OBSERVATION_SEQUENCE_LEN {
        return Err(format!(
            "HMM observations length ({}) must be in [1, {MAX_OBSERVATION_SEQUENCE_LEN}]",
            hmm.observations.len()
        ));
    }
    if let Some(&bad) = hmm.observations.iter().find(|&&o| o >= m) {
        return Err(format!(
            "HMM observation symbol {bad} out of bounds for {m} symbols"
        ));
    }
    if hmm.bw_max_iterations == 0 || hmm.bw_max_iterations > MAX_BAUM_WELCH_ITERATIONS {
        return Err(format!(
            "bw_max_iterations must be in [1, {MAX_BAUM_WELCH_ITERATIONS}]"
        ));
    }
    if !hmm.bw_tolerance.is_finite() || !(0.0..=1.0).contains(&hmm.bw_tolerance) {
        return Err("bw_tolerance must be finite in [0.0, 1.0]".to_string());
    }
    if !hmm.bw_dirichlet_alpha.is_finite() || !(0.0..=10.0).contains(&hmm.bw_dirichlet_alpha) {
        return Err("bw_dirichlet_alpha must be finite in [0.0, 10.0]".to_string());
    }
    if hmm.sim_length == 0 || hmm.sim_length > MAX_OBSERVATION_SEQUENCE_LEN {
        return Err(format!(
            "sim_length must be in [1, {MAX_OBSERVATION_SEQUENCE_LEN}]"
        ));
    }
    Ok(())
}

fn validate_ctmc_generator_row(i: usize, row: &[f64], n: usize) -> Result<(), String> {
    if row.len() != n {
        return Err(format!(
            "CTMC generator row {i} length ({}) must equal {n}",
            row.len()
        ));
    }
    let mut sum = 0.0;
    for (j, &rate) in row.iter().enumerate() {
        if !rate.is_finite() || rate.abs() > 1e4 {
            return Err(format!(
                "CTMC rate G[{i},{j}] must be finite within [-10000, 10000]"
            ));
        }
        if i == j && rate > 1e-6 {
            return Err(format!("CTMC diagonal G[{i},{i}] = {rate} must be <= 0"));
        }
        if i != j && rate < -1e-6 {
            return Err(format!(
                "CTMC off-diagonal G[{i},{j}] = {rate} must be >= 0"
            ));
        }
        sum += rate;
    }
    if sum.abs() > 1e-6 {
        return Err(format!(
            "CTMC generator row {i} sums to {sum:.6} instead of 0.0"
        ));
    }
    Ok(())
}

fn validate_ctmc(ctmc: &CtmcWorkbenchConfig) -> Result<(), String> {
    validate_labels(&ctmc.state_names, 2, MAX_MARKOV_STATES, "CTMC state")?;
    let n = ctmc.state_names.len();
    if ctmc.generator.len() != n {
        return Err(format!(
            "CTMC generator row count ({}) must equal {n}",
            ctmc.generator.len()
        ));
    }
    for (i, row) in ctmc.generator.iter().enumerate() {
        validate_ctmc_generator_row(i, row, n)?;
    }
    if ctmc.source_state >= n {
        return Err(format!(
            "CTMC source_state {} out of bounds",
            ctmc.source_state
        ));
    }
    if !ctmc.horizon_time.is_finite() || !(0.05..=1000.0).contains(&ctmc.horizon_time) {
        return Err("CTMC horizon_time must be finite in [0.05, 1000.0]".to_string());
    }
    if !ctmc.eval_time.is_finite() || !(0.0..=1000.0).contains(&ctmc.eval_time) {
        return Err("CTMC eval_time must be finite in [0.0, 1000.0]".to_string());
    }
    if ctmc.transient_states.len() >= n {
        return Err(
            "CTMC transient_states must leave at least one absorbing state when non-empty"
                .to_string(),
        );
    }
    for (pos, &s) in ctmc.transient_states.iter().enumerate() {
        if s >= n || ctmc.transient_states[..pos].contains(&s) {
            return Err(format!(
                "Invalid or duplicate CTMC transient state index {s}"
            ));
        }
    }
    Ok(())
}

fn validate_dtmc(dtmc: &AbsorbingDtmcWorkbenchConfig) -> Result<(), String> {
    validate_labels(&dtmc.state_names, 2, MAX_MARKOV_STATES, "DTMC state")?;
    let n = dtmc.state_names.len();
    validate_stochastic_rows(&dtmc.transition_matrix, n, n, "DTMC transition_matrix")?;
    if dtmc.state_types.len() != n {
        return Err(format!(
            "DTMC state_types length ({}) must equal {n}",
            dtmc.state_types.len()
        ));
    }
    let mut num_transient = 0;
    let mut num_absorbing = 0;
    for (i, &kind) in dtmc.state_types.iter().enumerate() {
        match kind {
            DtmcStateKind::Transient => num_transient += 1,
            DtmcStateKind::Absorbing => {
                num_absorbing += 1;
                for (j, &p) in dtmc.transition_matrix[i].iter().enumerate() {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    if (p - expected).abs() > 1e-6 {
                        return Err(format!(
                            "Absorbing state {i} must satisfy P[{i},{j}] = {expected}"
                        ));
                    }
                }
            }
        }
    }
    if num_transient == 0 || num_absorbing == 0 {
        return Err(
            "Absorbing DTMC requires at least one transient and one absorbing state".to_string(),
        );
    }
    if dtmc.absorbing_rewards.len() != num_absorbing {
        return Err(format!(
            "absorbing_rewards length ({}) must match absorbing state count ({num_absorbing})",
            dtmc.absorbing_rewards.len()
        ));
    }
    if dtmc
        .absorbing_rewards
        .iter()
        .any(|r| !r.is_finite() || !(-1e6..=1e6).contains(r))
    {
        return Err("absorbing_rewards must be finite in [-1e6, 1e6]".to_string());
    }
    Ok(())
}

impl MarkovWorkbenchPresetConfig {
    /// Validates all HMM, CTMC, and Absorbing DTMC preset invariants.
    pub fn validate(&self) -> Result<(), String> {
        validate_hmm(&self.hmm)?;
        validate_ctmc(&self.ctmc)?;
        validate_dtmc(&self.dtmc)
    }
}

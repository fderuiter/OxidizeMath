//! Unified client-side Markov Sequence & Reliability Workbench (`MarkovSequenceWorkbenchTool`).
// [cite:clinical_trials_statistics]

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::presets::{
    AbsorbingDtmcWorkbenchConfig, CtmcWorkbenchConfig, DtmcStateKind, HmmWorkbenchConfig,
    MarkovWorkbenchMode, MarkovWorkbenchPresetConfig,
};
use math_explorer::pure_math::statistics::markov::{
    BaumWelchReport, BaumWelchWorkspace, ViterbiWorkspace,
};

pub(crate) mod export;
pub(crate) mod io;
mod plots;
mod registry;
mod state;
mod ui;

#[cfg(test)]
mod tests;

/// Active plot lens for HMM Regime & Baum-Welch mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HmmPlotView {
    /// Time-by-state posterior heatmap with Viterbi decoded regime trajectory overlay.
    #[default]
    PosteriorHeatmap,
    /// Per-state posterior probability curves `\gamma_t(i)` and Viterbi step line.
    PosteriorLines,
    /// Monotonic Baum-Welch EM log-likelihood progression across iterations.
    LogLikelihoodCurve,
}

/// Active plot lens for CTMC Reliability mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CtmcPlotView {
    /// Continuous transition probabilities `P_{i,j}(t) = [exp(Gt)]_{i,j}` over `[0, T]`.
    #[default]
    TransitionSweep,
    /// Mean Time To Failure (`expected_absorption_times`) or ergodic steady-state bars.
    MttfAndSteadyState,
}

/// Active plot lens for Absorbing DTMC Valuation mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DtmcPlotView {
    /// Expected Possession Value (EPV) and terminal absorption probabilities `B = NR`.
    #[default]
    EpvAndAbsorption,
    /// Fundamental matrix `N = (I - Q)^{-1}` expected visit counts per transient state.
    FundamentalVisits,
}

/// Built-in reference scenarios adapted from `math_explorer/examples/markov_demo.rs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkovScenario {
    /// Three-state Bull/Bear/Sideways market regime HMM (`markov_demo.rs:L142-L169`).
    MarketRegimeHmm,
    /// Two-state Cold/Hot basketball shooting HMM (`markov_demo.rs:L66-L108`).
    HotHandHmm,
    /// Three-state redundant server reliability & MTTF CTMC.
    RedundantServerCtmc,
    /// Three-state birth-death service queue CTMC (`markov_demo.rs:L111-L139`).
    BirthDeathQueueCtmc,
    /// Four-state basketball possession EPV absorbing DTMC (`markov_demo.rs:L18-L63`).
    BasketballEpvDtmc,
    /// Five-state symmetric Gambler's Ruin absorbing DTMC (`markov_demo.rs:L171-L179`).
    GamblersRuinDtmc,
}

impl MarkovScenario {
    /// All built-in scenarios displayed in the workbench quick-loader.
    pub const ALL: [Self; 6] = [
        Self::MarketRegimeHmm,
        Self::HotHandHmm,
        Self::RedundantServerCtmc,
        Self::BirthDeathQueueCtmc,
        Self::BasketballEpvDtmc,
        Self::GamblersRuinDtmc,
    ];

    /// User-facing scenario label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::MarketRegimeHmm => "Market Regime HMM (3-State)",
            Self::HotHandHmm => "Hot-Hand Shooting HMM (2-State)",
            Self::RedundantServerCtmc => "Redundant Server MTTF (CTMC)",
            Self::BirthDeathQueueCtmc => "Birth-Death Queue (CTMC)",
            Self::BasketballEpvDtmc => "Basketball Possession EPV (DTMC)",
            Self::GamblersRuinDtmc => "Gambler's Ruin $0..$4 (DTMC)",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct HmmComputed {
    pub viterbi_path: Vec<usize>,
    pub posteriors: Vec<Vec<f64>>,
    pub filter_last: Vec<f64>,
    pub log_likelihood: f64,
    pub forward_likelihood: f64,
    pub bw_history: Vec<f64>,
    pub last_bw_report: Option<BaumWelchReport<f64>>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct CtmcComputed {
    pub sweep_times: Vec<f64>,
    pub sweep_probs: Vec<Vec<f64>>,
    pub eval_matrix: Vec<Vec<f64>>,
    pub steady_state: Option<Vec<f64>>,
    pub mttf: Option<Vec<f64>>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct DtmcComputed {
    pub transient_indices: Vec<usize>,
    pub absorbing_indices: Vec<usize>,
    pub fundamental: Vec<Vec<f64>>,
    pub absorption_probs: Vec<Vec<f64>>,
    pub expected_steps: Vec<f64>,
    pub epv: Vec<f64>,
}

/// Interactive Markov sequence, Baum-Welch EM, CTMC reliability, and Absorbing DTMC workbench.
pub struct MarkovSequenceWorkbenchTool {
    pub(crate) config: MarkovWorkbenchPresetConfig,
    pub(crate) hmm_plot_view: HmmPlotView,
    pub(crate) ctmc_plot_view: CtmcPlotView,
    pub(crate) dtmc_plot_view: DtmcPlotView,
    pub(crate) selected_scenario: MarkovScenario,
    pub(crate) hmm_out: HmmComputed,
    pub(crate) ctmc_out: CtmcComputed,
    pub(crate) dtmc_out: DtmcComputed,
    pub(crate) vit_ws: ViterbiWorkspace<f64>,
    pub(crate) bw_ws: BaumWelchWorkspace<f64>,
    pub(crate) last_compute_ms: f64,
    pub(crate) state_rev: u64,
    pub(crate) io_state: state::WorkbenchIoState,
}

pub(crate) fn market_regime_hmm_config() -> HmmWorkbenchConfig {
    HmmWorkbenchConfig {
        state_names: vec!["Bull".into(), "Bear".into(), "Sideways".into()],
        symbol_names: vec![
            "Large Up".into(),
            "Small Up".into(),
            "Flat".into(),
            "Small Down".into(),
            "Large Down".into(),
        ],
        initial: vec![0.33, 0.33, 0.34],
        transitions: vec![
            vec![0.70, 0.20, 0.10],
            vec![0.20, 0.70, 0.10],
            vec![0.25, 0.25, 0.50],
        ],
        emissions: vec![
            vec![0.30, 0.30, 0.20, 0.10, 0.10],
            vec![0.10, 0.10, 0.20, 0.30, 0.30],
            vec![0.10, 0.20, 0.40, 0.20, 0.10],
        ],
        observations: vec![1, 0, 0, 1, 2, 2, 3, 4, 4, 3, 2, 1],
        bw_max_iterations: 40,
        bw_tolerance: 1e-6,
        bw_dirichlet_alpha: 1e-4,
        sim_length: 80,
    }
}

pub(crate) fn hot_hand_hmm_config() -> HmmWorkbenchConfig {
    HmmWorkbenchConfig {
        state_names: vec!["Cold".into(), "Hot".into()],
        symbol_names: vec!["Miss".into(), "Make".into()],
        initial: vec![0.60, 0.40],
        transitions: vec![vec![0.75, 0.25], vec![0.30, 0.70]],
        emissions: vec![vec![0.65, 0.35], vec![0.25, 0.75]],
        observations: vec![1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1],
        bw_max_iterations: 40,
        bw_tolerance: 1e-6,
        bw_dirichlet_alpha: 1e-4,
        sim_length: 80,
    }
}

pub(crate) fn redundant_server_ctmc_config() -> CtmcWorkbenchConfig {
    CtmcWorkbenchConfig {
        state_names: vec![
            "Dual Healthy".into(),
            "Degraded (1 Up)".into(),
            "System Failed".into(),
        ],
        generator: vec![
            vec![-0.40, 0.40, 0.00],
            vec![1.20, -1.50, 0.30],
            vec![0.00, 0.00, 0.00],
        ],
        transient_states: vec![0, 1],
        source_state: 0,
        horizon_time: 12.0,
        eval_time: 3.0,
    }
}

pub(crate) fn birth_death_ctmc_config() -> CtmcWorkbenchConfig {
    CtmcWorkbenchConfig {
        state_names: vec!["Idle (0)".into(), "Busy (1)".into(), "Queued (2)".into()],
        generator: vec![
            vec![-2.0, 2.0, 0.0],
            vec![3.0, -5.0, 2.0],
            vec![0.0, 3.0, -3.0],
        ],
        transient_states: Vec::new(),
        source_state: 0,
        horizon_time: 4.0,
        eval_time: 1.0,
    }
}

pub(crate) fn basketball_epv_dtmc_config() -> AbsorbingDtmcWorkbenchConfig {
    AbsorbingDtmcWorkbenchConfig {
        state_names: vec![
            "Perimeter".into(),
            "Paint Advantage".into(),
            "Score (+2)".into(),
            "Turnover (0)".into(),
        ],
        transition_matrix: vec![
            vec![0.45, 0.30, 0.10, 0.15],
            vec![0.15, 0.35, 0.38, 0.12],
            vec![0.00, 0.00, 1.00, 0.00],
            vec![0.00, 0.00, 0.00, 1.00],
        ],
        state_types: vec![
            DtmcStateKind::Transient,
            DtmcStateKind::Transient,
            DtmcStateKind::Absorbing,
            DtmcStateKind::Absorbing,
        ],
        absorbing_rewards: vec![2.0, 0.0],
    }
}

pub(crate) fn gamblers_ruin_dtmc_config() -> AbsorbingDtmcWorkbenchConfig {
    AbsorbingDtmcWorkbenchConfig {
        state_names: vec![
            "$0 Ruin".into(),
            "$1 Stake".into(),
            "$2 Stake".into(),
            "$3 Stake".into(),
            "$4 Target".into(),
        ],
        transition_matrix: vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.5, 0.0, 0.5, 0.0, 0.0],
            vec![0.0, 0.5, 0.0, 0.5, 0.0],
            vec![0.0, 0.0, 0.5, 0.0, 0.5],
            vec![0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        state_types: vec![
            DtmcStateKind::Absorbing,
            DtmcStateKind::Transient,
            DtmcStateKind::Transient,
            DtmcStateKind::Transient,
            DtmcStateKind::Absorbing,
        ],
        absorbing_rewards: vec![0.0, 4.0],
    }
}

pub(crate) fn default_workbench_config() -> MarkovWorkbenchPresetConfig {
    MarkovWorkbenchPresetConfig {
        mode: MarkovWorkbenchMode::Hmm,
        rng_seed: 2026,
        hmm: market_regime_hmm_config(),
        ctmc: redundant_server_ctmc_config(),
        dtmc: basketball_epv_dtmc_config(),
    }
}

impl Default for MarkovSequenceWorkbenchTool {
    fn default() -> Self {
        let cfg = default_workbench_config();
        let mut tool = Self {
            vit_ws: ViterbiWorkspace::new(
                cfg.hmm.state_names.len(),
                cfg.hmm.symbol_names.len(),
                cfg.hmm.observations.len(),
            ),
            bw_ws: BaumWelchWorkspace::new(
                cfg.hmm.state_names.len(),
                cfg.hmm.symbol_names.len(),
                cfg.hmm.observations.len(),
            ),
            config: cfg.clone(),
            hmm_plot_view: HmmPlotView::PosteriorHeatmap,
            ctmc_plot_view: CtmcPlotView::TransitionSweep,
            dtmc_plot_view: DtmcPlotView::EpvAndAbsorption,
            selected_scenario: MarkovScenario::MarketRegimeHmm,
            hmm_out: HmmComputed::default(),
            ctmc_out: CtmcComputed::default(),
            dtmc_out: DtmcComputed::default(),
            last_compute_ms: 0.0,
            state_rev: 1,
            io_state: state::WorkbenchIoState::default(),
        };
        let _ = tool.apply_config(cfg);
        tool
    }
}

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::atlas::ParameterAtlasController;
use crate::framework::CalloutManager;
use math_explorer::pure_math::analysis::stochastic::GillespieSolver;
pub use math_explorer::pure_math::statistics::markov::parrondo::{
    advance_parrondo_mix_walkers, advance_parrondo_walkers, compute_parrondo_jump_metrics,
    parrondo_variant_win_probs, ParrondoGillespieSystem, ParrondoJumpMetrics as ParrondoMetrics,
    ParrondoSsaState, ParrondoVariant, ParrondoWalker,
};
use math_explorer::rng::OxidizeRng;
use serde::{Deserialize, Serialize};

/// Schedule highlighted against the analytical `MarkovChain` stationary slope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParrondoSchedule {
    /// Play losing Game A exclusively.
    PureA,
    /// Play state-dependent losing Game B exclusively.
    PureB,
    /// Deterministic periodic `ABB` ratchet schedule.
    PeriodicABB,
    /// Independent coin-flip mixture (`gamma * A + (1 - gamma) * B`).
    RandomMix,
}

/// Serializable bounded configuration for `ParrondoParadoxTool`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParrondoState {
    /// Losing bias parameter `epsilon` subtracted from fair coin thresholds.
    pub bias_epsilon: f64,
    /// Probability `gamma` of selecting Game A in the random mixture.
    pub mix_gamma: f64,
    /// Continuous-time switching rate `lambda` for `GillespieSolver`.
    pub jump_switch_rate: f64,
    /// Game variant (`CapitalDependent` or `HistoryDependent`).
    pub variant: ParrondoVariant,
    /// Schedule used for the analytical stationary slope overlay.
    pub focus_schedule: ParrondoSchedule,
    /// Number of discrete rounds simulated.
    pub steps: usize,
    /// Number of ensemble walkers averaged per schedule.
    pub walkers: usize,
    /// Deterministic `OxidizeRng` seed.
    pub seed: u64,
}

impl Default for ParrondoState {
    fn default() -> Self {
        Self {
            bias_epsilon: 0.005,
            mix_gamma: 0.5,
            jump_switch_rate: 1.2,
            variant: ParrondoVariant::CapitalDependent,
            focus_schedule: ParrondoSchedule::PeriodicABB,
            steps: 120,
            walkers: 256,
            seed: 42,
        }
    }
}

impl ParrondoState {
    pub(super) fn validate(&self) -> Result<(), String> {
        let valid_eps = self.bias_epsilon.is_finite() && (0.0..=0.20).contains(&self.bias_epsilon);
        let valid_mix = self.mix_gamma.is_finite() && (0.0..=1.0).contains(&self.mix_gamma);
        let valid_ssa =
            self.jump_switch_rate.is_finite() && (0.05..=10.0).contains(&self.jump_switch_rate);
        if !valid_eps || !valid_mix || !valid_ssa {
            return Err("Parrondo float parameters out of bounds".into());
        }
        if !(10..=600).contains(&self.steps) || !(16..=4096).contains(&self.walkers) {
            return Err("steps must be in [10, 600] and walkers in [16, 4096]".into());
        }
        Ok(())
    }
}

/// Interactive Parrondo's Paradox laboratory tool.
pub struct ParrondoParadoxTool {
    pub(super) state: ParrondoState,
    pub(super) metrics: ParrondoMetrics,
    pub(super) revision: u64,
    pub(super) current_step: usize,
    rng: OxidizeRng,
    walkers_a: Vec<ParrondoWalker>,
    walkers_b: Vec<ParrondoWalker>,
    walkers_abb: Vec<ParrondoWalker>,
    walkers_mix: Vec<ParrondoWalker>,
    ssa_solver: GillespieSolver<OxidizeRng>,
    ssa_system: ParrondoGillespieSystem,
    ssa_state: ParrondoSsaState,
    ssa_time: f64,
    pub(super) traj_a: Vec<[f64; 2]>,
    pub(super) traj_b: Vec<[f64; 2]>,
    pub(super) traj_abb: Vec<[f64; 2]>,
    pub(super) traj_mix: Vec<[f64; 2]>,
    pub(super) traj_exact: Vec<[f64; 2]>,
    pub(super) traj_ssa: Vec<[f64; 2]>,
    /// Progressive 2D parameter regime atlas controller.
    pub atlas: ParameterAtlasController,
    /// Plot callout overlay manager synchronized with stories, challenges, and atlas teleports.
    pub callouts: CalloutManager,
}

/// Canonical disambiguated type alias for [`crate::framework::ToolMetadata::id`] `"ParrondoJumpRatchetTool"`.
pub type ParrondoJumpRatchetTool = ParrondoParadoxTool;

impl Default for ParrondoParadoxTool {
    fn default() -> Self {
        let state = ParrondoState::default();
        let (win_a, win_b, _) = parrondo_variant_win_probs(state.variant, state.bias_epsilon);
        let mut tool = Self {
            metrics: ParrondoMetrics::default(),
            revision: 1,
            current_step: 0,
            rng: OxidizeRng::new(state.seed),
            walkers_a: Vec::new(),
            walkers_b: Vec::new(),
            walkers_abb: Vec::new(),
            walkers_mix: Vec::new(),
            ssa_solver: GillespieSolver::with_capacity(
                OxidizeRng::new(state.seed).with_stream(1),
                8,
            ),
            ssa_system: ParrondoGillespieSystem {
                win_a,
                win_b,
                variant: state.variant,
                switch_ab: state.jump_switch_rate * 1.6,
                switch_ba: state.jump_switch_rate * 0.8,
            },
            ssa_state: ParrondoSsaState::default(),
            ssa_time: 0.0,
            traj_a: Vec::new(),
            traj_b: Vec::new(),
            traj_abb: Vec::new(),
            traj_mix: Vec::new(),
            traj_exact: Vec::new(),
            traj_ssa: Vec::new(),
            atlas: ParameterAtlasController::new(vec![
                ("bias_epsilon", "mix_gamma"),
                ("bias_epsilon", "jump_switch_rate"),
            ]),
            callouts: CalloutManager::new(),
            state,
        };
        tool.rebuild_full_simulation();
        tool
    }
}

impl ParrondoParadoxTool {
    /// Returns analytical stationary-distribution drift metrics for all four schedules.
    pub fn metrics(&self) -> ParrondoMetrics {
        self.metrics
    }

    /// Pre-allocates walker and trajectory buffers at step 0 for zero-allocation stepping.
    pub fn prepare_stepping(&mut self) {
        self.metrics = compute_parrondo_jump_metrics(
            self.state.variant,
            self.state.bias_epsilon,
            self.state.mix_gamma,
        )
        .unwrap_or_default();
        let (win_a, win_b, _) =
            parrondo_variant_win_probs(self.state.variant, self.state.bias_epsilon);
        self.ssa_system = ParrondoGillespieSystem {
            win_a,
            win_b,
            variant: self.state.variant,
            switch_ab: self.state.jump_switch_rate * 1.6,
            switch_ba: self.state.jump_switch_rate * 0.8,
        };
        self.rng = OxidizeRng::new(self.state.seed);
        self.ssa_solver =
            GillespieSolver::with_capacity(OxidizeRng::new(self.state.seed).with_stream(1), 8);
        self.ssa_state = ParrondoSsaState::default();
        self.ssa_time = 0.0;
        self.current_step = 0;

        for wbuf in [
            &mut self.walkers_a,
            &mut self.walkers_b,
            &mut self.walkers_abb,
            &mut self.walkers_mix,
        ] {
            wbuf.clear();
            wbuf.resize(self.state.walkers, ParrondoWalker::default());
        }
        let cap = self.state.steps + 1;
        for tbuf in [
            &mut self.traj_a,
            &mut self.traj_b,
            &mut self.traj_abb,
            &mut self.traj_mix,
            &mut self.traj_exact,
            &mut self.traj_ssa,
        ] {
            tbuf.clear();
            tbuf.resize(cap, [0.0, 0.0]);
        }
    }

    /// Advances all ensembles and `GillespieSolver` by one round with zero heap allocations.
    pub fn step_parrondo_zero_alloc(&mut self) {
        if self.current_step >= self.state.steps {
            return;
        }
        let next_step = self.current_step + 1;
        let (variant, win_a, win_b) = (
            self.state.variant,
            self.ssa_system.win_a,
            self.ssa_system.win_b,
        );
        let mean_a = advance_parrondo_walkers(&mut self.walkers_a, &mut self.rng, variant, &win_a);
        let mean_b = advance_parrondo_walkers(&mut self.walkers_b, &mut self.rng, variant, &win_b);
        let abb_probs = if self.current_step.is_multiple_of(3) {
            &win_a
        } else {
            &win_b
        };
        let mean_abb =
            advance_parrondo_walkers(&mut self.walkers_abb, &mut self.rng, variant, abb_probs);
        let mean_mix = advance_parrondo_mix_walkers(
            &mut self.walkers_mix,
            &mut self.rng,
            variant,
            &win_a,
            &win_b,
            self.state.mix_gamma,
        );

        let target_time = next_step as f64;
        let mut jumps = 0usize;
        while self.ssa_time < target_time && jumps < 32 {
            if let Ok(tau) = self.ssa_solver.step(&self.ssa_system, &mut self.ssa_state) {
                if !tau.is_finite() {
                    break;
                }
                self.ssa_time += tau;
            }
            jumps += 1;
        }

        let x = next_step as f64;
        self.traj_a[next_step] = [x, mean_a];
        self.traj_b[next_step] = [x, mean_b];
        self.traj_abb[next_step] = [x, mean_abb];
        self.traj_mix[next_step] = [x, mean_mix];
        self.traj_exact[next_step] = [x, x * self.focus_slope()];
        self.traj_ssa[next_step] = [x, self.ssa_state.capital as f64];
        self.current_step = next_step;
    }

    pub(super) fn rebuild_full_simulation(&mut self) {
        self.prepare_stepping();
        while self.current_step < self.state.steps {
            self.step_parrondo_zero_alloc();
        }
        self.revision = self.revision.wrapping_add(1);
    }
}

// [cite:paradox_lab]

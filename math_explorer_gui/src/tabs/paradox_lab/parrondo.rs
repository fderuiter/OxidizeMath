#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use math_explorer::nalgebra::DMatrix;
use math_explorer::pure_math::analysis::stochastic::{
    GillespieSolver, StochasticError, StochasticSystem,
};
use math_explorer::pure_math::statistics::markov::dtmc::{MarkovChain, StateType};
use math_explorer::pure_math::statistics::markov::tensor::{TimeIndex, TransitionTensor};
use math_explorer::rand::Rng;
use math_explorer::rng::OxidizeRng;
use serde::{Deserialize, Serialize};

/// Parrondo ratchet state-space formulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParrondoVariant {
    /// Modulo-3 capital-dependent coin biases (`X_t mod 3`).
    CapitalDependent,
    /// Four-state two-step outcome history (`LL`, `LW`, `WL`, `WW`).
    HistoryDependent,
}

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

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Walker {
    pub capital: i32,
    pub state: u8,
}

/// Stack-allocated continuous-time state evolved by `GillespieSolver`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ParrondoSsaState {
    /// Cumulative bankroll.
    pub capital: i64,
    /// Current Markov state index.
    pub markov_state: usize,
    /// Active game (`0` = Game A, `1` = Game B).
    pub active_game: u8,
}

/// Zero-allocation continuous-time jump ratchet implementing `StochasticSystem`.
#[derive(Clone, Copy, Debug)]
pub struct ParrondoGillespieSystem {
    pub(super) win_a: [f64; 4],
    pub(super) win_b: [f64; 4],
    pub(super) variant: ParrondoVariant,
    pub(super) switch_ab: f64,
    pub(super) switch_ba: f64,
}

impl StochasticSystem<ParrondoSsaState> for ParrondoGillespieSystem {
    fn propensities(&self, state: &ParrondoSsaState, out: &mut Vec<f64>) {
        let idx = state.markov_state.min(3);
        let p_win = if state.active_game == 0 {
            self.win_a[idx]
        } else {
            self.win_b[idx]
        };
        out.push(p_win);
        out.push((1.0 - p_win).max(0.0));
        out.push(if state.active_game == 0 {
            self.switch_ab
        } else {
            0.0
        });
        out.push(if state.active_game == 1 {
            self.switch_ba
        } else {
            0.0
        });
    }

    fn react(&self, state: &mut ParrondoSsaState, idx: usize) -> Result<(), StochasticError> {
        match idx {
            0 => {
                state.capital += 1;
                state.markov_state = next_markov_state(self.variant, state.markov_state, true);
                Ok(())
            }
            1 => {
                state.capital -= 1;
                state.markov_state = next_markov_state(self.variant, state.markov_state, false);
                Ok(())
            }
            2 => {
                state.active_game = 1;
                Ok(())
            }
            3 => {
                state.active_game = 0;
                Ok(())
            }
            other => Err(StochasticError::InvalidReactionIndex(other)),
        }
    }
}

/// Analytical drift summary computed from `MarkovChain` and `TransitionTensor`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ParrondoMetrics {
    /// Exact stationary drift per turn for pure Game A.
    pub drift_a: f64,
    /// Exact stationary drift per turn for pure Game B.
    pub drift_b: f64,
    /// Exact stationary drift per turn for periodic `ABB`.
    pub drift_abb: f64,
    /// Exact stationary drift per turn for random mixture.
    pub drift_mix: f64,
    /// Entry `(0, 1)` of `TransitionTensor::average_transition` across `[0, 2]`.
    pub tensor_avg_p01: f64,
    /// Mid-cycle interpolated entry `(0, 1)` at `TimeIndex(0.5)`.
    pub tensor_mid_p01: f64,
}

/// Interactive Parrondo's Paradox laboratory tool.
pub struct ParrondoParadoxTool {
    pub(super) state: ParrondoState,
    pub(super) metrics: ParrondoMetrics,
    pub(super) revision: u64,
    pub(super) current_step: usize,
    rng: OxidizeRng,
    walkers_a: Vec<Walker>,
    walkers_b: Vec<Walker>,
    walkers_abb: Vec<Walker>,
    walkers_mix: Vec<Walker>,
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
}

#[inline]
fn next_markov_state(variant: ParrondoVariant, cur: usize, won: bool) -> usize {
    match variant {
        ParrondoVariant::CapitalDependent => (cur + if won { 1 } else { 2 }) % 3,
        ParrondoVariant::HistoryDependent => {
            usize::from(cur == 1 || cur == 3) * 2 + usize::from(won)
        }
    }
}

fn win_probs(variant: ParrondoVariant, eps: f64) -> ([f64; 4], [f64; 4], usize) {
    let win_a = [(0.5 - eps).clamp(0.01, 0.99); 4];
    match variant {
        ParrondoVariant::CapitalDependent => {
            let b0 = (0.10 - eps).clamp(0.01, 0.99);
            let b1 = (0.75 - eps).clamp(0.01, 0.99);
            (win_a, [b0, b1, b1, b1], 3)
        }
        ParrondoVariant::HistoryDependent => (
            win_a,
            [
                (0.90 - eps).clamp(0.01, 0.99),
                (0.25 - eps).clamp(0.01, 0.99),
                (0.25 - eps).clamp(0.01, 0.99),
                (0.70 - eps).clamp(0.01, 0.99),
            ],
            4,
        ),
    }
}

fn build_transition_matrix(variant: ParrondoVariant, probs: &[f64; 4], n: usize) -> DMatrix<f64> {
    let mut mat = DMatrix::zeros(n, n);
    for s in 0..n {
        mat[(s, next_markov_state(variant, s, true))] += probs[s];
        mat[(s, next_markov_state(variant, s, false))] += 1.0 - probs[s];
    }
    mat
}

fn stationary_drift(mat: &DMatrix<f64>, probs: &[f64; 4]) -> f64 {
    let n = mat.nrows();
    let chain = MarkovChain::new(mat.clone(), vec![StateType::Transient; n]).unwrap();
    let pi = chain.stationary_distribution().unwrap();
    (0..n).map(|s| pi[s] * (2.0 * probs[s] - 1.0)).sum()
}

fn compute_abb_tensor_metrics(
    p_a: &DMatrix<f64>,
    p_b: &DMatrix<f64>,
    win_a: &[f64; 4],
    win_b: &[f64; 4],
) -> (f64, f64, f64) {
    let n = p_a.nrows();
    let (t_min, t_mid, t_one, t_max) = (
        TimeIndex::new(0.0).unwrap(),
        TimeIndex::new(0.5).unwrap(),
        TimeIndex::new(1.0).unwrap(),
        TimeIndex::new(2.0).unwrap(),
    );
    let mut tensor = TransitionTensor::new(n, t_min, t_max);
    tensor.add_time_slice(t_min, p_a.clone()).unwrap();
    tensor.add_time_slice(t_one, p_b.clone()).unwrap();
    tensor.add_time_slice(t_max, p_b.clone()).unwrap();

    let s0 = tensor.transition_matrix_at(t_min).unwrap();
    let s1 = tensor.transition_matrix_at(t_one).unwrap();
    let s2 = tensor.transition_matrix_at(t_max).unwrap();
    let chain_abb = MarkovChain::new(&s0 * &s1 * &s2, vec![StateType::Transient; n]).unwrap();
    let pi0 = chain_abb.stationary_distribution().unwrap();
    let pi1 = s0.transpose() * &pi0;
    let pi2 = s1.transpose() * &pi1;
    let gain = |pi: &math_explorer::nalgebra::DVector<f64>, w: &[f64; 4]| -> f64 {
        (0..n).map(|s| pi[s] * (2.0 * w[s] - 1.0)).sum()
    };
    let drift_abb = (gain(&pi0, win_a) + gain(&pi1, win_b) + gain(&pi2, win_b)) / 3.0;
    let avg = tensor.average_transition(t_min, t_max, 3).unwrap();
    let mid = tensor.transition_matrix_at(t_mid).unwrap();
    (drift_abb, avg[(0, 1)], mid[(0, 1)])
}

fn compute_analytical_metrics(state: &ParrondoState) -> ParrondoMetrics {
    let (win_a, win_b, n) = win_probs(state.variant, state.bias_epsilon);
    let mut win_mix = [0.0; 4];
    for i in 0..n {
        win_mix[i] = state.mix_gamma * win_a[i] + (1.0 - state.mix_gamma) * win_b[i];
    }
    let p_a = build_transition_matrix(state.variant, &win_a, n);
    let p_b = build_transition_matrix(state.variant, &win_b, n);
    let p_mix = build_transition_matrix(state.variant, &win_mix, n);
    let (drift_abb, tensor_avg_p01, tensor_mid_p01) =
        compute_abb_tensor_metrics(&p_a, &p_b, &win_a, &win_b);
    ParrondoMetrics {
        drift_a: stationary_drift(&p_a, &win_a),
        drift_b: stationary_drift(&p_b, &win_b),
        drift_abb,
        drift_mix: stationary_drift(&p_mix, &win_mix),
        tensor_avg_p01,
        tensor_mid_p01,
    }
}

impl Default for ParrondoParadoxTool {
    fn default() -> Self {
        let state = ParrondoState::default();
        let (win_a, win_b, _) = win_probs(state.variant, state.bias_epsilon);
        let mut tool = Self {
            metrics: ParrondoMetrics::default(),
            revision: 1,
            current_step: 0,
            rng: OxidizeRng::new(state.seed),
            walkers_a: Vec::new(),
            walkers_b: Vec::new(),
            walkers_abb: Vec::new(),
            walkers_mix: Vec::new(),
            ssa_solver: GillespieSolver::with_capacity(OxidizeRng::new(state.seed ^ 0x9E37), 8),
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
        self.metrics = compute_analytical_metrics(&self.state);
        let (win_a, win_b, _) = win_probs(self.state.variant, self.state.bias_epsilon);
        self.ssa_system = ParrondoGillespieSystem {
            win_a,
            win_b,
            variant: self.state.variant,
            switch_ab: self.state.jump_switch_rate * 1.6,
            switch_ba: self.state.jump_switch_rate * 0.8,
        };
        self.rng = OxidizeRng::new(self.state.seed);
        self.ssa_solver =
            GillespieSolver::with_capacity(OxidizeRng::new(self.state.seed ^ 0x9E37), 8);
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
            wbuf.resize(self.state.walkers, Walker::default());
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
        let mean_a = advance_walkers(&mut self.walkers_a, &mut self.rng, variant, &win_a);
        let mean_b = advance_walkers(&mut self.walkers_b, &mut self.rng, variant, &win_b);
        let abb_probs = if self.current_step.is_multiple_of(3) {
            &win_a
        } else {
            &win_b
        };
        let mean_abb = advance_walkers(&mut self.walkers_abb, &mut self.rng, variant, abb_probs);
        let mean_mix = advance_mix_walkers(
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

fn advance_walkers(
    walkers: &mut [Walker],
    rng: &mut OxidizeRng,
    variant: ParrondoVariant,
    probs: &[f64; 4],
) -> f64 {
    let mut sum = 0.0;
    for w in walkers.iter_mut() {
        let won = rng.r#gen::<f64>() < probs[usize::from(w.state).min(3)];
        w.capital += if won { 1 } else { -1 };
        w.state = next_markov_state(variant, usize::from(w.state), won) as u8;
        sum += f64::from(w.capital);
    }
    sum / (walkers.len().max(1) as f64)
}

fn advance_mix_walkers(
    walkers: &mut [Walker],
    rng: &mut OxidizeRng,
    variant: ParrondoVariant,
    win_a: &[f64; 4],
    win_b: &[f64; 4],
    gamma: f64,
) -> f64 {
    let mut sum = 0.0;
    for w in walkers.iter_mut() {
        let idx = usize::from(w.state).min(3);
        let p = if rng.r#gen::<f64>() < gamma {
            win_a[idx]
        } else {
            win_b[idx]
        };
        let won = rng.r#gen::<f64>() < p;
        w.capital += if won { 1 } else { -1 };
        w.state = next_markov_state(variant, idx, won) as u8;
        sum += f64::from(w.capital);
    }
    sum / (walkers.len().max(1) as f64)
}

// [cite:paradox_lab]

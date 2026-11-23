//! Preallocated zero-allocation simulation pipeline and cooperative `SimulationRunner` for `ClosedLoopControlTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::controllers::{
    DiscreteQController, GrpoController, GRPO_GROUP_SIZE, ROLLOUT_HORIZON,
};
use super::plants::{
    sample_standard_normal, LorenzPlantTwin, PkPlantTwin, PlantStepOutput, SeirPlantTwin,
};
pub use super::runner::{ClosedLoopRunner, ClosedLoopTrainingSnapshot};
use super::state::{ClosedLoopState, ControllerKind, PlantKind};
use math_explorer::pure_math::analysis::ode::TimeStepper;
use math_explorer::rng::OxidizeRng;

/// Fixed ring capacity for synchronized time-series and phase-portrait traces.
pub const HISTORY_CAPACITY: usize = 160;
/// Maximum stored episodic training return points.
pub const MAX_RETURN_HISTORY: usize = 128;
const INITIAL_PREVIEW_STEPS: usize = 72;

/// Zero-allocation closed-loop EKF + RL simulation and training pipeline.
pub struct ClosedLoopPipeline {
    /// Live interactive Lorenz twin.
    pub live_lorenz: LorenzPlantTwin,
    /// Live interactive SEIR twin.
    pub live_seir: SeirPlantTwin,
    /// Live interactive ADME pharmacokinetic twin.
    pub live_pk: PkPlantTwin,
    rollout_lorenz: LorenzPlantTwin,
    rollout_seir: SeirPlantTwin,
    rollout_pk: PkPlantTwin,
    /// Critic-free GRPO Gaussian controller.
    pub grpo: GrpoController,
    /// Discretized Q-learning controller (`FixedArrayQFunction<64, 7>`).
    pub q_ctrl: DiscreteQController,
    live_rng: OxidizeRng,
    train_rng: OxidizeRng,
    /// Total completed training episodes for the active configuration.
    pub episodes_trained: u32,
    /// Latest live plant + EKF telemetry sample.
    pub last_output: PlantStepOutput,
    /// Last normalized `[dx, dy]` kick vector applied on the canvas, if any.
    pub last_kick_vec: Option<[f64; 2]>,
    /// Uncontrolled open-loop primary trajectory `[t, y_open]`.
    pub uncontrolled_trace: Vec<[f64; 2]>,
    /// True controlled primary trajectory `[t, y_ctrl]`.
    pub controlled_trace: Vec<[f64; 2]>,
    /// Noisy scalar sensor measurements `[t, z_k]`.
    pub noisy_meas_trace: Vec<[f64; 2]>,
    /// EKF posterior primary estimate `[t, y_hat]`.
    pub ekf_primary_trace: Vec<[f64; 2]>,
    /// EKF `+1σ` upper uncertainty bound `[t, y_hat + sigma]`.
    pub ekf_upper_trace: Vec<[f64; 2]>,
    /// EKF `-1σ` lower uncertainty bound `[t, y_hat - sigma]`.
    pub ekf_lower_trace: Vec<[f64; 2]>,
    /// Preallocated closed polygon vertices for the EKF `±1σ` confidence band.
    pub ekf_band_polygon: Vec<[f64; 2]>,
    /// True hidden/latent secondary trajectory `[t, latent_true]`.
    pub true_latent_trace: Vec<[f64; 2]>,
    /// EKF hidden/latent secondary estimate `[t, latent_hat]`.
    pub ekf_latent_trace: Vec<[f64; 2]>,
    /// Bounded control actuation trajectory `[t, u_t]`.
    pub actuation_trace: Vec<[f64; 2]>,
    /// EKF measurement innovation residual `[t, z_k - h(x_pred)]`.
    pub innovation_trace: Vec<[f64; 2]>,
    /// 2D phase trace of the uncontrolled twin.
    pub phase_open_trace: Vec<[f64; 2]>,
    /// 2D phase trace of the true controlled plant.
    pub phase_ctrl_trace: Vec<[f64; 2]>,
    /// 2D phase trace of the EKF posterior state.
    pub phase_ekf_trace: Vec<[f64; 2]>,
    /// Episodic training return curve `[episode, return]`.
    pub return_history: Vec<[f64; 2]>,
}

impl ClosedLoopPipeline {
    /// Allocates all simulation buffers up front and runs `cfg.warmup_episodes` + preview steps.
    pub fn new(cfg: &ClosedLoopState) -> Self {
        let mut pipeline = Self {
            live_lorenz: LorenzPlantTwin::new(cfg),
            live_seir: SeirPlantTwin::new(cfg),
            live_pk: PkPlantTwin::new(cfg),
            rollout_lorenz: LorenzPlantTwin::new(cfg),
            rollout_seir: SeirPlantTwin::new(cfg),
            rollout_pk: PkPlantTwin::new(cfg),
            grpo: GrpoController::new_untrained(),
            q_ctrl: DiscreteQController::new(cfg),
            live_rng: OxidizeRng::new(cfg.rng_seed),
            train_rng: OxidizeRng::new(cfg.rng_seed ^ 0x517C_C1B7_2722_0A95),
            episodes_trained: 0,
            last_output: zero_step_output(),
            last_kick_vec: None,
            uncontrolled_trace: Vec::with_capacity(HISTORY_CAPACITY),
            controlled_trace: Vec::with_capacity(HISTORY_CAPACITY),
            noisy_meas_trace: Vec::with_capacity(HISTORY_CAPACITY),
            ekf_primary_trace: Vec::with_capacity(HISTORY_CAPACITY),
            ekf_upper_trace: Vec::with_capacity(HISTORY_CAPACITY),
            ekf_lower_trace: Vec::with_capacity(HISTORY_CAPACITY),
            ekf_band_polygon: Vec::with_capacity(2 * HISTORY_CAPACITY),
            true_latent_trace: Vec::with_capacity(HISTORY_CAPACITY),
            ekf_latent_trace: Vec::with_capacity(HISTORY_CAPACITY),
            actuation_trace: Vec::with_capacity(HISTORY_CAPACITY),
            innovation_trace: Vec::with_capacity(HISTORY_CAPACITY),
            phase_open_trace: Vec::with_capacity(HISTORY_CAPACITY),
            phase_ctrl_trace: Vec::with_capacity(HISTORY_CAPACITY),
            phase_ekf_trace: Vec::with_capacity(HISTORY_CAPACITY),
            return_history: Vec::with_capacity(MAX_RETURN_HISTORY),
        };
        pipeline.rebuild_all(cfg);
        pipeline
    }

    /// Resets controllers, twins, and traces, executes `cfg.warmup_episodes`, and populates preview traces.
    pub fn rebuild_all(&mut self, cfg: &ClosedLoopState) {
        self.reset_untrained(cfg);
        for _ in 0..cfg.warmup_episodes {
            self.train_episode_zero_alloc(cfg);
        }
        self.reset_live_traces_only(cfg);
        for _ in 0..INITIAL_PREVIEW_STEPS {
            let _ = self.step_zero_alloc(cfg);
        }
    }

    /// Resets policies to zero/untrained initialization and clears live traces without heap allocations.
    pub fn reset_untrained(&mut self, cfg: &ClosedLoopState) {
        self.grpo.reset_untrained();
        self.q_ctrl.reset_untrained(cfg);
        self.train_rng = OxidizeRng::new(cfg.rng_seed ^ 0x517C_C1B7_2722_0A95);
        self.episodes_trained = 0;
        self.return_history.clear();
        self.reset_live_traces_only(cfg);
    }

    /// Resets only the live twin and trace buffers while preserving learned policy parameters.
    pub fn reset_live_traces_only(&mut self, cfg: &ClosedLoopState) {
        self.live_lorenz.reset_in_place([0.0, 0.0, 0.0]);
        self.live_seir.reset_in_place(0.0);
        self.live_pk.reset_in_place(0.0);
        self.live_rng = OxidizeRng::new(cfg.rng_seed);
        self.last_kick_vec = None;
        self.clear_trace_buffers();
    }

    fn clear_trace_buffers(&mut self) {
        self.uncontrolled_trace.clear();
        self.controlled_trace.clear();
        self.noisy_meas_trace.clear();
        self.ekf_primary_trace.clear();
        self.ekf_upper_trace.clear();
        self.ekf_lower_trace.clear();
        self.ekf_band_polygon.clear();
        self.true_latent_trace.clear();
        self.ekf_latent_trace.clear();
        self.actuation_trace.clear();
        self.innovation_trace.clear();
        self.phase_open_trace.clear();
        self.phase_ctrl_trace.clear();
        self.phase_ekf_trace.clear();
    }

    /// Advances the active live plant + EKF + controller by one step with zero heap allocations.
    pub fn step_zero_alloc(&mut self, cfg: &ClosedLoopState) -> PlantStepOutput {
        let (phi, bin) = self.live_features_and_bin(cfg);
        let raw_action = match cfg.controller {
            ControllerKind::GrpoPolicy => self.grpo.act_deterministic(&phi),
            ControllerKind::DiscreteQLearning => self.q_ctrl.act_greedy(bin),
        };
        let (out, open_latent) = match cfg.plant {
            PlantKind::LorenzChaos => {
                let step_out = self.live_lorenz.step(raw_action, cfg, &mut self.live_rng);
                (step_out, self.live_lorenz.uncontrolled.state.vec.y)
            }
            PlantKind::SeirEpidemic => {
                let step_out = self.live_seir.step(raw_action, cfg, &mut self.live_rng);
                (step_out, self.live_seir.uncontrolled.get_state().e)
            }
            PlantKind::Pharmacokinetics => {
                let step_out = self.live_pk.step(raw_action, cfg, &mut self.live_rng);
                (step_out, 0.28 * 10.0)
            }
        };
        self.record_step_telemetry(out, open_latent);
        self.last_output = out;
        out
    }

    /// Applies an interactive perturbation kick to the controlled plant and steps once to update EKF innovation.
    pub fn apply_state_kick(
        &mut self,
        dx_norm: f64,
        dy_norm: f64,
        cfg: &ClosedLoopState,
    ) -> PlantStepOutput {
        let mag = cfg.kick_magnitude.clamp(0.05, 3.0);
        let eff_dx = if dx_norm.abs() < 0.20 {
            if dx_norm >= 0.0 { 0.45 } else { -0.45 }
        } else {
            dx_norm.clamp(-1.0, 1.0)
        };
        let eff_dy = dy_norm.clamp(-1.0, 1.0);
        match cfg.plant {
            PlantKind::LorenzChaos => {
                let v = &mut self.live_lorenz.controlled.state.vec;
                v.x = (v.x + 6.0 * mag * eff_dx).clamp(-45.0, 45.0);
                v.y = (v.y + 6.0 * mag * eff_dy).clamp(-45.0, 45.0);
            }
            PlantKind::SeirEpidemic => {
                let st = self.live_seir.controlled.get_state_mut();
                let add_i = (42.0 * mag * eff_dx.abs()).min(st.s * 0.45);
                st.s -= add_i;
                st.i += add_i;
                let add_e = (50.0 * mag * eff_dy.abs().max(0.25)).min(st.s * 0.45);
                st.s -= add_e;
                st.e += add_e;
            }
            PlantKind::Pharmacokinetics => {
                let delta_cp = 1.4 * mag * eff_dx;
                self.live_pk.controlled_state[1] =
                    (self.live_pk.controlled_state[1] + delta_cp).clamp(0.05, 12.0);
                self.live_pk.controlled_state[0] =
                    (self.live_pk.controlled_state[0] + 22.0 * mag * eff_dy.max(0.0)).max(0.0);
            }
        }
        self.last_kick_vec = Some([eff_dx, eff_dy]);
        self.step_zero_alloc(cfg)
    }

    /// Executes one full zero-allocation RL training episode (GRPO group rollout or Q-learning trajectory).
    pub fn train_episode_zero_alloc(&mut self, cfg: &ClosedLoopState) -> f64 {
        let ep_return = match cfg.controller {
            ControllerKind::GrpoPolicy => self.run_grpo_group_episode(cfg),
            ControllerKind::DiscreteQLearning => self.run_q_learning_episode(cfg),
        };
        self.episodes_trained += 1;
        push_ring(
            &mut self.return_history,
            MAX_RETURN_HISTORY,
            [self.episodes_trained as f64, ep_return],
        );
        ep_return
    }

    fn run_grpo_group_episode(&mut self, cfg: &ClosedLoopState) -> f64 {
        let j0 = 0.25 * sample_standard_normal(&mut self.train_rng);
        let j1 = 0.25 * sample_standard_normal(&mut self.train_rng);
        for g in 0..GRPO_GROUP_SIZE {
            self.reset_rollout_plant(cfg.plant, j0, j1);
            let mut total_r = 0.0;
            for t in 0..ROLLOUT_HORIZON {
                let (phi, _) = self.rollout_features_and_bin(cfg);
                let act = self.grpo.sample_and_record(g, t, &phi, &mut self.train_rng);
                let step_out = self.step_rollout_plant(act, cfg);
                total_r += step_out.reward;
            }
            self.grpo.group_returns[g] = total_r;
        }
        self.grpo.finish_group_update(cfg);
        self.grpo.last_mean_return
    }

    fn run_q_learning_episode(&mut self, cfg: &ClosedLoopState) -> f64 {
        let j0 = 0.20 * sample_standard_normal(&mut self.train_rng);
        let j1 = 0.20 * sample_standard_normal(&mut self.train_rng);
        self.reset_rollout_plant(cfg.plant, j0, j1);
        let mut total_r = 0.0;
        let (_, mut bin) = self.rollout_features_and_bin(cfg);
        for _ in 0..ROLLOUT_HORIZON {
            let (disc_act, raw_act) = self.q_ctrl.select_exploratory_action(bin);
            let step_out = self.step_rollout_plant(raw_act, cfg);
            let (_, next_bin) = self.rollout_features_and_bin(cfg);
            self.q_ctrl
                .update_transition(bin, disc_act, step_out.reward, next_bin);
            total_r += step_out.reward;
            bin = next_bin;
        }
        self.q_ctrl.last_episode_return = total_r;
        total_r
    }

    fn reset_rollout_plant(&mut self, plant: PlantKind, j0: f64, j1: f64) {
        match plant {
            PlantKind::LorenzChaos => self.rollout_lorenz.reset_in_place([j0, j1, -0.5 * j0]),
            PlantKind::SeirEpidemic => self.rollout_seir.reset_in_place(j0 * 15.0),
            PlantKind::Pharmacokinetics => self.rollout_pk.reset_in_place(j0 * 0.02),
        }
    }

    fn rollout_features_and_bin(&self, cfg: &ClosedLoopState) -> ([f64; 4], usize) {
        match cfg.plant {
            PlantKind::LorenzChaos => self.rollout_lorenz.features_and_bin(),
            PlantKind::SeirEpidemic => self.rollout_seir.features_and_bin(),
            PlantKind::Pharmacokinetics => self.rollout_pk.features_and_bin(cfg),
        }
    }

    fn step_rollout_plant(&mut self, raw_action: f64, cfg: &ClosedLoopState) -> PlantStepOutput {
        match cfg.plant {
            PlantKind::LorenzChaos => {
                self.rollout_lorenz.step(raw_action, cfg, &mut self.train_rng)
            }
            PlantKind::SeirEpidemic => {
                self.rollout_seir.step(raw_action, cfg, &mut self.train_rng)
            }
            PlantKind::Pharmacokinetics => {
                self.rollout_pk.step(raw_action, cfg, &mut self.train_rng)
            }
        }
    }

    fn live_features_and_bin(&self, cfg: &ClosedLoopState) -> ([f64; 4], usize) {
        match cfg.plant {
            PlantKind::LorenzChaos => self.live_lorenz.features_and_bin(),
            PlantKind::SeirEpidemic => self.live_seir.features_and_bin(),
            PlantKind::Pharmacokinetics => self.live_pk.features_and_bin(cfg),
        }
    }

    fn record_step_telemetry(&mut self, out: PlantStepOutput, open_latent: f64) {
        let t = out.time;
        push_ring(&mut self.uncontrolled_trace, HISTORY_CAPACITY, [t, out.uncontrolled_primary]);
        push_ring(&mut self.controlled_trace, HISTORY_CAPACITY, [t, out.controlled_primary]);
        push_ring(&mut self.noisy_meas_trace, HISTORY_CAPACITY, [t, out.noisy_measurement]);
        push_ring(&mut self.ekf_primary_trace, HISTORY_CAPACITY, [t, out.ekf_primary]);
        push_ring(
            &mut self.ekf_upper_trace,
            HISTORY_CAPACITY,
            [t, out.ekf_primary + out.ekf_primary_std],
        );
        push_ring(
            &mut self.ekf_lower_trace,
            HISTORY_CAPACITY,
            [t, out.ekf_primary - out.ekf_primary_std],
        );
        push_ring(&mut self.true_latent_trace, HISTORY_CAPACITY, [t, out.true_latent]);
        push_ring(&mut self.ekf_latent_trace, HISTORY_CAPACITY, [t, out.ekf_latent]);
        push_ring(&mut self.actuation_trace, HISTORY_CAPACITY, [t, out.actuation]);
        push_ring(&mut self.innovation_trace, HISTORY_CAPACITY, [t, out.innovation]);
        self.record_phase_and_band(out, open_latent);
    }

    fn record_phase_and_band(&mut self, out: PlantStepOutput, open_latent: f64) {
        push_ring(
            &mut self.phase_open_trace,
            HISTORY_CAPACITY,
            [out.uncontrolled_primary, open_latent],
        );
        push_ring(
            &mut self.phase_ctrl_trace,
            HISTORY_CAPACITY,
            [out.controlled_primary, out.true_latent],
        );
        push_ring(
            &mut self.phase_ekf_trace,
            HISTORY_CAPACITY,
            [out.ekf_primary, out.ekf_latent],
        );
        self.rebuild_band_polygon_zero_alloc();
    }

    fn rebuild_band_polygon_zero_alloc(&mut self) {
        self.ekf_band_polygon.clear();
        for &pt in &self.ekf_upper_trace {
            self.ekf_band_polygon.push(pt);
        }
        for &pt in self.ekf_lower_trace.iter().rev() {
            self.ekf_band_polygon.push(pt);
        }
    }

    /// Synchronizes learned controller weights and Q-table from a background worker snapshot.
    pub fn apply_training_snapshot(&mut self, snap: &ClosedLoopTrainingSnapshot) {
        self.episodes_trained = snap.episodes_trained;
        self.grpo.weights = snap.grpo_weights;
        self.grpo.last_mean_return = snap.grpo_mean_return;
        self.grpo.last_surrogate_before = snap.grpo_surrogate_before;
        self.grpo.last_surrogate_after = snap.grpo_surrogate_after;
        self.q_ctrl.agent.q_func.copy_from_table(&snap.q_table);
        self.q_ctrl.last_episode_return = snap.q_last_return;
        self.return_history.clear();
        for &pt in snap.return_history.iter().take(MAX_RETURN_HISTORY) {
            self.return_history.push(pt);
        }
    }
}

fn push_ring(buf: &mut Vec<[f64; 2]>, cap: usize, pt: [f64; 2]) {
    if buf.len() < cap {
        buf.push(pt);
    } else if cap > 0 {
        buf.copy_within(1..cap, 0);
        buf[cap - 1] = pt;
    }
}

const fn zero_step_output() -> PlantStepOutput {
    PlantStepOutput {
        time: 0.0,
        uncontrolled_primary: 0.0,
        controlled_primary: 0.0,
        noisy_measurement: 0.0,
        ekf_primary: 0.0,
        ekf_primary_std: 0.0,
        true_latent: 0.0,
        ekf_latent: 0.0,
        ekf_latent_std: 0.0,
        actuation: 0.0,
        innovation: 0.0,
        reward: 0.0,
        tracking_metric: 0.0,
    }
}

// [cite:grpo]

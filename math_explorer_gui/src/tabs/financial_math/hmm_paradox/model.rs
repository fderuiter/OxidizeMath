//! HMM compilation, forward/backward/filter/Viterbi execution, and zero-allocation branching evaluation.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    HiddenMarkovParadoxTool, HmmParadoxConfig, TimelineSummary, MAX_HMM_STEPS, NUM_HMM_OBS,
    NUM_HMM_STATES,
};
use math_explorer::pure_math::statistics::markov::hmm::{HiddenMarkovModel, HmmScratch};
use nalgebra::{DMatrix, DVector};
use oxidize_core::rng::{OxidizeRng, BASE_SEED};

/// Builds a validated 3-state, 3-symbol `HiddenMarkovModel<f64>` from `cfg`.
pub fn build_hmm_from_config(cfg: &HmmParadoxConfig) -> HiddenMarkovModel<f64> {
    let initial = DVector::from_vec(vec![0.25, 0.05, 0.70]);

    let self_shock = cfg.shock_self_loop_prob.clamp(0.0, 0.60);
    let rem_shock = 1.0 - self_shock;
    let bridge_shock = cfg.bridge_to_shock_prob.clamp(0.10, 0.95);
    let transitions = DMatrix::from_row_slice(
        NUM_HMM_STATES,
        NUM_HMM_STATES,
        &[
            0.20,
            0.40,
            0.40,
            0.60 * rem_shock,
            self_shock,
            0.40 * rem_shock,
            1.0 - bridge_shock,
            bridge_shock,
            0.00,
        ],
    );

    let focus = cfg.shock_emission_focus.clamp(0.30, 0.90);
    let rem_emit = 1.0 - focus;
    let emissions = DMatrix::from_row_slice(
        NUM_HMM_STATES,
        NUM_HMM_OBS,
        &[
            0.50,
            0.15,
            0.35,
            rem_emit * (2.0 / 3.0),
            focus,
            rem_emit * (1.0 / 3.0),
            0.10,
            0.70,
            0.20,
        ],
    );

    HiddenMarkovModel::new(initial, transitions, emissions)
        .expect("stochastic 3x3 HMM matrices are normalized by construction")
}

impl HiddenMarkovParadoxTool {
    /// Clamps configuration parameters to valid bounds and active challenge locks.
    pub fn clamp_config_in_place(&mut self) {
        let c = &mut self.config;
        if c.observations.len() < 2 {
            c.observations = vec![0, 1, 1];
        } else if c.observations.len() > MAX_HMM_STEPS {
            c.observations.truncate(MAX_HMM_STEPS);
        }
        for tok in &mut c.observations {
            *tok = (*tok).min(NUM_HMM_OBS - 1);
        }
        c.fork_step = c.fork_step.min(c.observations.len() - 1);
        c.fork_token = c.fork_token.min(NUM_HMM_OBS - 1);
        c.shock_self_loop_prob = sanitize(c.shock_self_loop_prob, 0.0, 0.0, 0.60);
        c.bridge_to_shock_prob = sanitize(c.bridge_to_shock_prob, 0.95, 0.10, 0.95);
        c.shock_emission_focus = sanitize(c.shock_emission_focus, 0.85, 0.30, 0.90);
        if self.active_challenge.is_some() {
            self.enforce_challenge_locks();
        }
    }

    /// Resets the workbench to the canonical 3-state forbidden-marginal preset.
    pub fn reset_starter_preset(&mut self) {
        self.config = HmmParadoxConfig::default();
        if self.active_challenge.is_some() {
            self.enforce_challenge_locks();
        }
        self.recalculate();
    }

    /// Generates a new deterministic observation rollout using `OxidizeRng::from_seed(BASE_SEED ^ seed_counter)`.
    pub fn simulate_seeded_rollout(&mut self) {
        self.config.seed_counter = self.config.seed_counter.wrapping_add(1).max(1);
        let mut rng = OxidizeRng::from_seed(BASE_SEED ^ self.config.seed_counter);
        let target_len = self.config.observations.len().clamp(3, 8);
        if let Ok((_states, obs)) = self.hmm.generate(target_len, &mut rng) {
            self.config.observations = obs;
            self.config.fork_step = self.config.fork_step.min(target_len - 1);
        }
        self.recalculate();
    }

    /// Compiles the HMM and evaluates `forward`, `forward_log_likelihood`, `posterior_probabilities`,
    /// `filter`, `viterbi`, and zero-allocation `HmmScratch` across Baseline and Branched timelines.
    pub fn recalculate(&mut self) {
        self.clamp_config_in_place();
        self.hmm = build_hmm_from_config(&self.config);

        self.branched_observations.clear();
        self.branched_observations
            .extend_from_slice(&self.config.observations);
        self.branched_observations[self.config.fork_step] = self.config.fork_token;

        self.baseline = evaluate_timeline_full(
            &self.hmm,
            &self.config.observations,
            &mut self.baseline_scratch,
        );
        self.branched = evaluate_timeline_full(
            &self.hmm,
            &self.branched_observations,
            &mut self.branched_scratch,
        );

        self.sync_callouts();
    }

    fn sync_callouts(&mut self) {
        self.callout_manager.clear_auto();
        if let Some(step) = self.baseline.first_forbidden_step {
            let from_s = self.baseline.marginal_path[step] as f64;
            let to_s = self.baseline.marginal_path[step + 1] as f64;
            self.callout_manager.set_auto_callout(
                "hmm_forbidden_edge",
                [(step as f64) + 0.5, 0.5 * (from_s + to_s)],
                format!(
                    "Forbidden MAP Step t={}→{}: A[{},{}]=0 (P=0.0)",
                    step,
                    step + 1,
                    self.baseline.marginal_path[step],
                    self.baseline.marginal_path[step + 1]
                ),
                [16.0, -22.0],
            );
        }
        if let Some(&last_v) = self.baseline.viterbi_path.last() {
            let last_t = self.baseline.viterbi_path.len().saturating_sub(1) as f64;
            self.callout_manager.set_auto_callout(
                "hmm_viterbi_recovery",
                [last_t, last_v as f64],
                format!(
                    "Viterbi {:?} (P={:.5})",
                    self.baseline.viterbi_path, self.baseline.viterbi_joint_prob
                ),
                [-20.0, 20.0],
            );
        }
    }
}

fn evaluate_timeline_full(
    hmm: &HiddenMarkovModel<f64>,
    observations: &[usize],
    scratch: &mut HmmScratch<f64>,
) -> TimelineSummary {
    let fwd = hmm.forward(observations).unwrap_or(0.0);
    let fwd_log = hmm
        .forward_log_likelihood(observations)
        .unwrap_or(f64::NEG_INFINITY);
    let gamma_mat = hmm.posterior_probabilities(observations).ok();
    let filter_vec = hmm.filter(observations).ok();
    let viterbi_vec = hmm.viterbi(observations).unwrap_or_default();

    let scratch_eval = hmm
        .evaluate_with_scratch(observations, scratch)
        .expect("observations validated within scratch capacity");

    let t_len = observations.len();
    let marginal_path: Vec<usize> = if let Some(gm) = &gamma_mat {
        (0..t_len)
            .map(|t| {
                (0..NUM_HMM_STATES)
                    .max_by(|&a, &b| gm[(a, t)].partial_cmp(&gm[(b, t)]).unwrap())
                    .unwrap_or(0)
            })
            .collect()
    } else {
        scratch.marginal_slice(t_len).to_vec()
    };

    let mut final_belief = [0.0; NUM_HMM_STATES];
    if let Some(fv) = filter_vec {
        for i in 0..NUM_HMM_STATES {
            final_belief[i] = fv[i];
        }
    } else {
        final_belief.copy_from_slice(scratch.filter_slice(NUM_HMM_STATES));
    }

    let viterbi_joint_prob = hmm
        .path_joint_probability(&viterbi_vec, observations)
        .unwrap_or(scratch_eval.viterbi_joint_prob);
    let marginal_joint_prob = hmm
        .path_joint_probability(&marginal_path, observations)
        .unwrap_or(scratch_eval.marginal_joint_prob);

    TimelineSummary {
        forward_likelihood: fwd,
        forward_log_likelihood: fwd_log,
        viterbi_path: viterbi_vec,
        viterbi_joint_prob,
        marginal_path,
        marginal_joint_prob,
        final_filter_belief: final_belief,
        has_forbidden_marginal_step: scratch_eval.has_forbidden_marginal_step,
        first_forbidden_step: scratch_eval.first_forbidden_step,
    }
}

fn sanitize(val: f64, fallback: f64, min: f64, max: f64) -> f64 {
    if val.is_finite() {
        val.clamp(min, max)
    } else {
        fallback
    }
}

// [cite:clinical_trials_statistics]

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Log-space Hidden Markov Model (Viterbi + Baum-Welch EM) and exact Ornstein-Uhlenbeck
//! transition & parameter calibration workbench.

mod render;
pub mod state;

use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use crate::web_io::PlatformInstant;
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::nalgebra::{DMatrix, DVector};
use math_explorer::pure_math::statistics::markov::hmm::HiddenMarkovModel;
use math_explorer::pure_math::statistics::ou_process::{
    EulerMaruyama, ExactOuSolver, OuParams, TimeStep, estimate_ou_params,
};
use math_explorer::rng::OxidizeRng;
use scientific_metadata::theory::TheoryDescribable;
use state::{CalibrationMode, HmmOuState};

pub(super) const MIN_SEQ_LEN: usize = 40;
pub(super) const MAX_SEQ_LEN: usize = 600;
pub(super) const MIN_EM_ITERS: usize = 1;
pub(super) const MAX_EM_ITERS: usize = 35;
pub(super) const MIN_OU_STEPS: usize = 60;
pub(super) const MAX_OU_STEPS: usize = 800;

/// Interactive workbench for 600-step log-space HMM Viterbi decoding, floored Baum-Welch EM,
/// exact Gaussian Ornstein-Uhlenbeck transitions, and AR(1) OU volatility calibration.
pub struct HmmOuWorkbenchTool {
    pub(super) mode: CalibrationMode,
    pub(super) seq_len: usize,
    pub(super) stay_prob: f64,
    pub(super) emission_clarity: f64,
    pub(super) em_iters: usize,
    pub(super) prob_floor: f64,
    pub(super) ou_mu: f64,
    pub(super) ou_theta: f64,
    pub(super) ou_sigma: f64,
    pub(super) ou_x0: f64,
    pub(super) ou_dt: f64,
    pub(super) ou_steps: usize,
    pub(super) seed: u64,

    pub(super) true_log_lik: f64,
    pub(super) em_log_liks: Vec<PlotPoint>,
    pub(super) true_state_curve: Vec<PlotPoint>,
    pub(super) viterbi_curve: Vec<PlotPoint>,
    pub(super) viterbi_accuracy: f64,
    pub(super) em_converged: bool,

    pub(super) ou_exact_curve: Vec<PlotPoint>,
    pub(super) ou_euler_curve: Vec<PlotPoint>,
    pub(super) ou_mean_curve: Vec<PlotPoint>,
    pub(super) ou_upper_band: Vec<PlotPoint>,
    pub(super) ou_lower_band: Vec<PlotPoint>,
    pub(super) estimated_ou: Option<(f64, f64, f64)>,
    pub(super) error_msg: Option<String>,
    pub(super) last_compute_ms: f64,
}

impl Default for HmmOuWorkbenchTool {
    fn default() -> Self {
        let mut tool = Self {
            mode: CalibrationMode::HmmViterbiBaumWelch,
            seq_len: 240,
            stay_prob: 0.88,
            emission_clarity: 0.82,
            em_iters: 15,
            prob_floor: 0.01,
            ou_mu: 0.50,
            ou_theta: 2.50,
            ou_sigma: 0.35,
            ou_x0: 1.20,
            ou_dt: 0.05,
            ou_steps: 300,
            seed: 42,
            true_log_lik: 0.0,
            em_log_liks: Vec::new(),
            true_state_curve: Vec::new(),
            viterbi_curve: Vec::new(),
            viterbi_accuracy: 0.0,
            em_converged: false,
            ou_exact_curve: Vec::new(),
            ou_euler_curve: Vec::new(),
            ou_mean_curve: Vec::new(),
            ou_upper_band: Vec::new(),
            ou_lower_band: Vec::new(),
            estimated_ou: None,
            error_msg: None,
            last_compute_ms: 0.0,
        };
        tool.recompute_all();
        tool
    }
}

impl HmmOuWorkbenchTool {
    /// Switches between the HMM inference/EM mode and the OU exact transition/calibration mode.
    pub fn set_mode(&mut self, mode: CalibrationMode) {
        self.mode = mode;
        self.recompute_all();
    }

    pub(super) fn recompute_all(&mut self) {
        let start = PlatformInstant::now();
        self.error_msg = None;
        match self.mode {
            CalibrationMode::HmmViterbiBaumWelch => self.recompute_hmm(),
            CalibrationMode::OuExactAndCalibration => self.recompute_ou(),
        }
        self.last_compute_ms = start.elapsed().as_secs_f64() * 1000.0;
    }

    fn recompute_hmm(&mut self) {
        let stay = self.stay_prob.clamp(0.55, 0.97);
        let clarity = self.emission_clarity.clamp(0.55, 0.95);
        let true_hmm = match build_two_state_hmm(0.5, stay, clarity) {
            Ok(h) => h,
            Err(e) => {
                self.error_msg = Some(e);
                return;
            }
        };

        let mut rng = OxidizeRng::new(self.seed);
        let len = self.seq_len.clamp(MIN_SEQ_LEN, MAX_SEQ_LEN);
        let Ok((true_states, obs)) = true_hmm.generate(len, &mut rng) else {
            self.error_msg = Some("Failed to sample HMM sequence".to_string());
            return;
        };

        self.true_log_lik = true_hmm.forward_log_likelihood(&obs).unwrap_or(f64::NAN);
        let mut learner = match build_two_state_hmm(0.62, 0.65, 0.64) {
            Ok(h) => h,
            Err(e) => {
                self.error_msg = Some(e);
                return;
            }
        };

        if let Ok(res) = learner.baum_welch(&obs, self.em_iters, 1e-6, self.prob_floor) {
            self.em_converged = res.converged;
            self.em_log_liks = res
                .log_likelihoods
                .iter()
                .enumerate()
                .map(|(k, &ll)| PlotPoint::new(k as f64, ll))
                .collect();
        }

        if let Ok(decoded) = learner.viterbi(&obs) {
            let direct = decoded
                .iter()
                .zip(&true_states)
                .filter(|&(a, b)| a == b)
                .count();
            let flipped = len.saturating_sub(direct);
            let invert = flipped > direct;
            self.viterbi_accuracy = direct.max(flipped) as f64 / len as f64;
            let stride = (len / 200).max(1);
            self.true_state_curve = true_states
                .iter()
                .enumerate()
                .step_by(stride)
                .map(|(t, &s)| PlotPoint::new(t as f64, s as f64))
                .collect();
            self.viterbi_curve = decoded
                .iter()
                .enumerate()
                .step_by(stride)
                .map(|(t, &s)| {
                    let aligned = if invert { 1 - s } else { s };
                    PlotPoint::new(t as f64, aligned as f64 + 0.04)
                })
                .collect();
        }
    }

    fn recompute_ou(&mut self) {
        let params = match OuParams::from_values(self.ou_mu, self.ou_theta, self.ou_sigma) {
            Ok(p) => p,
            Err(e) => {
                self.error_msg = Some(format!("OU parameter error: {e}"));
                return;
            }
        };
        let dt = match TimeStep::new(self.ou_dt) {
            Ok(d) => d,
            Err(e) => {
                self.error_msg = Some(format!("OU time-step error: {e}"));
                return;
            }
        };

        let n = self.ou_steps.clamp(MIN_OU_STEPS, MAX_OU_STEPS);
        let exact_solver = ExactOuSolver::new(params, dt);
        let euler_solver = EulerMaruyama::new(params, dt);

        let mut exact_path = vec![0.0_f64; n + 1];
        let mut rng_exact = OxidizeRng::new(self.seed);
        exact_solver.simulate_into(self.ou_x0, &mut exact_path, &mut rng_exact);

        let mut rng_euler = OxidizeRng::new(self.seed);
        let euler_path = euler_solver.simulate(self.ou_x0, n, &mut rng_euler);

        self.estimated_ou = estimate_ou_params(&exact_path, self.ou_dt)
            .ok()
            .map(|est| (est.mu.value(), est.theta.value(), est.sigma.value()));

        self.ou_exact_curve.clear();
        self.ou_euler_curve.clear();
        self.ou_mean_curve.clear();
        self.ou_upper_band.clear();
        self.ou_lower_band.clear();

        for i in 0..=n {
            let t = i as f64 * self.ou_dt;
            let mean_t = exact_solver.conditional_mean(self.ou_x0, t);
            let sd_t = exact_solver.conditional_variance(t).sqrt();
            self.ou_exact_curve.push(PlotPoint::new(t, exact_path[i]));
            self.ou_euler_curve.push(PlotPoint::new(t, euler_path[i]));
            self.ou_mean_curve.push(PlotPoint::new(t, mean_t));
            self.ou_upper_band.push(PlotPoint::new(t, mean_t + sd_t));
            self.ou_lower_band.push(PlotPoint::new(t, mean_t - sd_t));
        }
    }
}

fn build_two_state_hmm(
    pi0: f64,
    stay: f64,
    clarity: f64,
) -> Result<HiddenMarkovModel<f64>, String> {
    let init = DVector::from_vec(vec![pi0, 1.0 - pi0]);
    let trans = DMatrix::from_row_slice(2, 2, &[stay, 1.0 - stay, 1.0 - stay, stay]);
    let emis = DMatrix::from_row_slice(2, 2, &[clarity, 1.0 - clarity, 1.0 - clarity, clarity]);
    HiddenMarkovModel::new(init, trans, emis).map_err(|e| format!("HMM construction error: {e}"))
}

impl InteractiveTool for HmmOuWorkbenchTool {
    fn name(&self) -> &'static str {
        "HMM & OU Calibration"
    }

    fn domain(&self) -> &'static str {
        "stochastic_processes"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn has_finite_parameters(&self) -> bool {
        [
            self.stay_prob,
            self.emission_clarity,
            self.prob_floor,
            self.ou_mu,
            self.ou_theta,
            self.ou_sigma,
            self.ou_x0,
            self.ou_dt,
        ]
        .into_iter()
        .all(f64::is_finite)
    }

    fn clamp_parameters(&mut self) -> bool {
        self.seq_len = self.seq_len.clamp(MIN_SEQ_LEN, MAX_SEQ_LEN);
        self.stay_prob = self.stay_prob.clamp(0.55, 0.97);
        self.emission_clarity = self.emission_clarity.clamp(0.55, 0.95);
        self.em_iters = self.em_iters.clamp(MIN_EM_ITERS, MAX_EM_ITERS);
        self.prob_floor = self.prob_floor.clamp(0.001, 0.10);
        self.ou_mu = self.ou_mu.clamp(-2.0, 2.0);
        self.ou_theta = self.ou_theta.clamp(0.10, 5.0);
        self.ou_sigma = self.ou_sigma.clamp(0.05, 1.50);
        self.ou_x0 = self.ou_x0.clamp(-2.0, 2.0);
        self.ou_dt = self.ou_dt.clamp(0.01, 0.25);
        self.ou_steps = self.ou_steps.clamp(MIN_OU_STEPS, MAX_OU_STEPS);
        self.recompute_all();
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&HmmOuState {
            mode: self.mode,
            seq_len: self.seq_len,
            stay_prob: self.stay_prob,
            emission_clarity: self.emission_clarity,
            em_iters: self.em_iters,
            prob_floor: self.prob_floor,
            ou_mu: self.ou_mu,
            ou_theta: self.ou_theta,
            ou_sigma: self.ou_sigma,
            ou_x0: self.ou_x0,
            ou_dt: self.ou_dt,
            ou_steps: self.ou_steps,
            seed: self.seed,
        })
        .ok()
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::new(crate::url_state::MAX_DECODED_URL_STATE_BYTES, 8, 64)
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: HmmOuState =
            serde_json::from_str(state).map_err(|e| format!("Invalid HMM/OU state: {e}"))?;
        parsed.validate()?;

        self.mode = parsed.mode;
        self.seq_len = parsed.seq_len;
        self.stay_prob = parsed.stay_prob;
        self.emission_clarity = parsed.emission_clarity;
        self.em_iters = parsed.em_iters;
        self.prob_floor = parsed.prob_floor;
        self.ou_mu = parsed.ou_mu;
        self.ou_theta = parsed.ou_theta;
        self.ou_sigma = parsed.ou_sigma;
        self.ou_x0 = parsed.ou_x0;
        self.ou_dt = parsed.ou_dt;
        self.ou_steps = parsed.ou_steps;
        self.seed = parsed.seed;
        self.recompute_all();
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_workspace(ui);
    }

    fn show_ui(&mut self, ui: &mut egui::Ui) {
        self.render_controls_panel(ui);
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.build_export_datasets()
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        self.build_export_primitives()
    }

    fn step_simulation(&mut self) {
        self.seed = self.seed.wrapping_add(1);
        self.recompute_all();
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

// [cite:clinical_trials_statistics]

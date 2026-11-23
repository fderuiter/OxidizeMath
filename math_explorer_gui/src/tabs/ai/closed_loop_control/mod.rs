//! Unified Closed-Loop EKF + GRPO / Q-Learning Control Workbench (`ClosedLoopControlTool`).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod canvas;
pub mod controllers;
pub mod ekf;
pub mod pipeline;
pub mod plants;
pub mod plants_pk;
pub mod runner;
pub mod state;
#[cfg(test)]
mod tests;
pub mod ui;

use crate::async_sim::{SimCommand, SimulationController};
use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use eframe::egui;
use pipeline::{ClosedLoopPipeline, ClosedLoopRunner, ClosedLoopTrainingSnapshot};
use plants::PlantStepOutput;
use state::ClosedLoopState;
use std::collections::HashMap;

/// Unified closed-loop Extended Kalman Filter + `GRPO` / `QLearningAgent` control workbench.
pub struct ClosedLoopControlTool {
    /// Serializable plant, observer, actuation-budget, and RL hyperparameter configuration.
    pub state: ClosedLoopState,
    /// Preallocated zero-allocation plant twins, `FastEkf` mirrors, and RL controllers.
    pub pipeline: ClosedLoopPipeline,
    /// Cooperative background `SimulationController` for batched episodic training.
    pub worker: Option<SimulationController>,
    /// Whether live closed-loop stepping is currently active.
    pub playing: bool,
    /// Canvas widget ID when the 2D phase-portrait perturbation pad owns keyboard focus.
    pub focused_canvas: Option<egui::Id>,
    /// Last rendered UI frame wall-clock duration in milliseconds.
    pub last_frame_ms: f64,
}

impl Default for ClosedLoopControlTool {
    fn default() -> Self {
        let state = ClosedLoopState::default();
        let pipeline = ClosedLoopPipeline::new(&state);
        Self {
            state,
            pipeline,
            worker: None,
            playing: false,
            focused_canvas: None,
            last_frame_ms: 0.0,
        }
    }
}

impl ClosedLoopControlTool {
    /// Advances the active plant twin, `FastEkf` observer, and RL controller by one zero-allocation step.
    pub fn step_closed_loop(&mut self) -> PlantStepOutput {
        self.pipeline.step_zero_alloc(&self.state)
    }

    /// Rebuilds the simulation pipeline, warm-start episodes, and preview traces from `self.state`.
    pub fn rebuild_simulation(&mut self) {
        self.playing = false;
        self.worker = None;
        self.pipeline.rebuild_all(&self.state);
    }

    /// Resets policies to untrained zero weights/Q-values and reruns open preview steps.
    pub fn reset_untrained_policy(&mut self) {
        self.playing = false;
        self.worker = None;
        self.pipeline.reset_untrained(&self.state);
        for _ in 0..64 {
            let _ = self.pipeline.step_zero_alloc(&self.state);
        }
    }

    /// Synchronously trains `episodes` zero-allocation GRPO or Q-learning rollouts and refreshes preview traces.
    pub fn train_episodes_sync(&mut self, episodes: usize) {
        for _ in 0..episodes {
            let _ = self.pipeline.train_episode_zero_alloc(&self.state);
        }
        self.pipeline.reset_live_traces_only(&self.state);
        for _ in 0..72 {
            let _ = self.pipeline.step_zero_alloc(&self.state);
        }
    }

    /// Launches cooperative asynchronous episodic training via `SimulationController` and `ClosedLoopRunner`.
    pub fn start_async_training(&mut self, episodes: usize) {
        let runner = ClosedLoopRunner::from_state_and_pipeline(self.state.clone(), &self.pipeline);
        let mut controller = SimulationController::new(runner);
        let _ = controller.send_command(SimCommand::TrainEpisodes(episodes));
        self.worker = Some(controller);
    }

    /// Polls the background `SimulationController`, applying new policy weights when available.
    pub fn poll_training_worker(&mut self) -> bool {
        let mut updated = false;
        let mut finished = false;
        if let Some(worker) = self.worker.as_mut() {
            if let Some(snap) = worker.update() {
                if let Some(payload) = snap
                    .structured_data
                    .as_ref()
                    .and_then(|d| d.downcast_ref::<ClosedLoopTrainingSnapshot>())
                {
                    self.pipeline.apply_training_snapshot(payload);
                    updated = true;
                }
            }
            finished = !worker.is_running();
        }
        if updated {
            self.pipeline.reset_live_traces_only(&self.state);
            for _ in 0..72 {
                let _ = self.pipeline.step_zero_alloc(&self.state);
            }
        }
        if finished && updated {
            self.worker = None;
        }
        updated
    }

    /// Applies a normalized `[dx, dy]` state perturbation kick and immediately updates EKF innovation.
    pub fn apply_kick(&mut self, dx_norm: f64, dy_norm: f64) -> PlantStepOutput {
        self.pipeline.apply_state_kick(dx_norm, dy_norm, &self.state)
    }
}

impl InteractiveTool for ClosedLoopControlTool {
    fn name(&self) -> &'static str {
        "Closed-Loop EKF Control"
    }

    fn domain(&self) -> &'static str {
        "ai"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn show(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            self.show_inside(ui);
        });
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        ui::show_tool_inside(self, ui);
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
    }

    fn has_finite_parameters(&self) -> bool {
        self.state.is_finite()
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        let out = self.pipeline.last_output;
        vec![
            ("controlled_primary", out.controlled_primary),
            ("uncontrolled_primary", out.uncontrolled_primary),
            ("ekf_primary", out.ekf_primary),
            ("ekf_primary_std", out.ekf_primary_std),
            ("innovation", out.innovation),
            ("actuation", out.actuation),
            ("tracking_metric", out.tracking_metric),
            ("episodes_trained", f64::from(self.pipeline.episodes_trained)),
        ]
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ClosedLoopState::STATE_BOUNDS
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, raw: &str) -> Result<(), String> {
        self.state_bounds().validate_json(raw)?;
        let loaded: ClosedLoopState = serde_json::from_str(raw)
            .map_err(|e| format!("Invalid ClosedLoopControlTool state: {e}"))?;
        loaded.validate_bounds()?;
        self.state = loaded;
        self.rebuild_simulation();
        Ok(())
    }

    fn clamp_parameters(&mut self) -> bool {
        self.state.clamp_in_place();
        self.rebuild_simulation();
        true
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        vec![
            (
                "Uncontrolled Open-Loop Twin".to_string(),
                self.pipeline.uncontrolled_trace.clone(),
            ),
            (
                "True Controlled State".to_string(),
                self.pipeline.controlled_trace.clone(),
            ),
            (
                "Noisy Sensor Measurement".to_string(),
                self.pipeline.noisy_meas_trace.clone(),
            ),
            (
                "EKF Posterior Estimate".to_string(),
                self.pipeline.ekf_primary_trace.clone(),
            ),
            (
                "EKF +1 Sigma Bound".to_string(),
                self.pipeline.ekf_upper_trace.clone(),
            ),
            (
                "EKF -1 Sigma Bound".to_string(),
                self.pipeline.ekf_lower_trace.clone(),
            ),
            (
                "Bounded Control Actuation".to_string(),
                self.pipeline.actuation_trace.clone(),
            ),
            (
                "EKF Innovation Residual".to_string(),
                self.pipeline.innovation_trace.clone(),
            ),
        ]
    }

    fn export_figure_options(&self) -> Option<egui_plot::export::ExportOptions> {
        Some(egui_plot::export::ExportOptions {
            line_thickness: 2.2,
            ..Default::default()
        })
    }

    fn step_simulation(&mut self) {
        let _ = self.step_closed_loop();
    }
}

impl scientific_metadata::theory::TheoryDescribable for ClosedLoopControlTool {
    fn theory_description(&self) -> String {
        "Nonlinear Extended Kalman Filtering propagates discrete RK4 tangent Jacobians $F_k = \\partial f_{\\text{RK4}} / \\partial x$ to reconstruct hidden state coordinates and posterior covariance $P_{k|k}$ from noisy scalar observations $z_k = h(x_k) + v_k$. Coupling $\\hat{x}_{k|k}$ to critic-free Group Relative Policy Optimization ($\\hat{A}_g = (R_g - \\mu_R)/(\\sigma_R + \\epsilon)$ with clipped importance ratios) or zero-allocation tabular Q-learning stabilizes unstable Lorenz periodic orbits via OGY Rayleigh thrusts, suppresses SEIR epidemic peaks from inferred latent exposures, and maintains Bateman ADME plasma drug levels inside $[\\text{MEC}, \\text{MTC}]$.".to_string()
    }

    fn phonetic_description(&self) -> String {
        "Closed-loop Extended Kalman Filter state estimation coupled to critic-free Group Relative Policy Optimization and discretized Q-learning across chaotic Lorenz convection, SEIR epidemic compartments, and Bateman pharmacokinetic dosing.".to_string()
    }

    fn theory_citation(&self) -> String {
        "Kalman, J. Basic Eng. 82(1), 35-45 (1960); Ott, Grebogi & Yorke, Phys. Rev. Lett. 64(11), 1196 (1990); Shao et al., DeepSeekMath GRPO, arXiv:2402.03300 (2024); Watkins & Dayan, Mach. Learn. 8, 279-292 (1992)".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "Nonlinear EKF & RK4 Tangent Jacobians".to_string(),
            "Propagates $P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q_k$ using exact 4-stage RK4 variational derivatives and updates $\\hat{x}_{k|k} = \\hat{x}_{k|k-1} + K_k (z_k - h(\\hat{x}_{k|k-1}))$ with floored measurement covariance $R \\ge 10^{-8}$.".to_string(),
        );
        map.insert(
            "Critic-Free GRPO & Zero-Alloc Q-Learning".to_string(),
            "GRPO normalizes group returns across $G$ sampled trajectories without a value critic and optimizes the PPO-clipped KL-regularized surrogate, while `FixedArrayQFunction` and `select_action_into` execute tabular Bellman control with zero heap allocations.".to_string(),
        );
        map
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "ClosedLoopControlTool",
        display_name: "Closed-Loop EKF Control",
        domain: "ai",
        tags: &["ekf", "kalman", "grpo", "q-learning", "control", "lorenz", "seir", "pharmacokinetics"],
        build: || Box::new(ClosedLoopControlTool::default()),
    }
}

// [cite:grpo]

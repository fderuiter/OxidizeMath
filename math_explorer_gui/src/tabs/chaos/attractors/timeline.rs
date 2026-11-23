//! Time-travel checkpoint and transport helpers for [`super::AttractorPlotter`] and [`super::AttractorRunner`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::flow::AttractorMode;
use super::{AttractorPlotter, AttractorRunner, SolverExecutionState};
use crate::async_sim::ui::WorkerFaultAction;
use crate::async_sim::{CheckpointParams, CheckpointPayload, SimCommand, TimeTravelCheckpoint};
use crate::framework::PlaybackAction;
use eframe::egui;
use nalgebra::Vector3;

impl AttractorPlotter {
    pub(super) fn apply_fault_action(&mut self, action: WorkerFaultAction, ctx: &egui::Context) {
        use crate::framework::recovery::RecoveryAction;
        match action {
            WorkerFaultAction::Reset => self.reset(),
            WorkerFaultAction::Recovery(RecoveryAction::AutoScaleStepSize) => {
                self.dt = if self.dt.is_finite() {
                    (self.dt * 0.5).max(0.001)
                } else {
                    0.005
                };
                self.reset();
            }
            WorkerFaultAction::Recovery(RecoveryAction::ClampParameters) => {
                self.system.sigma = if self.system.sigma.is_finite() {
                    self.system.sigma.clamp(0.0, 50.0)
                } else {
                    10.0
                };
                self.system.rho = if self.system.rho.is_finite() {
                    self.system.rho.clamp(0.0, 100.0)
                } else {
                    28.0
                };
                self.system.beta = if self.system.beta.is_finite() {
                    self.system.beta.clamp(0.0, 10.0)
                } else {
                    8.0 / 3.0
                };
                self.dt = if self.dt.is_finite() {
                    self.dt.clamp(0.001, 0.05)
                } else {
                    0.01
                };
                self.reset();
            }
            WorkerFaultAction::ResetParameters
            | WorkerFaultAction::Recovery(RecoveryAction::ResetParameters) => {
                self.reset_parameters();
            }
        }
        ctx.request_repaint();
    }

    pub(super) fn handle_transport_action(&mut self, action: PlaybackAction, ctx: &egui::Context) {
        match action {
            PlaybackAction::Play => {
                let _ = self.try_resume();
            }
            PlaybackAction::Pause => {
                self.execution_state = SolverExecutionState::Paused;
                let _ = self.worker.send_command(SimCommand::Pause);
            }
            PlaybackAction::Step => {
                self.execution_state = SolverExecutionState::Paused;
                let _ = self.worker.send_command(SimCommand::Step(1));
            }
            PlaybackAction::StepBack => {
                self.execution_state = SolverExecutionState::Paused;
                let _ = self.worker.send_command(SimCommand::StepBack(1));
            }
            PlaybackAction::ScrubTo(step) => {
                self.execution_state = SolverExecutionState::Paused;
                let _ = self.worker.send_command(SimCommand::SeekStep(step));
            }
            PlaybackAction::Reset => {
                self.reset();
                ctx.request_repaint();
            }
        }
    }
}

impl AttractorRunner {
    pub(super) fn capture_active_checkpoint_params(&self) -> Option<CheckpointParams> {
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        let mode_flag = match p.mode {
            AttractorMode::Lorenz => 0.0,
            AttractorMode::Custom => 1.0,
        };
        Some(CheckpointParams::scalars(&[
            p.sigma,
            p.rho,
            p.beta,
            p.dt,
            p.max_points as f64,
            mode_flag,
        ]))
    }

    pub(super) fn capture_time_travel_checkpoint(&self) -> Option<TimeTravelCheckpoint> {
        let params = self.capture_active_checkpoint_params()?;
        let v = self.system.state.vec;
        Some(TimeTravelCheckpoint {
            step: self.step_count,
            rng: self.rng.clone(),
            params,
            state: CheckpointPayload::scalars(&[v.x, v.y, v.z]),
        })
    }

    pub(super) fn restore_time_travel_checkpoint(&mut self, checkpoint: &TimeTravelCheckpoint) {
        if let Some(s) = checkpoint.state.as_scalars() {
            if s.len() >= 3 {
                self.system.state.vec = Vector3::new(s[0], s[1], s[2]);
            }
        }
        if let Some(p) = checkpoint.params.as_scalars() {
            if p.len() >= 3 {
                self.system.sigma = p[0];
                self.system.rho = p[1];
                self.system.beta = p[2];
            }
        }
        self.step_count = checkpoint.step;
        self.rng = checkpoint.rng.clone();
        self.twin = None;
        self.execution_state = if checkpoint.step == 0 {
            SolverExecutionState::Ready
        } else {
            SolverExecutionState::Paused
        };
    }

    pub(super) fn rebuild_time_travel_history(
        &mut self,
        front: &[TimeTravelCheckpoint],
        back: &[TimeTravelCheckpoint],
    ) {
        let max_points = back
            .last()
            .or_else(|| front.last())
            .and_then(|cp| cp.params.as_scalars()?.get(4).copied())
            .map_or(2000usize, |v| (v as usize).max(1));
        self.history.clear();
        let total_post_zero = front.iter().chain(back).filter(|cp| cp.step >= 1).count();
        let skip = total_post_zero.saturating_sub(max_points);
        for cp in front
            .iter()
            .chain(back)
            .filter(|cp| cp.step >= 1)
            .skip(skip)
        {
            if let Some(s) = cp.state.as_scalars().filter(|s| s.len() >= 3) {
                self.history.push_back(Vector3::new(s[0], s[1], s[2]));
            }
        }
    }

    pub(super) fn replay_time_travel_step(&mut self, params: &CheckpointParams) {
        let Some(p) = params.as_scalars() else {
            return;
        };
        if p.len() < 5 {
            return;
        }
        self.system.sigma = p[0];
        self.system.rho = p[1];
        self.system.beta = p[2];
        let dt = p[3];
        let max_points = (p[4] as usize).max(1);
        let mode = if p.get(5).copied().unwrap_or(0.0) > 0.5 {
            AttractorMode::Custom
        } else {
            AttractorMode::Lorenz
        };
        let custom_flow = (mode == AttractorMode::Custom)
            .then(|| {
                self.params
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .custom_flow
                    .clone()
            })
            .flatten();
        if self.advance_tick(dt, max_points, mode, custom_flow.as_deref(), false) {
            self.execution_state = SolverExecutionState::Paused;
        }
    }
}

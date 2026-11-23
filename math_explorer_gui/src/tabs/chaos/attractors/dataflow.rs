//! Synchronous stepping and `PaneDataflowBus` 3D landmark publishing for [`super::AttractorPlotter`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    AttractorMode, AttractorPlotter, CompiledFlowJob, SolverExecutionState,
    ATTRACTOR_RING_CAPACITY, MAX_SIMULATION_SPEED,
};
use crate::app::dataflow_bus::{publish_pane_landmarks, LandmarkCloud3DPayload};
use crate::tabs::geometry_topology::vietoris_rips::MAX_CHAOS_LANDMARKS;
use eframe::egui;
use math_explorer::pure_math::statistics::tda::{farthest_point_landmarks, Point3D};

/// Cached landmark frame for throttling farthest-point subsampling when publishing to `PaneDataflowBus`.
#[derive(Default, Clone)]
pub(crate) struct AttractorBusCache {
    key: Option<(u64, usize)>,
    payload: Option<LandmarkCloud3DPayload>,
}

impl AttractorPlotter {
    fn step_single_tick(&mut self, cap: usize, custom_job: Option<&CompiledFlowJob>) -> bool {
        let current_vec = self.system.state.vec;
        if !current_vec.x.is_finite() || !current_vec.y.is_finite() || !current_vec.z.is_finite() {
            self.execution_state = SolverExecutionState::Diverged {
                invalid_vec: current_vec,
                diverged_at_step: self.step_count,
            };
            return false;
        }
        if let Some(ref mut twin) = self.twin {
            if !twin.step_branch_a(self.dt, cap) {
                self.execution_state = SolverExecutionState::Diverged {
                    invalid_vec: twin.system_a.state.vec,
                    diverged_at_step: self.step_count,
                };
                return false;
            }
        }
        if let Some(job) = custom_job {
            self.system.state.vec = job.rk4_step(current_vec, self.dt);
        } else {
            self.system.step(self.dt);
        }
        self.step_count += 1;

        let vec = self.system.state.vec;
        if !vec.x.is_finite() || !vec.y.is_finite() || !vec.z.is_finite() {
            self.execution_state = SolverExecutionState::Diverged {
                invalid_vec: vec,
                diverged_at_step: self.step_count,
            };
            return false;
        }
        if self.history.len() >= cap {
            self.history.pop_front();
        }
        self.history.push_back(vec);
        if let Some(ref mut twin) = self.twin {
            twin.observe_lockstep(self.step_count, self.dt, vec);
        }
        true
    }

    pub(super) fn advance_step_simulation(&mut self) {
        if self.diverged() {
            return;
        }
        if self.execution_state == SolverExecutionState::Ready {
            self.execution_state = SolverExecutionState::Running;
        }
        let cap = self.max_points.clamp(1, ATTRACTOR_RING_CAPACITY);
        let custom_job = (self.mode == AttractorMode::Custom)
            .then(|| CompiledFlowJob::from_bank(&self.flow_bank));
        for _ in 0..self.simulation_speed.clamp(1, MAX_SIMULATION_SPEED) {
            if !self.step_single_tick(cap, custom_job.as_ref()) {
                break;
            }
        }
        self.sync_local_branch_engine();
    }

    pub(super) fn publish_bus_landmarks(&mut self, ctx: &egui::Context) {
        if self.diverged() || !self.is_finite_state() {
            return;
        }
        let cfg_hash = self.flow_config_hash();
        let step_bucket = self.step_count / 12;
        let key = (cfg_hash, step_bucket);
        if self.bus_cache.key != Some(key) {
            let traj = self.collect_or_seed_trajectory(900);
            let pts: Vec<Point3D> = traj
                .into_iter()
                .map(|v| Point3D::new(v.x, v.y, v.z))
                .collect();
            if let Ok(landmarks) = farthest_point_landmarks(&pts, MAX_CHAOS_LANDMARKS) {
                let handoff =
                    crate::tabs::geometry_topology::vietoris_rips::handoff::ChaosLandmarkHandoff {
                        points: landmarks,
                        sigma: self.system.sigma,
                        rho: self.system.rho,
                        beta: self.system.beta,
                        dt: self.dt,
                    };
                let rev = cfg_hash.wrapping_add((step_bucket as u64).wrapping_mul(0x9E37_79B9));
                self.bus_cache.payload = Some(handoff.to_bus_payload(rev.max(1)));
                self.bus_cache.key = Some(key);
            }
        }
        if let Some(ref payload) = self.bus_cache.payload {
            publish_pane_landmarks(ctx, payload.clone());
        }
    }
}

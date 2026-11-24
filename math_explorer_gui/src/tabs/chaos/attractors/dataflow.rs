//! `PaneDataflowBus` 3D landmark publishing for [`super::AttractorPlotter`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{AttractorPlotter, AttractorRunner};
use crate::app::dataflow_bus::{publish_pane_landmarks, LandmarkCloud3DPayload};
use crate::app::SharedSimulationBus;
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
    pub(crate) fn bind_sim_bus(&self, ctx: &egui::Context) {
        let bus = SharedSimulationBus::from_context_or_insert(ctx);
        let mut guard = self.params.write().unwrap_or_else(|e| e.into_inner());
        let needs_update = match &guard.sim_bus {
            Some(existing) => !existing.ptr_eq(&bus),
            None => true,
        };
        if needs_update {
            guard.sim_bus = Some(bus);
        }
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

impl AttractorRunner {
    pub(super) fn publish_live_landmarks(&self, dt: f64) {
        let sim_bus = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .sim_bus
            .clone();
        let Some(bus) = sim_bus else {
            return;
        };
        let pts: Vec<Point3D> = self
            .history
            .iter()
            .map(|v| Point3D::new(v.x, v.y, v.z))
            .collect();
        if let Ok(landmarks) = farthest_point_landmarks(&pts, MAX_CHAOS_LANDMARKS) {
            let raw_pts = landmarks.into_iter().map(|p| [p.x, p.y, p.z]).collect();
            let _ = bus.publish_point_cloud_3d(
                raw_pts,
                self.system.sigma,
                self.system.rho,
                self.system.beta,
                dt,
            );
        }
    }
}

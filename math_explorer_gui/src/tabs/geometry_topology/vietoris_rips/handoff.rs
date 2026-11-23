//! Workspace-scoped 3D chaotic landmark handoff and live stream adapters backed by
//! [`crate::app::SharedSimulationBus`] (with zero process-global statics).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::app::{SharedPointCloud3D, SharedSimulationBus};
use eframe::egui;
use math_explorer::pure_math::statistics::tda::Point3D;

/// Payload transferring up to `MAX_VIETORIS_RIPS_POINTS` 3D chaotic landmarks and
/// Lorenz system parameters from `AttractorPlotter` / `AttractorRunner` to `VietorisRipsTool`.
#[derive(Debug, Clone, PartialEq)]
pub struct ChaosLandmarkHandoff {
    /// Subsampled 3D landmark coordinates in state space.
    pub points: Vec<Point3D>,
    /// Prandtl parameter `σ`.
    pub sigma: f64,
    /// Rayleigh parameter `ρ`.
    pub rho: f64,
    /// Geometric parameter `β`.
    pub beta: f64,
    /// Integration time step `dt`.
    pub dt: f64,
}

impl ChaosLandmarkHandoff {
    /// Converts an immutable [`SharedPointCloud3D`] frame from [`SharedSimulationBus`] into [`Self`].
    #[must_use]
    pub fn from_shared_cloud(cloud: &SharedPointCloud3D) -> Self {
        Self {
            points: cloud
                .points
                .iter()
                .map(|&[x, y, z]| Point3D::new(x, y, z))
                .collect(),
            sigma: cloud.sigma,
            rho: cloud.rho,
            beta: cloud.beta,
            dt: cloud.dt,
        }
    }

    /// Converts [`Self`] landmark coordinates into `[x, y, z]` triples for [`SharedSimulationBus`].
    #[must_use]
    pub fn raw_points(&self) -> Vec<[f64; 3]> {
        self.points.iter().map(|p| [p.x, p.y, p.z]).collect()
    }
}

/// Queues a one-click 3D landmark handoff on the workspace [`SharedSimulationBus`] attached to `ctx`.
#[cfg(any(feature = "physics", test))]
pub fn submit_chaos_landmark_handoff(ctx: &egui::Context, handoff: ChaosLandmarkHandoff) {
    let bus = SharedSimulationBus::from_context_or_insert(ctx);
    let _ = bus.submit_vr_handoff(
        handoff.raw_points(),
        handoff.sigma,
        handoff.rho,
        handoff.beta,
        handoff.dt,
        false,
    );
}

/// Drains any pending one-click 3D landmark handoff queued on `ctx`'s [`SharedSimulationBus`].
pub fn take_chaos_landmark_handoff(ctx: &egui::Context) -> Option<ChaosLandmarkHandoff> {
    let bus = SharedSimulationBus::from_context(ctx)?;
    let cloud = bus.take_vr_handoff()?;
    Some(ChaosLandmarkHandoff::from_shared_cloud(&cloud))
}

/// Publishes the latest live 3D landmarks onto `ctx`'s [`SharedSimulationBus`].
#[cfg(test)]
pub fn publish_live_attractor_landmarks(ctx: &egui::Context, handoff: ChaosLandmarkHandoff) {
    let bus = SharedSimulationBus::from_context_or_insert(ctx);
    let _ = bus.publish_point_cloud_3d(
        handoff.raw_points(),
        handoff.sigma,
        handoff.rho,
        handoff.beta,
        handoff.dt,
    );
}

/// Reads a newer live landmark frame from `ctx`'s [`SharedSimulationBus`] if `revision > last_seen_seq`.
pub fn poll_live_attractor_landmarks(
    ctx: &egui::Context,
    last_seen_seq: u64,
) -> Option<(u64, ChaosLandmarkHandoff)> {
    let bus = SharedSimulationBus::from_context(ctx)?;
    let cloud = bus.latest_point_cloud_3d_since(last_seen_seq)?;
    Some((
        cloud.revision,
        ChaosLandmarkHandoff::from_shared_cloud(&cloud),
    ))
}

/// Signals `MathExplorerApp` on `ctx` to focus `Geometry & Topology` and select `VietorisRipsTool`.
#[cfg(any(feature = "physics", test))]
pub fn request_open_vietoris_rips(ctx: &egui::Context) {
    let bus = SharedSimulationBus::from_context_or_insert(ctx);
    bus.request_open_vietoris_rips();
    ctx.request_repaint();
}

/// Consumes any pending `VietorisRipsTool` cross-tab navigation request on `ctx`.
pub fn take_open_vietoris_rips_request(ctx: &egui::Context) -> bool {
    SharedSimulationBus::from_context(ctx).is_some_and(|bus| bus.take_open_vietoris_rips_request())
}

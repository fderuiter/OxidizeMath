//! Workspace-scoped 3D chaotic landmark handoff, live stream, and `PaneDataflowBus`
//! payload adapters backed by [`crate::app::SharedSimulationBus`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::MAX_CHAOS_LANDMARKS;
use crate::app::dataflow_bus::{LandmarkCloud3DPayload, ScalarField2DPayload};
use crate::app::{SharedPointCloud3D, SharedSimulationBus};
use eframe::egui;
use math_explorer::pure_math::statistics::tda::{farthest_point_landmarks, Point3D};

/// Payload transferring up to `MAX_CHAOS_LANDMARKS` 3D chaotic landmarks and
/// Lorenz system parameters from `AttractorPlotter` to `VietorisRipsTool`.
#[derive(Debug, Clone, Default, PartialEq)]
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
    #[cfg(any(feature = "physics", test))]
    #[must_use]
    pub fn raw_points(&self) -> Vec<[f64; 3]> {
        self.points.iter().map(|p| [p.x, p.y, p.z]).collect()
    }

    /// Converts this handoff into a sanitized [`LandmarkCloud3DPayload`] for [`crate::app::dataflow_bus::PaneDataflowBus`].
    #[cfg(any(feature = "physics", test))]
    #[must_use]
    pub fn to_bus_payload(&self, revision: u64) -> LandmarkCloud3DPayload {
        LandmarkCloud3DPayload::sanitized(
            revision,
            self.raw_points(),
            self.sigma,
            self.rho,
            self.beta,
            self.dt,
        )
    }

    /// Reconstructs a [`ChaosLandmarkHandoff`] from an upstream [`LandmarkCloud3DPayload`].
    #[must_use]
    pub fn from_bus_payload(payload: &LandmarkCloud3DPayload) -> Self {
        let points = payload
            .points
            .iter()
            .map(|&[x, y, z]| Point3D::new(x, y, z))
            .collect();
        Self {
            points,
            sigma: payload.sigma,
            rho: payload.rho,
            beta: payload.beta,
            dt: payload.dt,
        }
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

/// Extracts up to `max_landmarks` 3D heightfield landmarks from an upstream 2D scalar field.
pub fn landmarks_from_scalar_field(
    field: &ScalarField2DPayload,
    max_landmarks: usize,
) -> Result<Vec<Point3D>, String> {
    if field.width == 0 || field.height == 0 || field.values.len() < field.width * field.height {
        return Err("Empty scalar field payload".to_string());
    }
    let mut v_min = f64::INFINITY;
    let mut v_max = f64::NEG_INFINITY;
    for &val in &field.values[..(field.width * field.height)] {
        if val.is_finite() {
            v_min = v_min.min(val);
            v_max = v_max.max(val);
        }
    }
    let span = (v_max - v_min).max(1e-9);
    let mean = 0.5 * (v_min + v_max);
    let mut candidates = Vec::with_capacity(field.width * field.height);
    for row in 0..field.height {
        let ny = (row as f64 / (field.height.saturating_sub(1).max(1)) as f64) * 16.0 - 8.0;
        for col in 0..field.width {
            let idx = math_commons::math_kernel::types::flatten_2d_index(col, row, field.width);
            let val = field.values[idx];
            if !val.is_finite() {
                continue;
            }
            let nx = (col as f64 / (field.width.saturating_sub(1).max(1)) as f64) * 16.0 - 8.0;
            let nz = ((val - mean) / span) * 10.0;
            candidates.push(Point3D::new(nx, ny, nz));
        }
    }
    let target = max_landmarks.clamp(4, MAX_CHAOS_LANDMARKS);
    farthest_point_landmarks(&candidates, target).map_err(|e| format!("{e:?}"))
}

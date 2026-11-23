//! Context-scoped one-click landmark handoffs and `PaneDataflowBus` payload conversions.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::MAX_CHAOS_LANDMARKS;
use crate::app::dataflow_bus::{LandmarkCloud3DPayload, ScalarField2DPayload};
use eframe::egui;
use math_explorer::pure_math::statistics::tda::{farthest_point_landmarks, Point3D};

const OPEN_VR_REQUEST_ID: &str = "OPEN_VIETORIS_RIPS_HANDOFF";
const CHAOS_HANDOFF_PAYLOAD_ID: &str = "CHAOS_LANDMARK_HANDOFF_PAYLOAD";

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
    /// Converts this handoff into a sanitized [`LandmarkCloud3DPayload`] for [`crate::app::dataflow_bus::PaneDataflowBus`].
    pub fn to_bus_payload(&self, revision: u64) -> LandmarkCloud3DPayload {
        let raw: Vec<[f64; 3]> = self.points.iter().map(|p| [p.x, p.y, p.z]).collect();
        LandmarkCloud3DPayload::sanitized(
            revision, raw, self.sigma, self.rho, self.beta, self.dt,
        )
    }

    /// Reconstructs a [`ChaosLandmarkHandoff`] from an upstream [`LandmarkCloud3DPayload`].
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

/// Queues a one-click 3D landmark handoff inside `ctx` without process-global statics.
#[cfg(any(feature = "physics", test))]
pub fn submit_chaos_landmark_handoff(ctx: &egui::Context, handoff: ChaosLandmarkHandoff) {
    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::new(CHAOS_HANDOFF_PAYLOAD_ID), handoff);
    });
}

/// Drains any pending one-click 3D landmark handoff queued in `ctx`.
pub fn take_chaos_landmark_handoff(ctx: &egui::Context) -> Option<ChaosLandmarkHandoff> {
    ctx.data_mut(|d| d.remove_temp::<ChaosLandmarkHandoff>(egui::Id::new(CHAOS_HANDOFF_PAYLOAD_ID)))
}

/// Signals `MathExplorerApp` to focus the `Geometry & Topology` tab and select `VietorisRipsTool`.
#[cfg(any(feature = "physics", test))]
pub fn request_open_vietoris_rips(ctx: &egui::Context) {
    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::new(OPEN_VR_REQUEST_ID), true);
    });
    ctx.request_repaint();
}

/// Consumes any pending `VietorisRipsTool` cross-tab navigation request.
pub fn take_open_vietoris_rips_request(ctx: &egui::Context) -> bool {
    ctx.data_mut(|d| {
        d.remove_temp::<bool>(egui::Id::new(OPEN_VR_REQUEST_ID))
            .unwrap_or(false)
    })
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

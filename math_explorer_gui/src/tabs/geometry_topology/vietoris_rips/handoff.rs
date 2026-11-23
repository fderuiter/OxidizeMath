//! Context-scoped one-click landmark handoffs, `CrossToolBus` adapters, and `PaneDataflowBus` payload conversions.
//!
//! All cross-tool messages travel through instance-scoped [`crate::framework::CrossToolBus`]
//! queues or `egui::Context` temporary storage without process-global singletons.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    CloudSource, VietorisRipsTool, VrSourceMode, MAX_CHAOS_LANDMARKS, MAX_VIETORIS_RIPS_POINTS,
};
use crate::app::dataflow_bus::{LandmarkCloud3DPayload, ScalarField2DPayload};
use crate::framework::{CrossToolEnvelope, CrossToolPayload};
use eframe::egui;
use math_explorer::pure_math::statistics::tda::core::Point2D;
use math_explorer::pure_math::statistics::tda::{farthest_point_landmarks, Point3D};

/// Canonical tab display name hosting [`super::VietorisRipsTool`].
pub const VIETORIS_RIPS_TAB_NAME: &str = "Geometry & Topology";

/// Canonical tool identifier for [`super::VietorisRipsTool`].
pub const VIETORIS_RIPS_TOOL_ID: &str = "VietorisRipsTool";

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
    /// Converts this 3D landmark handoff into a typed [`CrossToolEnvelope`] targeting `VietorisRipsTool`.
    #[must_use]
    pub fn into_envelope(
        self,
        source_tool: &'static str,
        focus_target: bool,
        seq: u64,
    ) -> CrossToolEnvelope {
        let points = self.points.into_iter().map(|p| [p.x, p.y, p.z]).collect();
        let parameters = vec![
            ("sigma".to_string(), self.sigma),
            ("rho".to_string(), self.rho),
            ("beta".to_string(), self.beta),
            ("dt".to_string(), self.dt),
        ];
        CrossToolEnvelope {
            source_tool,
            target_tab: Some(VIETORIS_RIPS_TAB_NAME),
            target_tool: Some(VIETORIS_RIPS_TOOL_ID),
            focus_target,
            seq,
            payload: CrossToolPayload::PointCloud3D { points, parameters },
        }
    }

    /// Reconstructs a [`ChaosLandmarkHandoff`] from a [`CrossToolPayload::PointCloud3D`] variant.
    #[must_use]
    pub fn from_payload(payload: &CrossToolPayload) -> Option<Self> {
        let CrossToolPayload::PointCloud3D { points, parameters } = payload else {
            return None;
        };
        let lookup = |key: &str, default: f64| -> f64 {
            parameters
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(key))
                .map_or(default, |(_, v)| *v)
        };
        let pts = points
            .iter()
            .map(|&[x, y, z]| Point3D::new(x, y, z))
            .collect();
        Some(Self {
            points: pts,
            sigma: lookup("sigma", 10.0),
            rho: lookup("rho", 28.0),
            beta: lookup("beta", 8.0 / 3.0),
            dt: lookup("dt", 0.01),
        })
    }

    /// Converts this handoff into a sanitized [`LandmarkCloud3DPayload`] for [`crate::app::dataflow_bus::PaneDataflowBus`].
    #[must_use]
    pub fn to_bus_payload(&self, revision: u64) -> LandmarkCloud3DPayload {
        let raw: Vec<[f64; 3]> = self.points.iter().map(|p| [p.x, p.y, p.z]).collect();
        LandmarkCloud3DPayload::sanitized(revision, raw, self.sigma, self.rho, self.beta, self.dt)
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

/// Queues a one-click 3D landmark handoff inside `ctx` without process-global statics.
#[cfg(test)]
pub fn submit_chaos_landmark_handoff(ctx: &egui::Context, handoff: ChaosLandmarkHandoff) {
    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::new(CHAOS_HANDOFF_PAYLOAD_ID), handoff);
    });
}

/// Drains any pending one-click 3D landmark handoff queued in `ctx`.
pub fn take_chaos_landmark_handoff(ctx: &egui::Context) -> Option<ChaosLandmarkHandoff> {
    ctx.data_mut(|d| d.remove_temp::<ChaosLandmarkHandoff>(egui::Id::new(CHAOS_HANDOFF_PAYLOAD_ID)))
}

impl VietorisRipsTool {
    pub(crate) fn apply_cross_tool_payload(
        &mut self,
        payload: CrossToolPayload,
        is_live: bool,
        seq: u64,
    ) -> Result<bool, String> {
        match payload {
            ref cloud @ CrossToolPayload::PointCloud3D { .. } => {
                let Some(h) = ChaosLandmarkHandoff::from_payload(cloud) else {
                    return Ok(false);
                };
                if is_live {
                    self.chaos.live_stream_seq = seq.max(1);
                    self.chaos.cross_tool_live_frame = true;
                    self.ingest_live_stream_landmarks(h)?;
                } else {
                    self.ingest_chaos_handoff(h)?;
                }
                Ok(true)
            }
            CrossToolPayload::TimeSeries2D {
                series_name,
                points,
            } => self.apply_bus_time_series_2d(&series_name, points),
            CrossToolPayload::ParameterSet { parameters } => {
                self.apply_bus_parameter_set(&parameters)
            }
        }
    }

    fn apply_bus_time_series_2d(
        &mut self,
        series_name: &str,
        points: Vec<[f64; 2]>,
    ) -> Result<bool, String> {
        let pts: Vec<Point2D> = points
            .into_iter()
            .filter(|p| p[0].is_finite() && p[1].is_finite())
            .take(MAX_VIETORIS_RIPS_POINTS)
            .map(|[x, y]| Point2D { x, y })
            .collect();
        if pts.is_empty() {
            return Err("CrossToolBus TimeSeries2D payload contained no finite points".into());
        }
        self.rebuild_from_2d(CloudSource::Custom2D(pts), Vec::new(), self.radius)?;
        self.radius = (self.persistence.max_radius * 0.25).clamp(0.05, self.max_slider_radius);
        self.update_complex();
        self.status_msg = Some(format!("Imported series '{series_name}' via CrossToolBus"));
        Ok(true)
    }

    fn apply_bus_parameter_set(&mut self, parameters: &[(String, f64)]) -> Result<bool, String> {
        let mut touched_chaos = false;
        for &(ref k, v) in parameters {
            if !v.is_finite() {
                continue;
            }
            match k.as_str() {
                "sigma" => {
                    self.chaos.sigma = v.clamp(0.0, 50.0);
                    touched_chaos = true;
                }
                "rho" => {
                    self.chaos.rho = v.clamp(0.0, 100.0);
                    touched_chaos = true;
                }
                "beta" => {
                    self.chaos.beta = v.clamp(0.1, 10.0);
                    touched_chaos = true;
                }
                "dt" => {
                    self.chaos.dt = v.clamp(0.001, 0.05);
                    touched_chaos = true;
                }
                "radius" => self.set_radius(v.clamp(0.0, self.max_slider_radius)),
                _ => {}
            }
        }
        if touched_chaos {
            self.chaos.regenerate_trajectory();
            if matches!(self.mode, VrSourceMode::Lorenz3D | VrSourceMode::Takens3D) {
                let _ = self.rebuild_chaos_landmarks();
            }
        }
        self.bump_revision();
        Ok(true)
    }
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

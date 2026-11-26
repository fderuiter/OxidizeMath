//! Bilinear 2D scalar-field and radial landmark `ParametricSurface` adapters for `SurfaceViewer`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::SurfaceViewer;
use crate::app::dataflow_bus::{
    poll_pane_landmarks, poll_pane_scalar_field, LandmarkCloud3DPayload, ScalarField2DPayload,
};
use eframe::egui;
use math_explorer::pure_math::differential_geometry::surface::ParametricSurface;
use nalgebra::Point3;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Active upstream dataflow stream overriding `SurfaceViewer` geometry when connected.
#[derive(Clone, Default)]
pub(super) enum LiveSurfaceFeed {
    #[default]
    None,
    ScalarField {
        payload: Arc<ScalarField2DPayload>,
        mean: f64,
        inv_span: f64,
    },
    Landmarks {
        payload: Arc<LandmarkCloud3DPayload>,
        normalized_xy_z: Arc<Vec<[f64; 3]>>,
    },
}

/// Live dataflow stream state and user-adjustable `z` amplitude scale for `SurfaceViewer`.
#[derive(Clone)]
pub(super) struct LiveSurfaceFeedState {
    pub(super) feed: LiveSurfaceFeed,
    pub(super) z_amplitude: f64,
}

impl Default for LiveSurfaceFeedState {
    fn default() -> Self {
        Self {
            feed: LiveSurfaceFeed::None,
            z_amplitude: 1.5,
        }
    }
}

/// Smooth bilinear heightfield `z = A * (u_grid(x, y) - μ) / Δ` over an upstream 2D scalar field.
pub(super) struct DiscreteFieldHeightSurface {
    field: Arc<ScalarField2DPayload>,
    mean: f64,
    inv_span: f64,
    z_amplitude: f64,
}

impl ParametricSurface for DiscreteFieldHeightSurface {
    fn position(&self, u: f64, v: f64) -> Point3<f64> {
        let [u0, u1] = self.field.u_bounds;
        let [v0, v1] = self.field.v_bounds;
        let tx = ((u - u0) / (u1 - u0).max(1e-9)).clamp(0.0, 1.0);
        let ty = ((v - v0) / (v1 - v0).max(1e-9)).clamp(0.0, 1.0);
        let gx = tx * (self.field.width.saturating_sub(1)) as f64;
        let gy = ty * (self.field.height.saturating_sub(1)) as f64;
        let x0 = (gx.floor() as usize).min(self.field.width.saturating_sub(1));
        let y0 = (gy.floor() as usize).min(self.field.height.saturating_sub(1));
        let x1 = (x0 + 1).min(self.field.width.saturating_sub(1));
        let y1 = (y0 + 1).min(self.field.height.saturating_sub(1));
        let fx = gx - x0 as f64;
        let fy = gy - y0 as f64;
        let w = self.field.width;
        use math_commons::math_kernel::types::flatten_2d_index;
        let c00 = self.field.values[flatten_2d_index(x0, y0, w)];
        let c10 = self.field.values[flatten_2d_index(x1, y0, w)];
        let c01 = self.field.values[flatten_2d_index(x0, y1, w)];
        let c11 = self.field.values[flatten_2d_index(x1, y1, w)];
        let top = c00 + fx * (c10 - c00);
        let bot = c01 + fx * (c11 - c01);
        let val = top + fy * (bot - top);
        let z = ((val - self.mean) * self.inv_span) * self.z_amplitude;
        Point3::new(u, v, if z.is_finite() { z } else { 0.0 })
    }
}

/// Smooth inverse-distance Shepard heightfield reconstructed from upstream 3D landmarks.
pub(super) struct LandmarkShepardSurface {
    samples: Arc<Vec<[f64; 3]>>,
    z_amplitude: f64,
}

impl ParametricSurface for LandmarkShepardSurface {
    fn position(&self, u: f64, v: f64) -> Point3<f64> {
        if self.samples.is_empty() {
            return Point3::new(u, v, 0.0);
        }
        let mut num = 0.0;
        let mut den = 0.0;
        for &[sx, sy, sz] in self.samples.iter() {
            let d2 = (u - sx).powi(2) + (v - sy).powi(2) + 0.12;
            let w = 1.0 / (d2 * d2);
            num += w * sz;
            den += w;
        }
        let z = if den > 0.0 {
            (num / den) * self.z_amplitude
        } else {
            0.0
        };
        Point3::new(u, v, if z.is_finite() { z } else { 0.0 })
    }
}

fn field_stats(values: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        return (0.0, 1.0);
    }
    let span = (hi - lo).max(1e-6);
    (0.5 * (lo + hi), 1.0 / span)
}

fn normalize_landmarks(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut mins = [f64::INFINITY; 3];
    let mut maxs = [f64::NEG_INFINITY; 3];
    for p in points {
        for axis in 0..3 {
            mins[axis] = mins[axis].min(p[axis]);
            maxs[axis] = maxs[axis].max(p[axis]);
        }
    }
    let spans = [
        (maxs[0] - mins[0]).max(1e-6),
        (maxs[1] - mins[1]).max(1e-6),
        (maxs[2] - mins[2]).max(1e-6),
    ];
    points
        .iter()
        .map(|p| {
            [
                ((p[0] - mins[0]) / spans[0]) * 4.0 - 2.0,
                ((p[1] - mins[1]) / spans[1]) * 4.0 - 2.0,
                ((p[2] - mins[2]) / spans[2]) * 2.0 - 1.0,
            ]
        })
        .collect()
}

impl SurfaceViewer {
    pub(super) fn sync_live_dataflow(&mut self, ctx: &egui::Context) {
        if let Some(field) = poll_pane_scalar_field(ctx) {
            let needs_update = !matches!(
                &self.live_feed.feed,
                LiveSurfaceFeed::ScalarField { payload, .. } if payload.revision == field.revision
            );
            if needs_update {
                let (mean, inv_span) = field_stats(&field.values);
                self.live_feed.feed = LiveSurfaceFeed::ScalarField {
                    payload: field,
                    mean,
                    inv_span,
                };
            }
            return;
        }
        if let Some(landmarks) = poll_pane_landmarks(ctx) {
            let needs_update = !matches!(
                &self.live_feed.feed,
                LiveSurfaceFeed::Landmarks { payload, .. } if payload.revision == landmarks.revision
            );
            if needs_update {
                let norm = Arc::new(normalize_landmarks(&landmarks.points));
                self.live_feed.feed = LiveSurfaceFeed::Landmarks {
                    payload: landmarks,
                    normalized_xy_z: norm,
                };
            }
            return;
        }
        self.live_feed.feed = LiveSurfaceFeed::None;
    }

    pub(super) fn build_live_surface(&self) -> Option<Box<dyn ParametricSurface>> {
        match &self.live_feed.feed {
            LiveSurfaceFeed::None => None,
            LiveSurfaceFeed::ScalarField {
                payload,
                mean,
                inv_span,
            } => Some(Box::new(DiscreteFieldHeightSurface {
                field: Arc::clone(payload),
                mean: *mean,
                inv_span: *inv_span,
                z_amplitude: self.live_feed.z_amplitude,
            })),
            LiveSurfaceFeed::Landmarks {
                normalized_xy_z, ..
            } => Some(Box::new(LandmarkShepardSurface {
                samples: Arc::clone(normalized_xy_z),
                z_amplitude: self.live_feed.z_amplitude,
            })),
        }
    }

    pub(super) fn active_uv_domain(&self) -> ((f64, f64), (f64, f64)) {
        match &self.live_feed.feed {
            LiveSurfaceFeed::ScalarField { payload, .. } => (
                (payload.u_bounds[0], payload.u_bounds[1]),
                (payload.v_bounds[0], payload.v_bounds[1]),
            ),
            LiveSurfaceFeed::Landmarks { .. } => ((-2.0, 2.0), (-2.0, 2.0)),
            LiveSurfaceFeed::None => self.custom_sandbox.uv_domain_for(self.surface_type),
        }
    }

    pub(super) fn hash_live_feed<H: Hasher>(&self, state: &mut H) -> bool {
        match &self.live_feed.feed {
            LiveSurfaceFeed::None => false,
            LiveSurfaceFeed::ScalarField { payload, .. } => {
                1u8.hash(state);
                payload.revision.hash(state);
                self.live_feed.z_amplitude.to_bits().hash(state);
                true
            }
            LiveSurfaceFeed::Landmarks { payload, .. } => {
                2u8.hash(state);
                payload.revision.hash(state);
                self.live_feed.z_amplitude.to_bits().hash(state);
                true
            }
        }
    }

    pub(super) fn live_stream_summary(&self) -> Option<String> {
        match &self.live_feed.feed {
            LiveSurfaceFeed::None => None,
            LiveSurfaceFeed::ScalarField { payload, .. } => Some(format!(
                "⚡ Live 2D Field ({}×{}, rev {})",
                payload.width, payload.height, payload.revision
            )),
            LiveSurfaceFeed::Landmarks { payload, .. } => Some(format!(
                "⚡ Live 3D Landmarks ({} pts, rev {})",
                payload.points.len(),
                payload.revision
            )),
        }
    }
}

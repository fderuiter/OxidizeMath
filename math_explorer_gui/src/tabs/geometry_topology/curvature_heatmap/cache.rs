#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::custom_surface::SurfaceType;
use super::{CurvatureHeatmap, CurvatureType};
use eframe::egui;
use egui_plot::{PlotBounds, PlotPoint};
use math_commons::math_kernel::colormap::mapped_gradient;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::differential_geometry::surface::{
    KleinBottle, ParametricSurface, Sphere, SurfaceAnalysis, Torus,
};
use nalgebra::Point3;
use std::hash::{Hash as _, Hasher as _};
use std::sync::Arc;

pub(super) const CURVATURE_COLOR_BINS: usize = 64;

#[derive(Clone)]
pub(super) struct CurvatureColorBatch {
    pub(super) bin: usize,
    pub(super) color: egui::Color32,
    pub(super) segments: Arc<Vec<Vec<PlotPoint>>>,
    pub(super) bounds: PlotBounds,
}

#[derive(Default)]
pub(super) struct CurvatureGeometryCache {
    surface_key: Option<u64>,
    u_res: usize,
    v_res: usize,
    positions_3d: Vec<Point3<f64>>,
    curvatures: Vec<f64>,
    min_k: f64,
    max_k: f64,
    proj_key: Option<(u64, [u32; 3])>,
    batches: Vec<CurvatureColorBatch>,
    pub(super) curvature_eval_count: usize,
    pub(super) projection_count: usize,
}

fn canonical_bits(val: f64) -> u64 {
    if val == 0.0 {
        0
    } else {
        val.to_bits()
    }
}

impl CurvatureHeatmap {
    pub(super) fn curvature_geometry_key(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.surface_type as u8).hash(&mut h);
        (self.curvature_type as u8).hash(&mut h);
        self.u_resolution.hash(&mut h);
        self.v_resolution.hash(&mut h);
        match self.surface_type {
            SurfaceType::Sphere => canonical_bits(self.sphere_radius).hash(&mut h),
            SurfaceType::Torus => {
                canonical_bits(self.torus_major_radius).hash(&mut h);
                canonical_bits(self.torus_minor_radius).hash(&mut h);
            }
            SurfaceType::KleinBottle => canonical_bits(self.klein_radius).hash(&mut h),
            SurfaceType::CustomHeightfield | SurfaceType::CustomParametric => {
                self.custom_sandbox
                    .state_fingerprint(self.surface_type)
                    .hash(&mut h);
            }
        }
        h.finish().max(1)
    }

    pub(super) fn camera_pose_bits(&self) -> [u32; 3] {
        [
            self.camera.pitch.to_bits(),
            self.camera.yaw.to_bits(),
            self.camera.zoom.to_bits(),
        ]
    }

    fn evaluate_surface_curvature(&mut self) {
        let uv = self.surface_type.uv_domain();
        let u_res = self.u_resolution;
        let v_res = self.v_resolution;
        let ctype = self.curvature_type;
        let (pos, curv, min_k, max_k) = match self.surface_type {
            SurfaceType::Sphere => sample_positions_and_curvature(
                &Sphere {
                    radius: self.sphere_radius,
                },
                uv,
                u_res,
                v_res,
                ctype,
            ),
            SurfaceType::Torus => sample_positions_and_curvature(
                &Torus {
                    major_radius: self.torus_major_radius,
                    minor_radius: self.torus_minor_radius,
                },
                uv,
                u_res,
                v_res,
                ctype,
            ),
            SurfaceType::KleinBottle => sample_positions_and_curvature(
                &KleinBottle {
                    radius: self.klein_radius,
                },
                uv,
                u_res,
                v_res,
                ctype,
            ),
            SurfaceType::CustomHeightfield => sample_positions_and_curvature(
                &self.custom_sandbox.build_heightfield(),
                uv,
                u_res,
                v_res,
                ctype,
            ),
            SurfaceType::CustomParametric => sample_positions_and_curvature(
                &self.custom_sandbox.build_parametric(),
                uv,
                u_res,
                v_res,
                ctype,
            ),
        };
        self.cache.positions_3d = pos;
        self.cache.curvatures = curv;
        self.cache.min_k = min_k;
        self.cache.max_k = max_k;
        self.cache.u_res = u_res;
        self.cache.v_res = v_res;
        self.cache.curvature_eval_count += 1;
    }

    pub(super) fn ensure_projected_batches(&mut self) -> &[CurvatureColorBatch] {
        let surf_key = self.curvature_geometry_key();
        if self.cache.surface_key != Some(surf_key)
            || self.cache.u_res != self.u_resolution
            || self.cache.v_res != self.v_resolution
        {
            self.evaluate_surface_curvature();
            self.cache.surface_key = Some(surf_key);
        }
        let proj_key = (surf_key, self.camera_pose_bits());
        if self.cache.proj_key != Some(proj_key) {
            self.cache.batches = build_batched_curvature_lines(
                &self.cache.positions_3d,
                &self.cache.curvatures,
                self.cache.u_res,
                self.cache.v_res,
                self.cache.min_k,
                self.cache.max_k,
                &self.camera,
            );
            self.cache.proj_key = Some(proj_key);
            self.cache.projection_count += 1;
        }
        &self.cache.batches
    }
}

fn sample_positions_and_curvature<T: ParametricSurface>(
    surface: &T,
    uv_domain: ((f64, f64), (f64, f64)),
    u_res: usize,
    v_res: usize,
    curvature_type: CurvatureType,
) -> (Vec<Point3<f64>>, Vec<f64>, f64, f64) {
    let ((u_min, u_max), (v_min, v_max)) = uv_domain;
    let total = (u_res + 1) * (v_res + 1);
    let mut positions = Vec::with_capacity(total);
    let mut curvatures = Vec::with_capacity(total);
    let mut min_k = f64::INFINITY;
    let mut max_k = f64::NEG_INFINITY;

    for i in 0..=u_res {
        let u = u_min + (u_max - u_min) * (i as f64 / u_res.max(1) as f64);
        for j in 0..=v_res {
            let v = v_min + (v_max - v_min) * (j as f64 / v_res.max(1) as f64);
            positions.push(surface.position(u, v));
            let raw = match curvature_type {
                CurvatureType::Gaussian => surface.gaussian_curvature(u, v),
                CurvatureType::Mean => surface.mean_curvature(u, v),
            };
            let c = if raw.is_finite() {
                raw.clamp(-1e4, 1e4)
            } else {
                0.0
            };
            curvatures.push(c);
            min_k = min_k.min(c);
            max_k = max_k.max(c);
        }
    }
    if !min_k.is_finite() || !max_k.is_finite() {
        (positions, curvatures, 0.0, 1.0)
    } else {
        (positions, curvatures, min_k, max_k)
    }
}

fn quantize_curvature_bin(value: f64, min_k: f64, max_k: f64) -> usize {
    let span = max_k - min_k;
    let t = if !span.is_finite() || span.abs() < 1e-12 {
        0.5
    } else {
        ((value - min_k) / span).clamp(0.0, 1.0)
    };
    ((t * (CURVATURE_COLOR_BINS - 1) as f64).round() as usize).min(CURVATURE_COLOR_BINS - 1)
}

fn push_edge_to_run(
    edge: Option<(PlotPoint, PlotPoint, usize)>,
    active_bin: &mut Option<usize>,
    run: &mut Vec<PlotPoint>,
    buckets: &mut [Vec<Vec<PlotPoint>>],
) {
    match edge {
        Some((p1, p2, bin)) => {
            if *active_bin == Some(bin) {
                run.push(p2);
            } else {
                if let Some(prev) = active_bin.take() {
                    if run.len() >= 2 {
                        buckets[prev].push(std::mem::take(run));
                    }
                }
                run.clear();
                run.push(p1);
                run.push(p2);
                *active_bin = Some(bin);
            }
        }
        None => {
            if let Some(prev) = active_bin.take() {
                if run.len() >= 2 {
                    buckets[prev].push(std::mem::take(run));
                }
            }
            run.clear();
        }
    }
}

fn build_batched_curvature_lines(
    positions: &[Point3<f64>],
    curvatures: &[f64],
    u_res: usize,
    v_res: usize,
    min_k: f64,
    max_k: f64,
    camera: &crate::framework::Camera3D,
) -> Vec<CurvatureColorBatch> {
    let cols = v_res + 1;
    let pts_2d: Vec<Option<PlotPoint>> = positions
        .iter()
        .map(|p| {
            let p2 = camera.project(&[p.x, p.y, p.z]);
            (p2[0].is_finite() && p2[1].is_finite()).then(|| PlotPoint::new(p2[0], p2[1]))
        })
        .collect();

    let mut buckets: Vec<Vec<Vec<PlotPoint>>> = vec![Vec::new(); CURVATURE_COLOR_BINS];
    let mut run = Vec::new();
    let mut active_bin = None;

    for i in 0..=u_res {
        for j in 0..v_res {
            let idx0 = flatten_2d_index(j, i, cols);
            let idx1 = idx0 + 1;
            let edge = pts_2d[idx0].zip(pts_2d[idx1]).map(|(p1, p2)| {
                let bin = quantize_curvature_bin(
                    0.5 * (curvatures[idx0] + curvatures[idx1]),
                    min_k,
                    max_k,
                );
                (p1, p2, bin)
            });
            push_edge_to_run(edge, &mut active_bin, &mut run, &mut buckets);
        }
        push_edge_to_run(None, &mut active_bin, &mut run, &mut buckets);
    }

    for j in 0..=v_res {
        for i in 0..u_res {
            let idx0 = flatten_2d_index(j, i, cols);
            let idx1 = flatten_2d_index(j, i + 1, cols);
            let edge = pts_2d[idx0].zip(pts_2d[idx1]).map(|(p1, p2)| {
                let bin = quantize_curvature_bin(
                    0.5 * (curvatures[idx0] + curvatures[idx1]),
                    min_k,
                    max_k,
                );
                (p1, p2, bin)
            });
            push_edge_to_run(edge, &mut active_bin, &mut run, &mut buckets);
        }
        push_edge_to_run(None, &mut active_bin, &mut run, &mut buckets);
    }

    finalize_color_batches(buckets, min_k, max_k)
}

fn finalize_color_batches(
    buckets: Vec<Vec<Vec<PlotPoint>>>,
    min_k: f64,
    max_k: f64,
) -> Vec<CurvatureColorBatch> {
    let mut out = Vec::new();
    let max_idx = (CURVATURE_COLOR_BINS - 1) as f64;
    for (bin, segs) in buckets.into_iter().enumerate() {
        if segs.is_empty() {
            continue;
        }
        let mut bounds = PlotBounds::NOTHING;
        for seg in &segs {
            for pt in seg {
                bounds.extend_with(pt);
            }
        }
        let sample_val = min_k + (max_k - min_k) * (bin as f64 / max_idx);
        let rgb = mapped_gradient(sample_val, min_k, max_k);
        out.push(CurvatureColorBatch {
            bin,
            color: egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]),
            segments: Arc::new(segs),
            bounds,
        });
    }
    out
}

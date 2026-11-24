#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::CurvatureHeatmap;
use eframe::egui;
use egui_plot::LineStyle;
use math_commons::math_kernel::colormap::mapped_gradient;
use math_explorer::pure_math::differential_geometry::surface::{
    CurvatureKind, KleinBottle, ParametricSurface, ParametricSurfaceKind, Sphere, SurfaceAnalysis,
    Torus,
};
use nalgebra::Point3;
use std::hash::{Hash as _, Hasher as _};

pub(super) const CURVATURE_COLOR_BINS: usize = 64;

#[derive(Default)]
pub(super) struct CurvatureGeometryCache {
    surface_key: Option<u64>,
    u_res: usize,
    v_res: usize,
    positions_3d: Vec<Point3<f64>>,
    curvatures: Vec<f64>,
    min_k: f64,
    max_k: f64,
    proj_key: Option<(u64, u64, [u8; 4])>,
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
            ParametricSurfaceKind::Sphere => canonical_bits(self.sphere_radius).hash(&mut h),
            ParametricSurfaceKind::Torus => {
                canonical_bits(self.torus_major_radius).hash(&mut h);
                canonical_bits(self.torus_minor_radius).hash(&mut h);
            }
            ParametricSurfaceKind::KleinBottle => canonical_bits(self.klein_radius).hash(&mut h),
            ParametricSurfaceKind::CustomHeightfield | ParametricSurfaceKind::CustomParametric => {
                self.custom_sandbox
                    .state_fingerprint(self.surface_type)
                    .hash(&mut h);
            }
        }
        h.finish().max(1)
    }

    fn evaluate_surface_curvature(&mut self) {
        let uv = self.custom_sandbox.uv_domain_for(self.surface_type);
        let u_res = self.u_resolution;
        let v_res = self.v_resolution;
        let ctype = self.curvature_type;
        let (pos, curv, min_k, max_k) = match self.surface_type {
            ParametricSurfaceKind::Sphere => sample_positions_and_curvature(
                &Sphere {
                    radius: self.sphere_radius,
                },
                uv,
                u_res,
                v_res,
                ctype,
            ),
            ParametricSurfaceKind::Torus => sample_positions_and_curvature(
                &Torus {
                    major_radius: self.torus_major_radius,
                    minor_radius: self.torus_minor_radius,
                },
                uv,
                u_res,
                v_res,
                ctype,
            ),
            ParametricSurfaceKind::KleinBottle => sample_positions_and_curvature(
                &KleinBottle {
                    radius: self.klein_radius,
                },
                uv,
                u_res,
                v_res,
                ctype,
            ),
            ParametricSurfaceKind::CustomHeightfield => sample_positions_and_curvature(
                &self.custom_sandbox.build_heightfield(),
                uv,
                u_res,
                v_res,
                ctype,
            ),
            ParametricSurfaceKind::CustomParametric => sample_positions_and_curvature(
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

    pub(super) fn populate_cached_scene(&mut self, series_color: egui::Color32) {
        let surf_key = self.curvature_geometry_key();
        if self.cache.surface_key != Some(surf_key)
            || self.cache.u_res != self.u_resolution
            || self.cache.v_res != self.v_resolution
        {
            self.evaluate_surface_curvature();
            self.cache.surface_key = Some(surf_key);
        }
        let proj_key = (
            surf_key,
            self.camera.pose_fingerprint(),
            series_color.to_array(),
        );
        if self.cache.proj_key != Some(proj_key) {
            self.rebuild_curvature_scene(series_color);
            self.cache.proj_key = Some(proj_key);
            self.cache.projection_count += 1;
        }
    }

    fn rebuild_curvature_scene(&mut self, series_color: egui::Color32) {
        let lut = build_curvature_palette(self.cache.min_k, self.cache.max_k);
        let min_k = self.cache.min_k;
        let max_k = self.cache.max_k;
        let u_res = self.cache.u_res;
        let v_res = self.cache.v_res;
        let cols = v_res + 1;

        self.scene_builder.begin(self.camera);
        for i in 0..u_res {
            for j in 0..v_res {
                let idx = i * cols + j;
                let p1 = self.cache.positions_3d[idx];
                let p2 = self.cache.positions_3d[idx + 1];
                let p3 = self.cache.positions_3d[idx + cols];
                let c1 = self.cache.curvatures[idx];
                let col_v = lut[quantize_curvature_bin(
                    0.5 * (c1 + self.cache.curvatures[idx + 1]),
                    min_k,
                    max_k,
                )];
                let col_u = lut[quantize_curvature_bin(
                    0.5 * (c1 + self.cache.curvatures[idx + cols]),
                    min_k,
                    max_k,
                )];
                self.scene_builder.add_styled_segment(
                    "Curvature Surface",
                    [p1, p2],
                    egui::Stroke::new(1.25_f32, col_v),
                    LineStyle::Solid,
                    true,
                    false,
                );
                self.scene_builder.add_styled_segment(
                    "Curvature Surface",
                    [p1, p3],
                    egui::Stroke::new(1.25_f32, col_u),
                    LineStyle::Solid,
                    true,
                    false,
                );
            }
        }
        self.scene_builder.record_series(
            "Curvature Surface",
            &self.cache.positions_3d,
            series_color,
        );
        self.scene_builder.ensure_sorted();
    }
}

fn sample_positions_and_curvature<T: ParametricSurface>(
    surface: &T,
    uv_domain: ((f64, f64), (f64, f64)),
    u_res: usize,
    v_res: usize,
    curvature_type: CurvatureKind,
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
                CurvatureKind::Gaussian => surface.gaussian_curvature(u, v),
                CurvatureKind::Mean => surface.mean_curvature(u, v),
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

fn build_curvature_palette(min_k: f64, max_k: f64) -> [egui::Color32; CURVATURE_COLOR_BINS] {
    let max_idx = (CURVATURE_COLOR_BINS - 1) as f64;
    std::array::from_fn(|bin| {
        let sample = min_k + (max_k - min_k) * (bin as f64 / max_idx);
        let rgb = mapped_gradient(sample, min_k, max_k);
        egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2])
    })
}

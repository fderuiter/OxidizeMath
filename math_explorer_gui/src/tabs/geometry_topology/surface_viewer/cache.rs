#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::custom_surface::SurfaceType;
use super::SurfaceViewer;
use eframe::egui;
use egui_plot::LineStyle;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::differential_geometry::surface::ParametricSurface;
use nalgebra::Point3;
use std::hash::{Hash as _, Hasher as _};

#[derive(Default)]
pub(super) struct SurfaceGeometryCache {
    surface_key: Option<u64>,
    u_res: usize,
    v_res: usize,
    positions_3d: Vec<Point3<f64>>,
    line_scratch: Vec<Point3<f64>>,
    proj_key: Option<(u64, u64, [u8; 4])>,
    pub(super) surface_eval_count: usize,
    pub(super) projection_count: usize,
}

fn canonical_bits(val: f64) -> u64 {
    if val == 0.0 {
        0
    } else {
        val.to_bits()
    }
}

impl SurfaceViewer {
    pub(super) fn surface_geometry_key(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.surface_type as u8).hash(&mut h);
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

    fn ensure_surface_positions(&mut self) -> u64 {
        let key = self.surface_geometry_key();
        if self.cache.surface_key == Some(key)
            && self.cache.u_res == self.u_resolution
            && self.cache.v_res == self.v_resolution
        {
            return key;
        }
        let surface = self.create_surface();
        let uv_domain = self.custom_sandbox.uv_domain_for(self.surface_type);
        self.cache.positions_3d = sample_surface_positions(
            surface.as_ref(),
            uv_domain,
            self.u_resolution,
            self.v_resolution,
        );
        self.cache.surface_key = Some(key);
        self.cache.u_res = self.u_resolution;
        self.cache.v_res = self.v_resolution;
        self.cache.surface_eval_count += 1;
        key
    }

    pub(super) fn populate_cached_wireframe(&mut self, color: egui::Color32) {
        let surf_key = self.ensure_surface_positions();
        let proj_key = (surf_key, self.camera.pose_fingerprint(), color.to_array());
        if self.cache.proj_key != Some(proj_key) {
            self.rebuild_wireframe_scene(color);
            self.cache.proj_key = Some(proj_key);
            self.cache.projection_count += 1;
        }
    }

    fn rebuild_wireframe_scene(&mut self, color: egui::Color32) {
        let u_res = self.cache.u_res;
        let v_res = self.cache.v_res;
        let cols = v_res + 1;
        let stroke = egui::Stroke::new(1.25_f32, color);

        self.scene_builder.begin(self.camera);
        for i in 0..=u_res {
            self.cache.line_scratch.clear();
            for j in 0..=v_res {
                self.cache
                    .line_scratch
                    .push(self.cache.positions_3d[flatten_2d_index(j, i, cols)]);
            }
            self.scene_builder.add_polyline(
                "Surface",
                &self.cache.line_scratch,
                stroke,
                LineStyle::Solid,
                false,
            );
        }
        for j in 0..=v_res {
            self.cache.line_scratch.clear();
            for i in 0..=u_res {
                self.cache
                    .line_scratch
                    .push(self.cache.positions_3d[flatten_2d_index(j, i, cols)]);
            }
            self.scene_builder.add_polyline(
                "Surface",
                &self.cache.line_scratch,
                stroke,
                LineStyle::Solid,
                false,
            );
        }
        self.scene_builder
            .record_series("Surface", &self.cache.positions_3d, color);
        self.scene_builder.ensure_sorted();
    }
}

fn sample_surface_positions(
    surface: &dyn ParametricSurface,
    uv_domain: ((f64, f64), (f64, f64)),
    u_res: usize,
    v_res: usize,
) -> Vec<Point3<f64>> {
    let ((u_min, u_max), (v_min, v_max)) = uv_domain;
    let cols = v_res + 1;
    let mut out = Vec::with_capacity((u_res + 1) * cols);
    for i in 0..=u_res {
        let u = u_min + (u_max - u_min) * (i as f64 / u_res.max(1) as f64);
        for j in 0..=v_res {
            let v = v_min + (v_max - v_min) * (j as f64 / v_res.max(1) as f64);
            out.push(surface.position(u, v));
        }
    }
    out
}

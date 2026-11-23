#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::SurfaceViewer;
use egui_plot::{PlotBounds, PlotPoint};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::differential_geometry::surface::{
    ParametricSurface, ParametricSurfaceKind,
};
use nalgebra::Point3;
use std::hash::{Hash as _, Hasher as _};
use std::sync::Arc;

pub(super) struct SurfaceGeometryCache {
    surface_key: Option<u64>,
    u_res: usize,
    v_res: usize,
    positions_3d: Vec<Point3<f64>>,
    proj_key: Option<(u64, [u32; 3])>,
    segments_2d: Arc<Vec<Vec<PlotPoint>>>,
    bounds_2d: PlotBounds,
    pub(super) surface_eval_count: usize,
    pub(super) projection_count: usize,
}

impl Default for SurfaceGeometryCache {
    fn default() -> Self {
        Self {
            surface_key: None,
            u_res: 0,
            v_res: 0,
            positions_3d: Vec::new(),
            proj_key: None,
            segments_2d: Arc::new(Vec::new()),
            bounds_2d: PlotBounds::NOTHING,
            surface_eval_count: 0,
            projection_count: 0,
        }
    }
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

    pub(super) fn camera_pose_bits(&self) -> [u32; 3] {
        [
            self.camera.pitch.to_bits(),
            self.camera.yaw.to_bits(),
            self.camera.zoom.to_bits(),
        ]
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

    pub(super) fn ensure_projected_wireframe(&mut self) -> (Arc<Vec<Vec<PlotPoint>>>, PlotBounds) {
        let surf_key = self.ensure_surface_positions();
        let proj_key = (surf_key, self.camera_pose_bits());
        if self.cache.proj_key != Some(proj_key) {
            let (segs, bounds) = project_wireframe_grid(
                &self.cache.positions_3d,
                self.cache.u_res,
                self.cache.v_res,
                &self.camera,
            );
            self.cache.segments_2d = Arc::new(segs);
            self.cache.bounds_2d = bounds;
            self.cache.proj_key = Some(proj_key);
            self.cache.projection_count += 1;
        }
        (Arc::clone(&self.cache.segments_2d), self.cache.bounds_2d)
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

fn push_projected_point(
    pt_3d: Point3<f64>,
    camera: &crate::framework::Camera3D,
    current: &mut Vec<PlotPoint>,
    segments: &mut Vec<Vec<PlotPoint>>,
    bounds: &mut PlotBounds,
) {
    let p2 = camera.project(&[pt_3d.x, pt_3d.y, pt_3d.z]);
    if p2[0].is_finite() && p2[1].is_finite() {
        bounds.extend_with(&PlotPoint::new(p2[0], p2[1]));
        current.push(PlotPoint::new(p2[0], p2[1]));
    } else if current.len() >= 2 {
        segments.push(std::mem::take(current));
    } else {
        current.clear();
    }
}

fn project_wireframe_grid(
    positions: &[Point3<f64>],
    u_res: usize,
    v_res: usize,
    camera: &crate::framework::Camera3D,
) -> (Vec<Vec<PlotPoint>>, PlotBounds) {
    let cols = v_res + 1;
    let mut segments = Vec::with_capacity((u_res + 1) + (v_res + 1));
    let mut bounds = PlotBounds::NOTHING;
    let mut current = Vec::new();

    for i in 0..=u_res {
        current.clear();
        for j in 0..=v_res {
            push_projected_point(
                positions[flatten_2d_index(j, i, cols)],
                camera,
                &mut current,
                &mut segments,
                &mut bounds,
            );
        }
        if current.len() >= 2 {
            segments.push(std::mem::take(&mut current));
        }
    }
    for j in 0..=v_res {
        current.clear();
        for i in 0..=u_res {
            push_projected_point(
                positions[flatten_2d_index(j, i, cols)],
                camera,
                &mut current,
                &mut segments,
                &mut bounds,
            );
        }
        if current.len() >= 2 {
            segments.push(std::mem::take(&mut current));
        }
    }
    (segments, bounds)
}

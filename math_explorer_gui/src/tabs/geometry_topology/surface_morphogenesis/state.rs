// [cite:modular_polynomials_review]
//! State serialization, validation, and `FiniteStateGuard` recovery for `SurfaceMorphogenesisTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    MAX_SURFACE_MORPHO_RES, MIN_SURFACE_MORPHO_RES, MorphogenesisKineticsKind,
    SurfaceMorphogenesisTool, SurfaceOverlayMode,
};
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{Camera3D, InteractiveTool};
use crate::tabs::geometry_topology::custom_surface::SurfaceType;
use crate::tabs::geometry_topology::export_utils::{
    DEFAULT_WALL_THICKNESS, MAX_WALL_THICKNESS, MIN_WALL_THICKNESS,
};
use std::collections::BTreeMap;

pub(super) const fn default_wall_thickness() -> f64 {
    DEFAULT_WALL_THICKNESS
}
pub(super) const fn default_schnak_a() -> f64 {
    0.1
}
pub(super) const fn default_schnak_b() -> f64 {
    0.9
}
pub(super) const fn default_gs_feed() -> f64 {
    0.0367
}
pub(super) const fn default_gs_kill() -> f64 {
    0.0649
}
pub(super) const fn default_d_u() -> f64 {
    0.05
}
pub(super) const fn default_d_v() -> f64 {
    1.25
}
pub(super) const fn default_buckling_gain() -> f64 {
    0.28
}
pub(super) const fn default_tension() -> f64 {
    0.06
}
pub(super) const fn default_tether() -> f64 {
    0.12
}
pub(super) const fn default_dt() -> f64 {
    0.01
}
pub(super) const fn default_steps() -> usize {
    4
}

#[derive(serde::Serialize)]
pub(super) struct SurfaceMorphogenesisSaveState {
    pub surface_type: SurfaceType,
    pub kinetics_kind: MorphogenesisKineticsKind,
    pub overlay_mode: SurfaceOverlayMode,
    pub sphere_radius: f64,
    pub torus_major_radius: f64,
    pub torus_minor_radius: f64,
    pub klein_radius: f64,
    pub wall_thickness: f64,
    pub u_resolution: usize,
    pub v_resolution: usize,
    pub schnak_a: f64,
    pub schnak_b: f64,
    pub gs_feed: f64,
    pub gs_kill: f64,
    pub d_u: f64,
    pub d_v: f64,
    pub buckling_gain: f64,
    pub curvature_tension: f64,
    pub restoring_tether: f64,
    pub dt: f64,
    pub steps_per_frame: usize,
    pub camera: Camera3D,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_expr: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parametric_exprs: Option<[String; 3]>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub custom_params: BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub u_bounds: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub v_bounds: Option<[f64; 2]>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SurfaceMorphogenesisLoadState {
    pub surface_type: SurfaceType,
    #[serde(default)]
    pub kinetics_kind: MorphogenesisKineticsKind,
    #[serde(default)]
    pub overlay_mode: SurfaceOverlayMode,
    pub sphere_radius: f64,
    pub torus_major_radius: f64,
    pub torus_minor_radius: f64,
    pub klein_radius: f64,
    #[serde(default = "default_wall_thickness")]
    pub wall_thickness: f64,
    pub u_resolution: usize,
    pub v_resolution: usize,
    #[serde(default = "default_schnak_a")]
    pub schnak_a: f64,
    #[serde(default = "default_schnak_b")]
    pub schnak_b: f64,
    #[serde(default = "default_gs_feed")]
    pub gs_feed: f64,
    #[serde(default = "default_gs_kill")]
    pub gs_kill: f64,
    #[serde(default = "default_d_u")]
    pub d_u: f64,
    #[serde(default = "default_d_v")]
    pub d_v: f64,
    #[serde(default = "default_buckling_gain")]
    pub buckling_gain: f64,
    #[serde(default = "default_tension")]
    pub curvature_tension: f64,
    #[serde(default = "default_tether")]
    pub restoring_tether: f64,
    #[serde(default = "default_dt")]
    pub dt: f64,
    #[serde(default = "default_steps")]
    pub steps_per_frame: usize,
    #[serde(default)]
    pub camera: Option<Camera3D>,
    #[serde(default)]
    pub pitch: Option<f32>,
    #[serde(default)]
    pub yaw: Option<f32>,
    #[serde(default)]
    pub zoom: Option<f32>,
    #[serde(default)]
    pub custom_expr: Option<String>,
    #[serde(default)]
    pub parametric_exprs: Option<[String; 3]>,
    #[serde(default)]
    pub custom_params: BTreeMap<String, f64>,
    #[serde(default)]
    pub u_bounds: Option<[f64; 2]>,
    #[serde(default)]
    pub v_bounds: Option<[f64; 2]>,
}

fn sanitize_range(val: f64, lo: f64, hi: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val.clamp(lo, hi)
    } else {
        fallback
    }
}

pub(super) fn validate_loaded_state(s: &SurfaceMorphogenesisLoadState) -> Result<(), String> {
    if !(0.2..=5.0).contains(&s.sphere_radius)
        || !(1.0..=5.0).contains(&s.torus_major_radius)
        || !(0.15..=2.0).contains(&s.torus_minor_radius)
        || !(0.5..=5.0).contains(&s.klein_radius)
    {
        return Err("Surface radii out of valid bounds".to_string());
    }
    if !(MIN_WALL_THICKNESS..=MAX_WALL_THICKNESS).contains(&s.wall_thickness) {
        return Err("Wall thickness out of valid bounds".to_string());
    }
    let max_res = math_commons::registry::MAX_GRID_SIZE;
    if !(MIN_SURFACE_MORPHO_RES..=max_res).contains(&s.u_resolution)
        || !(MIN_SURFACE_MORPHO_RES..=max_res).contains(&s.v_resolution)
    {
        return Err("Surface morphogenesis resolution out of bounds".to_string());
    }
    if !s.dt.is_finite() || !(1e-6..=0.25).contains(&s.dt) || s.d_u <= 0.0 || s.d_v <= 0.0 {
        return Err("Reaction-diffusion time step or diffusivity out of bounds".to_string());
    }
    Ok(())
}

impl SurfaceMorphogenesisTool {
    /// Clamps any out-of-range or non-finite parameters and rolls back non-finite solver state.
    pub fn sanitize_and_recover(&mut self) {
        self.sphere_radius = sanitize_range(self.sphere_radius, 0.2, 5.0, 1.5);
        self.torus_major_radius = sanitize_range(self.torus_major_radius, 1.0, 5.0, 2.2);
        self.torus_minor_radius = sanitize_range(self.torus_minor_radius, 0.15, 2.0, 0.75);
        self.klein_radius = sanitize_range(self.klein_radius, 0.5, 5.0, 2.0);
        self.wall_thickness = sanitize_range(
            self.wall_thickness,
            MIN_WALL_THICKNESS,
            MAX_WALL_THICKNESS,
            DEFAULT_WALL_THICKNESS,
        );
        self.u_resolution = self.clamped_u_res();
        self.v_resolution = self.clamped_v_res();
        self.schnak_a = sanitize_range(self.schnak_a, 0.01, 1.0, 0.1);
        self.schnak_b = sanitize_range(self.schnak_b, 0.1, 2.5, 0.9);
        self.gs_feed = sanitize_range(self.gs_feed, 0.005, 0.12, 0.0367);
        self.gs_kill = sanitize_range(self.gs_kill, 0.01, 0.12, 0.0649);
        self.d_u = sanitize_range(self.d_u, 1e-4, 2.0, 0.05);
        self.d_v = sanitize_range(self.d_v, 1e-4, 5.0, 1.25);
        self.buckling_gain = sanitize_range(self.buckling_gain, -1.5, 1.5, 0.28);
        self.curvature_tension = sanitize_range(self.curvature_tension, 0.0, 1.0, 0.06);
        self.restoring_tether = sanitize_range(self.restoring_tether, 0.0, 2.0, 0.12);
        self.dt = sanitize_range(self.dt, 1e-5, 0.1, 0.01);
        self.steps_per_frame = self.steps_per_frame.clamp(1, 16);
        if let Some(eng) = self.engine.as_mut() {
            let finite_fields = eng.runner.u().iter().all(|x| x.is_finite())
                && eng.runner.v().iter().all(|x| x.is_finite())
                && eng
                    .flow
                    .points()
                    .iter()
                    .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
            if !finite_fields {
                eng.rollback_to_last_valid();
            }
        }
    }

    pub(super) fn has_finite_parameters_internal(&self) -> bool {
        (0.2..=5.0).contains(&self.sphere_radius)
            && (1.0..=5.0).contains(&self.torus_major_radius)
            && (0.15..=2.0).contains(&self.torus_minor_radius)
            && (0.5..=5.0).contains(&self.klein_radius)
            && (MIN_WALL_THICKNESS..=MAX_WALL_THICKNESS).contains(&self.wall_thickness)
            && (MIN_SURFACE_MORPHO_RES..=MAX_SURFACE_MORPHO_RES).contains(&self.u_resolution)
            && (MIN_SURFACE_MORPHO_RES..=MAX_SURFACE_MORPHO_RES).contains(&self.v_resolution)
            && self.schnak_a.is_finite()
            && self.schnak_b.is_finite()
            && self.gs_feed.is_finite()
            && self.gs_kill.is_finite()
            && self.d_u.is_finite()
            && self.d_u > 0.0
            && self.d_v.is_finite()
            && self.d_v > 0.0
            && self.buckling_gain.is_finite()
            && self.curvature_tension.is_finite()
            && self.curvature_tension >= 0.0
            && self.restoring_tether.is_finite()
            && self.restoring_tether >= 0.0
            && self.dt.is_finite()
            && (1e-6..=0.25).contains(&self.dt)
            && self.camera.is_valid()
    }

    pub(super) fn encode_state_json(&self) -> Option<String> {
        let (custom_expr, parametric_exprs, custom_params) =
            self.custom_sandbox.export_fields(self.surface_type);
        let (u_bounds, v_bounds) = self.custom_sandbox.export_domain_bounds(self.surface_type);
        serde_json::to_string(&SurfaceMorphogenesisSaveState {
            surface_type: self.surface_type,
            kinetics_kind: self.kinetics_kind,
            overlay_mode: self.overlay_mode,
            sphere_radius: self.sphere_radius,
            torus_major_radius: self.torus_major_radius,
            torus_minor_radius: self.torus_minor_radius,
            klein_radius: self.klein_radius,
            wall_thickness: self.wall_thickness,
            u_resolution: self.u_resolution,
            v_resolution: self.v_resolution,
            schnak_a: self.schnak_a,
            schnak_b: self.schnak_b,
            gs_feed: self.gs_feed,
            gs_kill: self.gs_kill,
            d_u: self.d_u,
            d_v: self.d_v,
            buckling_gain: self.buckling_gain,
            curvature_tension: self.curvature_tension,
            restoring_tether: self.restoring_tether,
            dt: self.dt,
            steps_per_frame: self.steps_per_frame,
            camera: self.camera,
            custom_expr,
            parametric_exprs,
            custom_params,
            u_bounds,
            v_bounds,
        })
        .ok()
    }

    pub(super) fn apply_loaded_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let s: SurfaceMorphogenesisLoadState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid Surface Morphogenesis state: {e}"))?;
        validate_loaded_state(&s)?;
        let camera = Camera3D::from_preset_fields(
            s.camera,
            s.pitch,
            s.yaw,
            s.zoom,
            Some(Camera3D::new(0.55, 0.50, 1.0)),
        )?;
        self.custom_sandbox
            .load_domain_bounds(s.surface_type, s.u_bounds, s.v_bounds)?;
        self.surface_type = s.surface_type;
        self.kinetics_kind = s.kinetics_kind;
        self.overlay_mode = s.overlay_mode;
        self.sphere_radius = s.sphere_radius;
        self.torus_major_radius = s.torus_major_radius;
        self.torus_minor_radius = s.torus_minor_radius;
        self.klein_radius = s.klein_radius;
        self.wall_thickness = s.wall_thickness;
        self.u_resolution = s.u_resolution.min(MAX_SURFACE_MORPHO_RES);
        self.v_resolution = s.v_resolution.min(MAX_SURFACE_MORPHO_RES);
        self.schnak_a = s.schnak_a;
        self.schnak_b = s.schnak_b;
        self.gs_feed = s.gs_feed;
        self.gs_kill = s.gs_kill;
        self.d_u = s.d_u;
        self.d_v = s.d_v;
        self.buckling_gain = s.buckling_gain;
        self.curvature_tension = s.curvature_tension;
        self.restoring_tether = s.restoring_tether;
        self.dt = s.dt;
        self.steps_per_frame = s.steps_per_frame.clamp(1, 16);
        self.camera.apply_pose(camera);
        self.custom_sandbox.load_fields(
            s.custom_expr.as_deref(),
            s.parametric_exprs.as_ref(),
            &s.custom_params,
        );
        self.engine = None;
        Ok(())
    }
}

impl FiniteStateGuard for SurfaceMorphogenesisTool {
    fn is_finite_state(&self) -> bool {
        if !self.has_finite_parameters_internal() {
            return false;
        }
        let Some(eng) = self.engine.as_ref() else {
            return true;
        };
        eng.runner.u().iter().all(|v| v.is_finite())
            && eng.runner.v().iter().all(|v| v.is_finite())
            && eng
                .flow
                .points()
                .iter()
                .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
    }
}

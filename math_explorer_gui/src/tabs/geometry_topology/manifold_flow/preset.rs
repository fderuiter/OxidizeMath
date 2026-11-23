//! Bounded JSON state serialization, validation, and `StarterPresetMetadata` cards for
//! [`super::ManifoldFlowStudio`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::custom_surface::{CustomSurfaceSandbox, SurfaceType};
use super::{ManifoldFlowMode, ManifoldFlowStudio};
use crate::camera::Camera3D;
use crate::presets::MAX_INTERACTIVE_SURFACE_RESOLUTION;
use crate::starter_gallery::StarterPresetMetadata;
use std::collections::BTreeMap;

/// Compact state snapshot for [`ManifoldFlowStudio`] fitting well inside `URL_JSON_BOUNDS` (`4 KiB`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifoldFlowState {
    /// Selected PDE evolution mode.
    pub mode: ManifoldFlowMode,
    /// Selected manifold surface type.
    pub surface_type: SurfaceType,
    /// Optional custom heightfield formula `z(x, y)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_expr: Option<String>,
    /// Optional custom parametric triple `[x(u,v), y(u,v), z(u,v)]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parametric_exprs: Option<[String; 3]>,
    /// Synthesized parameter slider map.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub custom_params: BTreeMap<String, f64>,
    /// Optional custom `u` domain bounds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub u_bounds: Option<[f64; 2]>,
    /// Optional custom `v` domain bounds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub v_bounds: Option<[f64; 2]>,
    /// Sphere radius (`0.3..=4.0`).
    pub sphere_radius: f64,
    /// Torus major radius (`1.2..=4.5`).
    pub torus_major_radius: f64,
    /// Torus minor radius (`0.25..=1.8`).
    pub torus_minor_radius: f64,
    /// Klein bottle scale (`0.6..=3.5`).
    pub klein_radius: f64,
    /// Initial normal ripple amplitude (`0.0..=0.5`).
    pub initial_perturbation: f64,
    /// Grid `u` resolution (`8..=48`).
    pub u_res: usize,
    /// Grid `v` resolution (`8..=48`).
    pub v_res: usize,
    /// Requested integration step `dt`.
    pub dt: f64,
    /// Thermal diffusivity `alpha`.
    pub alpha: f64,
    /// Substeps per UI frame (`1..=16`).
    pub substeps_per_frame: usize,
    /// Normalized probe coordinates `[u_frac, v_frac]`.
    pub probe_uv_frac: [f64; 2],
    /// Wireframe overlay toggle.
    pub show_wireframe: bool,
    /// 3D orbital camera pose.
    pub camera: Camera3D,
}

fn validate_state(s: &ManifoldFlowState) -> Result<(), String> {
    if !s.sphere_radius.is_finite() || !(0.3..=4.0).contains(&s.sphere_radius) {
        return Err("sphere_radius out of [0.3, 4.0]".to_string());
    }
    if !s.torus_major_radius.is_finite() || !(1.2..=4.5).contains(&s.torus_major_radius) {
        return Err("torus_major_radius out of [1.2, 4.5]".to_string());
    }
    if !s.torus_minor_radius.is_finite()
        || !(0.2..=1.8).contains(&s.torus_minor_radius)
        || s.torus_minor_radius >= s.torus_major_radius
    {
        return Err("torus_minor_radius out of valid range".to_string());
    }
    if !s.klein_radius.is_finite() || !(0.5..=3.5).contains(&s.klein_radius) {
        return Err("klein_radius out of [0.5, 3.5]".to_string());
    }
    if !s.initial_perturbation.is_finite() || !(0.0..=0.5).contains(&s.initial_perturbation) {
        return Err("initial_perturbation out of [0.0, 0.5]".to_string());
    }
    if !(8..=MAX_INTERACTIVE_SURFACE_RESOLUTION).contains(&s.u_res)
        || !(8..=MAX_INTERACTIVE_SURFACE_RESOLUTION).contains(&s.v_res)
    {
        return Err(format!(
            "Grid resolution out of 8..={MAX_INTERACTIVE_SURFACE_RESOLUTION}"
        ));
    }
    if !s.dt.is_finite() || !(1e-5..=0.1).contains(&s.dt) {
        return Err("dt out of [1e-5, 0.1]".to_string());
    }
    if !s.alpha.is_finite() || !(1e-4..=1.0).contains(&s.alpha) {
        return Err("alpha out of [1e-4, 1.0]".to_string());
    }
    if !(1..=16).contains(&s.substeps_per_frame) {
        return Err("substeps_per_frame out of 1..=16".to_string());
    }
    if !s.probe_uv_frac[0].is_finite()
        || !s.probe_uv_frac[1].is_finite()
        || !(0.0..=1.0).contains(&s.probe_uv_frac[0])
        || !(0.0..=1.0).contains(&s.probe_uv_frac[1])
    {
        return Err("probe_uv_frac out of [0, 1]^2".to_string());
    }
    if !s.camera.is_valid() {
        return Err("Invalid Camera3D state".to_string());
    }
    Ok(())
}

impl ManifoldFlowStudio {
    pub(super) fn parameters_are_valid(&self) -> bool {
        self.sphere_radius.is_finite()
            && (0.3..=4.0).contains(&self.sphere_radius)
            && self.torus_major_radius.is_finite()
            && (1.2..=4.5).contains(&self.torus_major_radius)
            && self.torus_minor_radius.is_finite()
            && (0.2..=1.8).contains(&self.torus_minor_radius)
            && self.klein_radius.is_finite()
            && (0.5..=3.5).contains(&self.klein_radius)
            && (8..=MAX_INTERACTIVE_SURFACE_RESOLUTION).contains(&self.u_res)
            && (8..=MAX_INTERACTIVE_SURFACE_RESOLUTION).contains(&self.v_res)
            && self.dt.is_finite()
            && (1e-5..=0.1).contains(&self.dt)
            && self.alpha.is_finite()
            && (1e-4..=1.0).contains(&self.alpha)
            && self.camera.is_valid()
    }

    pub(super) fn clamp_model_parameters(&mut self) {
        self.sphere_radius = self.sphere_radius.clamp(0.3, 4.0);
        self.torus_major_radius = self.torus_major_radius.clamp(1.2, 4.5);
        self.torus_minor_radius = self
            .torus_minor_radius
            .clamp(0.2, (self.torus_major_radius - 0.15).min(1.8));
        self.klein_radius = self.klein_radius.clamp(0.5, 3.5);
        self.initial_perturbation = self.initial_perturbation.clamp(0.0, 0.5);
        self.u_res = self.u_res.clamp(8, MAX_INTERACTIVE_SURFACE_RESOLUTION);
        self.v_res = self.v_res.clamp(8, MAX_INTERACTIVE_SURFACE_RESOLUTION);
        self.dt = if self.dt.is_finite() {
            self.dt.clamp(1e-5, 0.05)
        } else {
            0.005
        };
        self.alpha = if self.alpha.is_finite() {
            self.alpha.clamp(0.005, 0.5)
        } else {
            0.08
        };
        self.substeps_per_frame = self.substeps_per_frame.clamp(1, 16);
        self.probe_uv_frac[0] = self.probe_uv_frac[0].clamp(0.05, 0.95);
        self.probe_uv_frac[1] = self.probe_uv_frac[1].clamp(0.05, 0.95);
    }

    pub(super) fn serialize_studio_state(&self) -> Option<String> {
        let (custom_expr, parametric_exprs, custom_params) =
            self.custom_sandbox.export_fields(self.surface_type);
        let (u_bounds, v_bounds) = self.custom_sandbox.export_domain_bounds(self.surface_type);
        let state = ManifoldFlowState {
            mode: self.mode,
            surface_type: self.surface_type,
            custom_expr,
            parametric_exprs,
            custom_params,
            u_bounds,
            v_bounds,
            sphere_radius: self.sphere_radius,
            torus_major_radius: self.torus_major_radius,
            torus_minor_radius: self.torus_minor_radius,
            klein_radius: self.klein_radius,
            initial_perturbation: self.initial_perturbation,
            u_res: self.u_res,
            v_res: self.v_res,
            dt: self.dt,
            alpha: self.alpha,
            substeps_per_frame: self.substeps_per_frame,
            probe_uv_frac: self.probe_uv_frac,
            show_wireframe: self.show_wireframe,
            camera: self.camera,
        };
        serde_json::to_string(&state).ok()
    }

    pub(super) fn deserialize_studio_state(&mut self, raw: &str) -> Result<(), String> {
        let parsed: ManifoldFlowState = serde_json::from_str(raw)
            .map_err(|e| format!("Invalid ManifoldFlowStudio state: {e}"))?;
        validate_state(&parsed)?;

        let mut test_sandbox = CustomSurfaceSandbox::default();
        test_sandbox.load_fields(
            parsed.custom_expr.as_deref(),
            parsed.parametric_exprs.as_ref(),
            &parsed.custom_params,
        );
        if parsed.surface_type == SurfaceType::CustomHeightfield
            && !test_sandbox.heightfield_bar.is_valid()
        {
            return Err("Invalid custom heightfield formula in preset".to_string());
        }
        if parsed.surface_type == SurfaceType::CustomParametric
            && test_sandbox.parametric_error.is_some()
        {
            return Err("Invalid custom parametric triple in preset".to_string());
        }
        test_sandbox.load_domain_bounds(parsed.surface_type, parsed.u_bounds, parsed.v_bounds)?;

        self.mode = parsed.mode;
        self.surface_type = parsed.surface_type;
        self.custom_sandbox = test_sandbox;
        self.sphere_radius = parsed.sphere_radius;
        self.torus_major_radius = parsed.torus_major_radius;
        self.torus_minor_radius = parsed.torus_minor_radius;
        self.klein_radius = parsed.klein_radius;
        self.initial_perturbation = parsed.initial_perturbation;
        self.u_res = parsed.u_res;
        self.v_res = parsed.v_res;
        self.dt = parsed.dt;
        self.alpha = parsed.alpha;
        self.substeps_per_frame = parsed.substeps_per_frame;
        self.probe_uv_frac = parsed.probe_uv_frac;
        self.show_wireframe = parsed.show_wireframe;
        self.camera.apply_pose(parsed.camera);
        self.is_playing = false;
        self.reset_simulation();
        Ok(())
    }
}

/// Starter preset: Rippled Torus Mean Curvature Smoothing.
pub const PRESET_TORUS_MCF_SMOOTHING: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Rippled Torus Mean Curvature Smoothing",
    "domain": "geometry_topology",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "mode": "MeanCurvature",
    "surface_type": "Torus",
    "sphere_radius": 1.8,
    "torus_major_radius": 2.0,
    "torus_minor_radius": 0.75,
    "klein_radius": 1.5,
    "initial_perturbation": 0.18,
    "u_res": 24,
    "v_res": 20,
    "dt": 0.008,
    "alpha": 0.08,
    "substeps_per_frame": 2,
    "probe_uv_frac": [0.35, 0.45],
    "show_wireframe": true,
    "camera": { "pitch": 0.55, "yaw": 0.58, "zoom": 0.92 }
  }
}"#;

/// Starter preset: Huisken Sphere Mean Curvature Contraction.
pub const PRESET_SPHERE_HUISKEN_SHRINK: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Huisken Sphere Mean Curvature Shrinkage",
    "domain": "geometry_topology",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "mode": "MeanCurvature",
    "surface_type": "Sphere",
    "sphere_radius": 2.0,
    "torus_major_radius": 2.0,
    "torus_minor_radius": 0.75,
    "klein_radius": 1.5,
    "initial_perturbation": 0.0,
    "u_res": 24,
    "v_res": 16,
    "dt": 0.01,
    "alpha": 0.08,
    "substeps_per_frame": 3,
    "probe_uv_frac": [0.5, 0.5],
    "show_wireframe": true,
    "camera": { "pitch": 0.48, "yaw": 0.52, "zoom": 0.95 }
  }
}"#;

/// Starter preset: Torus Laplace-Beltrami Thermal Spot Diffusion.
pub const PRESET_TORUS_HEAT_DIFFUSION: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Torus Laplace-Beltrami Heat Diffusion",
    "domain": "geometry_topology",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "mode": "HeatDiffusion",
    "surface_type": "Torus",
    "sphere_radius": 1.8,
    "torus_major_radius": 2.1,
    "torus_minor_radius": 0.8,
    "klein_radius": 1.5,
    "initial_perturbation": 0.0,
    "u_res": 26,
    "v_res": 22,
    "dt": 0.012,
    "alpha": 0.12,
    "substeps_per_frame": 4,
    "probe_uv_frac": [0.5, 0.5],
    "show_wireframe": true,
    "camera": { "pitch": 0.56, "yaw": 0.64, "zoom": 0.9 }
  }
}"#;

inventory::submit! {
    StarterPresetMetadata {
        id: "torus_mean_curvature_smoothing",
        title: "Rippled Torus Mean Curvature Flow",
        description: "Evolves a normal-rippled torus along steepest area descent dr/dt = H n on flat vertex buffers.",
        category: "Differential Geometry",
        icon: "🍩",
        domain: "geometry_topology",
        tool_name: Some("ManifoldFlowStudio"),
        preset_generator: || PRESET_TORUS_MCF_SMOOTHING,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "sphere_huisken_contraction",
        title: "Huisken Sphere Contraction",
        description: "Demonstrates uniform spherical contraction under extrinsic Mean Curvature Flow with watertight mesh export.",
        category: "Differential Geometry",
        icon: "🫧",
        domain: "geometry_topology",
        tool_name: Some("ManifoldFlowStudio"),
        preset_generator: || PRESET_SPHERE_HUISKEN_SHRINK,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "torus_laplace_beltrami_heat",
        title: "Torus Laplace-Beltrami Heat Flow",
        description: "Diffuses a localized Gaussian thermal pulse on a curved torus via the metric Laplace-Beltrami operator.",
        category: "Differential Geometry",
        icon: "🔥",
        domain: "geometry_topology",
        tool_name: Some("ManifoldFlowStudio"),
        preset_generator: || PRESET_TORUS_HEAT_DIFFUSION,
    }
}

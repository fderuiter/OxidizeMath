//! Bounded state serialization, strict domain validation, and `StarterPresetMetadata` cards
//! for [`super::CurvilinearVectorStudio`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{CoordinateFrameKind, CurvilinearVectorStudio, MAX_HELMHOLTZ_GRID, MIN_HELMHOLTZ_GRID};
use crate::camera::Camera3D;
use crate::starter_gallery::StarterPresetMetadata;
use std::collections::BTreeMap;

/// Compact JSON state snapshot for [`CurvilinearVectorStudio`], comfortably fitting inside
/// `URL_JSON_BOUNDS` (`4 KiB`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurvilinearVectorState {
    /// Active curvilinear coordinate frame.
    pub frame: CoordinateFrameKind,
    /// Expression string for $F_1$.
    pub f1_expr: String,
    /// Expression string for $F_2$.
    pub f2_expr: String,
    /// Expression string for $F_3$.
    pub f3_expr: String,
    /// Expression string for scalar potential $\Phi$.
    pub scalar_expr: String,
    /// Synthesized parameter values.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub custom_params: BTreeMap<String, f64>,
    /// Gauss `Domain` lower bounds.
    pub domain_min: [f64; 3],
    /// Gauss `Domain` upper bounds.
    pub domain_max: [f64; 3],
    /// Stokes normal coordinate axis (`0..=2`).
    pub stokes_axis: usize,
    /// Stokes constant coordinate slice.
    pub stokes_fixed: f64,
    /// Stokes `u` tangent bounds.
    pub stokes_u: [f64; 2],
    /// Stokes `v` tangent bounds.
    pub stokes_v: [f64; 2],
    /// Probe point in curvilinear coordinates.
    pub probe_point: [f64; 3],
    /// 3D Helmholtz grid resolution (`5..=32`).
    pub grid_res: usize,
    /// Gauss/Stokes quadrature subdivisions (`6..=48`).
    pub integration_steps: usize,
    /// Poisson SOR sweeps (`20..=240`).
    pub poisson_iters: usize,
    /// 3D arrow visual scale.
    pub quiver_scale: f64,
    /// Visibility toggles `[total, irrotational, solenoidal, domain, stokes]`.
    pub layers: [bool; 5],
    /// 3D orbital camera pose.
    pub camera: Camera3D,
}

fn validate_bounds_for_frame(
    frame: CoordinateFrameKind,
    min: [f64; 3],
    max: [f64; 3],
) -> Result<(), String> {
    for ax in 0..3 {
        if !min[ax].is_finite() || !max[ax].is_finite() {
            return Err("Non-finite domain bound".to_string());
        }
        if max[ax] - min[ax] < 0.05 || min[ax] < -10.0 || max[ax] > 10.0 {
            return Err(format!("Invalid domain interval on axis {ax}"));
        }
    }
    if frame != CoordinateFrameKind::Cartesian && min[0] < 0.05 {
        return Err("Radial coordinate min[0] must be >= 0.05 to avoid origin singularity".into());
    }
    if frame == CoordinateFrameKind::Spherical
        && (min[1] < 0.05 || max[1] > std::f64::consts::PI - 0.05)
    {
        return Err("Spherical polar angle theta must lie inside [0.05, pi - 0.05]".into());
    }
    Ok(())
}

fn validate_state(s: &CurvilinearVectorState) -> Result<(), String> {
    validate_bounds_for_frame(s.frame, s.domain_min, s.domain_max)?;
    if s.stokes_axis > 2 {
        return Err("stokes_axis must be 0, 1, or 2".to_string());
    }
    if !s.stokes_fixed.is_finite() || !(-10.0..=10.0).contains(&s.stokes_fixed) {
        return Err("stokes_fixed out of range".to_string());
    }
    if !s.stokes_u[0].is_finite()
        || !s.stokes_u[1].is_finite()
        || s.stokes_u[1] - s.stokes_u[0] < 0.05
        || !s.stokes_v[0].is_finite()
        || !s.stokes_v[1].is_finite()
        || s.stokes_v[1] - s.stokes_v[0] < 0.05
    {
        return Err("Invalid Stokes patch interval".to_string());
    }
    if s.frame != CoordinateFrameKind::Cartesian && s.stokes_axis == 0 && s.stokes_fixed < 0.05 {
        return Err("Radial Stokes slice must be >= 0.05".to_string());
    }
    if !s
        .probe_point
        .iter()
        .all(|v| v.is_finite() && v.abs() <= 10.0)
    {
        return Err("probe_point out of finite range".to_string());
    }
    if !(MIN_HELMHOLTZ_GRID..=MAX_HELMHOLTZ_GRID).contains(&s.grid_res) {
        return Err(format!(
            "grid_res out of {MIN_HELMHOLTZ_GRID}..={MAX_HELMHOLTZ_GRID}"
        ));
    }
    if !(6..=48).contains(&s.integration_steps) {
        return Err("integration_steps out of 6..=48".to_string());
    }
    if !(20..=240).contains(&s.poisson_iters) {
        return Err("poisson_iters out of 20..=240".to_string());
    }
    if !s.quiver_scale.is_finite() || !(0.02..=2.0).contains(&s.quiver_scale) {
        return Err("quiver_scale out of [0.02, 2.0]".to_string());
    }
    if !s.camera.is_valid() {
        return Err("Invalid Camera3D configuration".to_string());
    }
    for (k, &v) in &s.custom_params {
        if k.is_empty() || !v.is_finite() || v.abs() > 1e3 {
            return Err("Invalid custom parameter value".to_string());
        }
    }
    Ok(())
}

impl CurvilinearVectorStudio {
    pub(super) fn parameters_are_valid(&self) -> bool {
        validate_bounds_for_frame(self.frame, self.domain_min, self.domain_max).is_ok()
            && self.stokes_axis <= 2
            && self.stokes_fixed.is_finite()
            && self.probe_point.iter().all(|v| v.is_finite())
            && (MIN_HELMHOLTZ_GRID..=MAX_HELMHOLTZ_GRID).contains(&self.grid_res)
            && self.quiver_scale.is_finite()
            && (0.02..=2.0).contains(&self.quiver_scale)
            && self.camera.is_valid()
    }

    pub(super) fn clamp_all_bounds(&mut self) {
        self.grid_res = self.grid_res.clamp(MIN_HELMHOLTZ_GRID, MAX_HELMHOLTZ_GRID);
        self.integration_steps = self.integration_steps.clamp(6, 48);
        self.poisson_iters = self.poisson_iters.clamp(20, 240);
        self.quiver_scale = if self.quiver_scale.is_finite() {
            self.quiver_scale.clamp(0.04, 0.8)
        } else {
            0.22
        };
        if self.frame != CoordinateFrameKind::Cartesian {
            self.domain_min[0] = self.domain_min[0].max(0.1);
            self.domain_max[0] = self.domain_max[0].max(self.domain_min[0] + 0.2);
        }
        if self.frame == CoordinateFrameKind::Spherical {
            self.domain_min[1] = self.domain_min[1].clamp(0.1, std::f64::consts::PI - 0.3);
            self.domain_max[1] =
                self.domain_max[1].clamp(self.domain_min[1] + 0.15, std::f64::consts::PI - 0.1);
        }
        for ax in 0..3 {
            if self.domain_max[ax] <= self.domain_min[ax] + 0.05 {
                self.domain_max[ax] = self.domain_min[ax] + 0.2;
            }
            self.probe_point[ax] =
                self.probe_point[ax].clamp(self.domain_min[ax], self.domain_max[ax]);
        }
        self.stokes_axis = self.stokes_axis.min(2);
        let au = (self.stokes_axis + 1) % 3;
        let av = (self.stokes_axis + 2) % 3;
        self.stokes_fixed = self.stokes_fixed.clamp(
            self.domain_min[self.stokes_axis],
            self.domain_max[self.stokes_axis],
        );
        self.stokes_u[0] = self.stokes_u[0].max(self.domain_min[au]);
        self.stokes_u[1] = self.stokes_u[1]
            .min(self.domain_max[au])
            .max(self.stokes_u[0] + 0.1);
        self.stokes_v[0] = self.stokes_v[0].max(self.domain_min[av]);
        self.stokes_v[1] = self.stokes_v[1]
            .min(self.domain_max[av])
            .max(self.stokes_v[0] + 0.1);
    }

    pub(super) fn serialize_studio_state(&self) -> Option<String> {
        let state = CurvilinearVectorState {
            frame: self.frame,
            f1_expr: self.f1_bar.raw_text.clone(),
            f2_expr: self.f2_bar.raw_text.clone(),
            f3_expr: self.f3_bar.raw_text.clone(),
            scalar_expr: self.scalar_bar.raw_text.clone(),
            custom_params: self.param_sliders.export_active_map(),
            domain_min: self.domain_min,
            domain_max: self.domain_max,
            stokes_axis: self.stokes_axis,
            stokes_fixed: self.stokes_fixed,
            stokes_u: self.stokes_u,
            stokes_v: self.stokes_v,
            probe_point: self.probe_point,
            grid_res: self.grid_res,
            integration_steps: self.integration_steps,
            poisson_iters: self.poisson_iters,
            quiver_scale: self.quiver_scale,
            layers: [
                self.show_total_field,
                self.show_irrotational,
                self.show_solenoidal,
                self.show_domain_wireframe,
                self.show_stokes_loop,
            ],
            camera: self.camera,
        };
        serde_json::to_string(&state).ok()
    }

    pub(super) fn deserialize_studio_state(&mut self, raw: &str) -> Result<(), String> {
        let parsed: CurvilinearVectorState = serde_json::from_str(raw)
            .map_err(|e| format!("Invalid CurvilinearVectorStudio state: {e}"))?;
        validate_state(&parsed)?;

        let vars = parsed.frame.axis_vars();
        let f1 = super::compute::build_field_bar("F_1", &vars, &parsed.f1_expr);
        let f2 = super::compute::build_field_bar("F_2", &vars, &parsed.f2_expr);
        let f3 = super::compute::build_field_bar("F_3", &vars, &parsed.f3_expr);
        let sc = super::compute::build_field_bar("Phi", &vars, &parsed.scalar_expr);
        if f1.expr().is_none() || f2.expr().is_none() || f3.expr().is_none() || sc.expr().is_none()
        {
            return Err("Failed to parse one or more field expressions in preset".to_string());
        }

        self.frame = parsed.frame;
        self.f1_bar = f1;
        self.f2_bar = f2;
        self.f3_bar = f3;
        self.scalar_bar = sc;
        self.recompile_expressions();
        self.param_sliders.set_params_map(&parsed.custom_params);
        if self.compile_error.is_some() {
            return Err("Preset expressions failed AST compilation".to_string());
        }

        self.domain_min = parsed.domain_min;
        self.domain_max = parsed.domain_max;
        self.stokes_axis = parsed.stokes_axis;
        self.stokes_fixed = parsed.stokes_fixed;
        self.stokes_u = parsed.stokes_u;
        self.stokes_v = parsed.stokes_v;
        self.probe_point = parsed.probe_point;
        self.grid_res = parsed.grid_res;
        self.integration_steps = parsed.integration_steps;
        self.poisson_iters = parsed.poisson_iters;
        self.quiver_scale = parsed.quiver_scale;
        self.show_total_field = parsed.layers[0];
        self.show_irrotational = parsed.layers[1];
        self.show_solenoidal = parsed.layers[2];
        self.show_domain_wireframe = parsed.layers[3];
        self.show_stokes_loop = parsed.layers[4];
        self.camera.apply_pose(parsed.camera);

        self.dirty = true;
        self.recompute_if_dirty();
        self.state_rev = self.state_rev.wrapping_add(1);
        Ok(())
    }
}

/// Starter preset: Cartesian vortex + radial source Helmholtz-Hodge split.
pub const PRESET_CARTESIAN_HELMHOLTZ: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Cartesian Vortex & Radial Source Split",
    "domain": "analysis",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "frame": "Cartesian",
    "f1_expr": "a * x - y",
    "f2_expr": "a * y + x",
    "f3_expr": "a * z",
    "scalar_expr": "0.5 * a * (x^2 + y^2 + z^2)",
    "custom_params": { "a": 1.0 },
    "domain_min": [-1.0, -1.0, -1.0],
    "domain_max": [1.0, 1.0, 1.0],
    "stokes_axis": 2,
    "stokes_fixed": 0.25,
    "stokes_u": [-0.8, 0.8],
    "stokes_v": [-0.8, 0.8],
    "probe_point": [0.4, 0.3, 0.2],
    "grid_res": 10,
    "integration_steps": 16,
    "poisson_iters": 100,
    "quiver_scale": 0.22,
    "layers": [true, true, true, true, true],
    "camera": { "pitch": 0.52, "yaw": 0.62, "zoom": 1.0 }
  }
}"#;

/// Starter preset: Cylindrical azimuthal swirl verifying Kelvin-Stokes circulation.
pub const PRESET_CYLINDRICAL_STOKES: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Cylindrical Azimuthal Stokes Circulation",
    "domain": "analysis",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "frame": "Cylindrical",
    "f1_expr": "0.5 * rho * sin(phi)",
    "f2_expr": "rho^2",
    "f3_expr": "0.6 * z",
    "scalar_expr": "0.5 * rho^2 * cos(phi)",
    "domain_min": [0.8, 0.15, -0.9],
    "domain_max": [2.0, 1.75, 0.9],
    "stokes_axis": 2,
    "stokes_fixed": 0.2,
    "stokes_u": [0.9, 1.9],
    "stokes_v": [0.2, 1.6],
    "probe_point": [1.35, 0.8, 0.25],
    "grid_res": 10,
    "integration_steps": 20,
    "poisson_iters": 110,
    "quiver_scale": 0.18,
    "layers": [true, false, true, true, true],
    "camera": { "pitch": 0.58, "yaw": 0.45, "zoom": 0.95 }
  }
}"#;

/// Starter preset: Spherical shell sector verifying Gauss divergence flux.
pub const PRESET_SPHERICAL_GAUSS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Spherical Radial Shell Gauss Flux",
    "domain": "analysis",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "frame": "Spherical",
    "f1_expr": "r^2",
    "f2_expr": "sin(theta)",
    "f3_expr": "r * sin(theta)",
    "scalar_expr": "0.333 * r^3",
    "domain_min": [1.0, 0.45, 0.15],
    "domain_max": [2.1, 1.55, 1.65],
    "stokes_axis": 0,
    "stokes_fixed": 1.6,
    "stokes_u": [0.5, 1.45],
    "stokes_v": [0.2, 1.5],
    "probe_point": [1.5, 0.95, 0.75],
    "grid_res": 10,
    "integration_steps": 20,
    "poisson_iters": 110,
    "quiver_scale": 0.16,
    "layers": [true, true, false, true, true],
    "camera": { "pitch": 0.48, "yaw": 0.72, "zoom": 0.9 }
  }
}"#;

inventory::submit! {
    StarterPresetMetadata {
        id: "cartesian_helmholtz_vortex_source",
        title: "Cartesian Helmholtz Vortex & Source",
        description: "Decomposes F = (ax - y, ay + x, az) into curl-free radial expansion and divergence-free z-vortex.",
        category: "Vector Calculus",
        icon: "🧭",
        domain: "analysis",
        tool_name: Some("CurvilinearVectorStudio"),
        preset_generator: || PRESET_CARTESIAN_HELMHOLTZ,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "cylindrical_stokes_azimuthal_swirl",
        title: "Cylindrical Stokes Circulation",
        description: "Verifies Kelvin-Stokes curl flux vs closed-loop circulation across an annular cylindrical sector.",
        category: "Vector Calculus",
        icon: "🌀",
        domain: "analysis",
        tool_name: Some("CurvilinearVectorStudio"),
        preset_generator: || PRESET_CYLINDRICAL_STOKES,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "spherical_gauss_radial_shell",
        title: "Spherical Gauss Shell Flux",
        description: "Verifies Gauss's divergence theorem and curvilinear scale factors on a 3D spherical shell sector.",
        category: "Vector Calculus",
        icon: "🌐",
        domain: "analysis",
        tool_name: Some("CurvilinearVectorStudio"),
        preset_generator: || PRESET_SPHERICAL_GAUSS,
    }
}

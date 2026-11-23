//! Bounded JSON state serialization, validation, and `StarterPresetMetadata` cards for
//! [`super::RiemannSurfaceStudio`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    CompanionInspectorMode, MeromorphicPreset, PolygonPreset, RiemannBranchFamily,
    RiemannSurfaceStudio, SurfaceHeightChannel,
};
use crate::camera::Camera3D;
use crate::starter_gallery::StarterPresetMetadata;

/// Compact JSON state snapshot for [`RiemannSurfaceStudio`] within `URL_JSON_BOUNDS` (`< 4 KiB`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RiemannSurfaceState {
    /// Active multi-sheet surface family.
    pub branch_family: RiemannBranchFamily,
    /// Root degree `2..=6`.
    pub root_n: usize,
    /// Symmetric logarithmic sheet span `1..=3`.
    pub log_sheet_span: usize,
    /// Scalar channel mapped to 3D height.
    pub height_channel: SurfaceHeightChannel,
    /// Outer polar radius `0.6..=2.5`.
    pub radial_max: f64,
    /// Radial mesh subdivisions `8..=28`.
    pub radial_steps: usize,
    /// Angular mesh subdivisions `16..=56`.
    pub angular_steps: usize,
    /// Initial sheet index `-5..=5`.
    pub start_sheet: isize,
    /// Signed winding turns `-4..=4`.
    pub winding_turns: isize,
    /// Closed contour radius `0.2..=2.0`.
    pub loop_radius: f64,
    /// Closed contour center `[re, im]`.
    pub loop_center: [f64; 2],
    /// Companion 2D complex plane mode.
    pub companion_mode: CompanionInspectorMode,
    /// Schwarz-Christoffel polygon preset.
    pub polygon_preset: PolygonPreset,
    /// Schwarz-Christoffel scale `0.3..=2.5`.
    pub sc_scale: f64,
    /// Schwarz reflection analytic continuation toggle.
    pub enable_schwarz_reflection: bool,
    /// Meromorphic preset for Cauchy & residue integration.
    pub meromorphic_preset: MeromorphicPreset,
    /// Optional custom real-axis numerator expression `g(x)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_numerator: Option<String>,
    /// Probe point `z_0`.
    pub probe_z: [f64; 2],
    /// Pole point `z_p`.
    pub pole_z: [f64; 2],
    /// Quadrature subdivisions `32..=400`.
    pub contour_steps: usize,
    /// Derivative order `1..=3`.
    pub derivative_order: usize,
    /// Visibility toggles `[wireframe, monodromy_path]`.
    pub layers: [bool; 2],
    /// 3D orbital camera pose.
    pub camera: Camera3D,
}

fn validate_state(s: &RiemannSurfaceState) -> Result<(), String> {
    if !(2..=6).contains(&s.root_n) {
        return Err("root_n must lie in 2..=6".to_string());
    }
    if !(1..=3).contains(&s.log_sheet_span) {
        return Err("log_sheet_span must lie in 1..=3".to_string());
    }
    if !s.radial_max.is_finite() || !(0.6..=2.5).contains(&s.radial_max) {
        return Err("radial_max must lie in [0.6, 2.5]".to_string());
    }
    let max_res = crate::presets::MAX_INTERACTIVE_SURFACE_RESOLUTION;
    if !(8..=max_res).contains(&s.radial_steps) || !(16..=max_res).contains(&s.angular_steps) {
        return Err("Mesh resolution out of bounds".to_string());
    }
    if !(-5..=5).contains(&s.start_sheet) || !(-4..=4).contains(&s.winding_turns) {
        return Err("Sheet or winding turns out of bounds".to_string());
    }
    if !s.loop_radius.is_finite() || !(0.2..=2.0).contains(&s.loop_radius) {
        return Err("loop_radius must lie in [0.2, 2.0]".to_string());
    }
    if !s
        .loop_center
        .iter()
        .all(|v| v.is_finite() && v.abs() <= 3.0)
    {
        return Err("loop_center out of [-3, 3]^2".to_string());
    }
    if !s.sc_scale.is_finite() || !(0.3..=2.5).contains(&s.sc_scale) {
        return Err("sc_scale out of [0.3, 2.5]".to_string());
    }
    if !s.probe_z.iter().all(|v| v.is_finite() && v.abs() <= 3.0)
        || !s.pole_z.iter().all(|v| v.is_finite() && v.abs() <= 3.0)
    {
        return Err("probe_z or pole_z out of [-3, 3]^2".to_string());
    }
    if !(32..=400).contains(&s.contour_steps) || !(1..=3).contains(&s.derivative_order) {
        return Err("contour_steps or derivative_order out of bounds".to_string());
    }
    if !s.camera.is_valid() {
        return Err("Invalid Camera3D state".to_string());
    }
    if let Some(expr) = &s.custom_numerator {
        if expr.len() > 256 {
            return Err("custom_numerator exceeds 256 bytes".to_string());
        }
        let ast =
            math_commons::ast::parse(expr).map_err(|e| format!("Invalid custom_numerator: {e}"))?;
        math_commons::ast::CompiledExpr::compile_with_vars(&ast, &["x"])
            .map_err(|e| format!("Unsupported variable in custom_numerator: {e}"))?;
    }
    Ok(())
}

impl RiemannSurfaceStudio {
    pub(super) fn to_state(&self) -> RiemannSurfaceState {
        let default_expr = "cos(x) + 0.5 * x^2";
        let cur_expr = self.custom_numerator_bar.raw_text.as_str();
        let custom_numerator = if cur_expr == default_expr {
            None
        } else {
            Some(cur_expr.to_string())
        };
        RiemannSurfaceState {
            branch_family: self.branch_family,
            root_n: self.root_n,
            log_sheet_span: self.log_sheet_span,
            height_channel: self.height_channel,
            radial_max: self.radial_max,
            radial_steps: self.radial_steps,
            angular_steps: self.angular_steps,
            start_sheet: self.start_sheet,
            winding_turns: self.winding_turns,
            loop_radius: self.loop_radius,
            loop_center: self.loop_center,
            companion_mode: self.companion_mode,
            polygon_preset: self.polygon_preset,
            sc_scale: self.sc_scale,
            enable_schwarz_reflection: self.enable_schwarz_reflection,
            meromorphic_preset: self.meromorphic_preset,
            custom_numerator,
            probe_z: self.probe_z,
            pole_z: self.pole_z,
            contour_steps: self.contour_steps,
            derivative_order: self.derivative_order,
            layers: [self.show_wireframe, self.show_monodromy_path],
            camera: self.camera,
        }
    }

    pub(super) fn serialize_studio_state(&self) -> Option<String> {
        serde_json::to_string(&self.to_state()).ok()
    }

    pub(super) fn deserialize_studio_state(&mut self, json: &str) -> Result<(), String> {
        let state: RiemannSurfaceState = serde_json::from_str(json)
            .map_err(|e| format!("Invalid RiemannSurfaceStudio state: {e}"))?;
        validate_state(&state)?;

        self.branch_family = state.branch_family;
        self.root_n = state.root_n;
        self.log_sheet_span = state.log_sheet_span;
        self.height_channel = state.height_channel;
        self.radial_max = state.radial_max;
        self.radial_steps = state.radial_steps;
        self.angular_steps = state.angular_steps;
        self.start_sheet = state.start_sheet;
        self.winding_turns = state.winding_turns;
        self.loop_radius = state.loop_radius;
        self.loop_center = state.loop_center;
        self.companion_mode = state.companion_mode;
        self.polygon_preset = state.polygon_preset;
        self.sc_scale = state.sc_scale;
        self.enable_schwarz_reflection = state.enable_schwarz_reflection;
        self.meromorphic_preset = state.meromorphic_preset;
        let num_str = state
            .custom_numerator
            .as_deref()
            .unwrap_or("cos(x) + 0.5 * x^2");
        self.custom_numerator_bar.set_expression(num_str);
        self.probe_z = state.probe_z;
        self.pole_z = state.pole_z;
        self.contour_steps = state.contour_steps;
        self.derivative_order = state.derivative_order;
        self.show_wireframe = state.layers[0];
        self.show_monodromy_path = state.layers[1];
        self.camera.apply_pose(state.camera);
        self.dirty = true;
        self.state_rev = self.state_rev.wrapping_add(1);
        self.recompute_if_dirty();
        Ok(())
    }
}

/// Starter preset: Square-Root Two-Sheet Monodromy.
pub const PRESET_SQRT_MONODROMY: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Square-Root Two-Sheet Monodromy",
    "domain": "analysis",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "branch_family": "NthRoot",
    "root_n": 2,
    "log_sheet_span": 1,
    "height_channel": "RealPart",
    "radial_max": 1.8,
    "radial_steps": 14,
    "angular_steps": 32,
    "start_sheet": 0,
    "winding_turns": 2,
    "loop_radius": 1.25,
    "loop_center": [0.0, 0.0],
    "companion_mode": "CauchyResidueContour",
    "polygon_preset": "EquilateralTriangle",
    "sc_scale": 1.0,
    "enable_schwarz_reflection": true,
    "meromorphic_preset": "SimplePoleRational",
    "probe_z": [0.35, 0.45],
    "pole_z": [0.25, -0.2],
    "contour_steps": 160,
    "derivative_order": 1,
    "layers": [true, true],
    "camera": { "pitch": 0.58, "yaw": 0.75, "zoom": 0.96 }
  }
}"#;

/// Starter preset: Logarithmic Helicoid Winding Ramp.
pub const PRESET_LOG_HELICOID: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Logarithmic Helicoid Winding Ramp",
    "domain": "analysis",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "branch_family": "Logarithm",
    "root_n": 2,
    "log_sheet_span": 2,
    "height_channel": "ImaginaryPart",
    "radial_max": 1.8,
    "radial_steps": 14,
    "angular_steps": 32,
    "start_sheet": -1,
    "winding_turns": 3,
    "loop_radius": 1.1,
    "loop_center": [0.0, 0.0],
    "companion_mode": "CauchyResidueContour",
    "polygon_preset": "EquilateralTriangle",
    "sc_scale": 1.0,
    "enable_schwarz_reflection": true,
    "meromorphic_preset": "SimplePoleRational",
    "probe_z": [0.35, 0.45],
    "pole_z": [0.25, -0.2],
    "contour_steps": 160,
    "derivative_order": 1,
    "layers": [true, true],
    "camera": { "pitch": 0.48, "yaw": 0.85, "zoom": 0.88 }
  }
}"#;

/// Starter preset: Schwarz-Christoffel & Cauchy Inspector.
pub const PRESET_SC_AND_CAUCHY: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Schwarz-Christoffel & Cauchy Inspector",
    "domain": "analysis",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "branch_family": "NthRoot",
    "root_n": 3,
    "log_sheet_span": 1,
    "height_channel": "ImaginaryPart",
    "radial_max": 1.8,
    "radial_steps": 14,
    "angular_steps": 32,
    "start_sheet": 0,
    "winding_turns": 1,
    "loop_radius": 1.15,
    "loop_center": [0.0, 0.0],
    "companion_mode": "SchwarzChristoffelPolygon",
    "polygon_preset": "SteppedChannel",
    "sc_scale": 1.15,
    "enable_schwarz_reflection": true,
    "meromorphic_preset": "ExponentialOverPole",
    "probe_z": [0.35, 0.45],
    "pole_z": [0.25, -0.2],
    "contour_steps": 200,
    "derivative_order": 2,
    "layers": [true, true],
    "camera": { "pitch": 0.52, "yaw": 0.62, "zoom": 0.92 }
  }
}"#;

inventory::submit! {
    StarterPresetMetadata {
        id: "riemann_sqrt_monodromy",
        title: "Square-Root Monodromy",
        description: "Lift closed loops onto the 2-sheeted w = z^(1/2) Riemann surface to observe monodromy sheet swapping and 4*pi closure.",
        category: "Complex Analysis",
        icon: "🌀",
        domain: "analysis",
        tool_name: Some("RiemannSurfaceStudio"),
        preset_generator: || PRESET_SQRT_MONODROMY,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "riemann_log_helicoid",
        title: "Logarithmic Helicoid Winding",
        description: "Ascend the infinite-sheeted w = ln(z) helicoid across multiple 2*pi*i branch jumps.",
        category: "Complex Analysis",
        icon: "🐚",
        domain: "analysis",
        tool_name: Some("RiemannSurfaceStudio"),
        preset_generator: || PRESET_LOG_HELICOID,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "riemann_sc_cauchy_inspector",
        title: "Schwarz-Christoffel & Cauchy",
        description: "Inspect Schwarz-Christoffel conformal polygon scaling, Schwarz reflection continuation, and Cauchy residue integrals.",
        category: "Complex Analysis",
        icon: "📐",
        domain: "analysis",
        tool_name: Some("RiemannSurfaceStudio"),
        preset_generator: || PRESET_SC_AND_CAUCHY,
    }
}

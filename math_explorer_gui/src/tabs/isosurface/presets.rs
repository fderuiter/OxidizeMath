//! Built-in triply periodic minimal surface (TPMS) and signed-distance equation presets
//! plus federated [`StarterPresetMetadata`] registrations for `IsosurfaceTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::camera::Camera3D;
use crate::framework::math_input::MAX_SYNTHESIZED_PARAMS;
use crate::framework::MathExpressionInputBar;
use crate::starter_gallery::StarterPresetMetadata;
use math_explorer::applied::isosurface::MAX_SYNC_INTERIOR_DIM;
use std::collections::BTreeMap;

/// Minimum interactive interior grid resolution per axis.
pub const MIN_GUI_RESOLUTION: usize = 16;

/// Maximum interactive interior grid resolution per axis (`64`).
pub const MAX_GUI_RESOLUTION: usize = MAX_SYNC_INTERIOR_DIM;

const MAX_ISOSURFACE_EXPR_CHARS: usize = 512;

/// Validated serialized configuration payload for `IsosurfaceTool`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct IsosurfaceState {
    pub half_extent: f64,
    pub iso_level: f64,
    pub thickness: f64,
    pub resolution: usize,
    pub sheet_mode: bool,
    pub show_wireframe: bool,
    pub expression: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub params: BTreeMap<String, f64>,
    #[serde(default)]
    pub camera: Option<Camera3D>,
    #[serde(default, skip_serializing)]
    pub pitch: Option<f32>,
    #[serde(default, skip_serializing)]
    pub yaw: Option<f32>,
    #[serde(default, skip_serializing)]
    pub zoom: Option<f32>,
}

pub(crate) fn parse_validated_state(raw: &str) -> Result<(IsosurfaceState, Camera3D), String> {
    let mut loaded: IsosurfaceState =
        serde_json::from_str(raw).map_err(|e| format!("Invalid IsosurfaceTool state: {e}"))?;
    if !loaded.half_extent.is_finite()
        || !(0.25..=20.0).contains(&loaded.half_extent)
        || !loaded.iso_level.is_finite()
        || !(-20.0..=20.0).contains(&loaded.iso_level)
        || !loaded.thickness.is_finite()
        || !(0.02..=5.0).contains(&loaded.thickness)
        || !(MIN_GUI_RESOLUTION..=MAX_GUI_RESOLUTION).contains(&loaded.resolution)
    {
        return Err("Isosurface parameters out of valid bounds".to_string());
    }
    if loaded.params.len() > MAX_SYNTHESIZED_PARAMS
        || loaded
            .params
            .values()
            .any(|v| !v.is_finite() || !(-100.0..=100.0).contains(v))
    {
        return Err("Isosurface free parameter values out of valid bounds".to_string());
    }
    let expr_trimmed = loaded.expression.trim().to_string();
    if expr_trimmed.is_empty() || expr_trimmed.chars().count() > MAX_ISOSURFACE_EXPR_CHARS {
        return Err("Isosurface expression empty or exceeds maximum length".to_string());
    }
    let test_bar = MathExpressionInputBar::new("f(x, y, z)", &expr_trimmed, Vec::new())
        .with_axes(&["x", "y", "z"]);
    if let Some(err) = test_bar.error_message() {
        return Err(format!("Invalid isosurface expression: {err}"));
    }
    let camera = Camera3D::from_preset_fields(
        loaded.camera,
        loaded.pitch,
        loaded.yaw,
        loaded.zoom,
        Some(Camera3D::new(0.55, 0.65, 1.0)),
    )?;
    loaded.expression = expr_trimmed;
    Ok((loaded, camera))
}

/// Built-in implicit equation preset specification for [`super::tool::IsosurfaceTool`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IsosurfacePresetSpec {
    /// Stable identifier slug used for export filenames and selection.
    pub id: &'static str,
    /// Human-readable preset title.
    pub title: &'static str,
    /// Brief mathematical / fabrication summary.
    pub summary: &'static str,
    /// Implicit scalar field $f(x, y, z)$.
    pub expression: &'static str,
    /// Symmetric cubic domain half-extent $L$ (`[-L, L]^3`).
    pub half_extent: f64,
    /// Scalar isovalue offset $c$.
    pub iso_level: f64,
    /// Whether thickened sheet mode $|f - c| - t/2$ is active (`false` = solid $f - c$).
    pub sheet_mode: bool,
    /// Full slab thickness $t$ used in sheet mode.
    pub thickness: f64,
    /// Interior voxel resolution $N$ per axis (`16..=64`).
    pub resolution: usize,
    /// Base filename stem for `.stl` and `.obj` downloads.
    pub filename_stem: &'static str,
}

/// Curated TPMS and signed-distance presets exposed in `IsosurfaceTool`.
pub const BUILTIN_ISOSURFACE_PRESETS: &[IsosurfacePresetSpec] = &[
    IsosurfacePresetSpec {
        id: "gyroid",
        title: "Gyroid TPMS",
        summary: "Schoen's bicontinuous triply periodic minimal surface thickened into a watertight lattice cell.",
        expression: "sin(x) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x)",
        half_extent: std::f64::consts::PI,
        iso_level: 0.0,
        sheet_mode: true,
        thickness: 0.45,
        resolution: 32,
        filename_stem: "gyroid_isosurface",
    },
    IsosurfacePresetSpec {
        id: "schwarz_p",
        title: "Schwarz P Surface",
        summary: "Primitive cubic minimal surface cos(x) + cos(y) + cos(z) with orthogonal pore channels.",
        expression: "cos(x) + cos(y) + cos(z)",
        half_extent: std::f64::consts::PI,
        iso_level: 0.0,
        sheet_mode: true,
        thickness: 0.45,
        resolution: 32,
        filename_stem: "schwarz_p_isosurface",
    },
    IsosurfacePresetSpec {
        id: "schwarz_d",
        title: "Schwarz D (Diamond)",
        summary: "Alternating tetrahedral diamond minimal lattice for isotropic mechanical metamaterials.",
        expression: "sin(x)*sin(y)*sin(z) + sin(x)*cos(y)*cos(z) + cos(x)*sin(y)*cos(z) + cos(x)*cos(y)*sin(z)",
        half_extent: std::f64::consts::PI,
        iso_level: 0.0,
        sheet_mode: true,
        thickness: 0.40,
        resolution: 32,
        filename_stem: "schwarz_d_isosurface",
    },
    IsosurfacePresetSpec {
        id: "torus",
        title: "Implicit Torus",
        summary: "Genus-1 revolution solid defined by radial signed-distance level set.",
        expression: "(sqrt(x^2 + y^2) - 1.8)^2 + z^2 - 0.45",
        half_extent: 2.8,
        iso_level: 0.0,
        sheet_mode: false,
        thickness: 0.40,
        resolution: 32,
        filename_stem: "torus_isosurface",
    },
    IsosurfacePresetSpec {
        id: "genus_two",
        title: "Genus-2 Bitorus",
        summary: "Closed algebraic surface of genus 2 (Euler characteristic chi = -2).",
        expression: "((x^2 + y^2)^2 - 2.2 * (x^2 - y^2))^2 + 3.5 * z^2 - 0.35",
        half_extent: 2.2,
        iso_level: 0.0,
        sheet_mode: false,
        thickness: 0.40,
        resolution: 36,
        filename_stem: "genus2_isosurface",
    },
    IsosurfacePresetSpec {
        id: "csg_slotted_sphere",
        title: "CSG Slotted Sphere (SDF)",
        summary: "Exact Boolean subtraction using max/min signed-distance primitives.",
        expression: "max(sqrt(x^2 + y^2 + z^2) - 2.2, -min(abs(x), abs(y)) + 0.45)",
        half_extent: 2.5,
        iso_level: 0.0,
        sheet_mode: false,
        thickness: 0.40,
        resolution: 32,
        filename_stem: "slotted_sphere_isosurface",
    },
];

/// Returns `(title, expression)` pairs for [`crate::framework::MathExpressionInputBar`].
#[must_use]
pub fn expression_bar_presets() -> Vec<(&'static str, &'static str)> {
    BUILTIN_ISOSURFACE_PRESETS
        .iter()
        .map(|spec| (spec.title, spec.expression))
        .collect()
}

/// Finds a built-in preset matching the given expression string, if any.
#[must_use]
pub fn find_preset_by_expression(expr: &str) -> Option<&'static IsosurfacePresetSpec> {
    let trimmed = expr.trim();
    BUILTIN_ISOSURFACE_PRESETS
        .iter()
        .find(|spec| spec.expression == trimmed)
}

const PRESET_GYROID_STARTER_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Gyroid TPMS Lattice",
    "domain": "isosurface",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "half_extent": 3.14,
    "iso_level": 0.0,
    "thickness": 0.5,
    "resolution": 32,
    "sheet_mode": true,
    "show_wireframe": false,
    "expression": "sin(x) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x)",
    "camera": {
      "pitch": 0.55,
      "yaw": 0.65,
      "zoom": 1.0
    }
  }
}"#;

const PRESET_SCHWARZ_P_STARTER_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Schwarz P Minimal Shell",
    "domain": "isosurface",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "half_extent": 3.14,
    "iso_level": 0.0,
    "thickness": 0.45,
    "resolution": 32,
    "sheet_mode": true,
    "show_wireframe": false,
    "expression": "cos(x) + cos(y) + cos(z)",
    "camera": {
      "pitch": 0.5,
      "yaw": 0.5,
      "zoom": 1.0
    }
  }
}"#;

const PRESET_CSG_SDF_STARTER_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "CSG Slotted Sphere SDF",
    "domain": "isosurface",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "half_extent": 2.5,
    "iso_level": 0.0,
    "thickness": 0.4,
    "resolution": 32,
    "sheet_mode": false,
    "show_wireframe": true,
    "expression": "max(sqrt(x^2 + y^2 + z^2) - 2.2, -min(abs(x), abs(y)) + 0.45)",
    "camera": {
      "pitch": 0.52,
      "yaw": 0.72,
      "zoom": 1.05
    }
  }
}"#;

inventory::submit! {
    StarterPresetMetadata {
        id: "isosurface_gyroid_starter",
        title: "Gyroid TPMS Lattice",
        description: "Watertight slab-offset Schoen Gyroid minimal surface ready for 3D slicer STL export.",
        category: "Computational Geometry",
        icon: "🧊",
        domain: "isosurface",
        tool_name: Some("IsosurfaceTool"),
        preset_generator: || PRESET_GYROID_STARTER_JSON,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "isosurface_schwarz_p_starter",
        title: "Schwarz P Minimal Shell",
        description: "Triply periodic primitive cubic minimal surface cos(x) + cos(y) + cos(z) with closed boundaries.",
        category: "Computational Geometry",
        icon: "🫧",
        domain: "isosurface",
        tool_name: Some("IsosurfaceTool"),
        preset_generator: || PRESET_SCHWARZ_P_STARTER_JSON,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "isosurface_csg_sdf_starter",
        title: "CSG Slotted Sphere SDF",
        description: "Constructive solid geometry equation combining sphere and cross-slot via min/max SDF intrinsics.",
        category: "Computational Geometry",
        icon: "⚙️",
        domain: "isosurface",
        tool_name: Some("IsosurfaceTool"),
        preset_generator: || PRESET_CSG_SDF_STARTER_JSON,
    }
}

impl scientific_metadata::theory::TheoryDescribable for super::tool::IsosurfaceTool {
    fn theory_description(&self) -> String {
        let mode_tex = if self.sheet_mode {
            format!(
                "|f(x, y, z) - {:.2}| - {:.2} \\le 0",
                self.iso_level,
                0.5 * self.thickness
            )
        } else {
            format!("f(x, y, z) - {:.2} \\le 0", self.iso_level)
        };
        format!(
            "Watertight Marching Cubes isosurface extraction over $(N+2)^3 = {}^3$ padded grid with box-SDF closure: ${mode_tex}$.",
            self.resolution + 2
        )
    }

    fn phonetic_description(&self) -> String {
        "Implicit isosurface Marching Cubes and triply periodic minimal surface theory".into()
    }

    fn theory_citation(&self) -> String {
        "Lorensen & Cline (1987), Marching Cubes: A High Resolution 3D Surface Construction Algorithm"
            .into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::from([
            (
                "Gyroid".into(),
                "Schoen (1970) triply periodic minimal surface free of straight lines and planar symmetries".into(),
            ),
            (
                "Watertight Closure".into(),
                "Intersection with domain box SDF and positive outer halo guarantees 0 open boundary half-edges".into(),
            ),
        ])
    }
}

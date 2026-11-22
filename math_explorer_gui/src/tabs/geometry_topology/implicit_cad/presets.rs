//! Tool registration, theory metadata, and bookmarkable starter presets for [`ImplicitCadStudio`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::ImplicitCadStudio;

/// Canonical Gyroid TPMS equation scaled to physical millimeters (`20 mm` unit cell).
pub const GYROID_EQUATION: &str =
    "(sin(pi*x/10)*cos(pi*y/10) + sin(pi*y/10)*cos(pi*z/10) + sin(pi*z/10)*cos(pi*x/10)) * 3.18";

/// Canonical Schwarz P minimal-surface equation scaled to physical millimeters (`20 mm` unit cell).
pub const SCHWARZ_P_EQUATION: &str = "(cos(pi*x/10) + cos(pi*y/10) + cos(pi*z/10)) * 3.18";

/// Constructive solid geometry equation intersecting a millimeter torus with a slab and central hub.
pub const TORUS_BOX_CSG_EQUATION: &str =
    "min(max(sqrt((sqrt(x^2 + y^2) - 12)^2 + z^2) - 5.5, max(abs(x) - 15, abs(y) - 15, abs(z) - 5)), sqrt(x^2 + y^2 + z^2) - 6.5)";

/// Preset JSON for a printable `40 mm` Gyroid heat-exchanger core with `1.6 mm` walls.
pub const PRESET_IMPLICIT_GYROID: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Gyroid Heat-Exchanger Core",
    "domain": "geometry_topology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "equation": "(sin(pi*x/10)*cos(pi*y/10) + sin(pi*y/10)*cos(pi*z/10) + sin(pi*z/10)*cos(pi*x/10)) * 3.18",
    "box_size_mm": 40.0,
    "shell_thickness_mm": 1.6,
    "resolution": 36,
    "seal_boundaries": true,
    "show_wireframe": false,
    "camera": {
      "pitch": 0.52,
      "yaw": 0.64,
      "zoom": 1.0
    }
  }
}"#;

/// Preset JSON for a sealed `40 mm` Schwarz P minimal lattice with `2.0 mm` walls.
pub const PRESET_IMPLICIT_SCHWARZ_P: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Schwarz P Minimal Lattice",
    "domain": "geometry_topology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "equation": "(cos(pi*x/10) + cos(pi*y/10) + cos(pi*z/10)) * 3.18",
    "box_size_mm": 40.0,
    "shell_thickness_mm": 2.0,
    "resolution": 36,
    "seal_boundaries": true,
    "show_wireframe": false,
    "camera": {
      "pitch": 0.48,
      "yaw": 0.58,
      "zoom": 1.0
    }
  }
}"#;

/// Preset JSON for a boolean CSG solid combining `min(f, g)` union and `max(f, g)` intersection.
pub const PRESET_IMPLICIT_TORUS_BOX: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Torus-Box Boolean CSG Solid",
    "domain": "geometry_topology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "equation": "min(max(sqrt((sqrt(x^2 + y^2) - 12)^2 + z^2) - 5.5, max(abs(x) - 15, abs(y) - 15, abs(z) - 5)), sqrt(x^2 + y^2 + z^2) - 6.5)",
    "box_size_mm": 40.0,
    "shell_thickness_mm": 0.0,
    "resolution": 40,
    "seal_boundaries": true,
    "show_wireframe": false,
    "camera": {
      "pitch": 0.56,
      "yaw": 0.46,
      "zoom": 1.0
    }
  }
}"#;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "ImplicitCadStudio",
        display_name: "Implicit CAD & TPMS Studio",
        domain: "geometry_topology",
        tags: &["cad", "tpms", "stl", "isosurface", "csg"],
        build: || Box::new(ImplicitCadStudio::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "implicit_cad_gyroid",
        title: "Gyroid Heat-Exchanger Core",
        description: "Watertight 40 mm Gyroid triply periodic minimal surface core with 1.6 mm walls for 3D printing.",
        category: "Implicit CAD & TPMS",
        icon: "🧬",
        domain: "geometry_topology",
        tool_name: Some("ImplicitCadStudio"),
        preset_generator: || PRESET_IMPLICIT_GYROID,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "implicit_cad_schwarz_p",
        title: "Schwarz P Minimal Lattice",
        description: "Boundary-sealed 40 mm Schwarz Primitive minimal lattice shell ready for millimeter STL slicing.",
        category: "Implicit CAD & TPMS",
        icon: "🧊",
        domain: "geometry_topology",
        tool_name: Some("ImplicitCadStudio"),
        preset_generator: || PRESET_IMPLICIT_SCHWARZ_P,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "implicit_cad_torus_box",
        title: "Torus-Box Boolean CSG Solid",
        description: "Constructive solid geometry combining torus-slab max() intersection with a central min() hub.",
        category: "Implicit CAD & TPMS",
        icon: "⚙️",
        domain: "geometry_topology",
        tool_name: Some("ImplicitCadStudio"),
        preset_generator: || PRESET_IMPLICIT_TORUS_BOX,
    }
}

impl scientific_metadata::theory::TheoryDescribable for ImplicitCadStudio {
    fn theory_description(&self) -> String {
        "Implicit solid modeling extracts closed 2-manifolds $M = \\{\\mathbf{x} \\in [-L/2, L/2]^3 : F(\\mathbf{x}) = 0\\}$ via Marching Cubes with rolling two-slice edge deduplication, physical central-difference gradients $\\nabla F / |\\nabla F|$, shell thickening $|f(\\mathbf{x})| - w/2$, and R-function CSG operators $\\min(f, g)$ (union) and $\\max(f, g)$ (intersection).".to_string()
    }

    fn phonetic_description(&self) -> String {
        "Implicit solid geometry and triply periodic minimal surface isosurface extraction"
            .to_string()
    }

    fn theory_citation(&self) -> String {
        "Lorensen & Cline, Marching Cubes: A High Resolution 3D Surface Construction Algorithm (SIGGRAPH 1987); Schoen, Infinite Periodic Minimal Surfaces Without Self-Intersections (NASA TN D-5541, 1970)".to_string()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::from([
            (
                "Gyroid".to_string(),
                "Triply periodic minimal surface sin(kx)cos(ky) + sin(ky)cos(kz) + sin(kz)cos(kx) = 0 with zero mean curvature".to_string(),
            ),
            (
                "SchwarzP".to_string(),
                "Cubic minimal lattice cos(kx) + cos(ky) + cos(kz) = 0 partitioning space into two congruent labyrinths".to_string(),
            ),
            (
                "CSG".to_string(),
                "Implicit Boolean operations: union min(f, g), intersection max(f, g), difference max(f, -g), and shell |f| - w/2".to_string(),
            ),
        ])
    }
}

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Starter gallery preset cards for [`super::VolumetricFieldStudioTool`].
// [cite:quantum_mechanics]
// [cite:favorite_child]

use crate::starter_gallery::StarterPresetMetadata;

/// Starter JSON for oscillating `1s + 2pz` electric-dipole superposition (`Delta l = +1`).
pub const PRESET_1S_2PZ_DIPOLE_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "1s + 2pz Dipole Oscillation",
    "domain": "quantum",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "source": "quantum_superposition",
    "render_mode": "hybrid_shell_and_cloud",
    "grid_resolution": 28,
    "n1": 1,
    "l1": 0,
    "m1": 0,
    "n2": 2,
    "l2": 1,
    "m2": 0,
    "superposition_mix": 0.5,
    "relative_phase": 0.0,
    "quantum_dt": 0.08,
    "observable": "signed_wavefunction",
    "iso_threshold": 0.032,
    "splat_scale": 1.0,
    "turing_a": 0.1,
    "turing_b": 0.9,
    "turing_du": 1.0,
    "turing_dv": 18.0,
    "turing_dt": 0.008,
    "cut_enabled": false,
    "cut_axis": "x",
    "cut_offset": 0.0,
    "probe_brush_enabled": true,
    "running": false,
    "camera": {
      "pitch": 0.42,
      "yaw": 0.58,
      "zoom": 1.05
    }
  }
}"#;

/// Starter JSON for `3dz²` polar lobes and equatorial torus nodal cone (`n=3, l=2, m=0`).
pub const PRESET_3DZ2_CONE_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "3dz2 Nodal Cone & Torus",
    "domain": "quantum",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "source": "quantum_superposition",
    "render_mode": "marching_cubes_mesh",
    "grid_resolution": 32,
    "n1": 3,
    "l1": 2,
    "m1": 0,
    "n2": 2,
    "l2": 0,
    "m2": 0,
    "superposition_mix": 0.0,
    "relative_phase": 0.0,
    "quantum_dt": 0.06,
    "observable": "signed_wavefunction",
    "iso_threshold": 0.011,
    "splat_scale": 0.95,
    "turing_a": 0.1,
    "turing_b": 0.9,
    "turing_du": 1.0,
    "turing_dv": 18.0,
    "turing_dt": 0.008,
    "cut_enabled": false,
    "cut_axis": "y",
    "cut_offset": 0.1,
    "probe_brush_enabled": true,
    "running": false,
    "camera": {
      "pitch": 0.48,
      "yaw": 0.65,
      "zoom": 1.1
    }
  }
}"#;

/// Starter JSON for 3D Schnakenberg reaction-diffusion gyroid labyrinth with active cutting plane.
pub const PRESET_3D_TURING_GYROID_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "3D Turing Gyroid Labyrinth",
    "domain": "quantum",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "source": "turing_morphogenesis",
    "render_mode": "hybrid_shell_and_cloud",
    "grid_resolution": 26,
    "n1": 1,
    "l1": 0,
    "m1": 0,
    "n2": 2,
    "l2": 1,
    "m2": 0,
    "superposition_mix": 0.5,
    "relative_phase": 0.0,
    "quantum_dt": 0.08,
    "observable": "signed_wavefunction",
    "iso_threshold": 0.06,
    "splat_scale": 0.9,
    "turing_a": 0.1,
    "turing_b": 0.9,
    "turing_du": 1.0,
    "turing_dv": 18.0,
    "turing_dt": 0.008,
    "cut_enabled": true,
    "cut_axis": "z",
    "cut_offset": 0.25,
    "probe_brush_enabled": true,
    "running": false,
    "camera": {
      "pitch": 0.52,
      "yaw": 0.72,
      "zoom": 1.08
    }
  }
}"#;

/// Starter JSON for thickened Gyroid TPMS lattice ready for millimeter `.stl` slicing.
pub const PRESET_GYROID_TPMS_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Gyroid TPMS Lattice",
    "domain": "quantum",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "source": "custom_implicit",
    "render_mode": "marching_cubes_mesh",
    "grid_resolution": 34,
    "n1": 1,
    "l1": 0,
    "m1": 0,
    "n2": 2,
    "l2": 1,
    "m2": 0,
    "superposition_mix": 0.5,
    "relative_phase": 0.0,
    "quantum_dt": 0.08,
    "observable": "signed_wavefunction",
    "iso_threshold": 0.032,
    "splat_scale": 0.9,
    "turing_a": 0.1,
    "turing_b": 0.9,
    "turing_du": 1.0,
    "turing_dv": 18.0,
    "turing_dt": 0.008,
    "implicit_expr": "sin(k * x) * cos(k * y) + sin(k * y) * cos(k * z) + sin(k * z) * cos(k * x)",
    "implicit_params": { "k": 1.5 },
    "implicit_level": 0.0,
    "implicit_closure": "thickened_wall",
    "wall_thickness": 0.55,
    "x_bounds": [-3.14159265, 3.14159265],
    "y_bounds": [-3.14159265, 3.14159265],
    "z_bounds": [-3.14159265, 3.14159265],
    "export_size_mm": 60.0,
    "cut_enabled": false,
    "cut_axis": "z",
    "cut_offset": 0.0,
    "probe_brush_enabled": false,
    "running": false,
    "camera": { "pitch": 0.48, "yaw": 0.68, "zoom": 1.04 }
  }
}"#;

/// Starter JSON for boundary-capped Schwarz-P triply periodic minimal shell.
pub const PRESET_SCHWARZ_P_SHELL_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Schwarz-P Minimal Shell",
    "domain": "quantum",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "source": "custom_implicit",
    "render_mode": "marching_cubes_mesh",
    "grid_resolution": 34,
    "n1": 1,
    "l1": 0,
    "m1": 0,
    "n2": 2,
    "l2": 1,
    "m2": 0,
    "superposition_mix": 0.5,
    "relative_phase": 0.0,
    "quantum_dt": 0.08,
    "observable": "signed_wavefunction",
    "iso_threshold": 0.032,
    "splat_scale": 0.9,
    "turing_a": 0.1,
    "turing_b": 0.9,
    "turing_du": 1.0,
    "turing_dv": 18.0,
    "turing_dt": 0.008,
    "implicit_expr": "cos(k * x) + cos(k * y) + cos(k * z)",
    "implicit_params": { "k": 1.5 },
    "implicit_level": 0.0,
    "implicit_closure": "thickened_wall",
    "wall_thickness": 0.5,
    "x_bounds": [-3.14159265, 3.14159265],
    "y_bounds": [-3.14159265, 3.14159265],
    "z_bounds": [-3.14159265, 3.14159265],
    "export_size_mm": 50.0,
    "cut_enabled": false,
    "cut_axis": "x",
    "cut_offset": 0.0,
    "probe_brush_enabled": false,
    "running": false,
    "camera": { "pitch": 0.46, "yaw": 0.62, "zoom": 1.05 }
  }
}"#;

/// Starter JSON for constructive solid geometry (`min`/`max`) carved hollow cube-sphere CAD block.
pub const PRESET_CSG_BOOLEAN_SOLID_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "CSG Carved Cube-Sphere",
    "domain": "quantum",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "source": "custom_implicit",
    "render_mode": "marching_cubes_mesh",
    "grid_resolution": 36,
    "n1": 1,
    "l1": 0,
    "m1": 0,
    "n2": 2,
    "l2": 1,
    "m2": 0,
    "superposition_mix": 0.5,
    "relative_phase": 0.0,
    "quantum_dt": 0.08,
    "observable": "signed_wavefunction",
    "iso_threshold": 0.032,
    "splat_scale": 0.9,
    "turing_a": 0.1,
    "turing_b": 0.9,
    "turing_du": 1.0,
    "turing_dv": 18.0,
    "turing_dt": 0.008,
    "implicit_expr": "max(max(abs(x) - a, max(abs(y) - a, abs(z) - a)), r - sqrt(x^2 + y^2 + z^2))",
    "implicit_params": { "a": 1.8, "r": 2.25 },
    "implicit_level": 0.0,
    "implicit_closure": "half_space_solid",
    "wall_thickness": 0.45,
    "x_bounds": [-2.5, 2.5],
    "y_bounds": [-2.5, 2.5],
    "z_bounds": [-2.5, 2.5],
    "export_size_mm": 45.0,
    "cut_enabled": false,
    "cut_axis": "x",
    "cut_offset": 0.0,
    "probe_brush_enabled": false,
    "running": false,
    "camera": { "pitch": 0.5, "yaw": 0.72, "zoom": 1.06 }
  }
}"#;

/// Starter JSON for inline 3D AST Schoen Gyroid minimal surface (`sin(k*x)*cos(k*y) + ...`).
pub const PRESET_3D_IMPLICIT_GYROID_AST_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "3D Implicit Schoen Gyroid AST",
    "domain": "quantum",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "source": "custom_implicit_ast",
    "render_mode": "hybrid_shell_and_cloud",
    "grid_resolution": 32,
    "n1": 1,
    "l1": 0,
    "m1": 0,
    "n2": 2,
    "l2": 1,
    "m2": 0,
    "superposition_mix": 0.5,
    "relative_phase": 0.0,
    "quantum_dt": 0.08,
    "observable": "signed_wavefunction",
    "iso_threshold": 0.25,
    "splat_scale": 0.95,
    "turing_a": 0.1,
    "turing_b": 0.9,
    "turing_du": 1.0,
    "turing_dv": 18.0,
    "turing_dt": 0.008,
    "custom_expr": "sin(k * x) * cos(k * y) + sin(k * y) * cos(k * z) + sin(k * z) * cos(k * x)",
    "custom_params": { "k": 2.4 },
    "cut_enabled": true,
    "cut_axis": "z",
    "cut_offset": 0.2,
    "probe_brush_enabled": true,
    "running": false,
    "camera": { "pitch": 0.46, "yaw": 0.64, "zoom": 1.08 }
  }
}"#;

inventory::submit! {
    StarterPresetMetadata {
        id: "volumetric_1s_2pz_dipole",
        title: "1s + 2pz Dipole Oscillation",
        description: "3D hydrogenic superposition demonstrating electric-dipole selection rules across MarchingCubes shells and Gaussian3D splats.",
        category: "Volumetric Fields",
        icon: "⚛",
        domain: "quantum",
        tool_name: Some("VolumetricFieldStudioTool"),
        preset_generator: || PRESET_1S_2PZ_DIPOLE_JSON,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "volumetric_3dz2_cone",
        title: "3dz² Nodal Cone & Torus",
        description: "Crisp MarchingCubes nodal surface of the hydrogenic 3dz² polar lobes and opposite-phase equatorial torus ready for OBJ export.",
        category: "Volumetric Fields",
        icon: "🔷",
        domain: "quantum",
        tool_name: Some("VolumetricFieldStudioTool"),
        preset_generator: || PRESET_3DZ2_CONE_JSON,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "volumetric_3d_turing_gyroid",
        title: "3D Turing Gyroid Labyrinth",
        description: "Live 3D Schnakenberg reaction-diffusion gyroid channels via FiniteDifference3D with cutting-plane slicing and activator pulse sculpting.",
        category: "Volumetric Fields",
        icon: "🧬",
        domain: "quantum",
        tool_name: Some("VolumetricFieldStudioTool"),
        preset_generator: || PRESET_3D_TURING_GYROID_JSON,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "volumetric_gyroid_tpms",
        title: "Gyroid TPMS Lattice",
        description: "Thickened triply periodic minimal surface sin(kx)cos(ky)+sin(ky)cos(kz)+sin(kz)cos(kx) with watertight boundary caps for 3D printing.",
        category: "Implicit CAD Solids",
        icon: "🕸",
        domain: "quantum",
        tool_name: Some("VolumetricFieldStudioTool"),
        preset_generator: || PRESET_GYROID_TPMS_JSON,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "volumetric_schwarz_p_shell",
        title: "Schwarz-P Minimal Shell",
        description: "Periodic Schwarz-P minimal shell cos(kx)+cos(ky)+cos(kz)=0 sealed at cubic box faces into a manifold STL/PLY solid.",
        category: "Implicit CAD Solids",
        icon: "🧊",
        domain: "quantum",
        tool_name: Some("VolumetricFieldStudioTool"),
        preset_generator: || PRESET_SCHWARZ_P_SHELL_JSON,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "volumetric_csg_boolean_solid",
        title: "CSG Carved Cube-Sphere",
        description: "Constructive solid geometry cube minus sphere evaluated via min/max implicit operators and scaled in millimeters.",
        category: "Implicit CAD Solids",
        icon: "🛠",
        domain: "quantum",
        tool_name: Some("VolumetricFieldStudioTool"),
        preset_generator: || PRESET_CSG_BOOLEAN_SOLID_JSON,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "volumetric_3d_implicit_gyroid_ast",
        title: "3D Implicit Gyroid Formula",
        description: "Zero-allocation CompiledExpr::eval_3d Schoen gyroid and 4f_xyz implicit surfaces with synthesized parameter sliders and 20^3 drag previews.",
        category: "Volumetric Fields",
        icon: "🔣",
        domain: "quantum",
        tool_name: Some("VolumetricFieldStudioTool"),
        preset_generator: || PRESET_3D_IMPLICIT_GYROID_AST_JSON,
    }
}

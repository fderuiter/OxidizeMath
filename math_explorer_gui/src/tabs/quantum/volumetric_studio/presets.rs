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

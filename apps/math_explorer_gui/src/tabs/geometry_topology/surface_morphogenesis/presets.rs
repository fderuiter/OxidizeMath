// [cite:modular_polynomials_review]
//! Curated starter gallery presets for [`super::SurfaceMorphogenesisTool`].

pub(super) const TORUS_STRIPES_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Torus Turing Stripes",
    "domain": "geometry_topology",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "surface_type": "Torus",
    "kinetics_kind": "Schnakenberg",
    "overlay_mode": "MorphogenConcentration",
    "sphere_radius": 1.4,
    "torus_major_radius": 2.2,
    "torus_minor_radius": 0.75,
    "klein_radius": 2.0,
    "wall_thickness": 0.16,
    "u_resolution": 30,
    "v_resolution": 30,
    "schnak_a": 0.1,
    "schnak_b": 0.9,
    "gs_feed": 0.037,
    "gs_kill": 0.06,
    "d_u": 0.05,
    "d_v": 1.25,
    "buckling_gain": 0.28,
    "curvature_tension": 0.06,
    "restoring_tether": 0.12,
    "dt": 0.01,
    "steps_per_frame": 4,
    "camera": {
      "pitch": 0.58,
      "yaw": 0.52,
      "zoom": 1.0
    }
  }
}"#;

pub(super) const KLEIN_MITOSIS_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Klein Bottle Mitosis",
    "domain": "geometry_topology",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "surface_type": "KleinBottle",
    "kinetics_kind": "GrayScott",
    "overlay_mode": "MorphogenConcentration",
    "sphere_radius": 1.4,
    "torus_major_radius": 2.2,
    "torus_minor_radius": 0.75,
    "klein_radius": 2.1,
    "wall_thickness": 0.16,
    "u_resolution": 30,
    "v_resolution": 30,
    "schnak_a": 0.1,
    "schnak_b": 0.9,
    "gs_feed": 0.0367,
    "gs_kill": 0.0649,
    "d_u": 0.04,
    "d_v": 0.02,
    "buckling_gain": 0.35,
    "curvature_tension": 0.05,
    "restoring_tether": 0.10,
    "dt": 0.02,
    "steps_per_frame": 6,
    "camera": {
      "pitch": 0.48,
      "yaw": 0.68,
      "zoom": 0.95
    }
  }
}"#;

pub(super) const SPHERE_POLAR_SPOTS_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Sphere Polar Spots",
    "domain": "geometry_topology",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "surface_type": "Sphere",
    "kinetics_kind": "Schnakenberg",
    "overlay_mode": "MorphogenConcentration",
    "sphere_radius": 1.5,
    "torus_major_radius": 2.2,
    "torus_minor_radius": 0.75,
    "klein_radius": 2.0,
    "wall_thickness": 0.16,
    "u_resolution": 32,
    "v_resolution": 24,
    "schnak_a": 0.12,
    "schnak_b": 0.88,
    "gs_feed": 0.037,
    "gs_kill": 0.06,
    "d_u": 0.04,
    "d_v": 0.95,
    "buckling_gain": 0.32,
    "curvature_tension": 0.07,
    "restoring_tether": 0.14,
    "dt": 0.008,
    "steps_per_frame": 4,
    "camera": {
      "pitch": 0.50,
      "yaw": 0.40,
      "zoom": 1.15
    }
  }
}"#;

pub(super) const SADDLE_ANISOTROPY_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Hyperbolic Saddle Buckling",
    "domain": "geometry_topology",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "surface_type": "CustomHeightfield",
    "kinetics_kind": "Schnakenberg",
    "overlay_mode": "MetricAnisotropy",
    "sphere_radius": 1.4,
    "torus_major_radius": 2.2,
    "torus_minor_radius": 0.75,
    "klein_radius": 2.0,
    "wall_thickness": 0.18,
    "u_resolution": 28,
    "v_resolution": 28,
    "schnak_a": 0.1,
    "schnak_b": 0.9,
    "gs_feed": 0.037,
    "gs_kill": 0.06,
    "d_u": 0.05,
    "d_v": 1.10,
    "buckling_gain": 0.26,
    "curvature_tension": 0.06,
    "restoring_tether": 0.12,
    "dt": 0.008,
    "steps_per_frame": 4,
    "camera": {
      "pitch": 0.55,
      "yaw": 0.45,
      "zoom": 1.0
    },
    "custom_expr": "a * (x^2 - y^2)",
    "custom_params": {
      "a": 0.45
    }
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "surface_morpho_torus_stripes",
        title: "Torus Turing Stripes",
        description: "Δ_g u + R(u,v): Schnakenberg stripes stretching across positive and negative Gaussian curvature on a torus.",
        category: "Differential Geometry",
        icon: "🍩",
        domain: "geometry_topology",
        tool_name: Some("SurfaceMorphogenesisTool"),
        preset_generator: || TORUS_STRIPES_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "surface_morpho_klein_mitosis",
        title: "Klein Bottle Mitosis",
        description: "Non-orientable Gray-Scott spot replication crossing the twisted Klein seam without boundary artifacts.",
        category: "Differential Geometry",
        icon: "🧬",
        domain: "geometry_topology",
        tool_name: Some("SurfaceMorphogenesisTool"),
        preset_generator: || KLEIN_MITOSIS_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "surface_morpho_sphere_poles",
        title: "Sphere Polar Spots",
        description: "Stokes-averaged polar caps coupling Laplace-Beltrami reaction-diffusion to radial lobe buckling.",
        category: "Differential Geometry",
        icon: "🌐",
        domain: "geometry_topology",
        tool_name: Some("SurfaceMorphogenesisTool"),
        preset_generator: || SPHERE_POLAR_SPOTS_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "surface_morpho_saddle_anisotropy",
        title: "Hyperbolic Saddle Buckling",
        description: "z = a*(x^2 - y^2): Principal metric anisotropy and buckled morphogenesis over a custom AST saddle.",
        category: "Differential Geometry",
        icon: "🏇",
        domain: "geometry_topology",
        tool_name: Some("SurfaceMorphogenesisTool"),
        preset_generator: || SADDLE_ANISOTROPY_PRESET_JSON,
    }
}

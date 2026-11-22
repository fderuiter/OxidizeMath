#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub(super) const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Gaussian Curvature Heatmap",
    "domain": "geometry_topology",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "surface_type": "Torus",
    "curvature_type": "Gaussian",
    "sphere_radius": 1.0,
    "torus_major_radius": 2.5,
    "torus_minor_radius": 0.8,
    "klein_radius": 2.0,
    "u_resolution": 36,
    "v_resolution": 36,
    "camera": {
      "pitch": 0.6,
      "yaw": 0.4,
      "zoom": 1.0
    }
  }
}"#;

pub(super) const CUSTOM_SADDLE_CURVATURE_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Hyperbolic Saddle Curvature",
    "domain": "geometry_topology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "surface_type": "CustomHeightfield",
    "curvature_type": "Gaussian",
    "sphere_radius": 1.0,
    "torus_major_radius": 2.0,
    "torus_minor_radius": 0.5,
    "klein_radius": 2.0,
    "u_resolution": 32,
    "v_resolution": 32,
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
        id: "curvature_heatmap_starter",
        title: "Gaussian Curvature Heatmap",
        description: "K = κ₁·κ₂: Intrinsic differential Gaussian and mean curvature distribution across manifolds.",
        category: "Differential Geometry",
        icon: "🌐",
        domain: "geometry_topology",
        tool_name: Some("CurvatureHeatmap"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "curvature_heatmap_custom_saddle",
        title: "Hyperbolic Saddle Curvature",
        description: "z = a*(x^2 - y^2): Negative Gaussian curvature heatmap over a parameterized hyperbolic paraboloid.",
        category: "Differential Geometry",
        icon: "🏇",
        domain: "geometry_topology",
        tool_name: Some("CurvatureHeatmap"),
        preset_generator: || CUSTOM_SADDLE_CURVATURE_PRESET_JSON,
    }
}

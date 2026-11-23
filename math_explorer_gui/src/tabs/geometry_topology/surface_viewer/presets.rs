pub(super) const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Parametric Surface Viewer",
    "domain": "geometry_topology",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "surface_type": "KleinBottle",
    "sphere_radius": 1.0,
    "torus_major_radius": 2.0,
    "torus_minor_radius": 0.5,
    "klein_radius": 2.2,
    "u_resolution": 40,
    "v_resolution": 40,
    "camera": {
      "pitch": 0.6,
      "yaw": 0.4,
      "zoom": 1.0
    }
  }
}"#;

pub(super) const CUSTOM_RIPPLE_MANIFOLD_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Custom Ripple Manifold",
    "domain": "geometry_topology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "surface_type": "CustomHeightfield",
    "sphere_radius": 1.0,
    "torus_major_radius": 2.0,
    "torus_minor_radius": 0.5,
    "klein_radius": 2.0,
    "u_resolution": 36,
    "v_resolution": 36,
    "camera": {
      "pitch": 0.55,
      "yaw": 0.45,
      "zoom": 1.0
    },
    "custom_expr": "a * cos(k * sqrt(x^2 + y^2 + 0.01))",
    "custom_params": {
      "a": 0.7,
      "k": 2.8
    }
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "surface_viewer_starter",
        title: "Parametric Surface Viewer",
        description: "Parametric 3D wireframe manifold visualization for Sphere, Torus, and Klein Bottle surfaces.",
        category: "Differential Geometry",
        icon: "🔮",
        domain: "geometry_topology",
        tool_name: Some("SurfaceViewer"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "surface_viewer_custom_ripple",
        title: "Custom Ripple Manifold",
        description: "z = a*cos(k*sqrt(x^2 + y^2)): Compiled Monge heightfield wireframe with .obj export and live sliders.",
        category: "Differential Geometry",
        icon: "🌊",
        domain: "geometry_topology",
        tool_name: Some("SurfaceViewer"),
        preset_generator: || CUSTOM_RIPPLE_MANIFOLD_PRESET_JSON,
    }
}

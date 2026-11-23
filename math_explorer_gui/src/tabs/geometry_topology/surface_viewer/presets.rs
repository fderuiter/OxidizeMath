#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::custom_surface::SurfaceType;
use super::SurfaceViewer;

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
        description: "z = a*cos(k*sqrt(x^2 + y^2)): Compiled Monge heightfield wireframe with .obj/.stl export and live sliders.",
        category: "Differential Geometry",
        icon: "🌊",
        domain: "geometry_topology",
        tool_name: Some("SurfaceViewer"),
        preset_generator: || CUSTOM_RIPPLE_MANIFOLD_PRESET_JSON,
    }
}

impl scientific_metadata::theory::TheoryDescribable for SurfaceViewer {
    fn theory_description(&self) -> String {
        let surf_desc = match self.surface_type {
            SurfaceType::Sphere => {
                "sphere $S^2$: $\\vec{r}(u, v) = (R \\sin v \\cos u, R \\sin v \\sin u, R \\cos v)$"
            }
            SurfaceType::Torus => {
                "torus $T^2$: $\\vec{r}(u, v) = ((R + r \\cos v)\\cos u, (R + r \\cos v)\\sin u, r \\sin v)$"
            }
            SurfaceType::KleinBottle => {
                "non-orientable Klein bottle embedded in 3D: $\\chi(M) = 0$"
            }
            SurfaceType::CustomHeightfield => {
                "custom Monge heightfield $z = f(x, y)$ compiled to stack bytecode"
            }
            SurfaceType::CustomParametric => {
                "custom 3D parametric immersion $\\vec{r}(u, v) = (x(u,v), y(u,v), z(u,v))$"
            }
        };
        format!("Parametric 2-manifold surface representation: {surf_desc}.")
    }

    fn phonetic_description(&self) -> String {
        "Parametric 2-manifold differential geometry theory".into()
    }

    fn theory_citation(&self) -> String {
        "Differential Geometry of Curves and Surfaces (do Carmo, 2016)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Sphere".into(),
            "Compact orientable manifold of genus 0 with Euler characteristic 2".into(),
        );
        map.insert(
            "Torus".into(),
            "Compact orientable manifold of genus 1 with Euler characteristic 0".into(),
        );
        map.insert(
            "KleinBottle".into(),
            "Non-orientable closed surface with Euler characteristic 0".into(),
        );
        map
    }
}

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{HyperbolicGeometryTool, LabMode};
use crate::framework::ToolMetadata;
use crate::starter_gallery::StarterPresetMetadata;

inventory::submit! {
    ToolMetadata {
        id: "HyperbolicGeometryTool",
        display_name: "Hyperbolic Geometry",
        domain: "geometry_topology",
        tags: &["hyperbolic", "poincare-disk", "mobius", "tiling", "billiards"],
        build: || Box::new(HyperbolicGeometryTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for HyperbolicGeometryTool {
    fn theory_description(&self) -> String {
        match self.state.mode {
            LabMode::TilingExplorer => {
                "Hyperbolic plane H^2 (K = -1) modeled on the Poincare disk D and upper half-plane H \
                 linked by the isometric Cayley map w = i(1 + z)/(1 - z), with PSU(1,1) / PSL(2,R) \
                 Mobius symmetries and Gauss-Bonnet area A = (p - 2)pi - sum(alpha_i)."
                    .into()
            }
            LabMode::ChaoticBilliards => {
                "Anosov-type geodesic billiard flow in constant negative curvature K = -1: twin rays \
                 separated by delta_0 diverge exponentially as delta_t ~ delta_0 * exp(t) across \
                 orthogonal circular-arc reflections."
                    .into()
            }
        }
    }

    fn phonetic_description(&self) -> String {
        "Poincare disk hyperbolic tilings, Cayley half-plane isometries, and chaotic geodesic billiards"
            .into()
    }

    fn theory_citation(&self) -> String {
        "Foundations of Hyperbolic Manifolds (Ratcliffe, 2019); Poincare (1882)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "PoincareDisk".into(),
            "Conformal metric ds = 2|dz| / (1 - |z|^2) with PSU(1,1) Mobius isometries".into(),
        );
        map.insert(
            "UpperHalfPlane".into(),
            "Conformal metric ds = |dw| / Im(w) with PSL(2,R) fractional linear isometries".into(),
        );
        map.insert(
            "GeodesicBilliards".into(),
            "Unit-speed negative-curvature billiard reflections and Lyapunov divergence ln(delta_t / delta_0)"
                .into(),
        );
        map
    }
}

pub(super) const PRESET_ESCHER_TILING_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Poincaré Disk {5,4} Möbius Tiling",
    "domain": "geometry_topology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "mode": "TilingExplorer",
    "chart_layout": "DualSynchronized",
    "tiling_preset": "Pentagonal54",
    "schlafli_p": 5,
    "schlafli_q": 4,
    "tiling_depth": 4,
    "show_horocycles": true,
    "horocycle_radius": 0.24,
    "billiard_preset": "CompactOctagon83",
    "billiard_seed": 42,
    "perturbation_log10": -6.0,
    "billiard_speed": 1.6,
    "billiards_running": false,
    "disk_isometry": {
      "a_re": 1.0235,
      "a_im": 0.0,
      "b_re": 0.18,
      "b_im": 0.125589
    },
    "half_plane_span": 3.2
  }
}"#;

pub(super) const PRESET_CHAOTIC_BILLIARDS_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Chaotic Ideal-Triangle Geodesic Billiards",
    "domain": "geometry_topology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "mode": "ChaoticBilliards",
    "chart_layout": "DualSynchronized",
    "tiling_preset": "Heptagonal73",
    "schlafli_p": 7,
    "schlafli_q": 3,
    "tiling_depth": 3,
    "show_horocycles": false,
    "horocycle_radius": 0.28,
    "billiard_preset": "IdealTriangle",
    "billiard_seed": 137,
    "perturbation_log10": -6.0,
    "billiard_speed": 2.0,
    "billiards_running": true,
    "disk_isometry": {
      "a_re": 1.0,
      "a_im": 0.0,
      "b_re": 0.0,
      "b_im": 0.0
    },
    "half_plane_span": 3.0
  }
}"#;

inventory::submit! {
    StarterPresetMetadata {
        id: "hyperbolic_escher_tiling_starter",
        title: "Poincaré Disk {5,4} Möbius Tiling",
        description: "ds = 2|dz|/(1-|z|²): Dual Poincaré disk and upper half-plane Coxeter tilings with live PSU(1,1) dragging.",
        category: "Hyperbolic Geometry",
        icon: "🌀",
        domain: "geometry_topology",
        tool_name: Some("HyperbolicGeometryTool"),
        preset_generator: || PRESET_ESCHER_TILING_JSON,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "hyperbolic_chaotic_billiards_starter",
        title: "Chaotic Ideal-Triangle Geodesic Billiards",
        description: "ln(δ_t/δ_0) ≈ λt: Exponential twin-ray divergence across negative-curvature circular-arc reflections.",
        category: "Hyperbolic Geometry",
        icon: "🎱",
        domain: "geometry_topology",
        tool_name: Some("HyperbolicGeometryTool"),
        preset_generator: || PRESET_CHAOTIC_BILLIARDS_JSON,
    }
}

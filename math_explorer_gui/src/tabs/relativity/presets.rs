#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Four curated starter presets for the General Relativity starter gallery.

use crate::starter_gallery::StarterPresetMetadata;

/// Starter preset 1: Schwarzschild near-critical photon ring and gravitational lensing.
pub const PRESET_PHOTON_RING_LENSING: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Geodesic Playground",
    "domain": "relativity",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "spacetime": "Schwarzschild",
    "mode": "NullPhoton",
    "scale_radius": 2.0,
    "spin_chi": 0.0,
    "initial_radius": 11.0,
    "impact_or_momentum": 5.21,
    "radial_bias": -0.88,
    "step_size": 0.04,
    "steps_per_frame": 6,
    "is_playing": false,
    "scrub_index": 220,
    "camera": {
      "pitch": 0.52,
      "yaw": -0.58,
      "zoom": 1.0
    }
  }
}"#;

/// Starter preset 2: Relativistic rosette precession and twin clock divergence near ISCO.
pub const PRESET_ISCO_ROSETTE_CLOCKS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Geodesic Playground",
    "domain": "relativity",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "spacetime": "Schwarzschild",
    "mode": "TimelikeParticle",
    "scale_radius": 2.0,
    "spin_chi": 0.0,
    "initial_radius": 7.5,
    "impact_or_momentum": 3.85,
    "radial_bias": -0.22,
    "step_size": 0.05,
    "steps_per_frame": 8,
    "is_playing": false,
    "scrub_index": 240,
    "camera": {
      "pitch": 0.44,
      "yaw": -0.42,
      "zoom": 0.95
    }
  }
}"#;

/// Starter preset 3: Traversable Morris-Thorne wormhole throat transit.
pub const PRESET_WORMHOLE_TRANSIT: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Geodesic Playground",
    "domain": "relativity",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "spacetime": "MorrisThorneWormhole",
    "mode": "NullPhoton",
    "scale_radius": 1.8,
    "spin_chi": 0.0,
    "initial_radius": 7.0,
    "impact_or_momentum": 1.15,
    "radial_bias": -0.92,
    "step_size": 0.05,
    "steps_per_frame": 6,
    "is_playing": false,
    "scrub_index": 210,
    "camera": {
      "pitch": 0.36,
      "yaw": -0.72,
      "zoom": 1.05
    }
  }
}"#;

/// Starter preset 4: Kerr equatorial frame-dragging Christoffel tensor cross-check.
pub const PRESET_CHRISTOFFEL_CROSS_CHECK: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Christoffel Inspector",
    "domain": "relativity",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "manifold": "KerrEquatorial4D",
    "scale_radius": 2.0,
    "spin_chi": 0.75,
    "radial_coord": 3.6,
    "theta": 1.5707963267948966,
    "selected_mu": 3
  }
}"#;

inventory::submit! {
    StarterPresetMetadata {
        id: "relativity_photon_ring_lensing",
        title: "Photon Ring & Gravitational Lensing",
        description: "Null ray grazing the unstable r = 1.5 Rs photon sphere across Flamm's paraboloid.",
        category: "Black Hole Optics",
        icon: "🕳️",
        domain: "relativity",
        tool_name: Some("GeodesicPlaygroundTool"),
        preset_generator: || PRESET_PHOTON_RING_LENSING,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "relativity_isco_rosette_clocks",
        title: "ISCO Rosette & Twin Clocks",
        description: "Precessing timelike orbit near r = 3 Rs comparing coordinate time t and proper time tau.",
        category: "Relativistic Orbits",
        icon: "⏱️",
        domain: "relativity",
        tool_name: Some("GeodesicPlaygroundTool"),
        preset_generator: || PRESET_ISCO_ROSETTE_CLOCKS,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "relativity_wormhole_transit",
        title: "Morris-Thorne Wormhole Transit",
        description: "Horizon-free geodesic crossing the l = 0 throat between upper and lower isometric sheets.",
        category: "Exotic Spacetimes",
        icon: "🌌",
        domain: "relativity",
        tool_name: Some("GeodesicPlaygroundTool"),
        preset_generator: || PRESET_WORMHOLE_TRANSIT,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "relativity_christoffel_inspector",
        title: "Christoffel Connection Cross-Check",
        description: "Compare closed-form Kerr equatorial Christoffel symbols against central-difference stencils.",
        category: "Tensor Calculus",
        icon: "📐",
        domain: "relativity",
        tool_name: Some("ChristoffelInspectorTool"),
        preset_generator: || PRESET_CHRISTOFFEL_CROSS_CHECK,
    }
}

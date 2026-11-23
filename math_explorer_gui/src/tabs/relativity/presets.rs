#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Curated starter presets, guided story tour, and playable AST goal challenges for General Relativity.

use crate::framework::CalloutSpec;
use crate::starter_gallery::{
    GoalChallengeMetadata, PresetStoryMetadata, StarterPresetMetadata, StoryStep,
};

const fn pin(
    tag: &'static str,
    target_point: [f64; 2],
    label_offset: [f32; 2],
    text: &'static str,
) -> CalloutSpec {
    CalloutSpec {
        tag,
        target_point,
        label_offset,
        text,
    }
}

/// Starter preset 1: Schwarzschild near-critical photon ring and twin worldline bifurcation.
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
    "impact_or_momentum": 5.205,
    "twin_delta_b": -0.03,
    "radial_bias": -0.88,
    "step_size": 0.05,
    "steps_per_frame": 6,
    "is_playing": false,
    "scrub_index": 560,
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
    "twin_delta_b": 0.15,
    "radial_bias": -0.22,
    "step_size": 0.05,
    "steps_per_frame": 8,
    "is_playing": false,
    "scrub_index": 360,
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
    "scrub_index": 320,
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

const STORY_STEP_WEAK_LENSING: &str = r#"{"metadata":{"schema_version":"1.0","title":"Weak-Field Deflection","domain":"relativity","timestamp":"2026-11-21T00:00:00Z"},"config":{"spacetime":"Schwarzschild","mode":"NullPhoton","scale_radius":2.0,"spin_chi":0.0,"initial_radius":11.0,"impact_or_momentum":6.8,"twin_delta_b":-0.08,"radial_bias":-0.88,"step_size":0.05,"steps_per_frame":6,"is_playing":false,"scrub_index":480,"camera":{"pitch":0.52,"yaw":-0.58,"zoom":1.0}}}"#;

const STORY_STEP_KERR_DRAG: &str = r#"{"metadata":{"schema_version":"1.0","title":"Kerr Ergosphere Skim","domain":"relativity","timestamp":"2026-11-21T00:00:00Z"},"config":{"spacetime":"KerrEquatorial","mode":"NullPhoton","scale_radius":2.0,"spin_chi":0.9,"initial_radius":10.5,"impact_or_momentum":2.96,"twin_delta_b":-0.08,"radial_bias":-0.9,"step_size":0.04,"steps_per_frame":6,"is_playing":false,"scrub_index":560,"camera":{"pitch":0.5,"yaw":-0.55,"zoom":1.05}}}"#;

const CHALLENGE_KERR_BASELINE: &str = r#"{"metadata":{"schema_version":"1.0","title":"Slow Kerr Baseline","domain":"relativity","timestamp":"2026-11-21T00:00:00Z"},"config":{"spacetime":"KerrEquatorial","mode":"NullPhoton","scale_radius":2.0,"spin_chi":0.25,"initial_radius":10.5,"impact_or_momentum":5.6,"twin_delta_b":-0.06,"radial_bias":-0.9,"step_size":0.04,"steps_per_frame":6,"is_playing":false,"scrub_index":560,"camera":{"pitch":0.5,"yaw":-0.55,"zoom":1.05}}}"#;

const CRITICAL_LENSING_STEPS: &[StoryStep] = &[
    StoryStep {
        title: "Weak-Field Centrifugal Reflection",
        narrative: "At b = 6.8 > b_crit = 5.196, the horizontal 0.5 e^2 = 0.50 line lies well below the V_eff(1.5 Rs) barrier peak, so both twin light rays reflect cleanly near r_min = 5.65.",
        preset_generator: || STORY_STEP_WEAK_LENSING,
        callouts: &[pin(
            "relativity_step",
            [3.0, 0.856],
            [18.0, -24.0],
            "Centrifugal peak at r_ph = 1.5 Rs",
        )],
        focus_observables: &["launch_energy", "r_min", "delta_phi"],
    },
    StoryStep {
        title: "Critical Photon-Ring Twin Bifurcation",
        narrative: "Lowering b to 5.205 brings 0.5 e^2 tangent to V_eff(3.0): the primary ray grazes r_min = 3.13 and whipsaws past the photon sphere while its b - 0.03 twin plunges across r+ = 2.0.",
        preset_generator: || PRESET_PHOTON_RING_LENSING,
        callouts: &[pin(
            "relativity_step",
            [3.0, 0.501],
            [20.0, -26.0],
            "0.5 e^2 grazes V_eff(1.5 Rs) Saddle",
        )],
        focus_observables: &["r_min", "twin_captured", "delta_phi"],
    },
    StoryStep {
        title: "Kerr Frame Dragging & Shrinking Horizon",
        narrative: "Spinning up to chi = 0.90 shrinks the outer horizon to r+ = 1.436 inside the r_ergo = 2.0 static limit, shifting the prograde photon ring inward to b_crit = 2.91.",
        preset_generator: || STORY_STEP_KERR_DRAG,
        callouts: &[pin(
            "relativity_step",
            [2.0, 0.50],
            [22.0, -24.0],
            "Static limit r_ergo = 2.0 > r+ = 1.436",
        )],
        focus_observables: &[
            "outer_horizon_radius",
            "ergosphere_radius",
            "frame_dragging_omega",
        ],
    },
];

inventory::submit! {
    StarterPresetMetadata {
        id: "relativity_photon_ring_lensing",
        title: "Photon Ring & Gravitational Lensing",
        description: "Twin null rays straddling b_crit = 3*sqrt(3)/2 Rs near the unstable r = 1.5 Rs photon sphere.",
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

inventory::submit! {
    PresetStoryMetadata {
        id: "relativity_critical_lensing_tour",
        title: "Critical Photon Ring & Frame-Dragging Tour",
        description: "From weak lensing to twin photon-sphere bifurcation and Kerr ergosphere frame dragging.",
        category: "Guided Story",
        icon: "📖",
        domain: "relativity",
        tool_name: "GeodesicPlaygroundTool",
        steps: CRITICAL_LENSING_STEPS,
    }
}

inventory::submit! {
    GoalChallengeMetadata {
        id: "relativity_photon_ring_bifurcation",
        title: "Split the Twin Photon Ring",
        description: "Tune impact parameter b so the primary ray escapes from r_min < 3.35 while its twin plunges.",
        category: "Challenge",
        icon: "🎯",
        domain: "relativity",
        tool_name: "GeodesicPlaygroundTool",
        initial_preset_generator: || STORY_STEP_WEAK_LENSING,
        goal_summary: "Keep the primary photon uncaptured (captured < 0.5) with periapsis r_min < 3.35 while the twin ray plunges into the horizon (twin_captured > 0.5).",
        predicate_expr: "captured < 0.5 && r_min < 3.35 && twin_captured > 0.5",
        hint: "Lower 'Impact Parameter b' to ~5.21 (or click just right of the +b_crit contour in the 2D Parameter Atlas) while keeping Twin Offset delta_b = -0.08.",
        win_message: "Challenge solved: Critical Photon-Ring Bifurcation! Your primary ray grazed r = 1.5 Rs and escaped while its twin crossed r+.",
        callouts: &[pin("relativity_challenge_ring", [3.0, 0.5], [18.0, -24.0], "Target r_ph = 3.0 saddle")],
        focus_observables: &["r_min", "captured", "twin_captured"],
    }
}

inventory::submit! {
    GoalChallengeMetadata {
        id: "relativity_kerr_ergosphere_skim",
        title: "Skim the Kerr Prograde Photon Sphere",
        description: "Spin up a Kerr black hole above chi = 0.75 and skim inside r < 2.85 without plunging.",
        category: "Challenge",
        icon: "🌀",
        domain: "relativity",
        tool_name: "GeodesicPlaygroundTool",
        initial_preset_generator: || CHALLENGE_KERR_BASELINE,
        goal_summary: "Raise Kerr Spin above 0.75 (outer_horizon_radius < 1.65) and dial prograde Impact Parameter b so r_min < 2.85 without capture (captured < 0.5).",
        predicate_expr: "spin_chi > 0.75 && outer_horizon_radius < 1.65 && captured < 0.5 && r_min < 2.85",
        hint: "Raise 'Kerr Spin chi' to ~0.90 and lower 'Impact Parameter b' near ~2.96, or open the 2D Parameter Atlas on (b, spin_chi) and click just outside b_crit(chi).",
        win_message: "Challenge solved: Prograde Kerr Frame Dragging! Co-rotating frame dragging let light dip inside r = 2.85 and escape.",
        callouts: &[pin("relativity_challenge_kerr", [2.0, 0.5], [20.0, -24.0], "Ergosphere r_ergo = 2.0")],
        focus_observables: &["spin_chi", "outer_horizon_radius", "r_min", "captured"],
    }
}

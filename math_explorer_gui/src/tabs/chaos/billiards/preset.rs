//! Bounded JSON state persistence, URL share state validation, and Starter Gallery presets.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{BilliardGeometryKind, DynamicalBilliardsTool, MAX_RAYS};
use crate::starter_gallery::StarterPresetMetadata;
use std::f64::consts::PI;

/// Compact parameter snapshot (`< 512` bytes) for [`DynamicalBilliardsTool`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BilliardsLabState {
    /// Central ray launch direction $\theta \in [-\pi, \pi]$.
    pub launch_angle: f64,
    /// Angular spread $\Delta \theta \in [0, 1.2]$ across the ray fan.
    pub fan_spread: f64,
    /// Number of rays in the fan (`1..=MAX_RAYS`).
    pub fan_count: usize,
    /// Specular collisions advanced per frame (`1..=64`).
    pub bounces_per_frame: usize,
    /// Selected Euclidean table geometry.
    pub geometry: BilliardGeometryKind,
    /// Circle radius $R$.
    pub circle_radius: f64,
    /// Ellipse semi-major axis $a$.
    pub ellipse_a: f64,
    /// Ellipse semi-minor axis $b \le a$.
    pub ellipse_b: f64,
    /// Stadium semicircle cap radius $R$.
    pub stadium_radius: f64,
    /// Stadium flat-wall half-length $a$.
    pub stadium_half_straight: f64,
    /// Sinai outer square half-side $L$.
    pub sinai_half_side: f64,
    /// Sinai central scatterer radius $R < L$.
    pub sinai_scatterer_radius: f64,
    /// Interior launch position `[x, y]`.
    pub launch_pos: [f64; 2],
    /// Whether continuous bounce animation is active.
    pub is_playing: bool,
}

impl DynamicalBilliardsTool {
    pub(super) fn serialize_billiards_state(&self) -> Option<String> {
        let snapshot = BilliardsLabState {
            launch_angle: self.launch_angle,
            fan_spread: self.fan_spread,
            fan_count: self.fan_count,
            bounces_per_frame: self.bounces_per_frame,
            geometry: self.geometry,
            circle_radius: self.circle_radius,
            ellipse_a: self.ellipse_a,
            ellipse_b: self.ellipse_b,
            stadium_radius: self.stadium_radius,
            stadium_half_straight: self.stadium_half_straight,
            sinai_half_side: self.sinai_half_side,
            sinai_scatterer_radius: self.sinai_scatterer_radius,
            launch_pos: self.launch_pos,
            is_playing: self.is_playing,
        };
        serde_json::to_string(&snapshot).ok()
    }

    pub(super) fn deserialize_billiards_state(&mut self, raw: &str) -> Result<(), String> {
        let parsed: BilliardsLabState = serde_json::from_str(raw)
            .map_err(|e| format!("Invalid DynamicalBilliardsTool state: {e}"))?;
        validate_billiards_state(&parsed)?;

        let mut candidate = Self {
            geometry: parsed.geometry,
            launch_angle: parsed.launch_angle,
            fan_spread: parsed.fan_spread,
            fan_count: parsed.fan_count,
            bounces_per_frame: parsed.bounces_per_frame,
            circle_radius: parsed.circle_radius,
            ellipse_a: parsed.ellipse_a,
            ellipse_b: parsed.ellipse_b,
            stadium_radius: parsed.stadium_radius,
            stadium_half_straight: parsed.stadium_half_straight,
            sinai_half_side: parsed.sinai_half_side,
            sinai_scatterer_radius: parsed.sinai_scatterer_radius,
            launch_pos: parsed.launch_pos,
            is_playing: parsed.is_playing,
            rays: Vec::with_capacity(MAX_RAYS),
            last_frame_ms: 0.0,
            state_rev: self.state_rev.wrapping_add(1),
            focused_canvas: self.focused_canvas,
            last_status_note: None,
            circle: self.circle,
            ellipse: self.ellipse,
            stadium: self.stadium,
            sinai: self.sinai,
        };
        candidate.sync_table_kernels();
        if !candidate.contains_interior_point(parsed.launch_pos) {
            return Err("launch_pos lies outside the active billiard table interior".into());
        }
        candidate.fire_fan_at(parsed.launch_pos)?;
        *self = candidate;
        Ok(())
    }
}

fn validate_billiards_state(s: &BilliardsLabState) -> Result<(), String> {
    if !s.launch_angle.is_finite() || !(-PI..=PI).contains(&s.launch_angle) {
        return Err(format!("launch_angle out of [-π, π]: {}", s.launch_angle));
    }
    if !s.fan_spread.is_finite() || !(0.0..=1.2).contains(&s.fan_spread) {
        return Err(format!("fan_spread out of [0, 1.2]: {}", s.fan_spread));
    }
    if !(1..=MAX_RAYS).contains(&s.fan_count) {
        return Err(format!("fan_count out of 1..={MAX_RAYS}: {}", s.fan_count));
    }
    if !(1..=64).contains(&s.bounces_per_frame) {
        return Err(format!(
            "bounces_per_frame out of 1..=64: {}",
            s.bounces_per_frame
        ));
    }
    if !s.circle_radius.is_finite() || !(0.3..=2.5).contains(&s.circle_radius) {
        return Err("circle_radius out of [0.3, 2.5]".into());
    }
    if !s.ellipse_a.is_finite()
        || !s.ellipse_b.is_finite()
        || !(0.6..=2.5).contains(&s.ellipse_a)
        || !(0.3..=s.ellipse_a).contains(&s.ellipse_b)
    {
        return Err("ellipse axes out of valid confocal bounds".into());
    }
    if !s.stadium_radius.is_finite()
        || !s.stadium_half_straight.is_finite()
        || !(0.2..=1.5).contains(&s.stadium_radius)
        || !(0.05..=2.0).contains(&s.stadium_half_straight)
    {
        return Err("stadium dimensions out of valid bounds".into());
    }
    if !s.sinai_half_side.is_finite()
        || !s.sinai_scatterer_radius.is_finite()
        || !(0.6..=2.0).contains(&s.sinai_half_side)
        || !(0.08..=0.9).contains(&s.sinai_scatterer_radius)
        || s.sinai_scatterer_radius >= s.sinai_half_side * 0.8
    {
        return Err("sinai dimensions out of valid bounds".into());
    }
    if !s.launch_pos[0].is_finite() || !s.launch_pos[1].is_finite() {
        return Err("launch_pos must be finite".into());
    }
    Ok(())
}

/// Starter preset: Confocal elliptical billiard whispering-gallery & hyperbolic caustics.
pub const PRESET_ELLIPTICAL_CAUSTICS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Confocal Elliptical Billiard Caustics",
    "domain": "chaos",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "launch_angle": 0.72,
    "fan_spread": 0.45,
    "fan_count": 6,
    "bounces_per_frame": 10,
    "geometry": "Ellipse",
    "circle_radius": 1.0,
    "ellipse_a": 1.45,
    "ellipse_b": 1.0,
    "stadium_radius": 0.75,
    "stadium_half_straight": 0.70,
    "sinai_half_side": 1.10,
    "sinai_scatterer_radius": 0.38,
    "launch_pos": [0.42, 0.25],
    "is_playing": false
  }
}"#;

/// Starter preset: Ergodic defocusing Bunimovich stadium filling Birkhoff phase space.
pub const PRESET_BUNIMOVICH_STADIUM: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Bunimovich Stadium Ergodic Billiard",
    "domain": "chaos",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "launch_angle": 0.51,
    "fan_spread": 0.02,
    "fan_count": 5,
    "bounces_per_frame": 12,
    "geometry": "Stadium",
    "circle_radius": 1.0,
    "ellipse_a": 1.35,
    "ellipse_b": 1.0,
    "stadium_radius": 0.75,
    "stadium_half_straight": 0.80,
    "sinai_half_side": 1.10,
    "sinai_scatterer_radius": 0.38,
    "launch_pos": [0.18, 0.22],
    "is_playing": false
  }
}"#;

/// Starter preset: Dispersing Sinai Lorentz-gas billiard with central convex scatterer.
pub const PRESET_SINAI_DISPERSING: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Sinai Dispersing Hyperbolic Billiard",
    "domain": "chaos",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "launch_angle": 2.38,
    "fan_spread": 0.015,
    "fan_count": 6,
    "bounces_per_frame": 10,
    "geometry": "Sinai",
    "circle_radius": 1.0,
    "ellipse_a": 1.35,
    "ellipse_b": 1.0,
    "stadium_radius": 0.75,
    "stadium_half_straight": 0.70,
    "sinai_half_side": 1.10,
    "sinai_scatterer_radius": 0.40,
    "launch_pos": [0.78, 0.20],
    "is_playing": false
  }
}"#;

inventory::submit! {
    StarterPresetMetadata {
        id: "elliptical_billiard_caustics",
        title: "Confocal Elliptical Caustics",
        description: "Integrable elliptical billiard conserving focal product L₁L₂ with whispering-gallery and hyperbolic Birkhoff curves.",
        category: "Dynamical Billiards",
        icon: "🥚",
        domain: "chaos",
        tool_name: Some("DynamicalBilliardsTool"),
        preset_generator: || PRESET_ELLIPTICAL_CAUSTICS,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "bunimovich_stadium_ergodic",
        title: "Bunimovich Stadium Ergodicity",
        description: "Defocusing semicircle caps induce K-mixing chaos that fills the Birkhoff (s/L, sin φ) phase cylinder.",
        category: "Dynamical Billiards",
        icon: "🏟️",
        domain: "chaos",
        tool_name: Some("DynamicalBilliardsTool"),
        preset_generator: || PRESET_BUNIMOVICH_STADIUM,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "sinai_dispersing_billiard",
        title: "Sinai Dispersing Billiard",
        description: "Convex central scatterer exponentially diverges nearby ray fans via hyperbolic wavefront defocusing.",
        category: "Dynamical Billiards",
        icon: "🎱",
        domain: "chaos",
        tool_name: Some("DynamicalBilliardsTool"),
        preset_generator: || PRESET_SINAI_DISPERSING,
    }
}

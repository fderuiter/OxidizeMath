//! Starter Gallery cards, URL-safe [`PhysarumToolState`] validation, and [`TheoryDescribable`]
//! metadata for [`super::PhysarumTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use math_explorer::biology::physarum::{
    PhysarumParams, PhysarumPresetKind, MAX_PHYSARUM_AGENTS, MAX_PHYSARUM_GRID,
    MAX_PHYSARUM_TERMINALS, MIN_PHYSARUM_AGENTS, MIN_PHYSARUM_GRID, MIN_PHYSARUM_TERMINALS,
};
use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub(crate) const MAX_SAVED_OBSTACLE_STROKES: usize = 24;

pub(crate) const fn default_brush_radius() -> f64 {
    4.0
}

pub(crate) const fn default_true() -> bool {
    true
}

/// Compact `< 4 KiB` URL-safe and preset-safe state for [`super::PhysarumTool`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysarumToolState {
    /// Left/right sensor angle $\theta_s$ (`0.10..=1.50` rad). Kept first for numeric perturbation tests.
    pub sensor_angle: f64,
    /// Heading turn angle $\theta_r$ (`0.05..=1.50` rad).
    pub turn_angle: f64,
    /// Sensor lookahead distance $d_s$ (`2.0..=32.0` cells).
    pub sensor_distance: f64,
    /// Forward propulsion speed $v$ (`0.20..=3.50` cells/step).
    pub step_size: f64,
    /// Chemoattractant trail deposition per agent step (`0.05..=5.00`).
    pub deposition: f64,
    /// Isotropic $3\times 3$ diffusion blend factor $\alpha \in [0.0, 1.0]$.
    pub diffusion_rate: f64,
    /// Linear trail evaporation rate $\gamma \in [0.005, 0.50]$.
    pub decay_rate: f64,
    /// Nonlinear sub-threshold Hill pruning rate $\gamma_{\text{prune}} \in [0.0, 0.35]$.
    pub prune_rate: f64,
    /// Nutrient beacon emission strength (`0.0..=5.0`).
    pub beacon_strength: f64,
    /// Square Eulerian grid side length (`64..=256`).
    pub grid_size: usize,
    /// Active SoA chemotactic agent count (`256..=16384`).
    pub agent_count: usize,
    /// Active spatial preset topology.
    pub preset: PhysarumPreset,
    /// Normalized `[u, v]` terminal coordinates (`2..=16` entries in `0.02..=0.98`).
    pub terminals: Vec<[f64; 2]>,
    /// Optional user-painted obstacle strokes `[u, v, radius, blocked_flag]` (`0..=24`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub obstacle_strokes: Vec<[f64; 4]>,
    /// Seeded `OxidizeRng` stream seed.
    pub rng_seed: u64,
    /// Interactive brush disc radius (`1.0..=24.0`).
    #[serde(default = "default_brush_radius")]
    pub brush_radius: f64,
    /// Active pointer/keyboard brush tool.
    #[serde(default)]
    pub brush_mode: PhysarumBrushMode,
    /// Whether extracted `Graph` skeleton and MST reference overlays are drawn.
    #[serde(default = "default_true")]
    pub show_skeleton_overlay: bool,
    /// Whether continuous cooperative stepping is active.
    #[serde(default)]
    pub playing: bool,
}

impl PhysarumToolState {
    /// Validates all scalar, grid, agent, terminal, and obstacle stroke bounds against `URL_JSON_BOUNDS`.
    pub fn validate(&self) -> Result<(), String> {
        let scalar_bounds = [
            ("sensor_angle", self.sensor_angle, 0.10, 1.50),
            ("turn_angle", self.turn_angle, 0.05, 1.50),
            ("sensor_distance", self.sensor_distance, 2.0, 32.0),
            ("step_size", self.step_size, 0.20, 3.50),
            ("deposition", self.deposition, 0.05, 5.00),
            ("diffusion_rate", self.diffusion_rate, 0.0, 1.0),
            ("decay_rate", self.decay_rate, 0.005, 0.50),
            ("prune_rate", self.prune_rate, 0.0, 0.35),
            ("beacon_strength", self.beacon_strength, 0.0, 5.0),
            ("brush_radius", self.brush_radius, 1.0, 24.0),
        ];
        for (name, val, min, max) in scalar_bounds {
            if !val.is_finite() || val < min || val > max {
                return Err(format!(
                    "Physarum parameter '{name}' = {val} must be finite within [{min}, {max}]"
                ));
            }
        }
        if !(MIN_PHYSARUM_GRID..=MAX_PHYSARUM_GRID).contains(&self.grid_size) {
            return Err(format!(
                "Physarum grid_size {} must be within {}..={}",
                self.grid_size, MIN_PHYSARUM_GRID, MAX_PHYSARUM_GRID
            ));
        }
        if !(MIN_PHYSARUM_AGENTS..=MAX_PHYSARUM_AGENTS).contains(&self.agent_count) {
            return Err(format!(
                "Physarum agent_count {} must be within {}..={}",
                self.agent_count, MIN_PHYSARUM_AGENTS, MAX_PHYSARUM_AGENTS
            ));
        }
        self.validate_terminals_and_strokes()
    }

    fn validate_terminals_and_strokes(&self) -> Result<(), String> {
        if !(MIN_PHYSARUM_TERMINALS..=MAX_PHYSARUM_TERMINALS).contains(&self.terminals.len()) {
            return Err(format!(
                "Physarum terminals length {} must be within {}..={}",
                self.terminals.len(),
                MIN_PHYSARUM_TERMINALS,
                MAX_PHYSARUM_TERMINALS
            ));
        }
        for (idx, &[u, v]) in self.terminals.iter().enumerate() {
            if !u.is_finite()
                || !v.is_finite()
                || !(0.01..=0.99).contains(&u)
                || !(0.01..=0.99).contains(&v)
            {
                return Err(format!(
                    "Physarum terminal[{idx}] = [{u}, {v}] must lie within [0.01, 0.99]^2"
                ));
            }
        }
        if self.obstacle_strokes.len() > MAX_SAVED_OBSTACLE_STROKES {
            return Err(format!(
                "Physarum obstacle_strokes length {} exceeds {}",
                self.obstacle_strokes.len(),
                MAX_SAVED_OBSTACLE_STROKES
            ));
        }
        for (idx, &[u, v, r, b]) in self.obstacle_strokes.iter().enumerate() {
            let ok = u.is_finite()
                && v.is_finite()
                && r.is_finite()
                && b.is_finite()
                && (0.0..=1.0).contains(&u)
                && (0.0..=1.0).contains(&v)
                && (1.0..=32.0).contains(&r)
                && (b == 0.0 || b == 1.0);
            if !ok {
                return Err(format!("Invalid Physarum obstacle_strokes[{idx}]"));
            }
        }
        Ok(())
    }
}

/// Serializable spatial preset enum mirroring [`PhysarumPresetKind`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PhysarumPreset {
    /// Equilateral three-terminal Fermat-Torricelli Steiner Y-junction.
    #[default]
    SteinerTriangle,
    /// Two-branch asymmetric maze with short vs detour corridors.
    AsymmetricMaze,
    /// Greater Tokyo urban rail hubs around a Tokyo Bay coastline barrier.
    TokyoRail,
}

impl PhysarumPreset {
    /// Ordered list of all Physarum spatial presets.
    pub const ALL: [PhysarumPreset; 3] =
        [Self::SteinerTriangle, Self::AsymmetricMaze, Self::TokyoRail];

    /// Maps to the `domain_biology` [`PhysarumPresetKind`].
    pub const fn to_domain(self) -> PhysarumPresetKind {
        match self {
            Self::SteinerTriangle => PhysarumPresetKind::SteinerTriangle,
            Self::AsymmetricMaze => PhysarumPresetKind::AsymmetricMaze,
            Self::TokyoRail => PhysarumPresetKind::TokyoRail,
        }
    }

    /// Concise user-facing button label.
    pub fn label(self) -> &'static str {
        self.to_domain().display_name()
    }
}

/// Interactive canvas brush mode for modifying terminals, walls, or active slime arteries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PhysarumBrushMode {
    /// Places a new city terminal or removes an existing terminal on click.
    PlaceCity,
    /// Paints impassable obstacle barriers onto the grid.
    PaintWall,
    /// Erases obstacle barriers to open new transport corridors.
    EraseWall,
    /// Slices active chemoattractant arteries to observe self-healing and rerouting.
    #[default]
    SeverTube,
}

/// [`TheoryDescribable`] metadata for chemotactic slime-mold transport networks.
#[derive(Clone, Debug, Default)]
pub struct PhysarumTheory {
    params: PhysarumParams,
}

impl TheoryDescribable for PhysarumTheory {
    fn theory_description(&self) -> String {
        "Lagrangian chemotactic agents coupled to a diffused, nonlinearly pruned Eulerian chemoattractant grid self-organize into Steiner trees, shortest-path maze solutions, and fault-tolerant rail meshes."
            .to_string()
    }

    fn phonetic_description(&self) -> String {
        self.params.phonetic_description()
    }

    fn theory_citation(&self) -> String {
        self.params.theory_citation()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = self.params.available_descriptions();
        map.insert(
            "steiner_benchmark".to_string(),
            "Zhang-Suen medial-axis thinning and Ramer-Douglas-Peucker simplification extract a weighted Graph compared against Dijkstra shortest paths and static minimum spanning trees."
                .to_string(),
        );
        map
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(
            r"T_{t+1} = (1 - \gamma)\,\mathcal{D}_\alpha(T_t) - \gamma_{\text{prune}}\frac{\mathcal{D}_\alpha(T_t)}{1 + (\mathcal{D}_\alpha(T_t)/T_c)^2} + \sum_{i=1}^{N} w_{\text{dep}} E_i \delta(\mathbf{x} - \mathbf{x}_i)"
                .to_string(),
        )
    }
}

/// JSON preset envelope for the 3-terminal Fermat-Torricelli Steiner Y-junction.
pub const PRESET_PHYSARUM_STEINER: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Physarum Network Lab - Fermat-Torricelli Steiner",
    "domain": "biology",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "sensor_angle": 0.76,
    "turn_angle": 0.62,
    "sensor_distance": 11.5,
    "step_size": 1.25,
    "deposition": 0.85,
    "diffusion_rate": 0.65,
    "decay_rate": 0.055,
    "prune_rate": 0.045,
    "beacon_strength": 1.4,
    "grid_size": 128,
    "agent_count": 6144,
    "preset": "SteinerTriangle",
    "terminals": [[0.5, 0.16], [0.794449, 0.67], [0.205551, 0.67]],
    "rng_seed": 42,
    "brush_radius": 4.0,
    "brush_mode": "SeverTube",
    "show_skeleton_overlay": true,
    "playing": true
  }
}"#;

/// JSON preset envelope for the two-branch Asymmetric Maze shortest-path solver.
pub const PRESET_PHYSARUM_MAZE: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Physarum Network Lab - Asymmetric Maze Solver",
    "domain": "biology",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "sensor_angle": 0.62,
    "turn_angle": 0.66,
    "sensor_distance": 8.0,
    "step_size": 1.4,
    "deposition": 0.85,
    "diffusion_rate": 0.55,
    "decay_rate": 0.075,
    "prune_rate": 0.095,
    "beacon_strength": 1.4,
    "grid_size": 128,
    "agent_count": 4096,
    "preset": "AsymmetricMaze",
    "terminals": [[0.22, 0.36], [0.78, 0.36]],
    "rng_seed": 42,
    "brush_radius": 4.0,
    "brush_mode": "PaintWall",
    "show_skeleton_overlay": true,
    "playing": true
  }
}"#;

/// JSON preset envelope for the 11-station Greater Tokyo rail mesh around Tokyo Bay.
pub const PRESET_PHYSARUM_TOKYO_RAIL: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Physarum Network Lab - Tokyo-Rail Network",
    "domain": "biology",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "sensor_angle": 0.68,
    "turn_angle": 0.58,
    "sensor_distance": 9.0,
    "step_size": 1.25,
    "deposition": 0.75,
    "diffusion_rate": 0.5,
    "decay_rate": 0.05,
    "prune_rate": 0.03,
    "beacon_strength": 1.35,
    "grid_size": 128,
    "agent_count": 6144,
    "preset": "TokyoRail",
    "terminals": [
      [0.54, 0.5], [0.4, 0.49], [0.39, 0.6], [0.42, 0.36],
      [0.57, 0.39], [0.35, 0.8], [0.44, 0.7], [0.84, 0.54],
      [0.42, 0.16], [0.16, 0.52], [0.72, 0.45]
    ],
    "rng_seed": 77,
    "brush_radius": 4.5,
    "brush_mode": "SeverTube",
    "show_skeleton_overlay": true,
    "playing": true
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "physarum_steiner",
        title: "Physarum Steiner Y-Junction",
        description: "Emergent 120° Fermat-Torricelli junction cutting total wire cost ~13% below the static 2-edge spanning tree.",
        category: "Adaptive Networks",
        icon: "🕸",
        domain: "biology",
        tool_name: Some("PhysarumTool"),
        preset_generator: || PRESET_PHYSARUM_STEINER,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "physarum_maze",
        title: "Physarum Shortest-Path Maze",
        description: "Positive-feedback flux pruning collapses detour tubes and routes >90% of plasmodial trail mass through the shorter corridor.",
        category: "Adaptive Networks",
        icon: "🧭",
        domain: "biology",
        tool_name: Some("PhysarumTool"),
        preset_generator: || PRESET_PHYSARUM_MAZE,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "physarum_tokyo_rail",
        title: "Physarum Tokyo-Rail Network",
        description: "Tero et al. (2010) 11-hub Tokyo Bay transport mesh balancing low wire cost, near-geodesic Dijkstra stretch, and fault survival.",
        category: "Adaptive Networks",
        icon: "🚄",
        domain: "biology",
        tool_name: Some("PhysarumTool"),
        preset_generator: || PRESET_PHYSARUM_TOKYO_RAIL,
    }
}

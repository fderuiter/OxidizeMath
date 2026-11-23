//! Euclidean Dynamical Billiards & Birkhoff Phase-Portrait Lab (`DynamicalBilliardsTool`).
//!
//! Simulates exact zero-allocation specular ray reflections in circular, confocal elliptical,
//! Bunimovich stadium, and Sinai dispersing Euclidean tables alongside a synchronized 2D
//! Birkhoff `(s / L, sin φ)` Poincaré boundary map.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:chaos_theory]

use crate::framework::{ExportCapabilities, InteractiveTool, ToolMetadata};
use eframe::egui;
use math_explorer::physics::chaos::billiards::{
    BilliardRay, CircularBilliard, EllipticalBilliard, SinaiBilliard, StadiumBilliard,
};
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

#[path = "billiards/canvas.rs"]
mod canvas;
#[path = "billiards/export.rs"]
mod export;
#[path = "billiards/preset.rs"]
pub mod preset;
#[path = "billiards/simulation.rs"]
mod simulation;
#[path = "billiards/ui.rs"]
mod ui;

#[cfg(test)]
#[path = "billiards/tests.rs"]
mod tests;

/// Maximum number of simultaneously traced billiard rays in a fan or ensemble.
pub const MAX_RAYS: usize = 16;

/// Maximum number of spatial `[x, y]` bounce vertices retained per ray trail.
pub const MAX_CHORD_VERTICES: usize = 320;

/// Maximum number of Birkhoff `[s / L, sin φ]` boundary collisions retained per ray.
pub const MAX_BIRKHOFF_POINTS: usize = 4_096;

/// Per-frame wall-clock ray-tracing budget in milliseconds.
pub const BILLIARD_FRAME_BUDGET_MS: f64 = 8.0;

/// Euclidean billiard table geometry selectable in [`DynamicalBilliardsTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BilliardGeometryKind {
    /// Integrable circular table conserving angular momentum $\sin \varphi$.
    Circle,
    /// Integrable confocal elliptical table conserving focal product $L_1 L_2$.
    Ellipse,
    /// Ergodic defocusing Bunimovich stadium table.
    Stadium,
    /// Hyperbolic dispersing Sinai table (square with central disk scatterer).
    Sinai,
}

impl BilliardGeometryKind {
    /// User-facing label for geometry selector menus.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Circle => "Circle (Integrable Angular Momentum)",
            Self::Ellipse => "Ellipse (Confocal Caustics & Separatrix)",
            Self::Stadium => "Bunimovich Stadium (Defocusing Ergodic)",
            Self::Sinai => "Sinai Billiard (Dispersing Hyperbolic)",
        }
    }
}

/// Per-ray trajectory slot holding active Euclidean ray state, spatial chords, and Birkhoff hits.
#[derive(Debug, Clone)]
pub struct RaySlot {
    /// Active Euclidean ray `(origin, direction)`.
    pub ray: BilliardRay,
    /// Palette series index for consistent coloring across spatial and Birkhoff panes.
    pub color_index: usize,
    /// Recent spatial `[x, y]` reflection vertices (`len() <= MAX_CHORD_VERTICES`).
    pub vertices: Vec<[f64; 2]>,
    /// Captured Birkhoff `[s / L, sin φ]` boundary map points (`len() <= MAX_BIRKHOFF_POINTS`).
    pub birkhoff_points: Vec<[f64; 2]>,
    pub(crate) birkhoff_write_idx: usize,
    /// Total specular collisions accumulated by this ray.
    pub bounce_count: usize,
}

/// Interactive Euclidean Dynamical Billiards & Birkhoff Phase-Portrait Tool.
pub struct DynamicalBilliardsTool {
    /// Active Euclidean table geometry.
    pub geometry: BilliardGeometryKind,
    /// Launch angle $\theta \in [-\pi, \pi]$ (radians) of the central ray.
    pub launch_angle: f64,
    /// Angular spread $\Delta \theta \in [0, 1.2]$ (radians) across the ray fan.
    pub fan_spread: f64,
    /// Number of rays launched in a fan (`1..=MAX_RAYS`).
    pub fan_count: usize,
    /// Specular collisions advanced per animation frame (`1..=64`).
    pub bounces_per_frame: usize,
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
    /// Launch / keyboard reticle position `[x, y]` inside the billiard table.
    pub launch_pos: [f64; 2],
    /// Whether continuous bounce playback is active.
    pub is_playing: bool,
    /// Active ensemble of traced billiard rays (`len() <= MAX_RAYS`).
    pub rays: Vec<RaySlot>,
    /// Last measured frame bounce computation time in milliseconds.
    pub last_frame_ms: f64,
    pub(crate) state_rev: u64,
    pub(crate) focused_canvas: Option<egui::Id>,
    pub(crate) last_status_note: Option<String>,
    pub(crate) circle: CircularBilliard,
    pub(crate) ellipse: EllipticalBilliard,
    pub(crate) stadium: StadiumBilliard,
    pub(crate) sinai: SinaiBilliard,
}

impl Default for DynamicalBilliardsTool {
    fn default() -> Self {
        let mut tool = Self {
            geometry: BilliardGeometryKind::Ellipse,
            launch_angle: 0.58,
            fan_spread: 0.24,
            fan_count: 5,
            bounces_per_frame: 8,
            circle_radius: 1.0,
            ellipse_a: 1.35,
            ellipse_b: 1.0,
            stadium_radius: 0.75,
            stadium_half_straight: 0.70,
            sinai_half_side: 1.10,
            sinai_scatterer_radius: 0.38,
            launch_pos: [0.35, 0.15],
            is_playing: false,
            rays: Vec::with_capacity(MAX_RAYS),
            last_frame_ms: 0.0,
            state_rev: 1,
            focused_canvas: None,
            last_status_note: None,
            circle: CircularBilliard::new(1.0),
            ellipse: EllipticalBilliard::new(1.35, 1.0),
            stadium: StadiumBilliard::new(0.70, 0.75),
            sinai: SinaiBilliard::new(1.10, 0.38),
        };
        tool.rebuild_tables_and_fire_fan();
        tool
    }
}

impl TheoryDescribable for DynamicalBilliardsTool {
    fn theory_description(&self) -> String {
        format!(
            "Euclidean dynamical billiards on {} with area-preserving Birkhoff boundary map $(s/L, \\sin\\varphi)$ ($ds \\wedge d(\\sin\\varphi)$ invariant). Integrable circles and ellipses foliate phase space into caustics, whereas Sinai dispersing and Bunimovich defocusing tables are K-mixing and ergodic.",
            self.geometry.label()
        )
    }

    fn phonetic_description(&self) -> String {
        format!(
            "Euclidean dynamical billiards lab on {} with {} active rays.",
            self.geometry.label(),
            self.rays.len()
        )
    }

    fn theory_citation(&self) -> String {
        "Birkhoff, G. D. (1927). Dynamical Systems. AMS; Sinai, Ya. G. (1970). Dynamical systems with elastic reflections. Russian Math. Surveys, 25(2), 137-189; Bunimovich, L. A. (1979). Commun. Math. Phys., 65(3), 295-312; Berry, M. V. (1981). Eur. J. Phys., 2(2), 91-102.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "birkhoff".to_string(),
            "Canonical Birkhoff coordinates (s/L, sin φ) preserve symplecticity ds ∧ d(sin φ) across specular Euclidean reflections.".to_string(),
        );
        map.insert(
            "ergodicity".to_string(),
            "Transition from integrable confocal elliptical caustics (L₁L₂ = const) to Sinai dispersing and Bunimovich defocusing chaos.".to_string(),
        );
        map
    }
}

impl InteractiveTool for DynamicalBilliardsTool {
    fn name(&self) -> &'static str {
        "Dynamical Billiards Lab"
    }

    fn domain(&self) -> &'static str {
        "chaos"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn state_revision(&self) -> u64 {
        self.state_rev
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
    }

    fn has_finite_parameters(&self) -> bool {
        self.launch_angle.is_finite()
            && self.fan_spread.is_finite()
            && self.circle_radius.is_finite()
            && self.ellipse_a.is_finite()
            && self.ellipse_b.is_finite()
            && self.stadium_radius.is_finite()
            && self.stadium_half_straight.is_finite()
            && self.sinai_half_side.is_finite()
            && self.sinai_scatterer_radius.is_finite()
            && self.launch_pos[0].is_finite()
            && self.launch_pos[1].is_finite()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.clamp_billiard_parameters();
        true
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.collect_export_datasets()
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        self.collect_export_primitives()
    }

    fn step_simulation(&mut self) {
        self.advance_bounces(self.bounces_per_frame.clamp(1, 64));
    }

    fn save_state(&self) -> Option<String> {
        self.serialize_billiards_state()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.deserialize_billiards_state(state)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        if self.is_playing {
            self.step_frame_with_budget();
            ui.ctx().request_repaint();
        }
        self.render_side_panel(ui);
        self.render_central_workspace(ui);
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

inventory::submit! {
    ToolMetadata {
        id: "DynamicalBilliardsTool",
        display_name: "Dynamical Billiards Lab",
        domain: "chaos",
        tags: &["billiards", "birkhoff", "stadium", "sinai", "ellipse", "caustics", "ergodic"],
        build: || Box::new(DynamicalBilliardsTool::default()),
    }
}

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! 3D and 2D Hamiltonian Loss Landscape Arena (`LossLandscapeTool`).

pub mod arena;
pub mod surfaces;
mod ui;

#[cfg(test)]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

use crate::camera::Camera3D;
use crate::framework::{enforce_ordered_drag_range, InteractionContext, InteractiveTool};
use arena::{ArenaConfig, ParticleArena};
use eframe::egui;
use surfaces::{
    map_2d_rect_pos, push_sculpt_bump, unproject_3d_rect_pos, LandscapeSurface, SculptBump,
    SurfaceGrid, MAX_SCULPT_BUMPS, SCULPT_AMP_MAX, SCULPT_AMP_MIN, SCULPT_SIGMA_MAX,
    SCULPT_SIGMA_MIN,
};

pub(crate) const LOSS_PARAM_MIN_BOUND: f64 = -100.0;
pub(crate) const LOSS_PARAM_MAX_BOUND: f64 = 100.0;
pub(super) const LOSS_MIN_SPAN: f64 = 0.1;

fn default_arena_camera() -> Camera3D {
    Camera3D::new(0.55, 0.65, 1.0)
}
const fn default_sculpt_amplitude() -> f64 {
    2.5
}
const fn default_sculpt_sigma() -> f64 {
    0.45
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum LandscapeViewMode {
    #[default]
    View3D,
    Contour2D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CanvasInteractionMode {
    #[default]
    Orbit,
    Slingshot,
    SculptBrush,
}

pub struct LossLandscapeTool {
    pub(crate) w_min: f64,
    pub(crate) w_max: f64,
    pub(crate) b_min: f64,
    pub(crate) b_max: f64,
    pub(crate) resolution: usize,
    pub surface: LandscapeSurface,
    pub view_mode: LandscapeViewMode,
    pub interaction_mode: CanvasInteractionMode,
    pub camera: Camera3D,
    pub arena_config: ArenaConfig,
    pub sculpt_amplitude: f64,
    pub sculpt_sigma: f64,
    pub sculpt_bumps: Vec<SculptBump>,
    pub is_playing: bool,
    pub arena: ParticleArena,
    pub(super) slingshot_anchor: Option<[f64; 2]>,
    pub(super) slingshot_current: Option<[f64; 2]>,
    pub(super) last_brush_wb: Option<[f64; 2]>,
}

impl Default for LossLandscapeTool {
    fn default() -> Self {
        let mut tool = Self {
            w_min: 0.0,
            w_max: 4.0,
            b_min: -1.0,
            b_max: 3.0,
            resolution: 20,
            surface: LandscapeSurface::LinearMse,
            view_mode: LandscapeViewMode::View3D,
            interaction_mode: CanvasInteractionMode::Orbit,
            camera: default_arena_camera(),
            arena_config: ArenaConfig::default(),
            sculpt_amplitude: default_sculpt_amplitude(),
            sculpt_sigma: default_sculpt_sigma(),
            sculpt_bumps: Vec::new(),
            is_playing: false,
            arena: ParticleArena::default(),
            slingshot_anchor: None,
            slingshot_current: None,
            last_brush_wb: None,
        };
        tool.reset_arena();
        tool
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LossLandscapeState {
    w_min: f64,
    w_max: f64,
    b_min: f64,
    b_max: f64,
    resolution: usize,
    #[serde(default)]
    surface: LandscapeSurface,
    #[serde(default)]
    view_mode: LandscapeViewMode,
    #[serde(default = "default_arena_camera")]
    camera: Camera3D,
    #[serde(default)]
    arena: ArenaConfig,
    #[serde(default = "default_sculpt_amplitude")]
    sculpt_amplitude: f64,
    #[serde(default = "default_sculpt_sigma")]
    sculpt_sigma: f64,
    #[serde(default)]
    sculpt_bumps: Vec<SculptBump>,
}

impl LossLandscapeTool {
    #[inline]
    #[must_use]
    pub const fn domain_bounds(&self) -> (f64, f64, f64, f64) {
        (self.w_min, self.w_max, self.b_min, self.b_max)
    }

    pub fn reset_arena(&mut self) {
        self.arena.reset(
            &self.arena_config,
            self.surface,
            &self.sculpt_bumps,
            self.domain_bounds(),
        );
    }

    pub fn apply_surface_preset(&mut self, surface: LandscapeSurface) {
        self.surface = surface;
        let (w0, w1, b0, b1) = surface.default_bounds();
        (self.w_min, self.w_max, self.b_min, self.b_max) = (w0, w1, b0, b1);
        self.sculpt_bumps.clear();
        let (spawn, vel, lr, damp, temp) = match surface {
            LandscapeSurface::LinearMse => ([0.4, 2.4], [0.8, -0.6], 0.03, 0.45, 0.6),
            LandscapeSurface::Rosenbrock => ([-1.4, 1.8], [1.5, -0.8], 0.002, 0.35, 0.5),
            LandscapeSurface::Rastrigin => ([-2.9, 2.9], [0.6, -0.6], 0.005, 1.2, 8.0),
            LandscapeSurface::MonkeySaddle => ([0.02, 0.01], [0.35, 0.2], 0.02, 0.5, 1.5),
        };
        self.arena_config.spawn = spawn;
        self.arena_config.slingshot_velocity = vel;
        self.arena_config.learning_rate = lr;
        self.arena_config.damping = damp;
        self.arena_config.temperature = temp;
        self.reset_arena();
    }

    pub(super) fn pointer_to_wb(&self, rect: egui::Rect, pos: egui::Pos2) -> [f64; 2] {
        match self.view_mode {
            LandscapeViewMode::View3D => {
                unproject_3d_rect_pos(&self.camera, rect, pos, self.domain_bounds())
            }
            LandscapeViewMode::Contour2D => map_2d_rect_pos(rect, pos, self.domain_bounds()),
        }
    }
}

impl InteractiveTool for LossLandscapeTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Loss Landscape"
    }

    fn domain(&self) -> &'static str {
        "ai"
    }

    fn has_finite_parameters(&self) -> bool {
        self.w_min.is_finite()
            && self.w_max.is_finite()
            && (LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&self.w_min)
            && (LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&self.w_max)
            && self.w_min < self.w_max
            && self.b_min.is_finite()
            && self.b_max.is_finite()
            && (LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&self.b_min)
            && (LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&self.b_max)
            && self.b_min < self.b_max
            && (10..=50).contains(&self.resolution)
            && self.camera.is_valid()
            && self.arena_config.is_valid()
            && self.sculpt_amplitude.is_finite()
            && (SCULPT_AMP_MIN..=SCULPT_AMP_MAX).contains(&self.sculpt_amplitude)
            && self.sculpt_sigma.is_finite()
            && (SCULPT_SIGMA_MIN..=SCULPT_SIGMA_MAX).contains(&self.sculpt_sigma)
            && self.sculpt_bumps.len() <= MAX_SCULPT_BUMPS
            && self.sculpt_bumps.iter().all(SculptBump::is_valid)
    }

    fn clamp_parameters(&mut self) -> bool {
        enforce_ordered_drag_range(
            &mut self.w_min,
            &mut self.w_max,
            LOSS_PARAM_MIN_BOUND,
            LOSS_PARAM_MAX_BOUND,
            LOSS_MIN_SPAN,
            false,
            false,
        );
        enforce_ordered_drag_range(
            &mut self.b_min,
            &mut self.b_max,
            LOSS_PARAM_MIN_BOUND,
            LOSS_PARAM_MAX_BOUND,
            LOSS_MIN_SPAN,
            false,
            false,
        );
        self.resolution = self.resolution.clamp(10, 50);
        self.arena_config.clamp_in_place();
        self.sculpt_amplitude = self.sculpt_amplitude.clamp(SCULPT_AMP_MIN, SCULPT_AMP_MAX);
        self.sculpt_sigma = self.sculpt_sigma.clamp(SCULPT_SIGMA_MIN, SCULPT_SIGMA_MAX);
        self.sculpt_bumps.truncate(MAX_SCULPT_BUMPS);
        true
    }

    fn on_brush(&mut self, ctx: &InteractionContext) {
        let Some(pos) = ctx.pointer_pos else {
            return;
        };
        let wb = self.pointer_to_wb(ctx.response.rect, pos);
        if let Some(prev) = self.last_brush_wb {
            if (wb[0] - prev[0]).hypot(wb[1] - prev[1]) < 0.02 {
                return;
            }
        }
        let amp = if ctx.modifiers.alt {
            -self.sculpt_amplitude
        } else {
            self.sculpt_amplitude
        };
        push_sculpt_bump(
            &mut self.sculpt_bumps,
            SculptBump::new(wb[0], wb[1], amp, self.sculpt_sigma),
        );
        self.last_brush_wb = Some(wb);
    }

    fn step_simulation(&mut self) {
        if self.has_finite_parameters() {
            let bounds = self.domain_bounds();
            self.arena
                .step_once(&self.arena_config, self.surface, &self.sculpt_bumps, bounds);
        }
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&LossLandscapeState {
            w_min: self.w_min,
            w_max: self.w_max,
            b_min: self.b_min,
            b_max: self.b_max,
            resolution: self.resolution,
            surface: self.surface,
            view_mode: self.view_mode,
            camera: self.camera,
            arena: self.arena_config.clone(),
            sculpt_amplitude: self.sculpt_amplitude,
            sculpt_sigma: self.sculpt_sigma,
            sculpt_bumps: self.sculpt_bumps.clone(),
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let p: LossLandscapeState =
            serde_json::from_str(state).map_err(|e| format!("Invalid loss landscape state: {e}"))?;
        let bounds_ok = p.w_min.is_finite()
            && p.w_max.is_finite()
            && (LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&p.w_min)
            && (LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&p.w_max)
            && p.w_min < p.w_max
            && p.b_min.is_finite()
            && p.b_max.is_finite()
            && (LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&p.b_min)
            && (LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&p.b_max)
            && p.b_min < p.b_max;
        let arena_ok = (10..=50).contains(&p.resolution)
            && p.camera.is_valid()
            && p.arena.is_valid()
            && p.sculpt_amplitude.is_finite()
            && (SCULPT_AMP_MIN..=SCULPT_AMP_MAX).contains(&p.sculpt_amplitude)
            && p.sculpt_sigma.is_finite()
            && (SCULPT_SIGMA_MIN..=SCULPT_SIGMA_MAX).contains(&p.sculpt_sigma)
            && p.sculpt_bumps.len() <= MAX_SCULPT_BUMPS
            && p.sculpt_bumps.iter().all(SculptBump::is_valid);
        if !bounds_ok || !arena_ok {
            return Err("Loss landscape parameter bounds or arena configuration invalid".into());
        }
        (self.w_min, self.w_max, self.b_min, self.b_max) = (p.w_min, p.w_max, p.b_min, p.b_max);
        (self.resolution, self.surface, self.view_mode) = (p.resolution, p.surface, p.view_mode);
        self.camera.apply_pose(p.camera);
        self.arena_config = p.arena;
        (self.sculpt_amplitude, self.sculpt_sigma) = (p.sculpt_amplitude, p.sculpt_sigma);
        self.sculpt_bumps = p
            .sculpt_bumps
            .into_iter()
            .map(|b| SculptBump::new(b.w, b.b, b.amplitude, b.sigma))
            .collect();
        self.is_playing = false;
        (self.slingshot_anchor, self.slingshot_current, self.last_brush_wb) = (None, None, None);
        self.reset_arena();
        Ok(())
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        if self.is_playing && self.has_finite_parameters() {
            self.step_simulation();
            parent_ui.ctx().request_repaint();
        }
        self.draw_controls(parent_ui);
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            if !self.has_finite_parameters() {
                return;
            }
            let grid = SurfaceGrid::sample(
                self.surface,
                &self.sculpt_bumps,
                self.domain_bounds(),
                self.resolution,
            );
            match self.view_mode {
                LandscapeViewMode::View3D => self.show_3d_arena(ui, &grid),
                LandscapeViewMode::Contour2D => self.show_2d_contours(ui, &grid),
            }
        });
    }
}

// [cite:modular_polynomials_review]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "LossLandscapeTool",
        display_name: "Loss Landscape",
        domain: "ai",
        tags: &["hamiltonian", "langevin", "optimization", "3d"],
        build: || Box::new(LossLandscapeTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for LossLandscapeTool {
    fn theory_description(&self) -> String {
        "3D and 2D non-convex optimization arena comparing first-order optimizers ($\\text{SGD}$, $\\text{Adam}$) against symplectic/conformal Hamiltonian dynamics $H(q, p) = \\frac{\\|p\\|^2}{2m} + L(q)$ and underdamped BAOAB Langevin diffusion $dp = -\\nabla L(q)\\,dt - \\gamma p\\,dt + \\sqrt{2\\gamma m k_B T}\\,dW_t$ across sculpted Gaussian potentials.".into()
    }

    fn phonetic_description(&self) -> String {
        "Hamiltonian and Langevin momentum optimization dynamics on 3D non-convex loss landscapes"
            .into()
    }

    fn theory_citation(&self) -> String {
        "Visualizing the Loss Landscape of Neural Nets (Li et al., 2018); Variational Perspectives on Accelerated Optimization (Wibisono, Wilson & Jordan, 2016); Molecular Dynamics (Leimkuhler & Matthews, 2015)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

const STARTER_PRESET_JSON: &str = r#"{"metadata":{"schema_version":"1.0","title":"Loss Landscape","domain":"ai","timestamp":"2026-11-16T00:00:00Z"},"config":{"w_min":-1.0,"w_max":5.0,"b_min":-2.0,"b_max":4.0,"resolution":30}}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "loss_landscape_starter",
        title: "Loss Landscape",
        description: "3D and 2D Hamiltonian & Langevin loss landscape arena over Linear MSE, Rosenbrock, Rastrigin, and sculpted Gaussian surfaces.",
        category: "Optimization",
        icon: "⛰️",
        domain: "ai",
        tool_name: Some("LossLandscapeTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

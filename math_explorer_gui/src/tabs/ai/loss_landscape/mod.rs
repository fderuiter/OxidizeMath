#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! 3D and 2D Hamiltonian Loss Landscape Arena (`LossLandscapeTool`).

pub mod arena;
pub mod evaluator;
pub mod presets;
pub mod surfaces;
mod ui;

#[cfg(test)]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

pub use presets::PredictorMode;

use crate::camera::Camera3D;
use crate::framework::{
    enforce_ordered_drag_range, InteractionContext, InteractiveTool, MathExpressionInputBar,
};
use arena::{ArenaConfig, ParticleArena};
use eframe::egui;
use presets::DEFAULT_LOSS_EXPR;
use std::collections::BTreeMap;
use surfaces::{
    map_2d_rect_pos, push_sculpt_bump, unproject_3d_rect_pos, LandscapeSurface, SculptBump,
    SurfaceEvaluator, SurfaceGrid, MAX_SCULPT_BUMPS, SCULPT_AMP_MAX, SCULPT_AMP_MIN,
    SCULPT_SIGMA_MAX, SCULPT_SIGMA_MIN,
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
    pub(crate) predictor_mode: PredictorMode,
    pub(crate) w_min: f64,
    pub(crate) w_max: f64,
    pub(crate) b_min: f64,
    pub(crate) b_max: f64,
    pub(crate) resolution: usize,
    pub(crate) math_input: MathExpressionInputBar,
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
            predictor_mode: PredictorMode::Linear,
            w_min: 0.0,
            w_max: 4.0,
            b_min: -1.0,
            b_max: 3.0,
            resolution: 20,
            math_input: MathExpressionInputBar::new(
                "f(x,w,b)",
                DEFAULT_LOSS_EXPR,
                vec![
                    ("Sinusoidal Non-Convex", "w * sin(b * x) + a"),
                    ("Shifted ReLU Neuron", "w * max(0, x + b)"),
                    ("Saturated Gate", "w * clamp(x + b, -1, 1)"),
                    ("Linear Baseline", "w * x + b"),
                ],
            )
            .with_axes(&["x", "w", "b"]),
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
    #[serde(default, skip_serializing_if = "PredictorMode::is_linear")]
    predictor_mode: PredictorMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    custom_expr: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    custom_params: BTreeMap<String, f64>,
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

    #[inline]
    #[must_use]
    pub(super) fn eval_from<'a>(
        surface: LandscapeSurface,
        mode: PredictorMode,
        bar: &'a MathExpressionInputBar,
        bumps: &'a [SculptBump],
    ) -> SurfaceEvaluator<'a> {
        let custom = (surface == LandscapeSurface::LinearMse && mode == PredictorMode::Custom)
            .then(|| bar.active_compiled().map(|c| (c, bar.param_slice())))
            .flatten();
        SurfaceEvaluator::new(surface, bumps, custom)
    }

    #[must_use]
    pub fn surface_evaluator(&self) -> SurfaceEvaluator<'_> {
        Self::eval_from(
            self.surface,
            self.predictor_mode,
            &self.math_input,
            &self.sculpt_bumps,
        )
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn evaluate_mse_at(&self, w: f64, b: f64) -> Option<f64> {
        match self.predictor_mode {
            PredictorMode::Linear => SurfaceEvaluator::evaluate_mse_only(None, w, b),
            PredictorMode::Custom => {
                let compiled = self.math_input.active_compiled()?;
                SurfaceEvaluator::evaluate_mse_only(
                    Some((compiled, self.math_input.param_slice())),
                    w,
                    b,
                )
            }
        }
    }

    pub fn reset_arena(&mut self) {
        let bounds = self.domain_bounds();
        let eval = Self::eval_from(
            self.surface,
            self.predictor_mode,
            &self.math_input,
            &self.sculpt_bumps,
        );
        self.arena.reset(&self.arena_config, eval, bounds);
    }

    pub fn apply_surface_preset(&mut self, surface: LandscapeSurface) {
        self.surface = surface;
        if surface != LandscapeSurface::LinearMse {
            self.predictor_mode = PredictorMode::Linear;
        }
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
            let eval = Self::eval_from(
                self.surface,
                self.predictor_mode,
                &self.math_input,
                &self.sculpt_bumps,
            );
            self.arena.step_once(&self.arena_config, eval, bounds);
        }
    }

    fn save_state(&self) -> Option<String> {
        let is_custom = self.predictor_mode == PredictorMode::Custom;
        let custom_expr = (is_custom || self.math_input.raw_text != DEFAULT_LOSS_EXPR)
            .then(|| self.math_input.raw_text.clone());
        let custom_params = if is_custom {
            self.math_input.params_map()
        } else {
            BTreeMap::new()
        };
        serde_json::to_string(&LossLandscapeState {
            w_min: self.w_min,
            w_max: self.w_max,
            b_min: self.b_min,
            b_max: self.b_max,
            resolution: self.resolution,
            predictor_mode: self.predictor_mode,
            custom_expr,
            custom_params,
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
        let p: LossLandscapeState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid loss landscape state: {e}"))?;
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
        let custom_ok = p.custom_expr.as_ref().is_none_or(|s| s.len() <= 512)
            && p.custom_params.len() <= crate::framework::math_input::MAX_SYNTHESIZED_PARAMS
            && p.custom_params
                .values()
                .all(|v| v.is_finite() && (-1e4..=1e4).contains(v));
        let arena_ok = (10..=50).contains(&p.resolution)
            && p.camera.is_valid()
            && p.arena.is_valid()
            && p.sculpt_amplitude.is_finite()
            && (SCULPT_AMP_MIN..=SCULPT_AMP_MAX).contains(&p.sculpt_amplitude)
            && p.sculpt_sigma.is_finite()
            && (SCULPT_SIGMA_MIN..=SCULPT_SIGMA_MAX).contains(&p.sculpt_sigma)
            && p.sculpt_bumps.len() <= MAX_SCULPT_BUMPS
            && p.sculpt_bumps.iter().all(SculptBump::is_valid);
        if !bounds_ok || !custom_ok || !arena_ok {
            return Err("Loss landscape parameter bounds or arena configuration invalid".into());
        }
        (self.w_min, self.w_max, self.b_min, self.b_max) = (p.w_min, p.w_max, p.b_min, p.b_max);
        (self.resolution, self.surface, self.view_mode) = (p.resolution, p.surface, p.view_mode);
        self.predictor_mode = p.predictor_mode;
        self.math_input
            .set_expression(p.custom_expr.as_deref().unwrap_or(DEFAULT_LOSS_EXPR));
        self.math_input.set_params_map(&p.custom_params);
        self.camera.apply_pose(p.camera);
        self.arena_config = p.arena;
        (self.sculpt_amplitude, self.sculpt_sigma) = (p.sculpt_amplitude, p.sculpt_sigma);
        self.sculpt_bumps = p
            .sculpt_bumps
            .into_iter()
            .map(|b| SculptBump::new(b.w, b.b, b.amplitude, b.sigma))
            .collect();
        self.is_playing = false;
        (
            self.slingshot_anchor,
            self.slingshot_current,
            self.last_brush_wb,
        ) = (None, None, None);
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
            let grid = SurfaceGrid::sample_eval(
                self.surface_evaluator(),
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

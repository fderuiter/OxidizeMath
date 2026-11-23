#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! 3D and 2D Hamiltonian Loss Landscape Arena (`LossLandscapeTool`).

pub mod arena;
mod canvas;
pub mod data;
pub mod evaluator;
mod persistence;
pub mod presets;
pub mod surfaces;
mod ui;

#[cfg(test)]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

pub use data::{LossMetric, TargetDataMode};
pub use presets::PredictorMode;

use crate::camera::Camera3D;
use crate::framework::{
    enforce_ordered_drag_range, InteractionContext, InteractiveTool, MathExpressionInputBar,
};
use arena::{ArenaConfig, ParticleArena};
use data::{
    default_preset_targets, default_target_formula_bar, parse_target_table, sample_formula_targets,
    DEFAULT_HUBER_DELTA, DEFAULT_TABLE_TEXT, DEFAULT_TARGET_SAMPLES, DEFAULT_TARGET_X_BOUNDS,
    MAX_TABLE_ROWS,
};
use eframe::egui;
use evaluator::EmpiricalConfig;
use presets::DEFAULT_LOSS_EXPR;
use surfaces::{
    map_2d_rect_pos, push_sculpt_bump, unproject_3d_rect_pos, LandscapeSurface, SculptBump,
    SurfaceEvaluator, SurfaceGrid, MAX_SCULPT_BUMPS, SCULPT_AMP_MAX, SCULPT_AMP_MIN,
    SCULPT_SIGMA_MAX, SCULPT_SIGMA_MIN,
};

pub(crate) const LOSS_PARAM_MIN_BOUND: f64 = -100.0;
pub(crate) const LOSS_PARAM_MAX_BOUND: f64 = 100.0;
pub(super) const LOSS_MIN_SPAN: f64 = 0.1;

pub(super) fn default_arena_camera() -> Camera3D {
    Camera3D::new(0.55, 0.65, 1.0)
}
pub(super) const fn default_sculpt_amplitude() -> f64 {
    2.5
}
pub(super) const fn default_sculpt_sigma() -> f64 {
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
    pub(super) cached_targets: Vec<[f64; 2]>,
    pub(crate) predictor_mode: PredictorMode,
    pub(crate) target_mode: TargetDataMode,
    pub(crate) loss_metric: LossMetric,
    pub(crate) huber_delta: f64,
    pub(crate) target_formula_bar: MathExpressionInputBar,
    pub(crate) target_x_bounds: [f64; 2],
    pub(crate) target_sample_count: usize,
    pub(crate) target_table_text: String,
    pub(crate) table_parse_error: Option<String>,
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
            cached_targets: default_preset_targets(),
            predictor_mode: PredictorMode::Linear,
            target_mode: TargetDataMode::PresetLinear,
            loss_metric: LossMetric::Mse,
            huber_delta: DEFAULT_HUBER_DELTA,
            target_formula_bar: default_target_formula_bar(),
            target_x_bounds: DEFAULT_TARGET_X_BOUNDS,
            target_sample_count: DEFAULT_TARGET_SAMPLES,
            target_table_text: DEFAULT_TABLE_TEXT.to_string(),
            table_parse_error: None,
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

impl LossLandscapeTool {
    #[inline]
    #[must_use]
    pub const fn domain_bounds(&self) -> (f64, f64, f64, f64) {
        (self.w_min, self.w_max, self.b_min, self.b_max)
    }

    #[cfg(test)]
    pub fn set_target_table_text(&mut self, text: &str) {
        self.target_mode = TargetDataMode::Table;
        self.target_table_text = text.to_string();
        self.refresh_target_samples();
    }

    pub fn refresh_target_samples(&mut self) {
        match self.target_mode {
            TargetDataMode::PresetLinear => {
                self.table_parse_error = None;
                self.cached_targets = default_preset_targets();
            }
            TargetDataMode::Formula => {
                self.table_parse_error = None;
                if let Some(pts) = sample_formula_targets(
                    &self.target_formula_bar,
                    self.target_x_bounds,
                    self.target_sample_count,
                ) {
                    self.cached_targets = pts;
                }
            }
            TargetDataMode::Table => match parse_target_table(&self.target_table_text) {
                Ok(rows) => {
                    self.table_parse_error = None;
                    self.cached_targets = rows;
                }
                Err(err) => {
                    self.table_parse_error = Some(err);
                }
            },
        }
    }

    #[inline]
    #[must_use]
    pub fn empirical_config(&self) -> EmpiricalConfig<'_> {
        let custom_predictor = (self.predictor_mode == PredictorMode::Custom)
            .then(|| {
                self.math_input
                    .active_compiled()
                    .map(|c| (c, self.math_input.param_slice()))
            })
            .flatten();
        EmpiricalConfig {
            targets: &self.cached_targets,
            metric: self.loss_metric,
            huber_delta: self.huber_delta,
            is_preset_linear_target: self.target_mode == TargetDataMode::PresetLinear,
            custom_predictor,
        }
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn evaluate_loss_at(&self, w: f64, b: f64) -> Option<f64> {
        self.empirical_config().evaluate_base(w, b)
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn evaluate_mse_at(&self, w: f64, b: f64) -> Option<f64> {
        self.evaluate_loss_at(w, b)
    }

    #[must_use]
    pub fn surface_evaluator(&self) -> SurfaceEvaluator<'_> {
        let empirical =
            (self.surface == LandscapeSurface::LinearMse).then(|| self.empirical_config());
        SurfaceEvaluator::new(self.surface, &self.sculpt_bumps, empirical)
    }

    #[inline]
    #[must_use]
    pub(super) fn build_empirical<'a>(
        surface: LandscapeSurface,
        predictor_mode: PredictorMode,
        target_mode: TargetDataMode,
        metric: LossMetric,
        huber_delta: f64,
        targets: &'a [[f64; 2]],
        bar: &'a MathExpressionInputBar,
    ) -> Option<EmpiricalConfig<'a>> {
        if surface != LandscapeSurface::LinearMse {
            return None;
        }
        let custom_predictor = (predictor_mode == PredictorMode::Custom)
            .then(|| bar.active_compiled().map(|c| (c, bar.param_slice())))
            .flatten();
        Some(EmpiricalConfig {
            targets,
            metric,
            huber_delta,
            is_preset_linear_target: target_mode == TargetDataMode::PresetLinear,
            custom_predictor,
        })
    }

    pub fn reset_arena(&mut self) {
        let bounds = self.domain_bounds();
        let emp = Self::build_empirical(
            self.surface,
            self.predictor_mode,
            self.target_mode,
            self.loss_metric,
            self.huber_delta,
            &self.cached_targets,
            &self.math_input,
        );
        let eval = SurfaceEvaluator::new(self.surface, &self.sculpt_bumps, emp);
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
            && self.huber_delta.is_finite()
            && (0.01..=50.0).contains(&self.huber_delta)
            && !self.cached_targets.is_empty()
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
        let [ref mut x0, ref mut x1] = self.target_x_bounds;
        enforce_ordered_drag_range(
            x0,
            x1,
            LOSS_PARAM_MIN_BOUND,
            LOSS_PARAM_MAX_BOUND,
            LOSS_MIN_SPAN,
            false,
            false,
        );
        self.resolution = self.resolution.clamp(10, 50);
        self.huber_delta = if self.huber_delta.is_finite() {
            self.huber_delta.clamp(0.01, 50.0)
        } else {
            DEFAULT_HUBER_DELTA
        };
        self.target_sample_count = self.target_sample_count.clamp(2, MAX_TABLE_ROWS);
        if self.cached_targets.is_empty() {
            self.cached_targets = default_preset_targets();
        }
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
            let emp = Self::build_empirical(
                self.surface,
                self.predictor_mode,
                self.target_mode,
                self.loss_metric,
                self.huber_delta,
                &self.cached_targets,
                &self.math_input,
            );
            let eval = SurfaceEvaluator::new(self.surface, &self.sculpt_bumps, emp);
            self.arena.step_once(&self.arena_config, eval, bounds);
        }
    }

    fn save_state(&self) -> Option<String> {
        persistence::save_tool_state(self)
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        persistence::try_load_tool_state(self, state)
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

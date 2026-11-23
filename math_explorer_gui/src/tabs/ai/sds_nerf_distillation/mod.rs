#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod presets;
mod tiles;
mod ui;

use crate::framework::{Camera3D, ExportCapabilities, InteractiveTool, Scene3DBuilder};
use eframe::egui;
use math_explorer::ai::sds::diffusion::{generate_noise_with_rng, inject_noise};
use math_explorer::ai::sds::gradient::compute_sds_gradient;
use math_explorer::ai::sds::models::{
    rgb_image_to_latent, AnalyticalScorePrior, AnalyticalVoxelNeRF, PromptPrototypeEncoder,
};
use math_explorer::ai::sds::rendering::{
    generate_ray_bundle, render_image_with_rng, RenderContext,
};
use math_explorer::ai::sds::score::{classifier_free_guidance, DiffusionModel, TextEncoder};
use math_explorer::ai::sds::training::{
    AdamOptimizer, DifferentiableNeRF, Optimizer, SgdOptimizer,
};
use math_explorer::nalgebra::{DMatrix, Matrix4, Point3, Vector3};
use math_explorer::primitives::{PositiveFloat, UnitInterval};
use oxidize_core::rng::OxidizeRng;
use scientific_metadata::theory::TheoryDescribable;
use tiles::{
    build_composite_export_image, latent_grad_to_colors, rgb_matrix_to_colors, SdsDiagnosticTile,
};

pub(crate) const MIN_RAY_GRID: usize = 8;
pub(crate) const MAX_RAY_GRID: usize = 32;
pub(crate) const DEFAULT_RAY_GRID: usize = 20;
pub(crate) const MIN_RAY_SAMPLES: usize = 6;
pub(crate) const MAX_RAY_SAMPLES: usize = 24;
pub(crate) const DEFAULT_RAY_SAMPLES: usize = 14;
pub(crate) const CFG_MIN: f64 = 1.0;
pub(crate) const CFG_MAX: f64 = 15.0;
pub(crate) const LR_MIN: f64 = 0.02;
pub(crate) const LR_MAX: f64 = 0.50;
pub(crate) const VOXEL_GRID_RES: usize = 8;
pub(crate) const VIEWPORT_EXTENT: f64 = 1.75;

/// Prompt shape prototypes for 2D-to-3D score distillation.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum SdsPromptPreset {
    /// Glowing amber torus sculpted around the origin.
    #[default]
    AmberTorus,
    /// Crystalline cyan octahedron volume.
    CyanOctahedron,
    /// Warm gold radiance sphere.
    GoldSphere,
    /// Dual-lobe symmetric radiance volume.
    TwinLobes,
}

impl SdsPromptPreset {
    /// Selectable prompt presets in UI order.
    pub const ALL: [Self; 4] = [
        Self::AmberTorus,
        Self::CyanOctahedron,
        Self::GoldSphere,
        Self::TwinLobes,
    ];

    /// Display label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::AmberTorus => "Glowing Amber Torus",
            Self::CyanOctahedron => "Cyan Crystal Octahedron",
            Self::GoldSphere => "Radiant Gold Sphere",
            Self::TwinLobes => "Symmetric Twin Lobes",
        }
    }

    /// Prompt string passed into [`PromptPrototypeEncoder`].
    pub const fn prompt_text(self) -> &'static str {
        match self {
            Self::AmberTorus => "glowing amber torus",
            Self::CyanOctahedron => "cyan crystal octahedron",
            Self::GoldSphere => "radiant gold sphere",
            Self::TwinLobes => "symmetric twin lobes",
        }
    }
}

/// First-order parameter optimizer selectable in `SdsNeRfDistillationTool`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum SdsOptimizerKind {
    /// Adaptive Moment Estimation (`AdamOptimizer`).
    #[default]
    Adam,
    /// Momentum Stochastic Gradient Descent (`SgdOptimizer`).
    Sgd,
}

impl SdsOptimizerKind {
    /// All optimizer choices.
    pub const ALL: [Self; 2] = [Self::Adam, Self::Sgd];

    /// Display label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Adam => "Adam Optimizer",
            Self::Sgd => "Momentum SGD",
        }
    }
}

/// Interactive 3D Score Distillation Sampling (SDS) NeRF playground built on `Plot3DViewport`.
pub struct SdsNeRfDistillationTool {
    /// Active text prompt prototype.
    pub prompt_preset: SdsPromptPreset,
    /// Active parameter optimizer (`Adam` or `SGD`).
    pub optimizer_kind: SdsOptimizerKind,
    /// Classifier-Free Guidance (CFG) scale $s \in [1, 15]$.
    pub guidance_scale: f64,
    /// Optimizer learning rate $\eta \in [0.02, 0.50]$.
    pub learning_rate: f64,
    /// Square ray-bundle grid dimension (`8..=32` pixels for responsive `wasm32`).
    pub ray_grid_size: usize,
    /// Stratified volumetric samples per ray (`6..=24` samples for responsive `wasm32`).
    pub samples_per_ray: usize,
    /// Shared 3D orbit, pan, projection, and zoom camera state.
    pub camera: Camera3D,
    /// Whether continuous distillation stepping is active.
    pub is_playing: bool,
    pub(super) nerf: AnalyticalVoxelNeRF,
    pub(super) render_ctx: RenderContext,
    pub(super) encoder: PromptPrototypeEncoder,
    pub(super) adam: AdamOptimizer,
    pub(super) sgd: SgdOptimizer,
    pub(super) rng: OxidizeRng,
    pub(super) steps_completed: usize,
    pub(super) initial_mse: f64,
    pub(super) current_mse: f64,
    pub(super) last_grad_norm: f64,
    pub(super) render_pixels: Vec<egui::Color32>,
    pub(super) grad_pixels: Vec<egui::Color32>,
    pub(super) backdrop_tex: Option<egui::TextureHandle>,
    pub(super) backdrop_dirty: bool,
    pub(super) render_tile: SdsDiagnosticTile,
    pub(super) grad_tile: SdsDiagnosticTile,
    pub(super) scene_builder: Scene3DBuilder,
    last_tick: Option<crate::web_io::PlatformInstant>,
}

impl Default for SdsNeRfDistillationTool {
    fn default() -> Self {
        let camera = Camera3D::new(0.45, 0.55, 1.0).with_perspective_distance(3.2);
        let lr = PositiveFloat::new(0.14).unwrap();
        let mom = UnitInterval::new(0.82).unwrap();
        let mut tool = Self {
            prompt_preset: SdsPromptPreset::AmberTorus,
            optimizer_kind: SdsOptimizerKind::Adam,
            guidance_scale: 6.5,
            learning_rate: 0.14,
            ray_grid_size: DEFAULT_RAY_GRID,
            samples_per_ray: DEFAULT_RAY_SAMPLES,
            camera,
            is_playing: false,
            nerf: AnalyticalVoxelNeRF::new(VOXEL_GRID_RES, 1.0).with_ray_march_bounds(
                1.2,
                4.2,
                DEFAULT_RAY_SAMPLES,
            ),
            render_ctx: RenderContext::new(),
            encoder: PromptPrototypeEncoder::new(),
            adam: AdamOptimizer::new(lr),
            sgd: SgdOptimizer::new(lr, mom),
            rng: OxidizeRng::from_seed(2026),
            steps_completed: 0,
            initial_mse: 0.0,
            current_mse: 0.0,
            last_grad_norm: 0.0,
            render_pixels: Vec::new(),
            grad_pixels: Vec::new(),
            backdrop_tex: None,
            backdrop_dirty: true,
            render_tile: SdsDiagnosticTile::new("sds_render_preview_tile"),
            grad_tile: SdsDiagnosticTile::new("sds_gradient_residual_tile"),
            scene_builder: Scene3DBuilder::new(camera),
            last_tick: None,
        };
        tool.reset_distillation();
        tool
    }
}

impl SdsNeRfDistillationTool {
    fn canonical_camera_pose() -> Matrix4<f64> {
        let eye = Point3::new(0.0, 0.0, 2.7);
        let target = Point3::origin();
        let up = Vector3::y();
        Matrix4::look_at_rh(&eye, &target, &up)
            .try_inverse()
            .unwrap_or_else(Matrix4::identity)
    }

    fn rebuild_optimizers(&mut self) {
        let lr_val = self.learning_rate.clamp(LR_MIN, LR_MAX);
        if let Ok(lr) = PositiveFloat::new(lr_val) {
            self.adam = AdamOptimizer::new(lr);
            if let Ok(mom) = UnitInterval::new(0.82) {
                self.sgd = SgdOptimizer::new(lr, mom);
            }
        }
    }

    /// Resets the 3D voxel radiance field, RNG, optimizer moments, and diagnostic raster tiles.
    pub fn reset_distillation(&mut self) {
        self.clamp_parameters();
        self.nerf = AnalyticalVoxelNeRF::new(VOXEL_GRID_RES, 1.0).with_ray_march_bounds(
            1.2,
            4.2,
            self.samples_per_ray,
        );
        self.rebuild_optimizers();
        self.rng = OxidizeRng::from_seed(2026);
        self.steps_completed = 0;
        self.refresh_diagnostic_rasters(false);
        self.initial_mse = self.current_mse;
    }

    /// Executes one Score Distillation Sampling step (`render -> noise -> CFG -> compute_sds_gradient -> backward -> optimizer`).
    pub fn step_distillation_once(&mut self) {
        self.clamp_parameters();
        self.refresh_diagnostic_rasters(true);
        self.steps_completed += 1;
    }

    fn refresh_diagnostic_rasters(&mut self, apply_optimizer_step: bool) {
        let grid = self.ray_grid_size.clamp(MIN_RAY_GRID, MAX_RAY_GRID);
        let samples = self.samples_per_ray.clamp(MIN_RAY_SAMPLES, MAX_RAY_SAMPLES);
        self.nerf = std::mem::replace(&mut self.nerf, AnalyticalVoxelNeRF::new(2, 1.0))
            .with_ray_march_bounds(1.2, 4.2, samples);

        let pose = Self::canonical_camera_pose();
        let bundle = generate_ray_bundle(&pose, grid, grid, std::f64::consts::FRAC_PI_3);
        let mut render_rng = OxidizeRng::from_seed(700 + self.steps_completed as u64);
        let rendered = render_image_with_rng(
            &mut self.render_ctx,
            &bundle,
            &self.nerf,
            1.2,
            4.2,
            samples,
            &mut render_rng,
        );
        let x0 = rgb_image_to_latent(&rendered);

        let prior = AnalyticalScorePrior::new(grid, grid, 100);
        let e_cond = self.encoder.encode(self.prompt_preset.prompt_text());
        let e_uncond = self.encoder.encode("");
        let target = prior.target_latent(&e_cond);

        let timestep = 24 + (self.steps_completed % 44);
        let (sig_scale, noise_scale) = prior.schedule().get_scales(timestep);
        let eps = generate_noise_with_rng(x0.nrows(), 3, &mut self.rng)
            .unwrap_or_else(|_| DMatrix::zeros(x0.nrows(), 3));
        let z_t = inject_noise(&x0, &eps, sig_scale, noise_scale);
        let eps_cond = prior.predict_noise(&z_t, timestep, &e_cond);
        let eps_uncond = prior.predict_noise(&z_t, timestep, &e_uncond);
        let eps_guided = classifier_free_guidance(
            &eps_cond,
            &eps_uncond,
            self.guidance_scale.clamp(CFG_MIN, CFG_MAX),
        );
        let sds_grad = compute_sds_gradient(&eps_guided, &eps, 0.16);
        self.last_grad_norm = sds_grad.norm() / ((sds_grad.nrows().max(1)) as f64).sqrt();

        if apply_optimizer_step {
            let param_grads = self.nerf.backward(&bundle, &sds_grad);
            let updated = match self.optimizer_kind {
                SdsOptimizerKind::Adam => self.adam.step(self.nerf.params(), &param_grads),
                SdsOptimizerKind::Sgd => self.sgd.step(self.nerf.params(), &param_grads),
            };
            if let Ok(next_params) = updated {
                self.nerf.set_params(next_params);
            }
        }
        self.update_mse_and_render_pixels(&bundle, samples, &target);

        latent_grad_to_colors(&sds_grad, grid, grid, &mut self.grad_pixels);
        self.backdrop_dirty = true;
        self.render_tile.mark_dirty();
        self.grad_tile.mark_dirty();
    }

    fn update_mse_and_render_pixels(
        &mut self,
        bundle: &math_explorer::ai::sds::rendering::RayBundle,
        samples: usize,
        target: &DMatrix<f64>,
    ) {
        let mut eval_rng = OxidizeRng::from_seed(99);
        let eval_img = render_image_with_rng(
            &mut self.render_ctx,
            bundle,
            &self.nerf,
            1.2,
            4.2,
            samples,
            &mut eval_rng,
        );
        let eval_x0 = rgb_image_to_latent(&eval_img);
        let denom = eval_x0.len().max(1) as f64;
        self.current_mse = (eval_x0 - target).norm_squared() / denom;
        rgb_matrix_to_colors(&eval_img, &mut self.render_pixels);
    }

    pub(super) fn tick_playback(&mut self, ctx: &egui::Context) {
        if !self.is_playing {
            self.last_tick = None;
            return;
        }
        let now = crate::web_io::PlatformInstant::now();
        let ready = match self.last_tick {
            Some(prev) => prev.elapsed().as_secs_f64() >= 0.08,
            None => true,
        };
        if ready {
            self.step_distillation_once();
            self.last_tick = Some(now);
            if self.steps_completed >= 120 {
                self.is_playing = false;
            }
        }
        ctx.request_repaint();
    }
}

impl InteractiveTool for SdsNeRfDistillationTool {
    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "SDS NeRF Distillation"
    }

    fn domain(&self) -> &'static str {
        "ai"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::raster_only()
    }

    fn export_raster_image(&self) -> Option<egui::ColorImage> {
        let grid = self.ray_grid_size.clamp(MIN_RAY_GRID, MAX_RAY_GRID);
        Some(build_composite_export_image(
            grid,
            &self.render_pixels,
            &self.grad_pixels,
        ))
    }

    fn step_simulation(&mut self) {
        self.step_distillation_once();
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.prompt_preset.hash(&mut h);
        self.optimizer_kind.hash(&mut h);
        self.guidance_scale.to_bits().hash(&mut h);
        self.learning_rate.to_bits().hash(&mut h);
        self.ray_grid_size.hash(&mut h);
        self.samples_per_ray.hash(&mut h);
        self.steps_completed.hash(&mut h);
        self.camera.hash_state(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        self.guidance_scale.is_finite()
            && (CFG_MIN..=CFG_MAX).contains(&self.guidance_scale)
            && self.learning_rate.is_finite()
            && (LR_MIN..=LR_MAX).contains(&self.learning_rate)
            && (MIN_RAY_GRID..=MAX_RAY_GRID).contains(&self.ray_grid_size)
            && (MIN_RAY_SAMPLES..=MAX_RAY_SAMPLES).contains(&self.samples_per_ray)
            && self.camera.is_valid()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.guidance_scale = if self.guidance_scale.is_finite() {
            self.guidance_scale.clamp(CFG_MIN, CFG_MAX)
        } else {
            6.5
        };
        self.learning_rate = if self.learning_rate.is_finite() {
            self.learning_rate.clamp(LR_MIN, LR_MAX)
        } else {
            0.14
        };
        self.ray_grid_size = self.ray_grid_size.clamp(MIN_RAY_GRID, MAX_RAY_GRID);
        self.samples_per_ray = self.samples_per_ray.clamp(MIN_RAY_SAMPLES, MAX_RAY_SAMPLES);
        if !self.camera.is_valid() {
            self.camera.reset();
        }
        true
    }

    fn save_state(&self) -> Option<String> {
        self.encode_state_json()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.decode_state_json(state)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.tick_playback(ui.ctx());
        self.draw_side_panel(ui);
        self.draw_viewport_3d(ui);
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

// [cite:gaussian_splatting]
// [cite:graph_parameters_rust]

#[cfg(test)]
mod tests;

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::{
    default_morpho_atlas_axes, is_default_morpho_atlas_axes, MORPHO_ATLAS_AXIS_PAIRS,
};
use super::presets::MorphogenesisTheory;
use super::runner::{MorphogenesisRunner, MorphogenesisSharedParams};
use crate::async_sim::atlas::ParameterAtlasController;
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use eframe::egui::ColorImage;
use math_explorer::biology::diffusion::{BoundaryMode, FiniteDifference2D};
use math_explorer::biology::morphogenesis::{
    GrayScottKinetics, SchnakenbergKinetics, TuringSystem,
};
use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

fn default_grid_dim() -> usize {
    100
}

fn default_feed_rate() -> f64 {
    0.037
}

fn default_kill_rate() -> f64 {
    0.060
}

/// Reaction-kinetics family selector for [`MorphogenesisTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MorphogenesisKineticsMode {
    /// Autocatalytic Schnakenberg reaction (`a - u + u^2 v`, `b - u^2 v`).
    #[default]
    Schnakenberg,
    /// Autocatalytic Gray-Scott reaction (`-u v^2 + f(1 - u)`, `u v^2 - (f + k)v`).
    GrayScott,
}

/// Spatial finite-difference boundary condition selector for [`MorphogenesisTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MorphogenesisBoundaryMode {
    /// Zero-flux clamped edges (`∂u/∂n = 0`).
    #[default]
    Neumann,
    /// Seamless toroidal wrap across opposite grid edges.
    Periodic,
}

impl MorphogenesisBoundaryMode {
    pub(crate) const fn to_domain_mode(self) -> BoundaryMode {
        match self {
            Self::Neumann => BoundaryMode::Neumann,
            Self::Periodic => BoundaryMode::Periodic,
        }
    }
}

/// Complete validated state schema for [`MorphogenesisTool`], compatible with `PRESET_TURING_*`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MorphogenesisToolState {
    /// Schnakenberg activator production rate `a` (`0.0..=1.0`).
    pub a: f64,
    /// Schnakenberg inhibitor production rate `b` (`0.0..=2.0`).
    pub b: f64,
    /// Activator diffusion coefficient `D_u` (`0.1..=5.0`).
    pub d_u: f64,
    /// Inhibitor diffusion coefficient `D_v` (`10.0..=200.0`).
    pub d_v: f64,
    /// Integration time step `dt` (`1e-4..=0.1`).
    pub dt: f64,
    /// Spatial grid width (`1..=512`).
    #[serde(default = "default_grid_dim")]
    pub width: usize,
    /// Spatial grid height (`1..=512`).
    #[serde(default = "default_grid_dim")]
    pub height: usize,
    /// Active reaction kinetics model (`Schnakenberg` or `GrayScott`).
    #[serde(default)]
    pub kinetics: MorphogenesisKineticsMode,
    /// Active boundary condition (`Neumann` or `Periodic`).
    #[serde(default)]
    pub boundary: MorphogenesisBoundaryMode,
    /// Gray-Scott feed rate `f` (`0.001..=0.12`).
    #[serde(default = "default_feed_rate")]
    pub feed_rate: f64,
    /// Gray-Scott kill rate `k` (`0.01..=0.12`).
    #[serde(default = "default_kill_rate")]
    pub kill_rate: f64,
    /// Optional worker execution state (`Running` when omitted unless `paused == Some(true)`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<SimWorkerState>,
    /// Legacy boolean pause flag accepted on deserialization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paused: Option<bool>,
    /// Deferred start flag keeping hidden tabs idle until first rendered frame.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub pending_initial_start: bool,
    /// Persisted visibility of the progressive 2D parameter atlas (never raster pixels).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub atlas_visible: bool,
    /// Persisted `(x, y)` axis parameter keys for the 2D parameter atlas.
    #[serde(
        default = "default_morpho_atlas_axes",
        skip_serializing_if = "is_default_morpho_atlas_axes"
    )]
    pub atlas_axes: [String; 2],
}

impl MorphogenesisToolState {
    /// Validates finiteness, parameter domain bounds, and atlas axis keys.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("a", self.a, 0.0, 1.0),
            ("b", self.b, 0.0, 2.0),
            ("d_u", self.d_u, 0.1, 5.0),
            ("d_v", self.d_v, 10.0, 200.0),
            ("dt", self.dt, 1e-4, 0.1),
            ("feed_rate", self.feed_rate, 0.001, 0.12),
            ("kill_rate", self.kill_rate, 0.01, 0.12),
        ];
        for (name, val, min, max) in checks {
            if !val.is_finite() || val < min || val > max {
                return Err(format!(
                    "Morphogenesis parameter '{name}' = {val} must be finite within [{min}, {max}]"
                ));
            }
        }
        if self.width == 0 || self.width > 512 || self.height == 0 || self.height > 512 {
            return Err(format!(
                "Morphogenesis dimensions {}x{} must be within 1..=512",
                self.width, self.height
            ));
        }
        let valid_axes = MORPHO_ATLAS_AXIS_PAIRS
            .iter()
            .any(|&(xk, yk)| xk == self.atlas_axes[0] && yk == self.atlas_axes[1]);
        if !valid_axes {
            return Err(format!(
                "Invalid Morphogenesis atlas axes ('{}', '{}')",
                self.atlas_axes[0], self.atlas_axes[1]
            ));
        }
        Ok(())
    }
}

/// Interactive tool for 2D Schnakenberg and Gray-Scott Turing reaction-diffusion morphogenesis.
#[cfg_attr(any(), verified(opt_out = "gui_tool"))]
pub struct MorphogenesisTool {
    /// Asynchronous simulation worker controller.
    pub worker: BackgroundWorker<MorphogenesisRunner>,
    /// Progressive 2D parameter regime atlas controller.
    pub atlas: ParameterAtlasController,
    pub(crate) params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) pixels: Vec<egui::Color32>,
    pub(crate) dirty: bool,
    pub(crate) stepped_pixels: Option<(usize, usize, Vec<egui::Color32>)>,
    pub(crate) stepped_scalar: Option<(usize, usize, Vec<f64>)>,
    pub(crate) step_runner: Option<MorphogenesisRunner>,
    pub(crate) pending_initial_start: bool,
    pub(crate) last_high_contrast: Option<bool>,
    pub(crate) theory: MorphogenesisTheory,
}

impl Default for MorphogenesisTool {
    fn default() -> Self {
        let width = 100;
        let height = 100;
        let params = Arc::new(std::sync::RwLock::new(MorphogenesisSharedParams {
            a: 0.1,
            b: 0.9,
            d_u: 1.0,
            d_v: 100.0,
            dt: 0.05,
            kinetics: MorphogenesisKineticsMode::Schnakenberg,
            boundary: MorphogenesisBoundaryMode::Neumann,
            feed_rate: 0.037,
            kill_rate: 0.060,
            high_contrast: false,
        }));

        let runner = Self::create_runner(&params, width, height);
        let worker = BackgroundWorker::new(runner);

        Self {
            worker,
            atlas: ParameterAtlasController::new(MORPHO_ATLAS_AXIS_PAIRS.to_vec()),
            params,
            width,
            height,
            texture: None,
            pixels: Vec::new(),
            dirty: true,
            stepped_pixels: None,
            stepped_scalar: None,
            step_runner: None,
            pending_initial_start: true,
            last_high_contrast: None,
            theory: MorphogenesisTheory::default(),
        }
    }
}

impl MorphogenesisTool {
    pub(crate) fn create_runner(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        width: usize,
        height: usize,
    ) -> MorphogenesisRunner {
        use math_explorer::biology::morphogenesis::DiffusionCoeff;
        use math_explorer::math_kernel::types::{Dimension, StepSize};
        let (a, b, d_u, d_v, kinetics, boundary, feed_rate, kill_rate) = {
            let p = params.read().unwrap_or_else(|e| e.into_inner());
            (
                p.a,
                p.b,
                p.d_u,
                p.d_v,
                p.kinetics,
                p.boundary,
                p.feed_rate,
                p.kill_rate,
            )
        };
        let bmode = boundary.to_domain_mode();
        let diffusion = FiniteDifference2D::new(
            Dimension(width),
            Dimension(height),
            StepSize(1.0),
            StepSize(1.0),
        )
        .with_boundary_mode(bmode);
        let mut system = TuringSystem::new_with_kinetics(
            Dimension(width * height),
            DiffusionCoeff(d_u),
            DiffusionCoeff(d_v),
            SchnakenbergKinetics { a, b },
            diffusion,
        );
        Self::initialize_system(&mut system, width, height);

        let gs_diffusion = FiniteDifference2D::new(
            Dimension(width),
            Dimension(height),
            StepSize(1.0),
            StepSize(1.0),
        )
        .with_boundary_mode(bmode);
        let mut gray_scott_system = TuringSystem::new_with_kinetics(
            Dimension(width * height),
            DiffusionCoeff(0.16),
            DiffusionCoeff(0.08),
            GrayScottKinetics {
                f: feed_rate,
                k: kill_rate,
            },
            gs_diffusion,
        );
        Self::initialize_gray_scott_system(&mut gray_scott_system, width, height);

        MorphogenesisRunner {
            system,
            gray_scott_system,
            active_kinetics: kinetics,
            params: Arc::clone(params),
            width,
            height,
        }
    }

    #[cfg(test)]
    #[allow(missing_docs)]
    pub fn restart_worker(&mut self) {
        let runner = Self::create_runner(&self.params, self.width, self.height);
        self.worker = BackgroundWorker::new(runner);
        self.stepped_pixels = None;
        self.stepped_scalar = None;
        self.step_runner = None;
        self.pending_initial_start = false;
        self.dirty = true;
    }

    #[cfg(test)]
    #[allow(missing_docs)]
    pub fn retry(&mut self) {
        self.restart_worker();
    }

    /// Resets the worker in place or respawns a disconnected worker in `Ready` state.
    pub fn reset(&mut self) {
        let params = Arc::clone(&self.params);
        let width = self.width;
        let height = self.height;
        self.worker
            .reset_or_respawn(|| Self::create_runner(&params, width, height));
        self.stepped_pixels = None;
        self.stepped_scalar = None;
        self.step_runner = None;
        self.pending_initial_start = false;
        self.dirty = true;
    }

    pub(crate) fn reset_system(&mut self) {
        self.reset();
        let _ = self.worker.send_command(SimCommand::Start);
    }
}

impl InteractiveTool for MorphogenesisTool {
    fn name(&self) -> &'static str {
        "Turing Morphogenesis"
    }

    fn domain(&self) -> &'static str {
        "biology"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::scalar_field()
    }

    fn export_raster_image(&self) -> Option<ColorImage> {
        self.build_export_raster_image()
    }

    fn export_scalar_field(
        &self,
    ) -> Option<math_commons::math_kernel::marching_squares::ScalarField2D> {
        self.build_export_scalar_field()
    }

    fn step_simulation(&mut self) {
        self.step_simulation_offscreen();
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let high_contrast = crate::app::ThemeMode::from_visuals(ui.visuals())
            == crate::app::ThemeMode::HighContrast;
        if self.last_high_contrast != Some(high_contrast) {
            self.last_high_contrast = Some(high_contrast);
            self.params
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .high_contrast = high_contrast;
            self.dirty = true;
        }
        if std::mem::take(&mut self.pending_initial_start) && self.worker.state().is_ready() {
            let _ = self.worker.send_command(SimCommand::Start);
        }
        if self.worker.update().is_some() {
            self.dirty = true;
        }
        self.draw_controls(ui);
        self.draw_canvas(ui);
        if matches!(
            self.worker.state(),
            SimWorkerState::Running | SimWorkerState::Stepping
        ) {
            ctx.request_repaint();
        }
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            p.a.to_bits().hash(&mut h);
            p.b.to_bits().hash(&mut h);
            p.d_u.to_bits().hash(&mut h);
            p.d_v.to_bits().hash(&mut h);
            p.dt.to_bits().hash(&mut h);
            p.kinetics.hash(&mut h);
            p.boundary.hash(&mut h);
            p.feed_rate.to_bits().hash(&mut h);
            p.kill_rate.to_bits().hash(&mut h);
        }
        self.width.hash(&mut h);
        self.height.hash(&mut h);
        std::mem::discriminant(&self.worker.state()).hash(&mut h);
        self.pending_initial_start.hash(&mut h);
        self.atlas.is_visible().hash(&mut h);
        self.atlas.axis_keys().hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        let state = self.snapshot_tool_state();
        serde_json::to_string(&state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.apply_parsed_state(state)
    }

    fn has_finite_parameters(&self) -> bool {
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        p.a.is_finite()
            && p.b.is_finite()
            && p.d_u.is_finite()
            && p.d_v.is_finite()
            && p.dt.is_finite()
            && p.feed_rate.is_finite()
            && p.kill_rate.is_finite()
    }

    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
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
    crate::framework::ToolMetadata {
        id: "MorphogenesisTool",
        display_name: "Turing Morphogenesis",
        domain: "biology",
        tags: &[
            "turing",
            "morphogenesis",
            "schnakenberg",
            "gray-scott",
            "reaction-diffusion",
            "spots",
            "stripes",
        ],
        build: || Box::new(MorphogenesisTool::default()),
    }
}

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::{
    default_morpho_atlas_axes, is_default_morpho_atlas_axes, MORPHO_ATLAS_AXIS_PAIRS,
};
use super::presets::MorphogenesisTheory;
use super::runner::{MorphogenesisRunner, MorphogenesisSharedParams};
use crate::async_sim::atlas::ParameterAtlasController;
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState, SimulationRunner};
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use eframe::egui::ColorImage;
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::SchnakenbergKinetics;
use math_explorer::biology::morphogenesis::TuringSystem;
use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

fn default_grid_dim() -> usize {
    100
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

    fn resolve_target_state(&self) -> SimWorkerState {
        self.state.unwrap_or_else(|| {
            if self.paused.unwrap_or(false) {
                SimWorkerState::Ready
            } else {
                SimWorkerState::Running
            }
        })
    }
}

/// Interactive tool for 2D Schnakenberg Turing reaction-diffusion morphogenesis.
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
        let (a, b, d_u, d_v) = {
            let p = params.read().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b, p.d_u, p.d_v)
        };
        let kinetics = SchnakenbergKinetics { a, b };
        let diffusion = FiniteDifference2D::new(
            Dimension(width),
            Dimension(height),
            StepSize(1.0),
            StepSize(1.0),
        );
        let mut system = TuringSystem::new_with_kinetics(
            Dimension(width * height),
            DiffusionCoeff(d_u),
            DiffusionCoeff(d_v),
            kinetics,
            diffusion,
        );
        Self::initialize_system(&mut system, width, height);
        MorphogenesisRunner {
            system,
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
        self.step_runner = None;
        self.pending_initial_start = false;
        self.dirty = true;
    }

    pub(crate) fn reset_system(&mut self) {
        self.reset();
        let _ = self.worker.send_command(SimCommand::Start);
    }

    pub(crate) fn apply_named_preset(&mut self, a: f64, b: f64, d_u: f64, d_v: f64, dt: f64) {
        let target_state = if self.worker.is_running() {
            SimWorkerState::Running
        } else {
            SimWorkerState::Ready
        };
        let (xk, yk) = self.atlas.axis_keys();
        let preset_state = MorphogenesisToolState {
            a,
            b,
            d_u,
            d_v,
            dt,
            width: self.width,
            height: self.height,
            state: Some(target_state),
            paused: None,
            pending_initial_start: false,
            atlas_visible: self.atlas.is_visible(),
            atlas_axes: [xk.to_string(), yk.to_string()],
        };
        if let Ok(json) = serde_json::to_string(&preset_state) {
            let _ = self.try_load_state(&json);
        }
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
        ExportCapabilities::plot()
    }

    fn export_tables(&self) -> Vec<crate::framework::ExportTable> {
        super::export::build_morphogenesis_tables(self)
    }

    fn export_raster_image(&self) -> Option<ColorImage> {
        if let Some((w, h, ref px)) = self.stepped_pixels {
            if w > 0 && h > 0 && px.len() == w * h {
                return Some(ColorImage {
                    size: [w, h],
                    pixels: px.clone(),
                    ..Default::default()
                });
            }
        }
        if !self.dirty && self.pixels.len() == self.width * self.height && !self.pixels.is_empty() {
            return Some(ColorImage {
                size: [self.width, self.height],
                pixels: self.pixels.clone(),
                ..Default::default()
            });
        }
        let snapshot = self.worker.latest_snapshot()?;
        let px = snapshot.pixels.read().unwrap_or_else(|e| e.into_inner());
        if snapshot.width == 0 || snapshot.height == 0 || px.len() != snapshot.width * snapshot.height
        {
            return None;
        }
        Some(ColorImage {
            size: [snapshot.width, snapshot.height],
            pixels: px.clone(),
            ..Default::default()
        })
    }

    fn step_simulation(&mut self) {
        let runner = self
            .step_runner
            .get_or_insert_with(|| Self::create_runner(&self.params, self.width, self.height));
        for _ in 0..5 {
            runner.step();
        }
        let snap = runner.get_snapshot();
        let px = snap.pixels.read().unwrap_or_else(|e| e.into_inner());
        self.stepped_pixels = Some((snap.width, snap.height, px.clone()));
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
        let (a, b, d_u, d_v, dt) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b, p.d_u, p.d_v, p.dt)
        };
        let (xk, yk) = self.atlas.axis_keys();
        serde_json::to_string(&MorphogenesisToolState {
            a,
            b,
            d_u,
            d_v,
            dt,
            width: self.width,
            height: self.height,
            state: Some(self.worker.state()),
            paused: None,
            pending_initial_start: self.pending_initial_start,
            atlas_visible: self.atlas.is_visible(),
            atlas_axes: [xk.to_string(), yk.to_string()],
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: MorphogenesisToolState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid Morphogenesis state: {e}"))?;
        parsed.validate()?;
        self.atlas
            .set_axis_keys(&parsed.atlas_axes[0], &parsed.atlas_axes[1])?;
        self.atlas.set_visible(parsed.atlas_visible);
        let target_state = parsed.resolve_target_state();
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.a = parsed.a;
            p.b = parsed.b;
            p.d_u = parsed.d_u;
            p.d_v = parsed.d_v;
            p.dt = parsed.dt;
        }
        if parsed.width != self.width || parsed.height != self.height {
            self.width = parsed.width;
            self.height = parsed.height;
            let runner = Self::create_runner(&self.params, self.width, self.height);
            self.worker = BackgroundWorker::new(runner);
            self.stepped_pixels = None;
            self.step_runner = None;
            self.dirty = true;
        } else {
            self.reset();
        }
        if target_state == SimWorkerState::Running {
            self.pending_initial_start = false;
            if !self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Start);
            }
        } else {
            self.pending_initial_start = parsed.pending_initial_start;
            if self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Pause);
            }
        }
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

    fn has_finite_parameters(&self) -> bool {
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        p.a.is_finite()
            && p.b.is_finite()
            && p.d_u.is_finite()
            && p.d_v.is_finite()
            && p.dt.is_finite()
    }

    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
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
            "reaction-diffusion",
            "spots",
            "stripes",
        ],
        build: || Box::new(MorphogenesisTool::default()),
    }
}

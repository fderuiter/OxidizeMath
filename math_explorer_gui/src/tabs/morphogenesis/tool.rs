#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets::MorphogenesisTheory;
use super::runner::{MorphogenesisRunner, MorphogenesisSharedParams};
use super::sculpt::default_morpho_brush_kit;
use crate::async_sim::{
    BackgroundWorker, SculptStatePayload, SimCommand, SimWorkerState, SimulationRunner,
};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{
    decode_rle_mask, ExportCapabilities, InteractionContext, InteractiveTool, SpatialBrushKit,
};
use eframe::egui;
use eframe::egui::ColorImage;
use math_explorer::biology::diffusion::MaskedFiniteDifference2D;
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
    /// Optional RLE-encoded `ReactionCellRole` mask (`[role, run_len]` pairs).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mask_rle: Vec<[u32; 2]>,
}

impl MorphogenesisToolState {
    /// Validates finiteness and domain bounds for all reaction-diffusion parameters.
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
        if !self.mask_rle.is_empty() {
            decode_rle_mask(&self.mask_rle, self.width * self.height)?;
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
#[cfg_attr(any(), verified(opt_out = "gui_tool"))]
pub struct MorphogenesisTool {
    /// Asynchronous simulation worker controller.
    pub worker: BackgroundWorker<MorphogenesisRunner<MaskedFiniteDifference2D>>,
    pub(crate) params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
    pub(crate) mask_rle: Arc<std::sync::RwLock<Vec<[u32; 2]>>>,
    pub(crate) brush_kit: SpatialBrushKit,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) pixels: Vec<egui::Color32>,
    pub(crate) dirty: bool,
    pub(crate) stepped_pixels: Option<(usize, usize, Vec<egui::Color32>)>,
    pub(crate) step_runner: Option<MorphogenesisRunner<MaskedFiniteDifference2D>>,
    pub(crate) pending_initial_start: bool,
    pub(crate) last_high_contrast: Option<bool>,
    theory: MorphogenesisTheory,
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

        let mask_rle = Arc::new(std::sync::RwLock::new(Vec::new()));
        let runner = Self::create_runner_with_mask(&params, &mask_rle, width, height);
        let worker = BackgroundWorker::new(runner);

        Self {
            worker,
            params,
            mask_rle,
            brush_kit: default_morpho_brush_kit(),
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
    #[cfg(test)]
    #[allow(missing_docs)]
    pub fn restart_worker(&mut self) {
        let runner =
            Self::create_runner_with_mask(&self.params, &self.mask_rle, self.width, self.height);
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
        let mask = Arc::clone(&self.mask_rle);
        let width = self.width;
        let height = self.height;
        self.worker
            .reset_or_respawn(|| Self::create_runner_with_mask(&params, &mask, width, height));
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
            mask_rle: self.mask_rle.read().unwrap_or_else(|e| e.into_inner()).clone(),
        };
        if let Ok(json) = serde_json::to_string(&preset_state) {
            let _ = self.try_load_state(&json);
        }
    }
}

impl FiniteStateGuard for MorphogenesisTool {
    fn is_finite_state(&self) -> bool {
        self.worker.state() != SimWorkerState::Faulted && self.worker.diagnostic_report().is_none()
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.worker.diagnostic_report()
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }

    fn auto_scale_step_size(&mut self) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.dt = (p.dt * 0.5).max(1e-4);
        drop(p);
        self.reset_system();
    }

    fn clamp_parameters(&mut self) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.a = p.a.clamp(0.0, 1.0);
        p.b = p.b.clamp(0.0, 2.0);
        p.d_u = p.d_u.clamp(0.1, 5.0);
        p.d_v = p.d_v.clamp(10.0, 200.0);
        p.dt = p.dt.clamp(0.001, 0.1);
        drop(p);
        self.reset_system();
    }

    fn reset_parameters(&mut self) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.a = 0.1;
        p.b = 0.9;
        p.d_u = 1.0;
        p.d_v = 100.0;
        p.dt = 0.05;
        drop(p);
        self.reset_system();
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
        ExportCapabilities::raster_only()
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
        let runner = self.step_runner.get_or_insert_with(|| {
            Self::create_runner_with_mask(&self.params, &self.mask_rle, self.width, self.height)
        });
        for _ in 0..5 {
            runner.step();
        }
        let snap = runner.get_snapshot();
        let px = snap.pixels.read().unwrap_or_else(|e| e.into_inner());
        self.stepped_pixels = Some((snap.width, snap.height, px.clone()));
    }

    fn on_brush(&mut self, ctx: &InteractionContext) {
        if let Some(stroke) = ctx.brush_stroke {
            self.dispatch_brush_stroke(stroke, Some(&ctx.response.ctx));
        }
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.brush_kit.focused_plot_id()
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
        self.mask_rle
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .hash(&mut h);
        std::mem::discriminant(&self.worker.state()).hash(&mut h);
        self.pending_initial_start.hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        let (a, b, d_u, d_v, dt) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b, p.d_u, p.d_v, p.dt)
        };
        let mask_rle = self
            .mask_rle
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
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
            mask_rle,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: MorphogenesisToolState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid Morphogenesis state: {e}"))?;
        parsed.validate()?;
        let target_state = parsed.resolve_target_state();
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.a = parsed.a;
            p.b = parsed.b;
            p.d_u = parsed.d_u;
            p.d_v = parsed.d_v;
            p.dt = parsed.dt;
        }
        *self.mask_rle.write().unwrap_or_else(|e| e.into_inner()) = parsed.mask_rle.clone();
        if parsed.width != self.width || parsed.height != self.height {
            self.width = parsed.width;
            self.height = parsed.height;
            let runner = Self::create_runner_with_mask(
                &self.params,
                &self.mask_rle,
                self.width,
                self.height,
            );
            self.worker = BackgroundWorker::new(runner);
            self.stepped_pixels = None;
            self.step_runner = None;
            self.dirty = true;
        } else {
            self.reset();
            let _ = self.worker.send_command(SimCommand::SyncSculptState(
                SculptStatePayload {
                    rle_mask: parsed.mask_rle,
                    sparse_potential: Vec::new(),
                },
            ));
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

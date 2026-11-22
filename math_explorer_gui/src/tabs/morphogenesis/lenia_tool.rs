#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub use super::lenia_runner::{
    LeniaPresetChoice, LeniaRunner, LeniaSharedParams, LeniaSnapshotPayload,
};
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState, SimulationRunner};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use eframe::egui::ColorImage;
use math_explorer::biology::lenia::{LeniaParams, LeniaPreset, LeniaTelemetry};
use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

fn default_lenia_dim() -> usize {
    128
}

/// Validated URL and preset state schema for [`LeniaTool`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LeniaToolState {
    /// Optimal neighborhood potential center `mu` (`0.05..=0.50`).
    pub mu: f64,
    /// Gaussian growth width `sigma` (`0.005..=0.15`).
    pub sigma: f64,
    /// Explicit integration step `dt` (`0.01..=0.50`).
    pub dt: f64,
    /// Kernel interaction radius in cells (`4..=32`).
    pub kernel_radius: usize,
    /// Selected organism seed preset.
    #[serde(default)]
    pub preset: LeniaPresetChoice,
    /// Periodic grid width (`32..=256`).
    #[serde(default = "default_lenia_dim")]
    pub width: usize,
    /// Periodic grid height (`32..=256`).
    #[serde(default = "default_lenia_dim")]
    pub height: usize,
    /// Optional worker execution state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<SimWorkerState>,
}

impl LeniaToolState {
    /// Validates finiteness and parameter ranges for spectral Lenia.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("mu", self.mu, 0.05, 0.50),
            ("sigma", self.sigma, 0.005, 0.15),
            ("dt", self.dt, 0.01, 0.50),
        ];
        for (name, val, min, max) in checks {
            if !val.is_finite() || val < min || val > max {
                return Err(format!(
                    "Lenia parameter '{name}' = {val} must be finite within [{min}, {max}]"
                ));
            }
        }
        if !(32..=256).contains(&self.width) || !(32..=256).contains(&self.height) {
            return Err(format!(
                "Lenia dimensions {}x{} must be within 32..=256",
                self.width, self.height
            ));
        }
        let max_r = (self.width.min(self.height) / 2).saturating_sub(1).min(32);
        if !(4..=max_r).contains(&self.kernel_radius) {
            return Err(format!(
                "Lenia kernel_radius {} must be within 4..={max_r}",
                self.kernel_radius
            ));
        }
        Ok(())
    }
}

/// Interactive GUI tool for 2D spectral continuous cellular automata (`LeniaSystem`).
pub struct LeniaTool {
    /// Asynchronous worker stepping [`LeniaRunner`].
    pub worker: BackgroundWorker<LeniaRunner>,
    pub(crate) params: Arc<std::sync::RwLock<LeniaSharedParams>>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) pixels: Vec<egui::Color32>,
    pub(crate) telemetry: LeniaTelemetry,
    pub(crate) dirty: bool,
    pub(crate) stepped_pixels: Option<(usize, usize, Vec<egui::Color32>)>,
    pub(crate) step_runner: Option<LeniaRunner>,
    pub(crate) last_high_contrast: Option<bool>,
    theory_params: LeniaParams,
}

impl Default for LeniaTool {
    fn default() -> Self {
        let width = 128;
        let height = 128;
        let orbium = LeniaPreset::Orbium.params();
        let params = Arc::new(std::sync::RwLock::new(LeniaSharedParams {
            mu: orbium.mu,
            sigma: orbium.sigma,
            dt: orbium.dt,
            kernel_radius: orbium.kernel_radius,
            preset: LeniaPresetChoice::Orbium,
            high_contrast: false,
        }));
        let runner = LeniaRunner::new(&params, width, height);
        let telemetry = runner.system.telemetry();
        let worker = BackgroundWorker::new(runner);

        Self {
            worker,
            params,
            width,
            height,
            texture: None,
            pixels: Vec::new(),
            telemetry,
            dirty: true,
            stepped_pixels: None,
            step_runner: None,
            last_high_contrast: None,
            theory_params: orbium,
        }
    }
}

impl LeniaTool {
    /// Resets or respawns the spectral Lenia worker with the active organism seed.
    pub fn reset(&mut self) {
        let params = Arc::clone(&self.params);
        let w = self.width;
        let h = self.height;
        self.worker
            .reset_or_respawn(|| LeniaRunner::new(&params, w, h));
        self.stepped_pixels = None;
        self.step_runner = None;
        self.dirty = true;
    }

    pub(crate) fn apply_preset_choice(&mut self, choice: LeniaPresetChoice) {
        let dp = choice.to_domain().params();
        let target_state = if self.worker.is_running() {
            SimWorkerState::Running
        } else {
            SimWorkerState::Ready
        };
        let state = LeniaToolState {
            mu: dp.mu,
            sigma: dp.sigma,
            dt: dp.dt,
            kernel_radius: dp.kernel_radius,
            preset: choice,
            width: self.width,
            height: self.height,
            state: Some(target_state),
        };
        if let Ok(json) = serde_json::to_string(&state) {
            let _ = self.try_load_state(&json);
        }
    }
}

impl FiniteStateGuard for LeniaTool {
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
        p.dt = (p.dt * 0.5).max(0.01);
        drop(p);
        self.reset();
    }

    fn clamp_parameters(&mut self) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.mu = if p.mu.is_nan() { 0.15 } else { p.mu.clamp(0.05, 0.50) };
        p.sigma = if p.sigma.is_nan() {
            0.015
        } else {
            p.sigma.clamp(0.005, 0.15)
        };
        p.dt = if p.dt.is_nan() { 0.1 } else { p.dt.clamp(0.01, 0.50) };
        p.kernel_radius = p.kernel_radius.clamp(4, 32);
        drop(p);
        self.reset();
    }

    fn reset_parameters(&mut self) {
        self.apply_preset_choice(LeniaPresetChoice::Orbium);
    }
}

impl InteractiveTool for LeniaTool {
    fn name(&self) -> &'static str {
        "Continuous Lenia"
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
        (px.len() == snapshot.width * snapshot.height && snapshot.width > 0).then(|| ColorImage {
            size: [snapshot.width, snapshot.height],
            pixels: px.clone(),
            ..Default::default()
        })
    }

    fn step_simulation(&mut self) {
        let runner = self
            .step_runner
            .get_or_insert_with(|| LeniaRunner::new(&self.params, self.width, self.height));
        for _ in 0..4 {
            runner.step();
        }
        self.telemetry = runner.system.telemetry();
        let snap = runner.get_snapshot();
        let px = snap.pixels.read().unwrap_or_else(|e| e.into_inner());
        self.stepped_pixels = Some((snap.width, snap.height, px.clone()));
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let hc = crate::app::ThemeMode::from_visuals(ui.visuals())
            == crate::app::ThemeMode::HighContrast;
        if self.last_high_contrast != Some(hc) {
            self.last_high_contrast = Some(hc);
            self.params
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .high_contrast = hc;
            self.dirty = true;
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
        &self.theory_params
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            p.mu.to_bits().hash(&mut h);
            p.sigma.to_bits().hash(&mut h);
            p.dt.to_bits().hash(&mut h);
            p.kernel_radius.hash(&mut h);
            p.preset.hash(&mut h);
        }
        self.width.hash(&mut h);
        self.height.hash(&mut h);
        std::mem::discriminant(&self.worker.state()).hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        let (mu, sigma, dt, kernel_radius, preset) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.mu, p.sigma, p.dt, p.kernel_radius, p.preset)
        };
        serde_json::to_string(&LeniaToolState {
            mu,
            sigma,
            dt,
            kernel_radius,
            preset,
            width: self.width,
            height: self.height,
            state: Some(self.worker.state()),
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: LeniaToolState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Lenia state: {e}"))?;
        parsed.validate()?;
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.mu = parsed.mu;
            p.sigma = parsed.sigma;
            p.dt = parsed.dt;
            p.kernel_radius = parsed.kernel_radius;
            p.preset = parsed.preset;
        }
        if parsed.width != self.width || parsed.height != self.height {
            self.width = parsed.width;
            self.height = parsed.height;
            self.worker = BackgroundWorker::new(LeniaRunner::new(
                &self.params,
                self.width,
                self.height,
            ));
            self.stepped_pixels = None;
            self.step_runner = None;
            self.dirty = true;
        } else {
            self.reset();
        }
        if parsed.state == Some(SimWorkerState::Running) {
            let _ = self.worker.send_command(SimCommand::Start);
        } else if self.worker.is_running() {
            let _ = self.worker.send_command(SimCommand::Pause);
        }
        Ok(())
    }

    fn has_finite_parameters(&self) -> bool {
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        p.mu.is_finite() && p.sigma.is_finite() && p.dt.is_finite()
    }

    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "LeniaTool",
        display_name: "Continuous Lenia",
        domain: "biology",
        tags: &[
            "lenia",
            "cellular-automata",
            "orbium",
            "hydrogeminium",
            "fft",
            "artificial-life",
        ],
        build: || Box::new(LeniaTool::default()),
    }
}

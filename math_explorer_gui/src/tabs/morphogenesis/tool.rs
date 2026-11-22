#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets::MorphogenesisTheory;
use super::runner::MorphogenesisRunner;
use super::state::MorphogenesisToolState;
use super::types::{
    MorphogenesisKineticsMode, MorphogenesisOverlayMode, MorphogenesisSharedParams,
};
use crate::app::ThemeMode;
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState, SimulationRunner};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use eframe::egui::ColorImage;
use scientific_metadata::theory::TheoryDescribable;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Interactive tool for 2D upwind advected Turing & Gray-Scott morphogenesis.
#[cfg_attr(any(), verified(opt_out = "gui_tool"))]
pub struct MorphogenesisTool {
    /// Asynchronous simulation worker controller.
    pub worker: BackgroundWorker<MorphogenesisRunner>,
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
    pub(crate) last_theme_mode: Option<ThemeMode>,
    pub(crate) last_overlay_mode: Option<MorphogenesisOverlayMode>,
    pub(crate) live_peclet: f64,
    pub(crate) live_damkohler: f64,
    pub(crate) theory: MorphogenesisTheory,
}

impl Default for MorphogenesisTool {
    fn default() -> Self {
        let width = 100;
        let height = 100;
        let params = Arc::new(std::sync::RwLock::new(MorphogenesisSharedParams::default()));
        let runner = Self::create_runner(&params, width, height);
        let peclet = runner.peclet;
        let damkohler = runner.damkohler;
        let worker = BackgroundWorker::new(runner);

        Self {
            worker,
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
            last_theme_mode: None,
            last_overlay_mode: None,
            live_peclet: peclet,
            live_damkohler: damkohler,
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
        MorphogenesisRunner::new(params, width, height)
    }

    #[cfg(test)]
    #[allow(missing_docs)]
    pub fn restart_worker(&mut self) {
        let runner = Self::create_runner(&self.params, self.width, self.height);
        self.live_peclet = runner.peclet;
        self.live_damkohler = runner.damkohler;
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
        let p = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let preset_state = MorphogenesisToolState {
            a,
            b,
            d_u,
            d_v,
            dt,
            width: self.width,
            height: self.height,
            kinetics: MorphogenesisKineticsMode::Schnakenberg,
            flow_mode: p.flow_mode,
            inlet_velocity: p.inlet_velocity,
            viscosity: p.viscosity,
            circulation: p.circulation,
            overlay_mode: p.overlay_mode,
            brush_mode: p.brush_mode,
            brush_radius: p.brush_radius,
            state: Some(target_state),
            paused: None,
            pending_initial_start: false,
        };
        if let Ok(json) = serde_json::to_string(&preset_state) {
            let _ = self.try_load_state(&json);
        }
    }

    fn sync_visual_preferences(&mut self, visuals: &egui::Visuals) {
        let theme_mode = ThemeMode::from_visuals(visuals);
        let high_contrast = theme_mode == ThemeMode::HighContrast;
        let (kinetics, overlay_mode) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.kinetics, p.overlay_mode)
        };
        self.theory.set_mode(kinetics);
        if self.last_theme_mode != Some(theme_mode)
            || self.last_high_contrast != Some(high_contrast)
            || self.last_overlay_mode != Some(overlay_mode)
        {
            self.last_theme_mode = Some(theme_mode);
            self.last_high_contrast = Some(high_contrast);
            self.last_overlay_mode = Some(overlay_mode);
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.theme_mode = theme_mode;
            p.high_contrast = high_contrast;
            self.dirty = true;
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
        p.d_u = p.d_u.clamp(0.01, 5.0);
        p.d_v = p.d_v.clamp(0.01, 200.0);
        p.dt = p.dt.clamp(0.001, 1.0);
        p.inlet_velocity = p.inlet_velocity.clamp(0.0, 2.0);
        p.viscosity = p.viscosity.clamp(0.003, 0.5);
        p.circulation = p.circulation.clamp(-25.0, 25.0);
        p.brush_radius = p.brush_radius.clamp(1, 25);
        drop(p);
        self.reset_system();
    }

    fn reset_parameters(&mut self) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        let hc = p.high_contrast;
        let tm = p.theme_mode;
        *p = MorphogenesisSharedParams {
            high_contrast: hc,
            theme_mode: tm,
            ..MorphogenesisSharedParams::default()
        };
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
        if snapshot.width == 0
            || snapshot.height == 0
            || px.len() != snapshot.width * snapshot.height
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
        self.live_peclet = runner.peclet;
        self.live_damkohler = runner.damkohler;
        let snap = runner.get_snapshot();
        let px = snap.pixels.read().unwrap_or_else(|e| e.into_inner());
        self.stepped_pixels = Some((snap.width, snap.height, px.clone()));
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.sync_visual_preferences(ui.visuals());
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
            p.flow_mode.hash(&mut h);
            p.inlet_velocity.to_bits().hash(&mut h);
            p.viscosity.to_bits().hash(&mut h);
            p.circulation.to_bits().hash(&mut h);
            p.overlay_mode.hash(&mut h);
            p.brush_mode.hash(&mut h);
            p.brush_radius.hash(&mut h);
        }
        self.width.hash(&mut h);
        self.height.hash(&mut h);
        std::mem::discriminant(&self.worker.state()).hash(&mut h);
        self.pending_initial_start.hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        let p = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        serde_json::to_string(&MorphogenesisToolState {
            a: p.a,
            b: p.b,
            d_u: p.d_u,
            d_v: p.d_v,
            dt: p.dt,
            width: self.width,
            height: self.height,
            kinetics: p.kinetics,
            flow_mode: p.flow_mode,
            inlet_velocity: p.inlet_velocity,
            viscosity: p.viscosity,
            circulation: p.circulation,
            overlay_mode: p.overlay_mode,
            brush_mode: p.brush_mode,
            brush_radius: p.brush_radius,
            state: Some(self.worker.state()),
            paused: None,
            pending_initial_start: self.pending_initial_start,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: MorphogenesisToolState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Morphogenesis state: {e}"))?;
        parsed.validate()?;
        let target_state = parsed.resolve_target_state();
        self.theory.set_mode(parsed.kinetics);
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.a = parsed.a;
            p.b = parsed.b;
            p.d_u = parsed.d_u;
            p.d_v = parsed.d_v;
            p.dt = parsed.dt;
            p.kinetics = parsed.kinetics;
            p.flow_mode = parsed.flow_mode;
            p.inlet_velocity = parsed.inlet_velocity;
            p.viscosity = parsed.viscosity;
            p.circulation = parsed.circulation;
            p.overlay_mode = parsed.overlay_mode;
            p.brush_mode = parsed.brush_mode;
            p.brush_radius = parsed.brush_radius;
        }
        self.width = parsed.width;
        self.height = parsed.height;
        let runner = Self::create_runner(&self.params, self.width, self.height);
        self.live_peclet = runner.peclet;
        self.live_damkohler = runner.damkohler;
        self.worker = BackgroundWorker::new(runner);
        self.stepped_pixels = None;
        self.step_runner = None;
        self.dirty = true;

        if target_state == SimWorkerState::Running {
            self.pending_initial_start = false;
            let _ = self.worker.send_command(SimCommand::Start);
        } else {
            self.pending_initial_start = parsed.pending_initial_start;
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
            && p.inlet_velocity.is_finite()
            && p.viscosity.is_finite()
            && p.circulation.is_finite()
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
            "gray-scott",
            "advection",
            "vorticity",
            "reaction-diffusion",
            "spots",
            "stripes",
        ],
        build: || Box::new(MorphogenesisTool::default()),
    }
}

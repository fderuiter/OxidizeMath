#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets::OpticsWaveTheory;
use super::runner::{
    render_wavefield_image, OpticsFrameData, OpticsSharedParams, OpticsWaveRunner, WaveRenderInput,
};
use super::state::{default_grid_size, OpticsWaveToolState};
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState, SimulationRunner};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui::{self, ColorImage};
use scientific_metadata::theory::TheoryDescribable;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Interactive 2D FDTD wave propagation, Hamiltonian ray caustics, and virtual detector playground.
#[cfg_attr(any(), verified(opt_out = "gui_tool"))]
pub struct OpticsWaveTool {
    /// Asynchronous simulation worker controller.
    pub worker: BackgroundWorker<OpticsWaveRunner>,
    pub(crate) params: Arc<std::sync::RwLock<OpticsSharedParams>>,
    pub(crate) grid_size: usize,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) pixels: Vec<egui::Color32>,
    pub(crate) latest_frame: Option<OpticsFrameData>,
    pub(crate) dirty: bool,
    pub(crate) stepped_pixels: Option<(usize, usize, Vec<egui::Color32>)>,
    pub(crate) step_runner: Option<OpticsWaveRunner>,
    pub(crate) pending_initial_start: bool,
    pub(crate) last_high_contrast: Option<bool>,
    pub(crate) canvas_id: Option<egui::Id>,
    pub(crate) theory: OpticsWaveTheory,
}

impl Default for OpticsWaveTool {
    fn default() -> Self {
        let grid_size = default_grid_size();
        let params = Arc::new(std::sync::RwLock::new(OpticsSharedParams::default()));
        let mut init_runner = OpticsWaveRunner::new(Arc::clone(&params), grid_size);
        init_runner.step();
        let init_snap = init_runner.get_snapshot();
        let init_pixels = init_snap
            .pixels
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let init_frame = init_snap
            .structured_data
            .and_then(|b| b.downcast::<OpticsFrameData>().ok().map(|d| *d));
        let worker = BackgroundWorker::new(init_runner);

        Self {
            worker,
            params,
            grid_size,
            texture: None,
            pixels: init_pixels,
            latest_frame: init_frame,
            dirty: true,
            stepped_pixels: None,
            step_runner: None,
            pending_initial_start: true,
            last_high_contrast: None,
            canvas_id: None,
            theory: OpticsWaveTheory,
        }
    }
}

impl OpticsWaveTool {
    /// Resets the worker in place or respawns a disconnected worker in `Ready` state.
    pub fn reset(&mut self) {
        let params = Arc::clone(&self.params);
        let grid_size = self.grid_size;
        self.worker
            .reset_or_respawn(|| OpticsWaveRunner::new(params, grid_size));
        self.refresh_initial_preview();
        self.stepped_pixels = None;
        self.step_runner = None;
        self.pending_initial_start = false;
        self.dirty = true;
    }

    pub(crate) fn reset_and_resume(&mut self) {
        let was_running = self.worker.is_running();
        self.reset();
        if was_running {
            let _ = self.worker.send_command(SimCommand::Start);
        }
    }

    pub(crate) fn rebuild_for_grid_size(&mut self, new_size: usize) {
        let clamped = new_size.clamp(64, 256);
        if clamped == self.grid_size {
            return;
        }
        let was_running = self.worker.is_running();
        self.grid_size = clamped;
        let mut runner = OpticsWaveRunner::new(Arc::clone(&self.params), self.grid_size);
        runner.step();
        let snap = runner.get_snapshot();
        self.pixels = snap
            .pixels
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        self.latest_frame = snap
            .structured_data
            .and_then(|b| b.downcast::<OpticsFrameData>().ok().map(|d| *d));
        self.worker = BackgroundWorker::new(runner);
        self.stepped_pixels = None;
        self.step_runner = None;
        self.dirty = true;
        if was_running {
            let _ = self.worker.send_command(SimCommand::Start);
        }
    }

    fn refresh_initial_preview(&mut self) {
        let mut preview = OpticsWaveRunner::new(Arc::clone(&self.params), self.grid_size);
        preview.step();
        let snap = preview.get_snapshot();
        self.pixels = snap
            .pixels
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        self.latest_frame = snap
            .structured_data
            .and_then(|b| b.downcast::<OpticsFrameData>().ok().map(|d| *d));
    }

    pub(crate) fn snapshot_state(&self) -> OpticsWaveToolState {
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        OpticsWaveToolState {
            preset: p.preset,
            grid_size: self.grid_size,
            frequency: p.frequency,
            slit_separation: p.slit_separation,
            aperture_width: p.aperture_width,
            lens_radius: p.lens_radius,
            detector_x: p.detector_x,
            cfl_scale: p.cfl_scale,
            show_rays: p.show_rays,
            show_irradiance: p.show_irradiance,
            ray_count: p.ray_count,
            adc_bits: p.adc_bits,
            state: Some(self.worker.state()),
            paused: None,
            pending_initial_start: self.pending_initial_start,
        }
    }
}

impl FiniteStateGuard for OpticsWaveTool {
    fn is_finite_state(&self) -> bool {
        self.worker.state() != SimWorkerState::Faulted
            && self.worker.diagnostic_report().is_none()
            && self.has_finite_parameters()
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.worker.diagnostic_report()
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }

    fn auto_scale_step_size(&mut self) {
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.cfl_scale = (p.cfl_scale * 0.5).clamp(0.10, 1.00);
            p.force_non_finite = false;
        }
        self.reset_and_resume();
    }

    fn clamp_parameters(&mut self) {
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.frequency = if p.frequency.is_finite() {
                p.frequency.clamp(4.0, 24.0)
            } else {
                14.0
            };
            p.slit_separation = if p.slit_separation.is_finite() {
                p.slit_separation.clamp(0.08, 0.40)
            } else {
                0.18
            };
            p.aperture_width = if p.aperture_width.is_finite() {
                p.aperture_width.clamp(0.015, 0.14)
            } else {
                0.04
            };
            p.lens_radius = if p.lens_radius.is_finite() {
                p.lens_radius.clamp(0.10, 0.34)
            } else {
                0.22
            };
            p.detector_x = if p.detector_x.is_finite() {
                p.detector_x.clamp(0.45, 0.90)
            } else {
                0.68
            };
            p.cfl_scale = if p.cfl_scale.is_finite() {
                p.cfl_scale.clamp(0.10, 1.00)
            } else {
                0.95
            };
            p.ray_count = p.ray_count.clamp(3, 31);
            p.adc_bits = p.adc_bits.clamp(4, 16);
            p.force_non_finite = false;
        }
        self.grid_size = self.grid_size.clamp(64, 256);
        self.reset_and_resume();
    }

    fn reset_parameters(&mut self) {
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            *p = OpticsSharedParams::default();
        }
        self.grid_size = default_grid_size();
        self.reset_and_resume();
    }
}

impl InteractiveTool for OpticsWaveTool {
    fn name(&self) -> &'static str {
        "2D Wave & Caustics Playground"
    }

    fn domain(&self) -> &'static str {
        "optics_waves"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory
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
        if let Some(ref frame) = self.latest_frame {
            let (show_irradiance, high_contrast) = {
                let p = self.params.read().unwrap_or_else(|e| e.into_inner());
                (p.show_irradiance, p.high_contrast)
            };
            if frame.field.len() == self.grid_size * self.grid_size {
                return Some(render_wavefield_image(WaveRenderInput {
                    field: &frame.field,
                    irradiance: &frame.irradiance,
                    refractive_index: &frame.refractive_index,
                    obstacle: &frame.obstacle,
                    width: self.grid_size,
                    height: self.grid_size,
                    show_irradiance,
                    high_contrast,
                }));
            }
        }
        if self.pixels.len() == self.grid_size * self.grid_size && !self.pixels.is_empty() {
            return Some(ColorImage {
                size: [self.grid_size, self.grid_size],
                pixels: self.pixels.clone(),
                ..Default::default()
            });
        }
        None
    }

    fn step_simulation(&mut self) {
        let runner = self
            .step_runner
            .get_or_insert_with(|| OpticsWaveRunner::new(Arc::clone(&self.params), self.grid_size));
        for _ in 0..4 {
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

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.canvas_id
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            p.preset.hash(&mut h);
            p.frequency.to_bits().hash(&mut h);
            p.slit_separation.to_bits().hash(&mut h);
            p.aperture_width.to_bits().hash(&mut h);
            p.lens_radius.to_bits().hash(&mut h);
            p.detector_x.to_bits().hash(&mut h);
            p.cfl_scale.to_bits().hash(&mut h);
            p.show_rays.hash(&mut h);
            p.show_irradiance.hash(&mut h);
            p.ray_count.hash(&mut h);
            p.adc_bits.hash(&mut h);
        }
        self.grid_size.hash(&mut h);
        std::mem::discriminant(&self.worker.state()).hash(&mut h);
        self.pending_initial_start.hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.snapshot_state()).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: OpticsWaveToolState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid OpticsWaveTool state: {e}"))?;
        parsed.validate()?;
        let target_state = parsed.resolve_target_state();
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.preset = parsed.preset;
            p.frequency = parsed.frequency;
            p.slit_separation = parsed.slit_separation;
            p.aperture_width = parsed.aperture_width;
            p.lens_radius = parsed.lens_radius;
            p.detector_x = parsed.detector_x;
            p.cfl_scale = parsed.cfl_scale;
            p.show_rays = parsed.show_rays;
            p.show_irradiance = parsed.show_irradiance;
            p.ray_count = parsed.ray_count;
            p.adc_bits = parsed.adc_bits;
            p.force_non_finite = false;
        }
        if parsed.grid_size != self.grid_size {
            self.grid_size = parsed.grid_size;
            let runner = OpticsWaveRunner::new(Arc::clone(&self.params), self.grid_size);
            self.worker = BackgroundWorker::new(runner);
            self.refresh_initial_preview();
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

    fn has_finite_parameters(&self) -> bool {
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        !p.force_non_finite
            && p.frequency.is_finite()
            && p.slit_separation.is_finite()
            && p.aperture_width.is_finite()
            && p.lens_radius.is_finite()
            && p.detector_x.is_finite()
            && p.cfl_scale.is_finite()
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
        id: "OpticsWaveTool",
        display_name: "2D Wave & Caustics Playground",
        domain: "optics_waves",
        tags: &[
            "optics",
            "waves",
            "fdtd",
            "caustics",
            "luneburg",
            "veselago",
            "double-slit",
            "refraction",
            "diffraction",
        ],
        build: || Box::new(OpticsWaveTool::default()),
    }
}

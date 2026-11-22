use crate::accessibility::AccessibleTheoryHover;
use crate::async_sim::ui::{
    render_empty_plot_guidance, render_worker_fault_banner, WorkerFaultAction,
};
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::framework::{InteractiveTool, PlaybackAction, PlaybackTransport};
use eframe::egui;
use math_explorer::physics::stat_mech::ising::SpinLattice;
use std::sync::Arc;

#[path = "ising/export.rs"]
mod export;
#[path = "ising/runner.rs"]
mod runner;
pub use runner::*;

const DEFAULT_ISING_SIZE: usize = 100;

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct IsingState {
    temperature: f64,
    j_coupling: f64,
    h_field: f64,
    #[serde(default)]
    state: Option<SimWorkerState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    running: Option<bool>,
    steps_per_frame: usize,
}

pub struct IsingModelTool {
    temperature: f64,
    j_coupling: f64,
    h_field: f64,
    steps_per_frame: usize,
    magnetization: f64,
    energy: f64,
    texture: Option<egui::TextureHandle>,
    pixels: Vec<egui::Color32>,
    stepped_pixels: Option<(usize, usize, Vec<egui::Color32>)>,
    stepped_spins: Option<(usize, usize, Vec<i8>)>,
    last_theme: Option<crate::app::ThemeMode>,
    dirty: bool,
    params: Arc<std::sync::RwLock<IsingSharedParams>>,
    pub worker: BackgroundWorker<IsingRunner>,
}

impl IsingModelTool {
    #[cfg(test)]
    #[allow(missing_docs)]
    pub fn restart_worker(&mut self) {
        let runner = IsingRunner {
            lattice: SpinLattice::new(DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE, None),
            params: Arc::clone(&self.params),
        };
        self.worker = BackgroundWorker::new(runner);
        self.stepped_pixels = None;
        self.stepped_spins = None;
        self.sync_metrics_from_latest();
        self.dirty = true;
    }

    #[cfg(test)]
    #[allow(missing_docs)]
    pub fn retry(&mut self) {
        self.restart_worker();
    }

    /// Resets the worker lattice in place or respawns a disconnected worker.
    pub fn reset(&mut self) {
        let params = Arc::clone(&self.params);
        self.worker.reset_or_respawn(|| IsingRunner {
            lattice: SpinLattice::new(DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE, None),
            params,
        });
        self.stepped_pixels = None;
        self.stepped_spins = None;
        self.sync_metrics_from_latest();
        self.dirty = true;
    }

    fn sync_metrics_from_latest(&mut self) {
        if let Some(snapshot) = self.worker.latest_snapshot() {
            if snapshot.custom_data.len() >= 2 {
                self.magnetization = snapshot.custom_data[0];
                self.energy = snapshot.custom_data[1];
            }
        }
    }

    fn request_metric_refresh(&mut self, ctx: &egui::Context) {
        if self.worker.state().is_ready()
            && self
                .worker
                .send_command(SimCommand::Custom("recompute_metrics".to_string()))
                .is_ok()
        {
            ctx.request_repaint();
        }
    }

    fn draw_parameter_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if ui
            .add(
                egui::Slider::new(&mut self.temperature, 0.1..=10.0)
                    .text("Temperature (T) - K (relative)"),
            )
            .changed()
        {
            self.params
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .temperature = self.temperature;
        }
        ui.small("Critical Temp Tc ≈ 2.269 (for J=1)");

        if ui
            .add(
                egui::Slider::new(&mut self.j_coupling, -2.0..=2.0)
                    .text("Coupling Constant (J) - J"),
            )
            .changed()
        {
            self.params
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .j_coupling = self.j_coupling;
            self.request_metric_refresh(ctx);
        }
        ui.small("Positive: Ferromagnetic\nNegative: Antiferromagnetic");

        if ui
            .add(egui::Slider::new(&mut self.h_field, -2.0..=2.0).text("External Field (H) - H"))
            .changed()
        {
            self.params
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .h_field = self.h_field;
            self.request_metric_refresh(ctx);
        }

        ui.separator();
        if ui
            .add(
                egui::Slider::new(&mut self.steps_per_frame, 100..=50000)
                    .text("Simulation Speed (Steps/Frame)")
                    .logarithmic(true),
            )
            .changed()
        {
            self.params
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .steps_per_frame = self.steps_per_frame;
        }
    }

    fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui, ctx: &egui::Context) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "ising_controls");
        crate::framework::PaneSidePanel::left(panel_id)
            .resizable(false)
            .default_width(220.0)
            .show_inside(parent_ui, |ui| {
                ui.heading("Ising Model Controls");
                ui.separator();

                if let Some(WorkerFaultAction::Reset | WorkerFaultAction::ResetParameters) =
                    render_worker_fault_banner(
                        ui,
                        self.worker.state(),
                        self.worker.diagnostic_report(),
                        false,
                    )
                {
                    self.reset();
                    ctx.request_repaint();
                    ui.separator();
                }

                self.draw_parameter_controls(ui, ctx);
                ui.separator();

                if let Some(action) = PlaybackTransport::from_worker_state(
                    "ising_transport",
                    "Ising Model",
                    self.worker.state(),
                )
                .can_reset(true)
                .show(ui)
                {
                    match action {
                        PlaybackAction::Play => {
                            let _ = self.worker.send_command(SimCommand::Start);
                        }
                        PlaybackAction::Pause => {
                            let _ = self.worker.send_command(SimCommand::Pause);
                        }
                        PlaybackAction::Step => {
                            let _ = self.worker.send_command(SimCommand::Step(1));
                        }
                        PlaybackAction::Reset => {
                            self.reset();
                            ctx.request_repaint();
                        }
                    }
                }

                ui.separator();
                ui.label(format!("Magnetization: {:.3}", self.magnetization));
                ui.label(format!("Energy: {:.3}", self.energy));
            });
    }

    fn sync_lattice_texture(&mut self, ctx: &egui::Context) {
        if !self.dirty && self.texture.is_some() {
            return;
        }
        let Some(snapshot) = self.worker.latest_snapshot() else {
            return;
        };
        let Ok(snap_pixels) = snapshot.pixels.try_read() else {
            return;
        };
        if snap_pixels.is_empty() {
            return;
        }
        let num_pixels = snapshot.width * snapshot.height;
        if self.pixels.len() != num_pixels {
            self.pixels.resize(num_pixels, egui::Color32::BLACK);
        }
        let palette = crate::app::SemanticPalette::from_visuals(&ctx.style().visuals);
        if let Some(spins) = snapshot
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<Vec<i8>>())
            .filter(|s| s.len() == num_pixels)
        {
            for (dst, &s) in self.pixels.iter_mut().zip(spins.iter()) {
                *dst = ising_spin_color(s, &palette);
            }
        } else if snap_pixels.len() == num_pixels {
            self.pixels.copy_from_slice(&snap_pixels);
        }

        let image = egui::ColorImage {
            size: [snapshot.width, snapshot.height],
            pixels: self.pixels.clone(),
            ..Default::default()
        };
        let texture_opts = egui::TextureOptions::NEAREST;
        if let Some(texture) = &mut self.texture {
            texture.set(image, texture_opts);
        } else {
            self.texture = Some(ctx.load_texture("ising_lattice", image, texture_opts));
        }
        self.dirty = false;
    }

    fn draw_central_panel(&mut self, parent_ui: &mut egui::Ui, ctx: &egui::Context) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.sync_lattice_texture(ctx);

            if let Some(texture) = &self.texture {
                let alt_label = format!(
                    "2D Ising Model spin lattice visualization (T = {:.2}, M = {:.3}, E = {:.3})",
                    self.temperature, self.magnetization, self.energy
                );
                let response = ui
                    .add(
                        egui::Image::new(texture)
                            .alt_text(alt_label)
                            .sense(egui::Sense::focusable_noninteractive()),
                    )
                    .accessible_theory_hover(self);
                ui.painter().rect_stroke(
                    response.rect,
                    0.0,
                    ui.visuals().widgets.noninteractive.bg_stroke,
                    egui::StrokeKind::Inside,
                );
                crate::accessibility::paint_canvas_focus_ring(ui, &response);
            } else {
                render_empty_plot_guidance(
                    ui,
                    "Spin lattice ready — click ▶ Play or ⏭ Step to run Metropolis spin flips.",
                );
            }
        });
    }
}

impl Default for IsingModelTool {
    fn default() -> Self {
        let temperature = 2.269;
        let j_coupling = 1.0;
        let h_field = 0.0;
        let steps_per_frame = 5000;
        let params = Arc::new(std::sync::RwLock::new(IsingSharedParams {
            temperature,
            j_coupling,
            h_field,
            steps_per_frame,
            theme: crate::app::ThemeMode::Dark,
        }));
        let runner = IsingRunner {
            lattice: SpinLattice::new(DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE, None),
            params: Arc::clone(&params),
        };
        let worker = BackgroundWorker::new(runner);
        let mut tool = Self {
            temperature,
            j_coupling,
            h_field,
            steps_per_frame,
            magnetization: 0.0,
            energy: 0.0,
            texture: None,
            pixels: Vec::new(),
            stepped_pixels: None,
            stepped_spins: None,
            last_theme: None,
            dirty: true,
            params,
            worker,
        };
        tool.sync_metrics_from_latest();
        tool
    }
}

impl InteractiveTool for IsingModelTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Ising Model"
    }
    fn domain(&self) -> &'static str {
        "solid_state"
    }

    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::discrete_lattice()
    }

    fn export_raster_image(&self) -> Option<egui::ColorImage> {
        self.build_ising_raster_image()
    }

    fn export_scalar_field(
        &self,
    ) -> Option<math_commons::math_kernel::marching_squares::ScalarField2D> {
        self.build_ising_scalar_field()
    }

    fn export_vector_svg(
        &self,
        options: &egui_plot::export::ExportOptions,
        _contour_levels: usize,
    ) -> Option<Result<String, String>> {
        Some(self.build_ising_vector_svg(options))
    }

    fn step_simulation(&mut self) {
        self.step_ising_offscreen();
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&IsingState {
            temperature: self.temperature,
            j_coupling: self.j_coupling,
            h_field: self.h_field,
            state: Some(self.worker.state()),
            running: None,
            steps_per_frame: self.steps_per_frame,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: IsingState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Ising state: {e}"))?;
        if !s.temperature.is_finite()
            || !(0.1..=10.0).contains(&s.temperature)
            || !s.j_coupling.is_finite()
            || !(-2.0..=2.0).contains(&s.j_coupling)
            || !s.h_field.is_finite()
            || !(-2.0..=2.0).contains(&s.h_field)
            || !(100..=50_000).contains(&s.steps_per_frame)
        {
            return Err("Ising parameters out of valid bounds".to_string());
        }
        self.temperature = s.temperature;
        self.j_coupling = s.j_coupling;
        self.h_field = s.h_field;
        self.steps_per_frame = s.steps_per_frame;
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.temperature = self.temperature;
            p.j_coupling = self.j_coupling;
            p.h_field = self.h_field;
            p.steps_per_frame = self.steps_per_frame;
        }
        let target_state = s.state.unwrap_or_else(|| {
            if s.running.unwrap_or(false) {
                SimWorkerState::Running
            } else {
                SimWorkerState::Ready
            }
        });
        if target_state == SimWorkerState::Running {
            if !self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Start);
            }
        } else if self.worker.is_running() {
            let _ = self.worker.send_command(SimCommand::Pause);
        }
        self.stepped_pixels = None;
        self.stepped_spins = None;
        self.dirty = true;
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let theme = crate::app::ThemeMode::from_visuals(ui.visuals());
        if self.last_theme != Some(theme) {
            self.last_theme = Some(theme);
            self.params.write().unwrap_or_else(|e| e.into_inner()).theme = theme;
            self.dirty = true;
        }
        if let Some(snapshot) = self.worker.update() {
            if snapshot.custom_data.len() >= 2 {
                self.magnetization = snapshot.custom_data[0];
                self.energy = snapshot.custom_data[1];
            }
            self.stepped_pixels = None;
            self.stepped_spins = None;
            self.dirty = true;
        }

        if matches!(
            self.worker.state(),
            SimWorkerState::Running | SimWorkerState::Stepping
        ) {
            ctx.request_repaint();
        }

        self.draw_side_panel(ui, &ctx);
        self.draw_central_panel(ui, &ctx);
    }
}

#[path = "ising/preset.rs"]
mod preset;

#[cfg(test)]
use crate::async_sim::SimulationRunner;
#[cfg(test)]
#[path = "ising/tests.rs"]
mod tests;

// [cite:clinical_trials_statistics]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "IsingModelTool",
        display_name: "Ising Model",
        domain: "solid_state",
        tags: &[],
        build: || Box::new(IsingModelTool::default()),
    }
}

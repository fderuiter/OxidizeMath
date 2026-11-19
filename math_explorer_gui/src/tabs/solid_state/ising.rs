use crate::accessibility::AccessibleHoverText;
use crate::async_sim::ui::{
    render_empty_plot_guidance, render_worker_fault_banner, WorkerFaultAction,
};
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::framework::InteractiveTool;
use eframe::egui;
use math_explorer::physics::stat_mech::ising::SpinLattice;
use std::sync::Arc;

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
                egui::Slider::new(&mut self.j_coupling, -2.0..=2.0).text("Coupling Constant (J) - J"),
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

    fn draw_side_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("ising_controls")
            .resizable(false)
            .default_width(200.0)
            .show(ctx, |ui| {
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

                ui.horizontal(|ui| {
                    let is_running = self.worker.state().is_running();
                    let can_toggle = matches!(
                        self.worker.state(),
                        SimWorkerState::Ready | SimWorkerState::Running
                    );
                    if ui
                        .add_enabled(
                            can_toggle,
                            egui::Button::new(if is_running { "⏸ Pause" } else { "▶ Play" }),
                        )
                        .accessible_hover_text(if is_running {
                            "Pause the Ising model simulation"
                        } else {
                            "Resume the Ising model simulation"
                        })
                        .clicked()
                    {
                        let cmd = if is_running {
                            SimCommand::Pause
                        } else {
                            SimCommand::Start
                        };
                        let _ = self.worker.send_command(cmd);
                    }

                    if ui
                        .button("🔄 Reset")
                        .accessible_hover_text("Reset the lattice spin state")
                        .clicked()
                    {
                        self.reset();
                        ctx.request_repaint();
                    }
                });

                ui.separator();
                ui.label(format!("Magnetization: {:.3}", self.magnetization));
                ui.label(format!("Energy: {:.3}", self.energy));
            });
    }

    fn draw_central_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.dirty || self.texture.is_none() {
                if let Some(snapshot) = self.worker.latest_snapshot() {
                    if let Ok(snap_pixels) = snapshot.pixels.try_read() {
                        if !snap_pixels.is_empty() {
                            let num_pixels = snapshot.width * snapshot.height;
                            if self.pixels.len() != num_pixels {
                                self.pixels.resize(num_pixels, egui::Color32::BLACK);
                            }
                            if snap_pixels.len() == num_pixels {
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
                                self.texture = Some(ctx.load_texture(
                                    "ising_lattice",
                                    image,
                                    texture_opts,
                                ));
                            }
                            self.dirty = false;
                        }
                    }
                }
            }

            if let Some(texture) = &self.texture {
                ui.image(texture);
            } else {
                render_empty_plot_guidance(
                    ui,
                    "Spin lattice ready — click ▶ Play to run Metropolis spin flips.",
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
        self.dirty = true;
        Ok(())
    }

    fn show(&mut self, ctx: &egui::Context) {
        if let Some(snapshot) = self.worker.update() {
            if snapshot.custom_data.len() >= 2 {
                self.magnetization = snapshot.custom_data[0];
                self.energy = snapshot.custom_data[1];
            }
            self.dirty = true;
        }

        if matches!(
            self.worker.state(),
            SimWorkerState::Running | SimWorkerState::Stepping
        ) {
            ctx.request_repaint();
        }

        self.draw_side_panel(ctx);
        self.draw_central_panel(ctx);
    }
}

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

impl scientific_metadata::theory::TheoryDescribable for IsingModelTool {
    fn theory_description(&self) -> String {
        "Two-dimensional square lattice ferromagnetic Ising model governed by Hamiltonian $H = -J \\sum_{\\langle i, j \\rangle} s_i s_j - h \\sum_i s_i$ with Onsager critical temperature $T_c = \\frac{2J}{k_B \\ln(1 + \\sqrt{2})} \\approx 2.269 \\frac{J}{k_B}$.".into()
    }
    fn phonetic_description(&self) -> String {
        "Two-dimensional Ising model ferromagnetic phase transition theory".into()
    }
    fn theory_citation(&self) -> String {
        "Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition (Onsager, 1944)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "2D Ising Model Ferromagnetism",
    "domain": "solid_state",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "temperature": 2.269,
    "j_coupling": 1.0,
    "h_field": 0.05,
    "steps_per_frame": 10000
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "ising_model_starter",
        title: "2D Ising Model Ferromagnetism",
        description: "H = -J ∑ s_i·s_j - h ∑ s_i: Monte Carlo Metropolis spin flip dynamics and Curie transition.",
        category: "Statistical Mechanics",
        icon: "🧲",
        domain: "solid_state",
        tool_name: Some("IsingModelTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

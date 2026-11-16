use crate::accessibility::AccessibleHoverText;
use crate::async_sim::{
    BackgroundWorker, SimCommand, SimWorkerState, SimulationRunner, StateSnapshot,
};
use crate::framework::InteractiveTool;
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::stat_mech::ising::SpinLattice;
use math_explorer::physics::stat_mech::KB;
use std::sync::Arc;

#[allow(missing_docs)]
pub struct IsingSharedParams {
    pub temperature: f64,
    pub j_coupling: f64,
    pub h_field: f64,
    pub steps_per_frame: usize,
}

#[allow(missing_docs)]
pub struct IsingRunner {
    pub lattice: SpinLattice,
    pub params: Arc<std::sync::RwLock<IsingSharedParams>>,
}

impl SimulationRunner for IsingRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::Reset => {
                self.lattice = SpinLattice::new(100, 100, None);
            }
            SimCommand::SetSpeed(speed) => {
                if let Ok(mut p) = self.params.write() {
                    p.steps_per_frame = speed;
                }
            }
            _ => {}
        }
    }

    fn step(&mut self) {
        let (temp, j, h, steps) = {
            let p = self.params.read().unwrap();
            (p.temperature, p.j_coupling, p.h_field, p.steps_per_frame)
        };
        let j_scale = if j.abs() < 1e-6 { 1.0 } else { j.abs() };
        let t_phys = temp * j_scale / KB;
        self.lattice.evolve(steps, t_phys, j, h);
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let width = self.lattice.width();
        let height = self.lattice.height();
        let spins_vec: Vec<i8> = self.lattice.spins().to_vec();
        let mut pixels = Vec::with_capacity(width * height);
        for &s in &spins_vec {
            pixels.push(if s > 0 {
                egui::Color32::from_rgb(200, 50, 50)
            } else {
                egui::Color32::from_rgb(50, 50, 200)
            });
        }
        let (j, h) = {
            let p = self.params.read().unwrap();
            (p.j_coupling, p.h_field)
        };
        let m = self.lattice.magnetization();
        let max_m = (width * height) as f64;
        let m_avg = m as f64 / max_m;
        let energy = self.lattice.hamiltonian(j, h);

        StateSnapshot {
            width,
            height,
            pixels: Arc::new(std::sync::RwLock::new(pixels)),
            custom_data: vec![m_avg, energy],
            structured_data: Some(Box::new(spins_vec)),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
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
    lattice: SpinLattice,
    temperature: f64,
    j_coupling: f64,
    h_field: f64,
    steps_per_frame: usize,
    texture: Option<egui::TextureHandle>,
    pixels: Vec<egui::Color32>,
    dirty: bool,
    params: Arc<std::sync::RwLock<IsingSharedParams>>,
    pub worker: BackgroundWorker<IsingRunner>,
}

impl IsingModelTool {
    #[allow(missing_docs)]
    pub fn restart_worker(&mut self) {
        let w = self.lattice.width();
        let h = self.lattice.height();
        let mut lattice = SpinLattice::new(w, h, None);
        for y in 0..h {
            for x in 0..w {
                lattice.set(x, y, self.lattice.get(x, y));
            }
        }
        let runner = IsingRunner {
            lattice,
            params: Arc::clone(&self.params),
        };
        self.worker = BackgroundWorker::new(runner);
        self.dirty = true;
    }

    #[allow(missing_docs)]
    pub fn retry(&mut self) {
        self.restart_worker();
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
            lattice: SpinLattice::new(100, 100, None),
            params: Arc::clone(&params),
        };
        let worker = BackgroundWorker::new(runner);
        Self {
            lattice: SpinLattice::new(100, 100, None),
            temperature,
            j_coupling,
            h_field,
            steps_per_frame,
            texture: None,
            pixels: Vec::new(),
            dirty: true,
            params,
            worker,
        }
    }
}

impl InteractiveTool for IsingModelTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
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
        }).ok()
    }

    fn load_state(&mut self, state: &str) {
        if let Ok(s) = serde_json::from_str::<IsingState>(state) {
            self.temperature = s.temperature;
            self.j_coupling = s.j_coupling;
            self.h_field = s.h_field;
            self.steps_per_frame = s.steps_per_frame;
            if let Ok(mut p) = self.params.write() {
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
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        // Controls
        egui::SidePanel::left("ising_controls")
            .resizable(false)
            .default_width(200.0)
            .show(ctx, |ui| {
                ui.heading("Ising Model Controls");
                ui.separator();

                if self.worker.state() == SimWorkerState::Faulted {
                    ui.group(|ui| {
                        ui.colored_label(
                            egui::Color32::RED,
                            "⚠ Simulation worker faulted. Channel disconnected or worker crashed.",
                        );
                        if ui
                            .button("🔄 Retry")
                            .accessible_hover_text("Restart the simulation worker")
                            .clicked()
                        {
                            self.retry();
                        }
                    });
                    ui.separator();
                }

                if ui
                    .add(egui::Slider::new(&mut self.temperature, 0.1..=10.0).text("Temperature (T) - K (relative)"))
                    .changed()
                {
                    self.params.write().unwrap().temperature = self.temperature;
                    self.dirty = true;
                }
                ui.small("Critical Temp Tc ≈ 2.269 (for J=1)");

                if ui
                    .add(egui::Slider::new(&mut self.j_coupling, -2.0..=2.0).text("Coupling Constant (J) - J"))
                    .changed()
                {
                    self.params.write().unwrap().j_coupling = self.j_coupling;
                    self.dirty = true;
                }
                ui.small("Positive: Ferromagnetic\nNegative: Antiferromagnetic");

                if ui
                    .add(egui::Slider::new(&mut self.h_field, -2.0..=2.0).text("External Field (H) - H"))
                    .changed()
                {
                    self.params.write().unwrap().h_field = self.h_field;
                    self.dirty = true;
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
                    self.params.write().unwrap().steps_per_frame = self.steps_per_frame;
                    self.dirty = true;
                }

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
                        if is_running {
                            let _ = self.worker.send_command(SimCommand::Pause);
                        } else {
                            let _ = self.worker.send_command(SimCommand::Start);
                        }
                    }

                    if ui
                        .button("🔄 Reset")
                        .accessible_hover_text("Reset the lattice spin state")
                        .clicked()
                    {
                        self.lattice = SpinLattice::new(100, 100, None);
                        if self.worker.state() == SimWorkerState::Faulted
                            || self.worker.state() == SimWorkerState::Terminated
                        {
                            self.restart_worker();
                        } else {
                            let _ = self.worker.send_command(SimCommand::Reset);
                        }
                        self.dirty = true;
                    }
                });

                ui.separator();

                let m = self.lattice.magnetization();
                let max_m = (self.lattice.width() * self.lattice.height()) as f64;
                let m_avg = m as f64 / max_m;
                ui.label(format!("Magnetization: {:.3}", m_avg));

                let energy = self.lattice.hamiltonian(self.j_coupling, self.h_field);
                ui.label(format!("Energy: {:.3}", energy));
            });

        // Simulation Update via Background Worker
        if let Some(snapshot) = self.worker.update() {
            if let Some(spins) = snapshot
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<Vec<i8>>())
            {
                let w = self.lattice.width();
                let h = self.lattice.height();
                for y in 0..h {
                    for x in 0..w {
                        self.lattice.set(x, y, spins[flatten_2d_index(x, y, w)]);
                    }
                }
            }
            self.dirty = true;
        }

        if matches!(
            self.worker.state(),
            SimWorkerState::Running | SimWorkerState::Stepping
        ) {
            ctx.request_repaint();
        }

        // Visualization
        egui::CentralPanel::default().show(ctx, |ui| {
            let width = self.lattice.width();
            let height = self.lattice.height();
            let num_pixels = width * height;

            if self.dirty || self.texture.is_none() {
                if self.pixels.len() != num_pixels {
                    self.pixels.resize(num_pixels, egui::Color32::BLACK);
                }

                for (pixel, &s) in self.pixels.iter_mut().zip(self.lattice.spins().iter()) {
                    *pixel = if s > 0 {
                        egui::Color32::from_rgb(200, 50, 50) // Red (Up)
                    } else {
                        egui::Color32::from_rgb(50, 50, 200) // Blue (Down)
                    };
                }

                let image = egui::ColorImage {
                    size: [width, height],
                    pixels: self.pixels.clone(),
                    ..Default::default()
                };

                let texture_opts = egui::TextureOptions::NEAREST; // Pixelated look
                if let Some(texture) = &mut self.texture {
                    texture.set(image, texture_opts);
                } else {
                    self.texture = Some(ctx.load_texture("ising_lattice", image, texture_opts));
                }

                self.dirty = false;
            }

            if let Some(texture) = &self.texture {
                ui.image(texture); // Show the texture
            }
        });
    }
}

#[cfg(test)]
#[path = "ising/tests.rs"]
mod tests;

// [cite:clinical_trials_statistics]


inventory::submit! {
    crate::framework::ToolMetadata {
        name: "IsingModelTool",
        domain: "solid_state",
        tags: &[],
        build: || Box::new(IsingModelTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for IsingModelTool {
    fn theory_description(&self) -> String { "Theoretical context not available.".into() }
    fn phonetic_description(&self) -> String { "Theoretical context not available.".into() }
    fn theory_citation(&self) -> String { "Uncited".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}

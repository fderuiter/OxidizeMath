use crate::accessibility::AccessibleTheoryHover;
use crate::async_sim::ui::{
    render_empty_plot_guidance, render_worker_fault_banner, WorkerFaultAction,
};
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::framework::{InteractiveTool, PlaybackAction, PlaybackTransport};
use eframe::egui;
use math_explorer::physics::stat_mech::ising::SpinLattice;
use math_explorer::physics::stat_mech::KB;
use std::sync::Arc;

#[path = "ising/graph_mode.rs"]
mod graph_mode;
#[path = "ising/runner.rs"]
mod runner;
pub use runner::*;

const DEFAULT_ISING_SIZE: usize = 100;

const fn default_true() -> bool {
    true
}

fn is_true(v: &bool) -> bool {
    *v
}

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
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    follow_workspace_graph: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seeded_graph: Option<crate::app::data_bus::SharedGraph>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    frozen_graph: Option<crate::app::data_bus::SharedGraph>,
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
    last_theme: Option<crate::app::ThemeMode>,
    dirty: bool,
    params: Arc<std::sync::RwLock<IsingSharedParams>>,
    pub worker: BackgroundWorker<IsingRunner>,
    graph_mode: graph_mode::GraphSpinMode,
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
        if self.reset_graph_spins() {
            return;
        }
        let params = Arc::clone(&self.params);
        self.worker.reset_or_respawn(|| IsingRunner {
            lattice: SpinLattice::new(DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE, None),
            params,
        });
        self.stepped_pixels = None;
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

    pub(super) fn request_metric_refresh(&mut self, ctx: &egui::Context) {
        if self.worker.state().is_ready()
            && self
                .worker
                .send_command(SimCommand::Custom("recompute_metrics".to_string()))
                .is_ok()
        {
            ctx.request_repaint();
        }
    }

    fn handle_transport_action(&mut self, action: PlaybackAction, ctx: &egui::Context) {
        match action {
            PlaybackAction::Play => {
                let _ = self.worker.send_command(SimCommand::Start);
            }
            PlaybackAction::Pause => {
                let _ = self.worker.send_command(SimCommand::Pause);
            }
            PlaybackAction::Step => {
                if !self.step_graph_spins(1) {
                    let _ = self.worker.send_command(SimCommand::Step(1));
                }
            }
            PlaybackAction::Reset => {
                self.reset();
                ctx.request_repaint();
            }
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
                self.render_graph_spin_controls(ui, ctx);
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
                    self.handle_transport_action(action, ctx);
                }

                ui.separator();
                ui.label(format!("Magnetization: {:.3}", self.magnetization));
                ui.label(format!("Energy: {:.3}", self.energy));
                if self.is_graph_spin_mode() {
                    ui.label(format!("Frustrated Edges: {}", self.frustrated_edge_count()));
                }
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
            if self.render_graph_spin_canvas(ui) {
                return;
            }
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
            last_theme: None,
            dirty: true,
            params,
            worker,
            graph_mode: graph_mode::GraphSpinMode::default(),
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
        crate::framework::ExportCapabilities::raster_only()
    }

    fn export_raster_image(&self) -> Option<egui::ColorImage> {
        if let Some((w, h, ref px)) = self.stepped_pixels {
            if w > 0 && h > 0 && !px.is_empty() {
                return Some(egui::ColorImage::new([w, h], px.clone()));
            }
        }
        let snap = self.worker.latest_snapshot()?;
        let px = snap.pixels.try_read().ok()?;
        if px.is_empty() || snap.width == 0 || snap.height == 0 {
            return None;
        }
        Some(egui::ColorImage::new([snap.width, snap.height], px.clone()))
    }

    fn step_simulation(&mut self) {
        let sweeps = (self.steps_per_frame / 500).clamp(1, 32);
        if self.step_graph_spins(sweeps) {
            return;
        }
        let mut lattice = SpinLattice::new(DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE, None);
        let j_scale = if self.j_coupling.abs() < 1e-6 {
            1.0
        } else {
            self.j_coupling.abs()
        };
        let t_phys = self.temperature * j_scale / KB;
        lattice.evolve(
            self.steps_per_frame.clamp(100, 5000),
            t_phys,
            self.j_coupling,
            self.h_field,
        );
        let palette = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .theme
            .palette();
        let pixels: Vec<egui::Color32> = lattice
            .spins()
            .iter()
            .map(|&s| ising_spin_color(s, &palette))
            .collect();
        self.stepped_pixels = Some((lattice.width(), lattice.height(), pixels));
        self.dirty = true;
    }

    fn save_state(&self) -> Option<String> {
        let frozen_graph = (!self.graph_mode.follow_workspace_graph)
            .then(|| self.graph_mode.active_graph.as_ref().map(|g| (**g).clone()))
            .flatten();
        serde_json::to_string(&IsingState {
            temperature: self.temperature,
            j_coupling: self.j_coupling,
            h_field: self.h_field,
            state: Some(self.worker.state()),
            running: None,
            steps_per_frame: self.steps_per_frame,
            follow_workspace_graph: self.graph_mode.follow_workspace_graph,
            seeded_graph: self.graph_mode.seeded_graph.clone(),
            frozen_graph,
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
        for g in [&s.seeded_graph, &s.frozen_graph].into_iter().flatten() {
            g.validate(
                crate::app::data_bus::SESSION_MAX_BUS_NODES,
                crate::app::data_bus::SESSION_MAX_BUS_EDGES,
            )?;
        }
        self.temperature = s.temperature;
        self.j_coupling = s.j_coupling;
        self.h_field = s.h_field;
        self.steps_per_frame = s.steps_per_frame;
        self.graph_mode.follow_workspace_graph = s.follow_workspace_graph;
        self.graph_mode.seeded_graph = s.seeded_graph.clone();
        self.graph_mode.pending_seed_publish = s.seeded_graph.is_some();
        if let Some(g) = s.frozen_graph.or(s.seeded_graph) {
            self.apply_shared_spin_graph(&Arc::new(g));
        } else if self.graph_mode.follow_workspace_graph {
            self.graph_mode.system = None;
            self.graph_mode.active_graph = None;
            self.graph_mode.applied_bus_revision = 0;
        }
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.temperature = self.temperature;
            p.j_coupling = self.j_coupling;
            p.h_field = self.h_field;
            p.steps_per_frame = self.steps_per_frame;
        }
        self.sync_worker_target_state(s.state, s.running);
        self.stepped_pixels = None;
        self.dirty = true;
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.sync_graph_from_bus(&ctx);
        let theme = crate::app::ThemeMode::from_visuals(ui.visuals());
        if self.last_theme != Some(theme) {
            self.last_theme = Some(theme);
            self.params.write().unwrap_or_else(|e| e.into_inner()).theme = theme;
            self.dirty = true;
        }
        let snap_metrics = self
            .worker
            .update()
            .map(|s| (s.custom_data.first().copied(), s.custom_data.get(1).copied()));
        if self.is_graph_spin_mode() {
            if self.worker.state() == SimWorkerState::Running {
                let sweeps = (self.steps_per_frame / 500).clamp(1, 32);
                let _ = self.step_graph_spins(sweeps);
            }
        } else if let Some((m_opt, e_opt)) = snap_metrics {
            if let (Some(m), Some(e)) = (m_opt, e_opt) {
                self.magnetization = m;
                self.energy = e;
            }
            self.stepped_pixels = None;
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

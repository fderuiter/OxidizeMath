use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::framework::{InteractionContext, InteractiveTool};
use eframe::egui;
use math_explorer::applied::game_theory::evolutionary::MatrixPayoff;
use math_explorer::physics::stat_mech::ising::SpinLattice;
use math_explorer::physics::stat_mech::KB;
use nalgebra::DMatrix;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[path = "ising/graph_mode.rs"]
mod graph_mode;
#[path = "ising/overlay.rs"]
mod overlay;
#[path = "ising/preset.rs"]
mod preset;
#[path = "ising/runner.rs"]
mod runner;
#[path = "ising/ui.rs"]
mod ui;

pub use preset::{FermiGamePreset, IsingInteractionMode};
pub use runner::*;

const DEFAULT_ISING_SIZE: usize = 100;
const MAX_SPATIAL_HISTORY: usize = 512;

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
    mode: IsingInteractionMode,
    #[serde(default)]
    game_preset: FermiGamePreset,
    #[serde(default = "preset::default_game_payoffs")]
    game_payoffs: [f64; 4],
    #[serde(default = "preset::default_brush_radius")]
    brush_radius: i32,
    #[serde(default = "preset::default_brush_cooperator")]
    brush_cooperator: bool,
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

impl IsingState {
    fn validate(&self) -> Result<(), String> {
        let valid = self.temperature.is_finite()
            && (1e-3..=10.0).contains(&self.temperature)
            && self.j_coupling.is_finite()
            && (-2.0..=2.0).contains(&self.j_coupling)
            && self.h_field.is_finite()
            && (-2.0..=2.0).contains(&self.h_field)
            && self
                .game_payoffs
                .iter()
                .all(|v| v.is_finite() && (-20.0..=20.0).contains(v))
            && (0..=99).contains(&self.brush_radius)
            && (100..=50_000).contains(&self.steps_per_frame);
        if valid {
            Ok(())
        } else {
            Err("Ising parameters out of valid bounds".to_string())
        }
    }
}

pub struct IsingModelTool {
    temperature: f64,
    j_coupling: f64,
    h_field: f64,
    mode: IsingInteractionMode,
    game_preset: FermiGamePreset,
    game_payoffs: [f64; 4],
    brush_radius: i32,
    brush_cooperator: bool,
    steps_per_frame: usize,
    magnetization: f64,
    energy: f64,
    spatial_history: VecDeque<[f64; 2]>,
    replicator_curve: Vec<[f64; 2]>,
    ode_anchor_sweep: f64,
    ode_initial_cooperator: f64,
    pending_ode_resync: bool,
    last_canvas_rect: Option<egui::Rect>,
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
        self.params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .total_trials
            .store(0, Ordering::Relaxed);
        let runner = IsingRunner {
            lattice: SpinLattice::new(DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE, None),
            params: Arc::clone(&self.params),
        };
        self.worker = BackgroundWorker::new(runner);
        self.stepped_pixels = None;
        self.sync_metrics_from_latest();
        self.reset_trajectory_overlays();
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
        self.params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .total_trials
            .store(0, Ordering::Relaxed);
        let params = Arc::clone(&self.params);
        self.worker.reset_or_respawn(|| IsingRunner {
            lattice: SpinLattice::new(DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE, None),
            params,
        });
        self.stepped_pixels = None;
        self.pending_ode_resync = true;
        self.sync_metrics_from_latest();
        self.reset_trajectory_overlays();
        self.dirty = true;
    }
}

impl Default for IsingModelTool {
    fn default() -> Self {
        let (temperature, j_coupling, h_field, steps_per_frame) = (2.269, 1.0, 0.0, 5000);
        let (mode, game_preset) = (
            IsingInteractionMode::Ferromagnetic,
            FermiGamePreset::default(),
        );
        let game_payoffs = preset::default_game_payoffs();
        let params = Arc::new(std::sync::RwLock::new(IsingSharedParams {
            temperature,
            j_coupling,
            h_field,
            mode,
            game_payoffs,
            steps_per_frame,
            theme: crate::app::ThemeMode::Dark,
            total_trials: AtomicU64::new(0),
        }));
        let worker = BackgroundWorker::new(IsingRunner {
            lattice: SpinLattice::new(DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE, None),
            params: Arc::clone(&params),
        });
        let mut tool = Self {
            temperature,
            j_coupling,
            h_field,
            mode,
            game_preset,
            game_payoffs,
            brush_radius: preset::default_brush_radius(),
            brush_cooperator: preset::default_brush_cooperator(),
            steps_per_frame,
            magnetization: 0.0,
            energy: 0.0,
            spatial_history: VecDeque::with_capacity(MAX_SPATIAL_HISTORY),
            replicator_curve: Vec::new(),
            ode_anchor_sweep: 0.0,
            ode_initial_cooperator: 0.5,
            pending_ode_resync: false,
            last_canvas_rect: None,
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
        tool.reset_trajectory_overlays();
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

    fn on_brush(&mut self, ctx: &InteractionContext) {
        let Some(pos) = ctx.pointer_pos else {
            return;
        };
        let rect = ctx.response.rect;
        if rect.width() <= 0.0 || rect.height() <= 0.0 {
            return;
        }
        let (w, h) = self
            .worker
            .latest_snapshot()
            .map_or((DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE), |s| {
                (s.width.max(1), s.height.max(1))
            });
        let (max_x, max_y) = ((w as i32) - 1, (h as i32) - 1);
        let rel_x = ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 1.0);
        let rel_y = ((pos.y - rect.min.y) / rect.height()).clamp(0.0, 1.0);
        let cx = ((rel_x * (max_x as f32)).round() as i32).clamp(0, max_x);
        let cy = ((rel_y * (max_y as f32)).round() as i32).clamp(0, max_y);
        let r = self.brush_radius.clamp(0, max_x.max(max_y));
        let invert = ctx.modifiers.shift
            || ctx.response.secondary_clicked()
            || ctx.response.dragged_by(egui::PointerButton::Secondary);
        let is_obstacle = if invert {
            !self.brush_cooperator
        } else {
            self.brush_cooperator
        };

        if self
            .worker
            .send_command(SimCommand::ApplyBrush {
                cx,
                cy,
                r,
                is_obstacle,
            })
            .is_ok()
        {
            if !self.worker.is_running() {
                self.pending_ode_resync = true;
            }
            ctx.response.ctx.request_repaint();
        }
    }

    fn step_simulation(&mut self) {
        let sweeps = (self.steps_per_frame / 500).clamp(1, 32);
        if self.step_graph_spins(sweeps) {
            return;
        }
        let mut lattice = SpinLattice::new(DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE, None);
        let steps = self.steps_per_frame.clamp(100, 5000);
        match self.mode {
            IsingInteractionMode::Ferromagnetic => {
                let j_scale = if self.j_coupling.abs() < 1e-6 {
                    1.0
                } else {
                    self.j_coupling.abs()
                };
                let t_phys = self.temperature.max(1e-3) * j_scale / KB;
                lattice.evolve(steps, t_phys, self.j_coupling, self.h_field);
            }
            IsingInteractionMode::FermiGame => {
                let mat = DMatrix::from_row_slice(2, 2, &self.game_payoffs);
                if let Ok(payoff) = MatrixPayoff::new(mat) {
                    lattice.evolve_fermi_game(steps, self.temperature, &payoff);
                }
            }
        }
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
            mode: self.mode,
            game_preset: self.game_preset,
            game_payoffs: self.game_payoffs,
            brush_radius: self.brush_radius,
            brush_cooperator: self.brush_cooperator,
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
        s.validate()?;
        for g in [&s.seeded_graph, &s.frozen_graph].into_iter().flatten() {
            g.validate(
                crate::app::data_bus::SESSION_MAX_BUS_NODES,
                crate::app::data_bus::SESSION_MAX_BUS_EDGES,
            )?;
        }
        self.temperature = s.temperature;
        self.j_coupling = s.j_coupling;
        self.h_field = s.h_field;
        self.mode = s.mode;
        self.game_preset = s.game_preset;
        self.game_payoffs = s.game_payoffs;
        self.brush_radius = s.brush_radius;
        self.brush_cooperator = s.brush_cooperator;
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
        self.sync_shared_params();
        self.recompute_replicator_curve();
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
        let snap_metrics = self.worker.update().map(|s| {
            (
                s.custom_data.first().copied(),
                s.custom_data.get(1).copied(),
            )
        });
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
            self.record_snapshot_progress();
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

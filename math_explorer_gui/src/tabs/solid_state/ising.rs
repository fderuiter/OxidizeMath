use crate::async_sim::atlas::ParameterAtlasController;
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::framework::{ChallengeEvaluation, InteractionContext, InteractiveTool};
use eframe::egui;
use math_explorer::applied::game_theory::evolutionary::MatrixPayoff;
use math_explorer::physics::stat_mech::ising::SpinLattice;
use math_explorer::physics::stat_mech::KB;
use nalgebra::DMatrix;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[path = "ising/atlas.rs"]
pub mod atlas;
#[path = "ising/challenge.rs"]
pub mod challenge;
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

use atlas::ISING_ATLAS_AXIS_PAIRS;
use challenge::IsingChallengeWorkspace;
pub(crate) use preset::IsingState;
pub use preset::{FermiGamePreset, IsingInteractionMode};
pub use runner::*;

const DEFAULT_ISING_SIZE: usize = 100;
const MAX_SPATIAL_HISTORY: usize = 512;

pub struct IsingModelTool {
    temperature: f64,
    j_coupling: f64,
    h_field: f64,
    mutation_rate: f64,
    mode: IsingInteractionMode,
    game_preset: FermiGamePreset,
    game_payoffs: [f64; 4],
    brush_radius: i32,
    brush_cooperator: bool,
    defector_sea: bool,
    droplet_seeds: Vec<DropletSeed>,
    steps_per_frame: usize,
    magnetization: f64,
    uniform_magnetization: f64,
    staggered_magnetization: f64,
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
    pub topology_source: crate::app::TopologySource,
    pub(super) graph_system: Option<math_explorer::physics::stat_mech::GraphSpinSystem>,
    pub(super) shared_topology: Option<Arc<crate::app::SharedGraphTopology>>,
    pub(super) synced_bus_revision: Option<u64>,
    pub(super) graph_running: bool,
    pub(super) graph_sweeps: f64,
    pub(crate) atlas: ParameterAtlasController,
    pub(crate) active_challenge: Option<&'static str>,
    pub(crate) challenge_workspace: IsingChallengeWorkspace,
    params: Arc<std::sync::RwLock<IsingSharedParams>>,
    pub worker: BackgroundWorker<IsingRunner>,
}

impl IsingModelTool {
    #[cfg(test)]
    #[allow(missing_docs)]
    pub fn restart_worker(&mut self) {
        self.reinitialize_lattice_and_metrics();
    }

    fn reinitialize_lattice_and_metrics(&mut self) {
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
        self.pending_ode_resync = false;
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
        if self.active_challenge.is_none() {
            self.defector_sea = false;
            self.droplet_seeds.clear();
            self.sync_shared_params();
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
        if self.active_challenge.is_some() {
            self.push_seed_layout_to_worker();
        }
        self.stepped_pixels = None;
        self.pending_ode_resync = true;
        self.sync_metrics_from_latest();
        self.reset_trajectory_overlays();
        self.dirty = true;
    }
}

impl Default for IsingModelTool {
    fn default() -> Self {
        let params = Arc::new(std::sync::RwLock::new(IsingSharedParams::default()));
        let worker = BackgroundWorker::new(IsingRunner {
            lattice: SpinLattice::new(DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE, None),
            params: Arc::clone(&params),
        });
        let mut tool = Self {
            temperature: 2.269,
            j_coupling: 1.0,
            h_field: 0.0,
            mutation_rate: 0.0,
            mode: IsingInteractionMode::Ferromagnetic,
            game_preset: FermiGamePreset::default(),
            game_payoffs: preset::default_game_payoffs(),
            brush_radius: preset::default_brush_radius(),
            brush_cooperator: preset::default_brush_cooperator(),
            defector_sea: false,
            droplet_seeds: Vec::new(),
            steps_per_frame: 5000,
            magnetization: 0.0,
            uniform_magnetization: 0.0,
            staggered_magnetization: 0.0,
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
            topology_source: crate::app::TopologySource::SharedBus,
            graph_system: None,
            shared_topology: None,
            synced_bus_revision: None,
            graph_running: false,
            graph_sweeps: 0.0,
            atlas: ParameterAtlasController::new(ISING_ATLAS_AXIS_PAIRS.to_vec()),
            active_challenge: None,
            challenge_workspace: IsingChallengeWorkspace::default(),
            params,
            worker,
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

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.temperature.to_bits().hash(&mut h);
        self.j_coupling.to_bits().hash(&mut h);
        self.h_field.to_bits().hash(&mut h);
        self.mutation_rate.to_bits().hash(&mut h);
        self.brush_radius.hash(&mut h);
        self.brush_cooperator.hash(&mut h);
        self.defector_sea.hash(&mut h);
        self.droplet_seeds.len().hash(&mut h);
        self.steps_per_frame.hash(&mut h);
        self.topology_source.hash(&mut h);
        self.atlas.is_visible().hash(&mut h);
        self.atlas.axis_keys().hash(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        [
            self.temperature,
            self.j_coupling,
            self.h_field,
            self.mutation_rate,
        ]
        .into_iter()
        .all(f64::is_finite)
            && self.game_payoffs.iter().all(|v| v.is_finite())
    }

    fn clamp_parameters(&mut self) -> bool {
        let fin =
            |v: f64, lo: f64, hi: f64, d: f64| if v.is_finite() { v.clamp(lo, hi) } else { d };
        self.temperature = fin(self.temperature, 1e-3, 10.0, 2.269);
        self.j_coupling = fin(self.j_coupling, -2.0, 2.0, 1.0);
        self.h_field = fin(self.h_field, -2.0, 2.0, 0.0);
        self.mutation_rate = fin(self.mutation_rate, 0.0, 0.05, 0.0);
        let defaults = preset::default_game_payoffs();
        for (slot, d) in self.game_payoffs.iter_mut().zip(defaults) {
            *slot = fin(*slot, -20.0, 20.0, d);
        }
        self.enforce_ising_challenge_locks();
        self.sync_shared_params();
        self.recompute_replicator_curve();
        true
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("magnetization", self.magnetization),
            ("uniform_magnetization", self.uniform_magnetization),
            ("staggered_magnetization", self.staggered_magnetization),
            ("energy", self.energy),
            ("cooperator_fraction", self.cooperator_fraction()),
            (
                "graph_node_count",
                self.active_graph_node_count().unwrap_or(0) as f64,
            ),
            (
                "synced_bus_revision",
                self.synced_bus_revision.unwrap_or(0) as f64,
            ),
        ]
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
        let max_r = if self.active_challenge.is_some() {
            10
        } else {
            max_x.max(max_y)
        };
        let r = self.brush_radius.clamp(0, max_r);
        let invert = ctx.modifiers.shift
            || ctx.response.secondary_clicked()
            || ctx.response.dragged_by(egui::PointerButton::Secondary);
        let is_obstacle = if invert {
            !self.brush_cooperator
        } else {
            self.brush_cooperator
        };
        self.record_droplet_seed(DropletSeed {
            cx,
            cy,
            r,
            spin: if is_obstacle { 1 } else { -1 },
        });
        self.sync_shared_params();

        if self.active_challenge.is_some() {
            self.push_seed_layout_to_worker();
            self.refresh_challenge_workspace();
            ctx.response.ctx.request_repaint();
            return;
        }
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
        if self.is_graph_mode() {
            let trials = self.steps_per_frame.clamp(16, 2_000);
            self.step_graph_trials(trials);
            return;
        }
        let mut lattice = SpinLattice::new(DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE, None);
        let steps = self.steps_per_frame.clamp(100, 5000);
        match self.mode {
            IsingInteractionMode::Ferromagnetic => {
                let t_phys = self.temperature.max(1e-3) / KB;
                lattice.evolve(steps, t_phys, self.j_coupling, self.h_field);
            }
            IsingInteractionMode::FermiGame => {
                let mat = DMatrix::from_row_slice(2, 2, &self.game_payoffs);
                if let Ok(payoff) = MatrixPayoff::new(mat) {
                    lattice.evolve_fermi_game_with_mutation(
                        steps,
                        self.temperature,
                        &payoff,
                        self.mutation_rate,
                    );
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
        self.serialize_ising_state()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.deserialize_ising_state(state)
    }

    fn apply_challenge(&mut self, challenge_id: Option<&str>) -> Result<(), String> {
        self.set_ising_challenge(challenge_id)
    }

    fn active_challenge_id(&self) -> Option<&'static str> {
        self.active_challenge
    }

    fn evaluate_challenge(&self) -> Option<ChallengeEvaluation> {
        self.eval_ising_challenge()
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.sync_shared_topology(&ctx);
        let theme = crate::app::ThemeMode::from_visuals(ui.visuals());
        if self.last_theme != Some(theme) {
            self.last_theme = Some(theme);
            self.params.write().unwrap_or_else(|e| e.into_inner()).theme = theme;
            self.dirty = true;
        }
        if self.is_graph_mode() {
            if self.graph_running {
                let trials = self.steps_per_frame.clamp(16, 4_000);
                self.step_graph_trials(trials);
                ctx.request_repaint();
            }
        } else {
            let fallback_m = self.magnetization;
            let fallback_e = self.energy;
            if let Some((m, e, mu, ms)) = self
                .worker
                .update()
                .map(|s| Self::extract_snapshot_metrics(s, fallback_m, fallback_e))
            {
                self.magnetization = m;
                self.energy = e;
                self.uniform_magnetization = mu;
                self.staggered_magnetization = ms;
                self.record_snapshot_progress();
                self.stepped_pixels = None;
                self.dirty = true;
            }
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
#[path = "ising/atlas_challenge_tests.rs"]
mod atlas_challenge_tests;
#[cfg(test)]
#[path = "ising/tests.rs"]
mod tests;

// [cite:clinical_trials_statistics]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "IsingModelTool",
        display_name: "Ising Model",
        domain: "solid_state",
        tags: &["ferromagnetism", "metropolis", "monte carlo", "curie", "phase transition"],
        build: || Box::new(IsingModelTool::default()),
    }
}

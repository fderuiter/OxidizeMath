use crate::async_sim::atlas::ParameterAtlasController;
use crate::async_sim::{BackgroundWorker, SimWorkerState};
use crate::framework::{
    ChallengeEvaluation, InteractionContext, InteractiveTool, RasterBrushState,
};
use eframe::egui;
use math_explorer::physics::stat_mech::ising::SpinLattice;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[path = "ising/atlas.rs"]
pub mod atlas;
#[path = "ising/brush.rs"]
mod brush;
#[path = "ising/challenge.rs"]
pub mod challenge;
#[path = "ising/graph_canvas.rs"]
mod graph_canvas;
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

pub(crate) const DEFAULT_ISING_SIZE: usize = 100;
const MAX_SPATIAL_HISTORY: usize = 512;

pub struct IsingModelTool {
    pub(crate) temperature: f64,
    pub(crate) j_coupling: f64,
    pub(crate) h_field: f64,
    pub(crate) mutation_rate: f64,
    pub(crate) mode: IsingInteractionMode,
    pub(crate) game_preset: FermiGamePreset,
    pub(crate) game_payoffs: [f64; 4],
    pub(crate) brush_radius: i32,
    pub(crate) brush_cooperator: bool,
    pub(crate) last_synced_brush_radius: i32,
    pub(crate) last_synced_brush_cooperator: bool,
    pub(crate) defector_sea: bool,
    pub(crate) droplet_seeds: Vec<DropletSeed>,
    pub(crate) steps_per_frame: usize,
    pub(crate) magnetization: f64,
    pub(crate) uniform_magnetization: f64,
    pub(crate) staggered_magnetization: f64,
    pub(crate) energy: f64,
    pub(crate) spatial_history: VecDeque<[f64; 2]>,
    pub(crate) replicator_curve: Vec<[f64; 2]>,
    pub(crate) ode_anchor_sweep: f64,
    pub(crate) ode_initial_cooperator: f64,
    pub(crate) pending_ode_resync: bool,
    pub(crate) last_canvas_rect: Option<egui::Rect>,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) pixels: Vec<egui::Color32>,
    pub(crate) stepped_pixels: Option<(usize, usize, Vec<egui::Color32>)>,
    pub(crate) last_theme: Option<crate::app::ThemeMode>,
    pub(crate) dirty: bool,
    pub topology_source: crate::app::TopologySource,
    pub(super) graph_system: Option<math_explorer::physics::stat_mech::GraphSpinSystem>,
    pub(super) shared_topology: Option<Arc<crate::app::SharedGraphTopology>>,
    pub(super) synced_bus_revision: Option<u64>,
    pub(super) graph_running: bool,
    pub(super) graph_sweeps: f64,
    pub(crate) atlas: ParameterAtlasController,
    pub(crate) active_challenge: Option<&'static str>,
    pub(crate) challenge_workspace: IsingChallengeWorkspace,
    pub(crate) brush_state: RasterBrushState,
    pub(crate) brush_mode: IsingBrushMode,
    pub(crate) pin_cells: Vec<u8>,
    pub(crate) params: Arc<std::sync::RwLock<IsingSharedParams>>,
    pub worker: BackgroundWorker<IsingRunner>,
    graph_mode: graph_mode::GraphSpinMode,
}

impl IsingModelTool {
    fn build_runner(params: &Arc<std::sync::RwLock<IsingSharedParams>>) -> IsingRunner {
        let mut lattice = SpinLattice::new(DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE, None);
        let pinned_rle = params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .pinned_rle
            .clone();
        apply_pinned_rle_to_lattice(&mut lattice, pinned_rle.as_deref());
        IsingRunner {
            lattice,
            params: Arc::clone(params),
        }
    }

    #[cfg(test)]
    #[allow(missing_docs)]
    pub fn restart_worker(&mut self) {
        self.params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .total_trials
            .store(0, Ordering::Relaxed);
        let runner = Self::build_runner(&self.params);
        self.worker.replace_runner(runner);
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
        if self.is_graph_mode() && self.reset_graph_spins() {
            return;
        }
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
        self.worker.reset_or_respawn(|| Self::build_runner(&params));
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
        let worker = BackgroundWorker::new(Self::build_runner(&params));
        let brush_radius = preset::default_brush_radius();
        let brush_cooperator = preset::default_brush_cooperator();
        let mut tool = Self {
            temperature: 2.269,
            j_coupling: 1.0,
            h_field: 0.0,
            mutation_rate: 0.0,
            mode: IsingInteractionMode::Ferromagnetic,
            game_preset: FermiGamePreset::default(),
            game_payoffs: preset::default_game_payoffs(),
            brush_radius,
            brush_cooperator,
            last_synced_brush_radius: brush_radius,
            last_synced_brush_cooperator: brush_cooperator,
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
            brush_state: RasterBrushState::with_radius(brush_radius as usize),
            brush_mode: IsingBrushMode::default(),
            pin_cells: vec![0; DEFAULT_ISING_SIZE * DEFAULT_ISING_SIZE],
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

    fn set_suspended(&mut self, suspended: bool) {
        self.worker.set_suspended(suspended);
        self.atlas.set_suspended(suspended);
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        (!self.is_graph_mode())
            .then_some(self.brush_state.focused_id)
            .flatten()
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.temperature.to_bits().hash(&mut h);
        self.j_coupling.to_bits().hash(&mut h);
        self.h_field.to_bits().hash(&mut h);
        self.mutation_rate.to_bits().hash(&mut h);
        self.mode.hash(&mut h);
        self.game_preset.hash(&mut h);
        for v in self.game_payoffs {
            v.to_bits().hash(&mut h);
        }
        self.brush_radius.hash(&mut h);
        self.brush_cooperator.hash(&mut h);
        self.defector_sea.hash(&mut h);
        self.droplet_seeds.len().hash(&mut h);
        self.steps_per_frame.hash(&mut h);
        self.topology_source.hash(&mut h);
        self.atlas.is_visible().hash(&mut h);
        self.atlas.axis_keys().hash(&mut h);
        self.graph_mode.follow_workspace_graph.hash(&mut h);
        self.graph_mode.applied_bus_revision.hash(&mut h);
        self.params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .pinned_rle
            .hash(&mut h);
        std::mem::discriminant(&self.worker.state()).hash(&mut h);
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
        self.handle_plot_brush(ctx);
    }

    fn step_simulation(&mut self) {
        if self.is_graph_mode() {
            let trials = self.steps_per_frame.clamp(16, 2_000);
            self.step_graph_trials(trials);
            return;
        }
        self.step_ising_simulation();
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
        if !crate::app::SharedSimulationBus::is_link_topology_enabled(&ctx) {
            self.sync_graph_from_bus(&ctx);
        }
        self.sync_brush_fields_before_frame();
        let theme = crate::app::ThemeMode::from_visuals(ui.visuals());
        if self.last_theme != Some(theme) {
            self.last_theme = Some(theme);
            self.params.write().unwrap_or_else(|e| e.into_inner()).theme = theme;
            self.dirty = true;
        }
        if self.is_graph_mode() {
            if self.graph_running || self.worker.state() == SimWorkerState::Running {
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
use crate::async_sim::{SimCommand, SimulationRunner};
#[cfg(test)]
#[path = "ising/atlas_challenge_tests.rs"]
mod atlas_challenge_tests;
#[cfg(test)]
#[path = "ising/tests.rs"]
mod tests;

// [cite:solid_state]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "IsingModelTool",
        display_name: "Ising Model",
        domain: "solid_state",
        tags: &["ferromagnetism", "metropolis", "monte carlo", "curie", "phase transition"],
        build: || Box::new(IsingModelTool::default()),
    }
}

//! Interactive **Physarum Network Lab** (`PhysarumTool`) inside `MorphogenesisTab`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod canvas;
pub mod presets;
mod tool_impl;
mod ui;

#[cfg(test)]
mod tests;

pub use presets::{PhysarumBrushMode, PhysarumPreset, PhysarumTheory, PhysarumToolState};

use crate::framework::recovery::DiagnosticReport;
use eframe::egui;
use math_explorer::biology::physarum::{
    asymmetric_maze_corridors, extract_physarum_benchmark, PhysarumEngine,
    PhysarumNetworkBenchmark, PhysarumParams, PhysarumTerminal, MAX_PHYSARUM_AGENTS,
    MAX_PHYSARUM_GRID, MAX_PHYSARUM_TERMINALS, MIN_PHYSARUM_AGENTS, MIN_PHYSARUM_GRID,
    MIN_PHYSARUM_TERMINALS,
};
use math_explorer::rng::OxidizeRng;
use presets::default_brush_radius;
use std::collections::VecDeque;

const MAX_BENCHMARK_HISTORY: usize = 200;
const COOPERATIVE_FRAME_BUDGET_SECS: f64 = 0.008;
const BENCHMARK_REFRESH_STRIDE: u64 = 8;

/// Historical scalar benchmark sample plotted in the live `egui_plot` inspector.
#[derive(Clone, Copy, Debug)]
pub struct PhysarumBenchmarkSample {
    /// Simulation step time.
    pub time: f64,
    /// Wire cost normalized by static MST cost (`wire_cost / mst_cost`).
    pub cost_ratio: f64,
    /// Mean `dijkstra` shortest-path route stretch across connected terminal pairs.
    pub mean_route_stretch: f64,
    /// Mean pairwise connectivity surviving single-edge deletion (`0.0..=1.0`).
    pub pairwise_fault_survival: f64,
    /// Short-corridor trail share (`AsymmetricMaze`) or connected fraction (`0.0..=1.0`).
    pub corridor_focus: f64,
    /// Total integrated chemoattractant trail mass.
    pub total_trail_mass: f64,
}

/// Interactive zero-allocation chemotactic slime-mold and Steiner/Dijkstra network studio.
pub struct PhysarumTool {
    pub(crate) engine: PhysarumEngine,
    /// Left/right sensor angle $\theta_s$ (`0.10..=1.50` rad).
    pub sensor_angle: f64,
    /// Heading turn angle $\theta_r$ (`0.05..=1.50` rad).
    pub turn_angle: f64,
    /// Forward sensor offset distance $d_s$ (`2.0..=32.0` cells).
    pub sensor_distance: f64,
    /// Forward propulsion step $v$ (`0.20..=3.50` cells/step).
    pub step_size: f64,
    /// Trail deposition per agent step (`0.05..=5.00`).
    pub deposition: f64,
    /// Isotropic $3\times 3$ diffusion blend $\alpha \in [0.0, 1.0]$.
    pub diffusion_rate: f64,
    /// Linear trail decay $\gamma \in [0.005, 0.50]$.
    pub decay_rate: f64,
    /// Sub-threshold Hill pruning rate $\gamma_{\text{prune}} \in [0.0, 0.35]$.
    pub prune_rate: f64,
    /// Nutrient beacon emission amplitude (`0.0..=5.0`).
    pub beacon_strength: f64,
    /// Square grid dimension (`64..=256`).
    pub grid_size: usize,
    /// Active SoA chemotactic agent count (`256..=16384`).
    pub agent_count: usize,
    /// Active spatial preset topology.
    pub preset: PhysarumPreset,
    /// Normalized `[u, v]` city terminal coordinates (`2..=16`).
    pub terminals: Vec<[f64; 2]>,
    /// User-painted obstacle strokes `[u, v, r, blocked]` (`0..=24`).
    pub obstacle_strokes: Vec<[f64; 4]>,
    /// Deterministic PRNG seed.
    pub rng_seed: u64,
    /// Brush disc radius (`1.0..=24.0`).
    pub brush_radius: f64,
    /// Active pointer/keyboard brush mode.
    pub brush_mode: PhysarumBrushMode,
    /// Whether skeleton `Graph` and MST overlays are visible.
    pub show_skeleton_overlay: bool,
    /// Whether continuous cooperative stepping is active.
    pub playing: bool,
    pub(crate) cursor: [usize; 2],
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) pixels: Vec<egui::Color32>,
    pub(crate) texture_dirty: bool,
    pub(crate) high_contrast: bool,
    pub(crate) obstacle_rgb: [u8; 3],
    pub(crate) beacon_rgb: [u8; 3],
    pub(crate) latest_benchmark: PhysarumNetworkBenchmark,
    pub(crate) history: VecDeque<PhysarumBenchmarkSample>,
    pub(crate) last_step_ms: f64,
    pub(crate) diagnostic: Option<DiagnosticReport>,
    pub(crate) focused_canvas: Option<egui::Id>,
    pub(crate) theory: PhysarumTheory,
}

impl Default for PhysarumTool {
    fn default() -> Self {
        let grid_size = 128;
        let agent_count = 6144;
        let preset = PhysarumPreset::SteinerTriangle;
        let rng_seed = 42;
        let engine = PhysarumEngine::from_preset(
            grid_size,
            grid_size,
            agent_count,
            preset.to_domain(),
            OxidizeRng::new(rng_seed),
        );
        let p = engine.params();
        let terminals = preset
            .to_domain()
            .normalized_terminals()
            .into_iter()
            .map(|[u, v]| [(u * 1e6).round() / 1e6, (v * 1e6).round() / 1e6])
            .collect();
        let latest_benchmark = extract_physarum_benchmark(&engine);
        let mut tool = Self {
            engine,
            sensor_angle: p.sensor_angle,
            turn_angle: p.turn_angle,
            sensor_distance: p.sensor_distance,
            step_size: p.step_size,
            deposition: p.deposition,
            diffusion_rate: p.diffusion_rate,
            decay_rate: p.decay_rate,
            prune_rate: p.prune_rate,
            beacon_strength: p.beacon_strength,
            grid_size,
            agent_count,
            preset,
            terminals,
            obstacle_strokes: Vec::new(),
            rng_seed,
            brush_radius: default_brush_radius(),
            brush_mode: PhysarumBrushMode::SeverTube,
            show_skeleton_overlay: true,
            playing: false,
            cursor: [grid_size / 2, grid_size / 2],
            texture: None,
            pixels: vec![egui::Color32::BLACK; grid_size * grid_size],
            texture_dirty: true,
            high_contrast: false,
            obstacle_rgb: [46, 50, 58],
            beacon_rgb: [250, 204, 56],
            latest_benchmark,
            history: VecDeque::with_capacity(MAX_BENCHMARK_HISTORY),
            last_step_ms: 0.0,
            diagnostic: None,
            focused_canvas: None,
            theory: PhysarumTheory::default(),
        };
        tool.record_benchmark_sample();
        tool.refresh_pixels_in_place();
        tool
    }
}

impl PhysarumTool {
    /// Applies a canonical [`PhysarumPreset`] topology, resetting parameters, terminals, and obstacles.
    pub fn apply_preset(&mut self, preset: PhysarumPreset) {
        self.preset = preset;
        let p = preset.to_domain().default_params();
        self.sensor_angle = p.sensor_angle;
        self.turn_angle = p.turn_angle;
        self.sensor_distance = p.sensor_distance;
        self.step_size = p.step_size;
        self.deposition = p.deposition;
        self.diffusion_rate = p.diffusion_rate;
        self.decay_rate = p.decay_rate;
        self.prune_rate = p.prune_rate;
        self.beacon_strength = p.beacon_strength;
        self.terminals = preset
            .to_domain()
            .normalized_terminals()
            .into_iter()
            .map(|[u, v]| [(u * 1e6).round() / 1e6, (v * 1e6).round() / 1e6])
            .collect();
        self.obstacle_strokes.clear();
        self.diagnostic = None;
        self.rebuild_from_state();
    }

    /// Rebuilds the pre-allocated engine from current sliders, terminals, and saved obstacle strokes.
    pub fn rebuild_from_state(&mut self) {
        let g = self.grid_size.clamp(MIN_PHYSARUM_GRID, MAX_PHYSARUM_GRID);
        let n = self
            .agent_count
            .clamp(MIN_PHYSARUM_AGENTS, MAX_PHYSARUM_AGENTS);
        self.grid_size = g;
        self.agent_count = n;
        self.engine = PhysarumEngine::from_preset(
            g,
            g,
            n,
            self.preset.to_domain(),
            OxidizeRng::new(self.rng_seed),
        );
        self.cursor = [self.cursor[0].min(g - 1), self.cursor[1].min(g - 1)];
        self.sync_terminals_and_strokes();
        self.sync_engine_parameters();
        self.history.clear();
        self.refresh_benchmark_and_record();
        self.refresh_pixels_in_place();
    }

    fn sync_terminals_and_strokes(&mut self) {
        let w = self.engine.width();
        let h = self.engine.height();
        let mapped: Vec<PhysarumTerminal> = self
            .terminals
            .iter()
            .take(MAX_PHYSARUM_TERMINALS)
            .map(|&[u, v]| {
                PhysarumTerminal::clamped(u * (w as f64), v * (h as f64), 3.4, 1.0, w, h)
            })
            .collect();
        if mapped.len() >= MIN_PHYSARUM_TERMINALS {
            self.engine.set_terminals(&mapped);
        }
        for &[u, v, r, blocked] in &self.obstacle_strokes {
            let cx = ((u * (w as f64)).round() as usize).min(w - 1);
            let cy = ((v * (h as f64)).round() as usize).min(h - 1);
            self.engine.paint_obstacle_disc(cx, cy, r, blocked >= 0.5);
        }
    }

    pub(crate) fn current_params(&self) -> PhysarumParams {
        PhysarumParams {
            sensor_angle: self.sensor_angle,
            turn_angle: self.turn_angle,
            sensor_distance: self.sensor_distance,
            step_size: self.step_size,
            deposition: self.deposition,
            diffusion_rate: self.diffusion_rate,
            decay_rate: self.decay_rate,
            prune_rate: self.prune_rate,
            beacon_strength: self.beacon_strength,
        }
    }

    pub(crate) fn parameters_within_safe_bounds(&self) -> bool {
        self.current_params().is_within_bounds()
            && (MIN_PHYSARUM_GRID..=MAX_PHYSARUM_GRID).contains(&self.grid_size)
            && (MIN_PHYSARUM_AGENTS..=MAX_PHYSARUM_AGENTS).contains(&self.agent_count)
            && (MIN_PHYSARUM_TERMINALS..=MAX_PHYSARUM_TERMINALS).contains(&self.terminals.len())
            && self.brush_radius.is_finite()
            && (1.0..=24.0).contains(&self.brush_radius)
    }

    pub(crate) fn sync_engine_parameters(&mut self) {
        if !self.parameters_within_safe_bounds() {
            return;
        }
        self.engine.set_params(self.current_params());
        self.engine.set_agent_count(self.agent_count);
    }

    pub(crate) fn check_and_latch_instability(&mut self) -> bool {
        if !self.parameters_within_safe_bounds() {
            self.playing = false;
            self.diagnostic = Some(DiagnosticReport::new(
                self.engine.telemetry().time,
                self.engine.telemetry().step_count as usize,
                "parameters",
                "Out-of-range Physarum chemotaxis or speed parameter detected; simulation halted by FiniteStateGuard.",
            ));
            return false;
        }
        if !self.engine.trail().iter().all(|v| v.is_finite()) {
            self.playing = false;
            self.diagnostic = Some(DiagnosticReport::new(
                self.engine.telemetry().time,
                self.engine.telemetry().step_count as usize,
                "trail",
                "Non-finite chemoattractant concentration detected; simulation halted by FiniteStateGuard.",
            ));
            return false;
        }
        true
    }

    /// Executes one timed zero-allocation engine step and periodically updates the graph benchmark.
    pub fn step_once(&mut self) {
        if !self.check_and_latch_instability() {
            return;
        }
        self.sync_engine_parameters();
        let start = crate::web_io::PlatformInstant::now();
        self.engine.step();
        self.last_step_ms = start.elapsed().as_secs_f64() * 1000.0;
        if self.check_and_latch_instability() {
            if self
                .engine
                .telemetry()
                .step_count
                .is_multiple_of(BENCHMARK_REFRESH_STRIDE)
            {
                self.latest_benchmark = extract_physarum_benchmark(&self.engine);
            }
            self.record_benchmark_sample();
            self.refresh_pixels_in_place();
        }
    }

    pub(crate) fn step_frame_cooperative(&mut self) {
        let frame_start = crate::web_io::PlatformInstant::now();
        for _ in 0..3 {
            self.step_once();
            if !self.playing || frame_start.elapsed().as_secs_f64() >= COOPERATIVE_FRAME_BUDGET_SECS
            {
                break;
            }
        }
    }

    pub(crate) fn refresh_benchmark_and_record(&mut self) {
        self.latest_benchmark = extract_physarum_benchmark(&self.engine);
        self.record_benchmark_sample();
    }

    pub(crate) fn compute_corridor_focus(&self) -> f64 {
        if self.preset == PhysarumPreset::AsymmetricMaze {
            let regions = asymmetric_maze_corridors(self.engine.width(), self.engine.height());
            let [sx0, sy0, sx1, sy1] = regions.short_branch;
            let [lx0, ly0, lx1, ly1] = regions.long_branch;
            let s = self.engine.trail_mass_in_rect(sx0, sy0, sx1, sy1);
            let l = self.engine.trail_mass_in_rect(lx0, ly0, lx1, ly1);
            if s + l > 1e-6 {
                return (s / (s + l)).clamp(0.0, 1.0);
            }
        }
        self.latest_benchmark.connected_pair_fraction
    }

    pub(crate) fn record_benchmark_sample(&mut self) {
        if self.history.len() >= MAX_BENCHMARK_HISTORY {
            self.history.pop_front();
        }
        let tel = self.engine.telemetry();
        let b = &self.latest_benchmark;
        self.history.push_back(PhysarumBenchmarkSample {
            time: tel.time,
            cost_ratio: b.cost_ratio,
            mean_route_stretch: b.mean_route_stretch,
            pairwise_fault_survival: b.pairwise_fault_survival,
            corridor_focus: self.compute_corridor_focus(),
            total_trail_mass: tel.total_trail_mass,
        });
    }

    pub(crate) fn current_state_snapshot(&self) -> PhysarumToolState {
        PhysarumToolState {
            sensor_angle: self.sensor_angle,
            turn_angle: self.turn_angle,
            sensor_distance: self.sensor_distance,
            step_size: self.step_size,
            deposition: self.deposition,
            diffusion_rate: self.diffusion_rate,
            decay_rate: self.decay_rate,
            prune_rate: self.prune_rate,
            beacon_strength: self.beacon_strength,
            grid_size: self.grid_size,
            agent_count: self.agent_count,
            preset: self.preset,
            terminals: self.terminals.clone(),
            obstacle_strokes: self.obstacle_strokes.clone(),
            rng_seed: self.rng_seed,
            brush_radius: self.brush_radius,
            brush_mode: self.brush_mode,
            show_skeleton_overlay: self.show_skeleton_overlay,
            playing: self.playing,
        }
    }
}

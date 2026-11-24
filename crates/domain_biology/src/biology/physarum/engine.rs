//! Zero-allocation Structure-of-Arrays (SoA) Lagrangian-Eulerian `PhysarumEngine`.

use super::dynamics::rng_unit_f64;
use super::params::{PhysarumParams, PhysarumTelemetry, PhysarumTerminal};
use super::presets::{PhysarumPresetKind, seed_preset_into_engine};
use super::{
    MAX_PHYSARUM_AGENTS, MAX_PHYSARUM_GRID, MAX_PHYSARUM_TERMINALS, MIN_PHYSARUM_AGENTS,
    MIN_PHYSARUM_GRID, MIN_PHYSARUM_TERMINALS,
};
use math_commons::math_kernel::types::flatten_2d_index;
use oxidize_core::rng::OxidizeRng;
use std::f64::consts::TAU;

/// Zero-allocation Structure-of-Arrays (SoA) chemotactic slime-mold engine.
///
/// Pre-allocates `(agent_x, agent_y, agent_heading, agent_energy, agent_origin)` buffers up to
/// [`MAX_PHYSARUM_AGENTS`] alongside contiguous `trail`, `trail_scratch`, and `obstacles` slices so
/// [`PhysarumEngine::step`] performs zero heap allocations under `VerifiedAllocator`.
pub struct PhysarumEngine {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) agent_count: usize,
    pub(crate) params: PhysarumParams,
    pub(crate) agent_x: Vec<f64>,
    pub(crate) agent_y: Vec<f64>,
    pub(crate) agent_heading: Vec<f64>,
    pub(crate) agent_energy: Vec<f64>,
    pub(crate) agent_origin: Vec<u8>,
    pub(crate) trail: Vec<f64>,
    pub(crate) trail_scratch: Vec<f64>,
    pub(crate) obstacles: Vec<u8>,
    pub(crate) terminals: [PhysarumTerminal; MAX_PHYSARUM_TERMINALS],
    pub(crate) terminal_count: usize,
    pub(crate) rng: OxidizeRng,
    pub(crate) telemetry: PhysarumTelemetry,
}

impl PhysarumEngine {
    /// Constructs a pre-allocated `PhysarumEngine` on a `width x height` grid (`64..=256`)
    /// with `agent_count` (`256..=16384`) chemotactic agents seeded deterministically via `rng`.
    #[verified_engine::verified]
    pub fn new(
        width: usize,
        height: usize,
        agent_count: usize,
        params: PhysarumParams,
        mut rng: OxidizeRng,
    ) -> Self {
        let w = width.clamp(MIN_PHYSARUM_GRID, MAX_PHYSARUM_GRID);
        let h = height.clamp(MIN_PHYSARUM_GRID, MAX_PHYSARUM_GRID);
        let n_agents = agent_count.clamp(MIN_PHYSARUM_AGENTS, MAX_PHYSARUM_AGENTS);
        let cells = w * h;
        assert!(cells >= MIN_PHYSARUM_GRID * MIN_PHYSARUM_GRID);
        assert!(n_agents >= MIN_PHYSARUM_AGENTS);

        let mut terminals = [PhysarumTerminal::default(); MAX_PHYSARUM_TERMINALS];
        terminals[0] =
            PhysarumTerminal::clamped(0.25 * (w as f64), 0.5 * (h as f64), 3.2, 1.0, w, h);
        terminals[1] =
            PhysarumTerminal::clamped(0.75 * (w as f64), 0.5 * (h as f64), 3.2, 1.0, w, h);

        let mut agent_x = vec![0.5 * (w as f64); MAX_PHYSARUM_AGENTS];
        let mut agent_y = vec![0.5 * (h as f64); MAX_PHYSARUM_AGENTS];
        let mut agent_heading = vec![0.0; MAX_PHYSARUM_AGENTS];
        let agent_energy = vec![1.0; MAX_PHYSARUM_AGENTS];
        let mut agent_origin = vec![u8::MAX; MAX_PHYSARUM_AGENTS];
        for i in 0..MAX_PHYSARUM_AGENTS {
            agent_x[i] = 2.0 + rng_unit_f64(&mut rng) * ((w - 4) as f64);
            agent_y[i] = 2.0 + rng_unit_f64(&mut rng) * ((h - 4) as f64);
            agent_heading[i] = rng_unit_f64(&mut rng) * TAU;
            agent_origin[i] = (i % 2) as u8;
        }

        let mut engine = Self {
            width: w,
            height: h,
            agent_count: n_agents,
            params: params.sanitized(),
            agent_x,
            agent_y,
            agent_heading,
            agent_energy,
            agent_origin,
            trail: vec![0.0; cells],
            trail_scratch: vec![0.0; cells],
            obstacles: vec![0; cells],
            terminals,
            terminal_count: MIN_PHYSARUM_TERMINALS,
            rng,
            telemetry: PhysarumTelemetry::default(),
        };
        engine.refresh_telemetry(0.0);
        engine
    }

    /// Creates a pre-allocated engine configured and seeded with a canonical [`PhysarumPresetKind`].
    #[verified_engine::verified]
    pub fn from_preset(
        width: usize,
        height: usize,
        agent_count: usize,
        preset: PhysarumPresetKind,
        rng: OxidizeRng,
    ) -> Self {
        let mut engine = Self::new(width, height, agent_count, preset.default_params(), rng);
        seed_preset_into_engine(&mut engine, preset);
        assert!(engine.terminal_count() >= MIN_PHYSARUM_TERMINALS);
        debug_assert!(engine.width() >= MIN_PHYSARUM_GRID);
        engine
    }

    /// Grid width $W$ (`64..=256`).
    #[verified_engine::verified]
    pub fn width(&self) -> usize {
        debug_assert!(self.width >= MIN_PHYSARUM_GRID);
        assert!(self.width <= MAX_PHYSARUM_GRID);
        self.width
    }

    /// Grid height $H$ (`64..=256`).
    #[verified_engine::verified]
    pub fn height(&self) -> usize {
        debug_assert!(self.height >= MIN_PHYSARUM_GRID);
        assert!(self.height <= MAX_PHYSARUM_GRID);
        self.height
    }

    /// Active Lagrangian agent count (`256..=16384`).
    #[verified_engine::verified]
    pub fn agent_count(&self) -> usize {
        debug_assert!(self.agent_count >= MIN_PHYSARUM_AGENTS);
        assert!(self.agent_count <= MAX_PHYSARUM_AGENTS);
        self.agent_count
    }

    /// Adjusts the active agent count within `256..=16384` without heap allocation.
    #[verified_engine::verified]
    pub fn set_agent_count(&mut self, count: usize) {
        let next = count.clamp(MIN_PHYSARUM_AGENTS, MAX_PHYSARUM_AGENTS);
        assert!(next <= self.agent_x.len());
        debug_assert!(next >= MIN_PHYSARUM_AGENTS);
        self.agent_count = next;
    }

    /// Active chemotaxis and trail parameters.
    #[verified_engine::verified]
    pub fn params(&self) -> PhysarumParams {
        debug_assert!(self.params.is_within_bounds());
        assert!(self.params.step_size > 0.0);
        self.params
    }

    /// Updates chemotaxis and trail parameters after clamping to safe bounds.
    #[verified_engine::verified]
    pub fn set_params(&mut self, params: PhysarumParams) {
        self.params = params.sanitized();
        debug_assert!(self.params.is_within_bounds());
        assert!(self.params.sensor_distance >= 2.0);
    }

    /// Row-major chemoattractant trail concentration slice of length `width * height`.
    #[verified_engine::verified]
    pub fn trail(&self) -> &[f64] {
        assert_eq!(self.trail.len(), self.width * self.height);
        debug_assert!(!self.trail.is_empty());
        &self.trail
    }

    /// Mutable row-major chemoattractant trail slice of length `width * height`.
    #[verified_engine::verified]
    pub fn trail_mut(&mut self) -> &mut [f64] {
        assert_eq!(self.trail.len(), self.width * self.height);
        debug_assert!(!self.trail.is_empty());
        &mut self.trail
    }

    /// Row-major obstacle mask (`0` = open corridor, `1` = impassable wall) of length `width * height`.
    #[verified_engine::verified]
    pub fn obstacles(&self) -> &[u8] {
        assert_eq!(self.obstacles.len(), self.width * self.height);
        debug_assert!(!self.obstacles.is_empty());
        &self.obstacles
    }

    /// Mutable row-major obstacle mask (`0` = open corridor, `1` = impassable wall).
    #[verified_engine::verified]
    pub fn obstacles_mut(&mut self) -> &mut [u8] {
        assert_eq!(self.obstacles.len(), self.width * self.height);
        debug_assert!(!self.obstacles.is_empty());
        &mut self.obstacles
    }

    /// Active Structure-of-Arrays horizontal agent coordinates `x[0..agent_count]`.
    #[verified_engine::verified]
    pub fn agent_x(&self) -> &[f64] {
        assert!(self.agent_count <= self.agent_x.len());
        debug_assert!(self.agent_count >= MIN_PHYSARUM_AGENTS);
        &self.agent_x[..self.agent_count]
    }

    /// Active Structure-of-Arrays vertical agent coordinates `y[0..agent_count]`.
    #[verified_engine::verified]
    pub fn agent_y(&self) -> &[f64] {
        assert!(self.agent_count <= self.agent_y.len());
        debug_assert!(self.agent_count >= MIN_PHYSARUM_AGENTS);
        &self.agent_y[..self.agent_count]
    }

    /// Active Structure-of-Arrays agent headings `heading[0..agent_count]` in radians.
    #[verified_engine::verified]
    pub fn agent_heading(&self) -> &[f64] {
        assert!(self.agent_count <= self.agent_heading.len());
        debug_assert!(self.agent_count >= MIN_PHYSARUM_AGENTS);
        &self.agent_heading[..self.agent_count]
    }

    /// Active nutrient beacon terminals (`2..=16`).
    #[verified_engine::verified]
    pub fn terminals(&self) -> &[PhysarumTerminal] {
        assert!(self.terminal_count <= MAX_PHYSARUM_TERMINALS);
        debug_assert!(self.terminal_count >= MIN_PHYSARUM_TERMINALS);
        &self.terminals[..self.terminal_count]
    }

    /// Number of active nutrient beacon terminals.
    #[verified_engine::verified]
    pub fn terminal_count(&self) -> usize {
        assert!(self.terminal_count <= MAX_PHYSARUM_TERMINALS);
        debug_assert!(self.terminal_count >= MIN_PHYSARUM_TERMINALS);
        self.terminal_count
    }

    /// Latest simulation telemetry snapshot.
    #[verified_engine::verified]
    pub fn telemetry(&self) -> PhysarumTelemetry {
        debug_assert!(self.telemetry.total_trail_mass.is_finite());
        assert!(self.telemetry.active_cells >= 0.0);
        self.telemetry
    }

    /// Replaces the internal deterministic RNG stream without reallocating buffers.
    #[verified_engine::verified]
    pub fn set_rng(&mut self, rng: OxidizeRng) {
        self.rng = rng;
        debug_assert!(self.agent_count >= MIN_PHYSARUM_AGENTS);
        assert_eq!(self.trail.len(), self.width * self.height);
    }

    #[verified_engine::verified]
    pub(crate) fn rng_mut(&mut self) -> &mut OxidizeRng {
        debug_assert!(self.width >= MIN_PHYSARUM_GRID);
        assert_eq!(self.obstacles.len(), self.width * self.height);
        &mut self.rng
    }

    #[verified_engine::verified]
    pub(crate) fn set_agent_slot(
        &mut self,
        idx: usize,
        x: f64,
        y: f64,
        heading: f64,
        energy: f64,
        origin: u8,
    ) {
        assert!(idx < self.agent_x.len());
        debug_assert!(x.is_finite() && y.is_finite());
        self.agent_x[idx] = x;
        self.agent_y[idx] = y;
        self.agent_heading[idx] = heading;
        self.agent_energy[idx] = energy;
        self.agent_origin[idx] = origin;
    }

    /// Replaces the active nutrient terminal list (clamped to `2..=16` entries) and clears obstacles under terminals.
    #[verified_engine::verified]
    pub fn set_terminals(&mut self, list: &[PhysarumTerminal]) {
        let count = list
            .len()
            .clamp(MIN_PHYSARUM_TERMINALS, MAX_PHYSARUM_TERMINALS);
        for idx in 0..count {
            let src = list.get(idx).copied().unwrap_or(PhysarumTerminal {
                x: if idx == 0 {
                    0.25 * (self.width as f64)
                } else {
                    0.75 * (self.width as f64)
                },
                y: 0.5 * (self.height as f64),
                radius: 3.2,
                weight: 1.0,
            });
            self.terminals[idx] = PhysarumTerminal::clamped(
                src.x,
                src.y,
                src.radius,
                src.weight,
                self.width,
                self.height,
            );
        }
        self.terminal_count = count;
        self.clear_obstacles_under_terminals();
        assert!(self.terminal_count >= MIN_PHYSARUM_TERMINALS);
        debug_assert!(self.terminal_count <= MAX_PHYSARUM_TERMINALS);
    }

    /// Appends a new nutrient terminal at `(x, y)` up to `16`, or removes an existing terminal within `4.5` cells if `> 2` exist.
    #[verified_engine::verified]
    pub fn toggle_terminal_at(&mut self, x: f64, y: f64) -> bool {
        for idx in 0..self.terminal_count {
            let dist = (self.terminals[idx].x - x).hypot(self.terminals[idx].y - y);
            if dist <= 4.5 {
                if self.terminal_count <= MIN_PHYSARUM_TERMINALS {
                    return false;
                }
                for shift in idx..(self.terminal_count - 1) {
                    self.terminals[shift] = self.terminals[shift + 1];
                }
                self.terminal_count -= 1;
                assert!(self.terminal_count >= MIN_PHYSARUM_TERMINALS);
                return true;
            }
        }
        if self.terminal_count >= MAX_PHYSARUM_TERMINALS {
            return false;
        }
        let term = PhysarumTerminal::clamped(x, y, 3.2, 1.0, self.width, self.height);
        self.terminals[self.terminal_count] = term;
        self.terminal_count += 1;
        self.clear_obstacles_under_terminals();
        debug_assert!(self.terminal_count <= MAX_PHYSARUM_TERMINALS);
        true
    }

    /// Resets trail concentrations and telemetry counters while preserving obstacles and terminals.
    #[verified_engine::verified]
    pub fn clear_trails(&mut self) {
        self.trail.fill(0.0);
        self.trail_scratch.fill(0.0);
        self.telemetry = PhysarumTelemetry::default();
        assert_eq!(self.telemetry.step_count, 0);
        debug_assert_eq!(self.telemetry.total_trail_mass, 0.0);
    }

    /// Applies a full row-major `0`/`1` obstacle mask of length `width * height` in place without heap allocations,
    /// keeping active nutrient beacons unobstructed and relocating any newly trapped agents.
    #[verified_engine::verified]
    pub fn apply_obstacle_mask(&mut self, mask: &[u8]) {
        assert_eq!(mask.len(), self.width * self.height);
        debug_assert_eq!(self.obstacles.len(), mask.len());
        for (idx, &cell) in mask.iter().enumerate() {
            let blocked = u8::from(cell != 0);
            self.obstacles[idx] = blocked;
            if blocked != 0 {
                self.trail[idx] = 0.0;
            }
        }
        self.clear_obstacles_under_terminals();
        self.relocate_trapped_agents();
        self.refresh_telemetry(self.telemetry.mean_flux);
    }

    /// Paints or erases a circular obstacle barrier centered at `(cx, cy)`, keeping nutrient terminals open.
    #[verified_engine::verified]
    pub fn paint_obstacle_disc(&mut self, cx: usize, cy: usize, radius: f64, blocked: bool) {
        let r = radius.clamp(1.0, 32.0);
        let span = r.ceil() as isize;
        let fill = u8::from(blocked);
        for dy in -span..=span {
            let ny = cy as isize + dy;
            if ny < 0 || ny >= self.height as isize {
                continue;
            }
            for dx in -span..=span {
                let nx = cx as isize + dx;
                if nx >= 0 && nx < self.width as isize && (dx as f64).hypot(dy as f64) <= r {
                    let idx = flatten_2d_index(nx as usize, ny as usize, self.width);
                    self.obstacles[idx] = fill;
                    if blocked {
                        self.trail[idx] = 0.0;
                    }
                }
            }
        }
        self.clear_obstacles_under_terminals();
        if blocked {
            self.relocate_trapped_agents();
        }
        self.refresh_telemetry(self.telemetry.mean_flux);
        assert!(self.terminal_count >= MIN_PHYSARUM_TERMINALS);
    }

    /// Slices active chemoattractant tubes inside `radius` around `(cx, cy)` to test live rerouting and self-healing.
    #[verified_engine::verified]
    pub fn sever_trail_disc(&mut self, cx: usize, cy: usize, radius: f64) {
        let r = radius.clamp(1.0, 32.0);
        let span = r.ceil() as isize;
        for dy in -span..=span {
            let ny = cy as isize + dy;
            if ny < 0 || ny >= self.height as isize {
                continue;
            }
            for dx in -span..=span {
                let nx = cx as isize + dx;
                if nx >= 0 && nx < self.width as isize && (dx as f64).hypot(dy as f64) <= r {
                    let idx = flatten_2d_index(nx as usize, ny as usize, self.width);
                    self.trail[idx] = 0.0;
                }
            }
        }
        self.refresh_telemetry(self.telemetry.mean_flux);
        debug_assert!(self.telemetry.total_trail_mass >= 0.0);
        assert_eq!(self.trail.len(), self.width * self.height);
    }

    /// Integrates chemoattractant trail mass over the inclusive grid rectangle `[x0..=x1] x [y0..=y1]`.
    #[verified_engine::verified]
    pub fn trail_mass_in_rect(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
        if self.width == 0 || self.height == 0 {
            return 0.0;
        }
        let xa = x0.min(x1).min(self.width - 1);
        let xb = x0.max(x1).min(self.width - 1);
        let ya = y0.min(y1).min(self.height - 1);
        let yb = y0.max(y1).min(self.height - 1);
        let mut sum = 0.0;
        for y in ya..=yb {
            for x in xa..=xb {
                let idx = flatten_2d_index(x, y, self.width);
                if self.obstacles[idx] == 0 {
                    sum += self.trail[idx];
                }
            }
        }
        debug_assert!(sum.is_finite());
        assert!(sum >= 0.0);
        sum
    }

    /// Advances one complete Lagrangian-Eulerian chemotaxis step with **zero dynamic heap allocations**.
    #[verified_engine::verified]
    pub fn step(&mut self) {
        assert_eq!(self.trail.len(), self.width * self.height);
        debug_assert!(self.agent_count <= self.agent_x.len());

        self.emit_nutrient_beacons();
        let moved_fraction = self.step_soa_agents();
        self.diffuse_and_prune_trails();
        self.telemetry.step_count += 1;
        self.telemetry.time += 1.0;
        self.refresh_telemetry(moved_fraction);
    }
}

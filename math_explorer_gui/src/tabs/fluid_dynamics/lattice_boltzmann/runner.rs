//! Background worker [`LbmRunner`] and raster colormap rendering for [`super::LatticeBoltzmannTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::{
    compute_aerodynamics, LbmViewMode, LBM_CELL_COUNT, LBM_HEIGHT, LBM_INLET_GUARD_COLS, LBM_WIDTH,
};
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use eframe::egui::{self, Color32};
use math_commons::math_kernel::colormap::{
    ensure_dark_canvas_contrast, heatmap_color, heatmap_color_high_contrast, mapped_gradient,
};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    BgkCollision, LatticeBoltzmannD2Q9, LatticeState,
};
use std::sync::{Arc, RwLock};

const VORTICITY_FULL_SCALE: f64 = 0.045;

/// Shared UI-to-worker configuration synchronized without restarting the worker thread.
#[derive(Clone, Debug)]
pub struct LbmSharedConfig {
    /// Kinematic viscosity `nu = (tau - 0.5) / 3`.
    pub viscosity: f64,
    /// Horizontal inlet velocity `U` in lattice units.
    pub inflow_speed: f64,
    /// Number of LBM steps advanced per worker tick.
    pub steps_per_frame: usize,
    /// Scalar visualization field (`Speed` or `Vorticity`).
    pub view_mode: LbmViewMode,
    /// Whether WCAG high-contrast colormap lifting is active.
    pub high_contrast: bool,
    /// Authoritative `100 x 50` obstacle mask (`y * 100 + x`).
    pub obstacles: Vec<bool>,
    /// Monotonic generation incremented whenever obstacles are sculpted or replaced.
    pub mask_generation: u64,
}

/// Initializes a `100 x 50` D2Q9 wind-tunnel solver with uniform inflow equilibrium and obstacles.
#[must_use]
pub fn build_wind_tunnel_solver(
    viscosity: f64,
    inflow_speed: f64,
    obstacles: &[bool],
) -> LatticeBoltzmannD2Q9<BgkCollision> {
    let safe_visc = if viscosity.is_finite() && viscosity > 0.0 {
        viscosity
    } else {
        0.02
    };
    let safe_u = if inflow_speed.is_finite() {
        inflow_speed.max(0.0)
    } else {
        0.10
    };
    let tau = 3.0 * safe_visc + 0.5;
    let mut solver = LatticeBoltzmannD2Q9::new(LBM_WIDTH, LBM_HEIGHT, tau);
    solver.set_inlet(0, 0, LBM_WIDTH, LBM_HEIGHT, safe_u, 0.0);
    sync_solver_obstacles(&mut solver, obstacles);
    solver.set_inlet(0, 0, LBM_INLET_GUARD_COLS, LBM_HEIGHT, safe_u, 0.0);
    solver
}

/// Synchronizes a `100 x 50` obstacle slice onto an existing solver, re-equilibrating newly freed fluid cells.
pub fn sync_solver_obstacles(solver: &mut LatticeBoltzmannD2Q9<BgkCollision>, obstacles: &[bool]) {
    if obstacles.len() < LBM_CELL_COUNT {
        return;
    }
    for y in 0..LBM_HEIGHT {
        for x in 0..LBM_WIDTH {
            let want_solid =
                x >= LBM_INLET_GUARD_COLS && obstacles[flatten_2d_index(x, y, LBM_WIDTH)];
            if solver.is_obstacle(x, y) != want_solid {
                solver.set_obstacle(x, y, want_solid);
            }
        }
    }
}

fn render_cell_color(
    solver: &LatticeBoltzmannD2Q9<BgkCollision>,
    x: usize,
    y: usize,
    inflow_speed: f64,
    view_mode: LbmViewMode,
    high_contrast: bool,
) -> Color32 {
    if solver.is_obstacle(x, y) {
        return if high_contrast {
            crate::app::ThemeMode::HighContrast.palette().muted_text
        } else {
            crate::app::ThemeMode::Dark.palette().muted_text
        };
    }
    let rgb = match view_mode {
        LbmViewMode::Speed => {
            let v = solver.get_velocity_magnitude(x, y);
            let safe_v = if v.is_finite() { v } else { 0.0 };
            let scale = (inflow_speed.abs() * 2.2).max(0.05);
            let norm = (safe_v / scale).clamp(0.0, 1.0);
            if high_contrast {
                heatmap_color_high_contrast(norm)
            } else {
                heatmap_color(norm)
            }
        }
        LbmViewMode::Vorticity => {
            let curl = solver.vorticity_curl(x, y);
            let safe_curl = if curl.is_finite() { curl } else { 0.0 };
            let span = (VORTICITY_FULL_SCALE * (inflow_speed.abs() / 0.10).max(0.25)).max(1e-4);
            let raw = mapped_gradient(safe_curl, -span, span);
            if high_contrast {
                ensure_dark_canvas_contrast(raw)
            } else {
                raw
            }
        }
    };
    Color32::from_rgb(rgb[0], rgb[1], rgb[2])
}

/// Renders a solver state into an `egui::ColorImage` (`y = 0` at bottom of image).
#[must_use]
pub fn render_lbm_color_image(
    solver: &LatticeBoltzmannD2Q9<BgkCollision>,
    inflow_speed: f64,
    view_mode: LbmViewMode,
    high_contrast: bool,
) -> egui::ColorImage {
    let width = solver.width();
    let height = solver.height();
    let mut pixels = vec![Color32::BLACK; width * height];
    for py in 0..height {
        let gy = height - 1 - py;
        for gx in 0..width {
            let idx = flatten_2d_index(gx, py, width);
            pixels[idx] = render_cell_color(solver, gx, gy, inflow_speed, view_mode, high_contrast);
        }
    }
    egui::ColorImage::new([width, height], pixels)
}

/// Dedicated background simulation runner for the D2Q9 wind tunnel.
pub struct LbmRunner {
    pub(crate) solver: LatticeBoltzmannD2Q9<BgkCollision>,
    pub(crate) config: Arc<RwLock<LbmSharedConfig>>,
    pub(crate) shared_lattice: Arc<RwLock<LatticeState<9>>>,
    pub(crate) applied_mask_gen: u64,
    pub(crate) step_count: usize,
    pub(crate) diverged: bool,
    pub(crate) diagnostic_report: Option<DiagnosticReport>,
}

impl LbmRunner {
    /// Creates a new wind-tunnel runner bound to `config` and `shared_lattice`.
    #[must_use]
    pub fn new(
        config: Arc<RwLock<LbmSharedConfig>>,
        shared_lattice: Arc<RwLock<LatticeState<9>>>,
    ) -> Self {
        let (visc, u0, mask, gen) = {
            let cfg = config.read().unwrap_or_else(|e| e.into_inner());
            (
                cfg.viscosity,
                cfg.inflow_speed,
                cfg.obstacles.clone(),
                cfg.mask_generation,
            )
        };
        let solver = build_wind_tunnel_solver(visc, u0, &mask);
        *shared_lattice.write().unwrap_or_else(|e| e.into_inner()) = solver.state.clone();
        Self {
            solver,
            config,
            shared_lattice,
            applied_mask_gen: gen,
            step_count: 0,
            diverged: false,
            diagnostic_report: None,
        }
    }

    fn sync_from_shared_config(&mut self) -> (f64, f64, usize) {
        let (visc, u0, spf, gen, maybe_mask) = {
            let cfg = self.config.read().unwrap_or_else(|e| e.into_inner());
            let mask = (cfg.mask_generation != self.applied_mask_gen).then(|| cfg.obstacles.clone());
            (
                cfg.viscosity,
                cfg.inflow_speed,
                cfg.steps_per_frame,
                cfg.mask_generation,
                mask,
            )
        };
        if let Some(mask) = maybe_mask {
            sync_solver_obstacles(&mut self.solver, &mask);
            self.applied_mask_gen = gen;
        }
        (visc, u0, spf.max(1))
    }

    fn advance_steps(&mut self, count: usize) {
        if self.diverged {
            return;
        }
        let (visc, u0, _) = self.sync_from_shared_config();
        if !visc.is_finite() || visc <= 0.0 || !u0.is_finite() || u0 < 0.0 {
            self.diverged = true;
            self.diagnostic_report = Some(DiagnosticReport::new(
                self.step_count as f64,
                self.step_count,
                "parameters",
                format!("Invalid non-finite wind-tunnel parameters (nu={visc}, U={u0})"),
            ));
            return;
        }
        self.solver.collision_model.tau = 3.0 * visc + 0.5;
        for _ in 0..count.max(1) {
            self.solver
                .set_inlet(0, 0, LBM_INLET_GUARD_COLS, LBM_HEIGHT, u0, 0.0);
            self.solver.step();
            self.step_count += 1;
            if !FiniteStateGuard::is_finite_state(self) {
                self.diverged = true;
                self.diagnostic_report = Some(DiagnosticReport::new(
                    self.step_count as f64,
                    self.step_count,
                    "velocity",
                    "Non-finite velocity or density detected in D2Q9 wind-tunnel lattice.",
                ));
                break;
            }
        }
        *self
            .shared_lattice
            .write()
            .unwrap_or_else(|e| e.into_inner()) = self.solver.state.clone();
    }

    fn reset_flow(&mut self) {
        let (visc, u0, mask, gen) = {
            let cfg = self.config.read().unwrap_or_else(|e| e.into_inner());
            (
                cfg.viscosity,
                cfg.inflow_speed,
                cfg.obstacles.clone(),
                cfg.mask_generation,
            )
        };
        self.solver = build_wind_tunnel_solver(visc, u0, &mask);
        self.applied_mask_gen = gen;
        self.step_count = 0;
        self.diverged = false;
        self.diagnostic_report = None;
        *self
            .shared_lattice
            .write()
            .unwrap_or_else(|e| e.into_inner()) = self.solver.state.clone();
    }
}

impl FiniteStateGuard for LbmRunner {
    fn is_finite_state(&self) -> bool {
        !self.diverged
            && self.solver.state.density().iter().all(|v| v.is_finite())
            && self.solver.state.velocity_x().iter().all(|v| v.is_finite())
            && self.solver.state.velocity_y().iter().all(|v| v.is_finite())
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.diagnostic_report.as_ref()
    }

    fn diverged(&self) -> bool {
        self.diverged || !FiniteStateGuard::is_finite_state(self)
    }

    fn reset_parameters(&mut self) {
        self.reset_flow();
    }
}

impl SimulationRunner for LbmRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::Reset => self.reset_flow(),
            SimCommand::SetSpeed(spf) => {
                self.config
                    .write()
                    .unwrap_or_else(|e| e.into_inner())
                    .steps_per_frame = spf.max(1);
            }
            SimCommand::Custom(ref action) if action == "sync_mask" => {
                let _ = self.sync_from_shared_config();
                *self
                    .shared_lattice
                    .write()
                    .unwrap_or_else(|e| e.into_inner()) = self.solver.state.clone();
            }
            _ => {}
        }
    }

    fn step(&mut self) {
        let spf = self
            .config
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .steps_per_frame
            .max(1);
        self.advance_steps(spf);
    }

    fn step_once(&mut self) {
        self.advance_steps(1);
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let (visc, u0, view_mode, high_contrast, obstacles) = {
            let cfg = self.config.read().unwrap_or_else(|e| e.into_inner());
            (
                cfg.viscosity,
                cfg.inflow_speed,
                cfg.view_mode,
                cfg.high_contrast,
                cfg.obstacles.clone(),
            )
        };
        let image = render_lbm_color_image(&self.solver, u0, view_mode, high_contrast);
        let telemetry = compute_aerodynamics(
            self.solver.net_obstacle_force(),
            u0,
            visc,
            &obstacles,
            self.step_count,
        );
        StateSnapshot {
            width: image.width(),
            height: image.height(),
            pixels: Arc::new(RwLock::new(image.pixels)),
            custom_data: vec![
                telemetry.cd,
                telemetry.cl,
                telemetry.reynolds,
                telemetry.solid_cells as f64,
            ],
            structured_data: Some(Box::new(telemetry)),
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn is_finite_state(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        self.diagnostic_report.clone()
    }
}

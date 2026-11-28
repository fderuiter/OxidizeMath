//! Zero-allocation Lattice Boltzmann (`D2Q9` BGK) wind-tunnel coupling, obstacle mask
//! synchronization, live Péclet/Courant diagnostics, and combined CFL stabilization for
//! [`super::MorphogenesisTool`] and [`super::runner::MorphogenesisKineticsRunner`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::runner::MorphogenesisKineticsRunner;
use super::MorphogenesisTool;
use crate::app::SemanticPalette;
use eframe::egui;
use math_commons::math_kernel::types::{flatten_2d_index, StepSize};
use math_explorer::biology::diffusion::{CellMask2D, ReactionCellRole};
#[cfg(feature = "physics")]
pub use math_explorer::physics::fluid_dynamics::lattice_boltzmann::MAX_LATTICE_JET_SPEED;
#[cfg(feature = "physics")]
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    advance_wind_tunnel_step, project_velocity_row_major_into, sync_obstacles_row_major,
    BgkCollision, LatticeBoltzmannD2Q9,
};
#[cfg(not(feature = "physics"))]
/// Maximum interactive Lattice Boltzmann freestream inflow speed in lattice units.
pub const MAX_LATTICE_JET_SPEED: f64 = 0.16;
use math_explorer::pure_math::analysis::pde::advection_2d::{
    advection_courant_number, advection_diffusion_cfl_limit, grid_peclet_number,
};

/// Default Lattice Boltzmann horizontal inflow velocity in lattice units.
pub const DEFAULT_MORPHO_INLET_U: f64 = 0.10;
/// Default Lattice Boltzmann kinematic viscosity $\nu = (\tau - 0.5) / 3$.
pub const DEFAULT_MORPHO_VISCOSITY: f64 = 0.02;
/// Minimum interactive kinematic viscosity bound keeping BGK $\tau \ge 0.515$.
pub const MIN_MORPHO_VISCOSITY: f64 = 0.005;
/// Maximum interactive kinematic viscosity bound.
pub const MAX_MORPHO_VISCOSITY: f64 = 0.20;

/// Serde default helper for [`super::MorphogenesisToolState::inlet_u`].
#[must_use]
pub fn default_morpho_inlet_u() -> f64 {
    DEFAULT_MORPHO_INLET_U
}

/// Serde default helper for [`super::MorphogenesisToolState::viscosity`].
#[must_use]
pub fn default_morpho_viscosity() -> f64 {
    DEFAULT_MORPHO_VISCOSITY
}

/// Serde predicate omitting default `inlet_u` from URL/preset JSON payloads.
#[must_use]
pub fn is_default_morpho_inlet_u(val: &f64) -> bool {
    (*val - DEFAULT_MORPHO_INLET_U).abs() < 1e-12
}

/// Serde predicate omitting default `viscosity` from URL/preset JSON payloads.
#[must_use]
pub fn is_default_morpho_viscosity(val: &f64) -> bool {
    (*val - DEFAULT_MORPHO_VISCOSITY).abs() < 1e-12
}

/// Shared Lattice Boltzmann wind-tunnel coupling state synchronized with the worker thread.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MorphogenesisAdvectionState {
    /// Whether Lattice Boltzmann wind-tunnel co-stepping and 2D upwind advection are enabled.
    pub enabled: bool,
    /// Horizontal freestream inflow speed `u_in` (`0.0..=MAX_LATTICE_JET_SPEED`).
    pub inlet_u: f64,
    /// Lattice kinematic viscosity $\nu$ (`MIN_MORPHO_VISCOSITY..=MAX_MORPHO_VISCOSITY`).
    pub viscosity: f64,
}

impl Default for MorphogenesisAdvectionState {
    fn default() -> Self {
        Self {
            enabled: false,
            inlet_u: DEFAULT_MORPHO_INLET_U,
            viscosity: DEFAULT_MORPHO_VISCOSITY,
        }
    }
}

/// Pre-allocated `D2Q9` Lattice Boltzmann wind tunnel and row-major velocity/obstacle buffers.
pub struct MorphoWindTunnel {
    #[cfg(feature = "physics")]
    pub(crate) solver: LatticeBoltzmannD2Q9<BgkCollision>,
    #[cfg(not(feature = "physics"))]
    pub(crate) dims: (usize, usize),
    pub(crate) ux_buf: Vec<f64>,
    pub(crate) uy_buf: Vec<f64>,
    pub(crate) obstacle_buf: Vec<bool>,
    pub(crate) max_abs_ux: f64,
    pub(crate) max_abs_uy: f64,
    pub(crate) twin_step_ready: bool,
}

#[cfg(feature = "physics")]
fn detect_serpentine_baffles(mask: &CellMask2D) -> Option<(usize, usize)> {
    let w = mask.width();
    let h = mask.height();
    if w < 8 || h < 8 {
        return None;
    }
    let mut y_top = None;
    for y in 1..(h - 2) {
        let left_anchored = mask.get(0, y) == ReactionCellRole::ZeroFluxWall
            && mask.get(w / 2, y) == ReactionCellRole::ZeroFluxWall;
        let right_open = mask.get(w - 2, y) != ReactionCellRole::ZeroFluxWall;
        if left_anchored && right_open {
            y_top = Some(y);
            break;
        }
    }
    let top = y_top?;
    let mut y_bot = None;
    for y in (top + 2)..(h - 1) {
        let right_anchored = mask.get(w - 1, y) == ReactionCellRole::ZeroFluxWall
            && mask.get(w / 2, y) == ReactionCellRole::ZeroFluxWall;
        let left_open = mask.get(2, y) != ReactionCellRole::ZeroFluxWall;
        if right_anchored && left_open {
            y_bot = Some(y);
        }
    }
    y_bot.map(|bot| (top, bot))
}

impl MorphoWindTunnel {
    /// Allocates a `width x height` BGK wind tunnel and row-major projection slices.
    #[must_use]
    pub fn new(width: usize, height: usize, viscosity: f64) -> Self {
        let safe_v = viscosity.clamp(MIN_MORPHO_VISCOSITY, MAX_MORPHO_VISCOSITY);
        let _tau = (3.0 * safe_v + 0.5).max(0.515);
        let total = width * height;
        Self {
            #[cfg(feature = "physics")]
            solver: LatticeBoltzmannD2Q9::new(width, height, _tau),
            #[cfg(not(feature = "physics"))]
            dims: (width, height),
            ux_buf: vec![0.0; total],
            uy_buf: vec![0.0; total],
            obstacle_buf: vec![false; total],
            max_abs_ux: 0.0,
            max_abs_uy: 0.0,
            twin_step_ready: false,
        }
    }

    /// Returns `true` when all Lattice Boltzmann density and velocity cells are finite.
    #[must_use]
    pub fn is_finite_state(&self) -> bool {
        #[cfg(feature = "physics")]
        {
            self.solver.state.density().iter().all(|&d| d.is_finite())
                && self
                    .solver
                    .state
                    .velocity_x()
                    .iter()
                    .all(|&u| u.is_finite())
                && self
                    .solver
                    .state
                    .velocity_y()
                    .iter()
                    .all(|&v| v.is_finite())
        }
        #[cfg(not(feature = "physics"))]
        {
            self.dims.0 > 0
                && self.dims.1 > 0
                && self.ux_buf.iter().all(|&u| u.is_finite())
                && self.uy_buf.iter().all(|&v| v.is_finite())
        }
    }

    /// Synchronizes `ReactionCellRole::ZeroFluxWall` cells from `mask` into `D2Q9` bounce-back
    /// obstacles with zero heap allocations.
    pub fn sync_from_cell_mask(&mut self, mask: &CellMask2D) {
        #[cfg(feature = "physics")]
        {
            let (w, h) = (self.solver.width(), self.solver.height());
            if mask.width() != w || mask.height() != h {
                return;
            }
            let roles = mask.roles();
            for y in 0..h {
                for x in 0..w {
                    let idx = flatten_2d_index(x, y, w);
                    self.obstacle_buf[idx] = roles[idx] == ReactionCellRole::ZeroFluxWall as u8;
                }
            }
            if let Some((y_top, y_bot)) = detect_serpentine_baffles(mask) {
                for y in y_top..h {
                    for x in 0..2.min(w) {
                        self.obstacle_buf[flatten_2d_index(x, y, w)] = true;
                    }
                }
                for y in 0..=y_bot.min(h - 1) {
                    self.obstacle_buf[flatten_2d_index(w - 1, y, w)] = true;
                }
            }
            sync_obstacles_row_major(&mut self.solver, &self.obstacle_buf);
            for i in 0..self.obstacle_buf.len() {
                if self.obstacle_buf[i] {
                    self.ux_buf[i] = 0.0;
                    self.uy_buf[i] = 0.0;
                }
            }
        }
        #[cfg(not(feature = "physics"))]
        let _ = (
            mask,
            flatten_2d_index(0, 0, 1),
            ReactionCellRole::ZeroFluxWall,
        );
    }

    /// Resets `D2Q9` populations to rest equilibrium while preserving sculpted wall obstacles.
    pub fn reset_preserving_mask(&mut self, mask: &CellMask2D, viscosity: f64) {
        let safe_v = if viscosity.is_finite() {
            viscosity.clamp(MIN_MORPHO_VISCOSITY, MAX_MORPHO_VISCOSITY)
        } else {
            DEFAULT_MORPHO_VISCOSITY
        };
        let _tau = (3.0 * safe_v + 0.5).max(0.515);
        #[cfg(feature = "physics")]
        {
            self.solver.collision_model.tau = _tau;
            self.solver.clear_obstacles();
            self.solver.init_equilibrium();
        }
        self.ux_buf.fill(0.0);
        self.uy_buf.fill(0.0);
        self.max_abs_ux = 0.0;
        self.max_abs_uy = 0.0;
        self.twin_step_ready = false;
        self.sync_from_cell_mask(mask);
    }

    /// Advances one wind-tunnel LBM step and projects column-major velocities into row-major
    /// `ux_buf` and `uy_buf` slices with zero heap allocations.
    pub fn step_and_project(&mut self, inlet_u: f64, viscosity: f64) {
        let safe_v = if viscosity.is_finite() {
            viscosity.clamp(MIN_MORPHO_VISCOSITY, MAX_MORPHO_VISCOSITY)
        } else {
            DEFAULT_MORPHO_VISCOSITY
        };
        let _tau = (3.0 * safe_v + 0.5).max(0.515);
        let safe_u = if inlet_u.is_finite() {
            inlet_u.clamp(0.0, MAX_LATTICE_JET_SPEED)
        } else {
            0.0
        };
        #[cfg(feature = "physics")]
        {
            self.solver.collision_model.tau = _tau;
            advance_wind_tunnel_step(&mut self.solver, safe_u);
            project_velocity_row_major_into(&self.solver, &mut self.ux_buf, &mut self.uy_buf);
        }
        #[cfg(not(feature = "physics"))]
        self.ux_buf.fill(safe_u);
        let (mut peak_x, mut peak_y) = (0.0_f64, 0.0_f64);
        for (&vx, &vy) in self.ux_buf.iter().zip(self.uy_buf.iter()) {
            peak_x = peak_x.max(vx.abs());
            peak_y = peak_y.max(vy.abs());
        }
        self.max_abs_ux = peak_x;
        self.max_abs_uy = peak_y;
    }

    /// Borrowed row-major horizontal velocity slice (`width * height`).
    #[inline]
    #[must_use]
    pub fn ux(&self) -> &[f64] {
        &self.ux_buf
    }

    /// Borrowed row-major vertical velocity slice (`width * height`).
    #[inline]
    #[must_use]
    pub fn uy(&self) -> &[f64] {
        &self.uy_buf
    }
}

impl MorphogenesisKineticsRunner {
    pub(crate) fn read_advection(&self) -> MorphogenesisAdvectionState {
        *self
            .advection_state
            .read()
            .unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn step_coupled_or_isotropic(&mut self, dt: f64) {
        let adv = self.read_advection();
        if !adv.enabled {
            self.wind_tunnel.twin_step_ready = false;
            let (subs, sub_dt) = if self.auto_dt.load(std::sync::atomic::Ordering::Relaxed) {
                math_explorer::biology::morphogenesis::FusedEulerSolver::cfl_substeps_2d(
                    dt,
                    1.0,
                    self.system.diffusion_coeffs[0],
                    self.system.diffusion_coeffs[1],
                    super::runner::MAX_CFL_SUBSTEPS,
                )
            } else {
                (1, dt)
            };
            for _ in 0..subs {
                self.system.step(sub_dt);
            }
            return;
        }
        if !std::mem::take(&mut self.wind_tunnel.twin_step_ready) {
            self.wind_tunnel
                .step_and_project(adv.inlet_u, adv.viscosity);
        }
        let d_max = self.system.diffusion_coeffs[0]
            .max(self.system.diffusion_coeffs[1])
            .max(0.0);
        let cfl_limit = advection_diffusion_cfl_limit(
            StepSize(1.0),
            StepSize(1.0),
            d_max,
            self.wind_tunnel.max_abs_ux,
            self.wind_tunnel.max_abs_uy,
        );
        let safe_dt = if cfl_limit.is_finite() && cfl_limit > 0.0 {
            dt.min(0.98 * cfl_limit)
        } else {
            dt
        };
        self.system
            .step_advected(safe_dt, self.wind_tunnel.ux(), self.wind_tunnel.uy());
    }
}

impl MorphogenesisTool {
    /// Returns `(peclet, courant, cfl_limit, combined_cfl)` for the active morphogenesis parameters.
    #[must_use]
    pub fn advection_diagnostics(&self) -> (f64, f64, f64, f64) {
        let (d_u, d_v, dt) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.d_u, p.d_v, p.dt)
        };
        let adv = *self
            .advection_state
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let (max_ux, max_uy) = if adv.enabled {
            let safe_inlet = adv.inlet_u.clamp(0.0, MAX_LATTICE_JET_SPEED);
            (
                self.live_velocity_peaks[0].max(safe_inlet),
                self.live_velocity_peaks[1],
            )
        } else {
            (0.0, 0.0)
        };
        let speed = max_ux.hypot(max_uy);
        let d_max = d_u.max(d_v).max(0.0);
        let d_min = d_u.min(d_v).max(1e-12);
        let peclet = grid_peclet_number(speed, 1.0, d_min);
        let courant = advection_courant_number(dt, StepSize(1.0), StepSize(1.0), max_ux, max_uy);
        let cfl_limit =
            advection_diffusion_cfl_limit(StepSize(1.0), StepSize(1.0), d_max, max_ux, max_uy);
        let combined_cfl = if cfl_limit.is_finite() && cfl_limit > 0.0 {
            dt / cfl_limit
        } else {
            0.0
        };
        (peclet, courant, cfl_limit, combined_cfl)
    }

    pub(crate) fn stabilize_combined_cfl_step(
        &mut self,
        ctx: &egui::Context,
        min_dt: f64,
        max_dt: f64,
    ) {
        let (_, _, cfl_limit, _) = self.advection_diagnostics();
        let safe_dt = {
            let mut params = self.params.write().unwrap_or_else(|e| e.into_inner());
            let raw_dt = if cfl_limit.is_finite() && cfl_limit > 0.0 {
                0.95 * cfl_limit
            } else {
                0.2485 / params.d_u.max(params.d_v).max(1e-6)
            };
            let clamped = raw_dt.clamp(min_dt, max_dt);
            params.dt = clamped;
            clamped
        };
        self.reset_system();
        crate::accessibility::announce_live_status(
            ctx,
            format!("Time step stabilized to CFL bound dt = {safe_dt:.4}"),
        );
    }

    pub(crate) fn clamp_advection_and_cfl(&self, d_u: f64, d_v: f64, dt: &mut f64) {
        let (adv_enabled, safe_inlet) = {
            let mut adv = self
                .advection_state
                .write()
                .unwrap_or_else(|e| e.into_inner());
            adv.inlet_u = if adv.inlet_u.is_finite() {
                adv.inlet_u.clamp(0.0, MAX_LATTICE_JET_SPEED)
            } else {
                0.0
            };
            adv.viscosity = if adv.viscosity.is_finite() {
                adv.viscosity
                    .clamp(MIN_MORPHO_VISCOSITY, MAX_MORPHO_VISCOSITY)
            } else {
                default_morpho_viscosity()
            };
            (adv.enabled, adv.inlet_u)
        };
        if adv_enabled {
            let cfl = advection_diffusion_cfl_limit(
                StepSize(1.0),
                StepSize(1.0),
                d_u.max(d_v),
                safe_inlet,
                0.0,
            );
            if cfl.is_finite() && cfl > 0.0 {
                *dt = (*dt).min(0.95 * cfl).max(1e-4);
            }
        }
    }

    pub(crate) fn draw_wind_tunnel_controls(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.label("Fluid Wind-Tunnel Coupling");
        let mut toggled = None;
        {
            let mut adv = self
                .advection_state
                .write()
                .unwrap_or_else(|e| e.into_inner());
            if ui
                .checkbox(&mut adv.enabled, "🌊 Enable LBM Wind Tunnel")
                .changed()
            {
                toggled = Some(adv.enabled);
            }
            if adv.enabled {
                ui.add(
                    egui::Slider::new(&mut adv.inlet_u, 0.0..=MAX_LATTICE_JET_SPEED)
                        .step_by(0.005)
                        .text("Inflow u_in"),
                );
                ui.add(
                    egui::Slider::new(
                        &mut adv.viscosity,
                        MIN_MORPHO_VISCOSITY..=MAX_MORPHO_VISCOSITY,
                    )
                    .step_by(0.005)
                    .text("Viscosity ν"),
                );
            }
        }
        if let Some(enabled) = toggled {
            if !enabled {
                self.live_velocity_peaks = [0.0, 0.0];
            }
            self.dirty = true;
            let status = if enabled {
                "Coupled Lattice Boltzmann wind-tunnel flow enabled"
            } else {
                "Coupled Lattice Boltzmann wind-tunnel flow disabled"
            };
            crate::accessibility::announce_live_status(ui.ctx(), status.to_string());
        }
        let (peclet, courant, _, combined_cfl) = self.advection_diagnostics();
        let cfl_color = if combined_cfl >= 1.0 {
            palette.warning
        } else {
            palette.muted_text
        };
        ui.colored_label(
            palette.info,
            format!("Grid Péclet Pe = {peclet:.3} | Advective Co = {courant:.3}"),
        );
        ui.colored_label(
            cfl_color,
            format!("Combined Advective-Diffusive CFL = {combined_cfl:.3}"),
        );
    }
}

#[cfg(all(test, feature = "physics"))]
pub use super::starter_presets::PRESET_VORTEX_WAKE_MORPHOGENESIS;

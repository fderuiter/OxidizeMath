//! Preallocated zero-allocation co-stepped `LatticeBoltzmannD2Q9` + `TuringSystem` engine
//! for [`super::AdvectiveMorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets::{AdvectiveKineticsPreset, AdvectiveToolState};
use crate::framework::{
    LBM_CELL_COUNT, LBM_HEIGHT, LBM_INLET_COLS, LBM_REF_DIAMETER, LBM_RHO_0, LBM_U_IN, LBM_WIDTH,
};
use crate::tabs::morphogenesis::SimpleRng;
use math_commons::math_kernel::types::{flatten_2d_index, Dimension, StepSize};
use math_explorer::biology::diffusion::AdvectiveFiniteDifference2D;
use math_explorer::biology::morphogenesis::{
    DiffusionCoeff, GrayScottKinetics, SchnakenbergKinetics, TuringSystem,
};
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    advance_wind_tunnel_step, compute_drag_coefficient, compute_vorticity_into, BgkCollision,
    LatticeBoltzmannD2Q9,
};

const BASE_RD_SUBSTEPS: usize = 2;
const MAX_RD_SUBSTEPS: usize = 8;
const CFL_SAFETY: f64 = 0.45;
const MAX_COUPLED_SPEED: f64 = 0.22;

/// Live quantitative hydrodynamic and chemical wake metrics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AdvectiveWakeMetrics {
    /// Bounce-back horizontal drag force `F_D`.
    pub f_drag: f64,
    /// Bounce-back vertical lift force `F_L`.
    pub f_lift: f64,
    /// Dimensionless drag coefficient `C_D`.
    pub c_d: f64,
    /// Estimated obstacle Reynolds number `Re = U_in * D_ref / nu`.
    pub reynolds: f64,
    /// Peak fluid vorticity magnitude `max |omega_z|`.
    pub max_vorticity: f64,
    /// CFL-clamped reaction-diffusion step size `dt_eff`.
    pub effective_dt: f64,
    /// Combined diffusive + advective CFL ceiling `dt_max`.
    pub cfl_limit: f64,
    /// Whether `requested_dt` was clamped by `cfl_limit`.
    pub cfl_clamped: bool,
    /// Wake transverse-to-streamwise gradient ratio (`> 1` indicates streamwise plume stretching).
    pub plume_elongation: f64,
    /// Solid obstacle cell count.
    pub solid_cells: usize,
    /// Total co-stepped fluid ticks executed since last reset.
    pub step_count: u64,
}

/// Co-stepped `100 x 50` D2Q9 fluid + reaction-diffusion engine with preallocated row-major buffers.
pub struct AdvectiveMorphogenesisEngine {
    /// D2Q9 BGK fluid wind-tunnel solver (`100 x 50`).
    pub lbm: LatticeBoltzmannD2Q9<BgkCollision>,
    /// Preallocated Schnakenberg advective reaction-diffusion solver.
    pub schnak: TuringSystem<2, SchnakenbergKinetics, AdvectiveFiniteDifference2D>,
    /// Preallocated Gray-Scott advective reaction-diffusion solver.
    pub gray_scott: TuringSystem<2, GrayScottKinetics, AdvectiveFiniteDifference2D>,
    /// Row-major discrete fluid curl $\omega_z$ (`5,000` cells).
    pub vorticity: Vec<f64>,
    /// Row-major solid obstacle mask (`5,000` cells).
    pub mask: [bool; LBM_CELL_COUNT],
    /// Latest quantitative hydrodynamic and chemical wake metrics.
    pub metrics: AdvectiveWakeMetrics,
}

impl AdvectiveMorphogenesisEngine {
    /// Allocates all `100 x 50` fluid and reaction-diffusion buffers up front without heavy stepping.
    #[must_use]
    pub fn new(cfg: &AdvectiveToolState) -> Self {
        let tau = 3.0 * cfg.viscosity.clamp(0.018, 0.25) + 0.5;
        let lbm = LatticeBoltzmannD2Q9::new(LBM_WIDTH, LBM_HEIGHT, tau);
        let dims = (Dimension(LBM_WIDTH), Dimension(LBM_HEIGHT));
        let spacing = (StepSize(1.0), StepSize(1.0));
        let schnak = TuringSystem::new_advective_2d(
            dims,
            DiffusionCoeff(cfg.d_u),
            DiffusionCoeff(cfg.d_v),
            spacing,
            cfg.peclet,
            SchnakenbergKinetics::new(cfg.a, cfg.b),
        );
        let gray_scott = TuringSystem::new_advective_2d(
            dims,
            DiffusionCoeff(cfg.d_u),
            DiffusionCoeff(cfg.d_v),
            spacing,
            cfg.peclet,
            GrayScottKinetics::new(cfg.f, cfg.k),
        );
        let mut mask = [false; LBM_CELL_COUNT];
        cfg.stamp.write_mask(&mut mask);
        let mut engine = Self {
            lbm,
            schnak,
            gray_scott,
            vorticity: vec![0.0; LBM_CELL_COUNT],
            mask,
            metrics: AdvectiveWakeMetrics::default(),
        };
        engine.reset_fluid_wake(cfg.viscosity);
        engine.reseed_morphogens(cfg);
        engine.refresh_metrics(cfg);
        engine
    }

    /// Synchronizes obstacle cells onto `lbm` and both `TuringSystem` masks without heap allocation.
    pub fn sync_obstacle_mask(&mut self, cfg: &AdvectiveToolState) {
        let mut n_solid = 0_usize;
        for y in 0..LBM_HEIGHT {
            for x in 0..LBM_WIDTH {
                let idx = flatten_2d_index(x, y, LBM_WIDTH);
                let is_obs = x >= LBM_INLET_COLS && self.mask[idx];
                self.mask[idx] = is_obs;
                if is_obs {
                    n_solid += 1;
                }
                self.lbm.set_obstacle(x, y, is_obs);
            }
        }
        self.metrics.solid_cells = n_solid;
        let (_, _, obs_s) = self.schnak.velocity_and_obstacles_mut();
        obs_s.copy_from_slice(&self.mask);
        let (_, _, obs_g) = self.gray_scott.velocity_and_obstacles_mut();
        obs_g.copy_from_slice(&self.mask);
        self.pin_solid_cells(cfg);
    }

    /// Re-initializes the D2Q9 wind tunnel cleanly with uniform freestream inflow `U_in = 0.1`.
    pub fn reset_fluid_wake(&mut self, viscosity: f64) {
        let clean_visc = viscosity.clamp(0.018, 0.25);
        self.lbm.collision_model.tau = 3.0 * clean_visc + 0.5;
        self.lbm.init_equilibrium();
        for y in 0..LBM_HEIGHT {
            for x in 0..LBM_WIDTH {
                let idx = flatten_2d_index(x, y, LBM_WIDTH);
                let is_obs = x >= LBM_INLET_COLS && self.mask[idx];
                self.lbm.set_obstacle(x, y, is_obs);
            }
        }
        self.lbm
            .set_inlet(0, 0, LBM_WIDTH, LBM_HEIGHT, LBM_U_IN, 0.0);
        // Tiny localized transverse perturbation in the lee to seed unsteady vortex shedding cleanly
        for y in 20..30 {
            for x in 37..43 {
                if !self.lbm.is_obstacle(x, y) {
                    let kick = if y < 25 { 0.004 } else { -0.004 };
                    self.lbm.set_inlet(x, y, 1, 1, LBM_U_IN, kick);
                }
            }
        }
        self.vorticity.fill(0.0);
        self.metrics.step_count = 0;
        self.metrics.f_drag = 0.0;
        self.metrics.f_lift = 0.0;
        self.metrics.c_d = 0.0;
    }

    /// Seeds deterministic activator/inhibitor perturbations across fluid cells outside `mask`.
    pub fn reseed_morphogens(&mut self, cfg: &AdvectiveToolState) {
        self.sync_parameters(cfg);
        let mut rng = SimpleRng::new(cfg.rng_seed);
        let u0 = (cfg.a + cfg.b).max(0.05);
        let v0 = cfg.b / (u0 * u0);

        for y in 0..LBM_HEIGHT {
            for x in 0..LBM_WIDTH {
                let idx = flatten_2d_index(x, y, LBM_WIDTH);
                if self.mask[idx] {
                    self.schnak.u_mut()[idx] = u0;
                    self.schnak.v_mut()[idx] = v0;
                    self.gray_scott.u_mut()[idx] = 1.0;
                    self.gray_scott.v_mut()[idx] = 0.0;
                    continue;
                }
                let wave = ((x as f64 * 0.62).cos() * (y as f64 * 0.62).sin()) * 0.16;
                let jitter = (rng.next_f64() - 0.5) * 0.08;
                self.schnak.u_mut()[idx] = (u0 + wave + jitter).max(0.01);
                self.schnak.v_mut()[idx] = (v0 - 0.5 * wave - 0.5 * jitter).max(0.01);
                let gs_seed = gray_scott_wake_seed(x, y, rng.next_f64());
                self.gray_scott.u_mut()[idx] = 1.0 - 0.5 * gs_seed;
                self.gray_scott.v_mut()[idx] = 0.28 * gs_seed;
            }
        }
        self.sync_obstacle_mask(cfg);
    }

    /// Updates kinetic constants, diffusivities, viscosity, and Péclet gain without reallocating.
    pub fn sync_parameters(&mut self, cfg: &AdvectiveToolState) {
        let clean_visc = if cfg.viscosity.is_finite() {
            cfg.viscosity.clamp(0.018, 0.25)
        } else {
            0.020
        };
        self.lbm.collision_model.tau = 3.0 * clean_visc + 0.5;
        self.schnak.kinetics = SchnakenbergKinetics::new(cfg.a, cfg.b);
        self.schnak.diffusion_coeffs = [cfg.d_u, cfg.d_v];
        self.schnak.set_peclet(cfg.peclet);

        self.gray_scott.kinetics = GrayScottKinetics::new(cfg.f, cfg.k);
        self.gray_scott.diffusion_coeffs = [cfg.d_u, cfg.d_v];
        self.gray_scott.set_peclet(cfg.peclet);
    }

    /// Executes one zero-heap-allocation co-stepped D2Q9 + reaction-diffusion tick.
    pub fn step_coupled(&mut self, cfg: &AdvectiveToolState) {
        self.sync_parameters(cfg);
        advance_wind_tunnel_step(&mut self.lbm, LBM_U_IN);
        compute_vorticity_into(&self.lbm, &mut self.vorticity);

        if cfg.preset.is_gray_scott() {
            {
                let (ux, uy, obs) = self.gray_scott.velocity_and_obstacles_mut();
                self.lbm.copy_velocity_and_obstacles_row_major(ux, uy, obs);
                clamp_coupled_velocities(ux, uy);
            }
            self.gray_scott.pin_solid_species([1.0, 0.0]);
            let cfl = self.gray_scott.max_cfl_dt(CFL_SAFETY);
            let (substeps, dt_eff) = cfl_substep_plan(cfg.dt, cfl);
            for _ in 0..substeps {
                self.gray_scott.step(dt_eff);
                clamp_species_slice(self.gray_scott.u_mut(), 1.0);
                clamp_species_slice(self.gray_scott.v_mut(), 0.0);
            }
        } else {
            let u0 = (cfg.a + cfg.b).max(0.05);
            let v0 = cfg.b / (u0 * u0);
            {
                let (ux, uy, obs) = self.schnak.velocity_and_obstacles_mut();
                self.lbm.copy_velocity_and_obstacles_row_major(ux, uy, obs);
                clamp_coupled_velocities(ux, uy);
            }
            self.schnak.pin_solid_species([u0, v0]);
            let cfl = self.schnak.max_cfl_dt(CFL_SAFETY);
            let (substeps, dt_eff) = cfl_substep_plan(cfg.dt, cfl);
            for _ in 0..substeps {
                self.schnak.step(dt_eff);
            }
        }
        self.metrics.step_count = self.metrics.step_count.saturating_add(1);
        self.refresh_metrics(cfg);
    }

    /// Returns the primary visualized morphogen slice (`u` for Schnakenberg, `v` for Gray-Scott).
    #[must_use]
    pub fn active_morphogen(&self, preset: AdvectiveKineticsPreset) -> &[f64] {
        if preset.is_gray_scott() {
            self.gray_scott.v()
        } else {
            self.schnak.u()
        }
    }

    /// Returns the secondary species slice (`v` for Schnakenberg, `u` for Gray-Scott).
    #[must_use]
    pub fn secondary_morphogen(&self, preset: AdvectiveKineticsPreset) -> &[f64] {
        if preset.is_gray_scott() {
            self.gray_scott.u()
        } else {
            self.schnak.v()
        }
    }

    /// Recomputes live hydrodynamic forces, CFL bounds, and wake plume elongation in place.
    pub fn refresh_metrics(&mut self, cfg: &AdvectiveToolState) {
        let (raw_fd, raw_fl) = self.lbm.hydrodynamic_forces();
        let (fd, fl) = if self.metrics.step_count <= 1 {
            (raw_fd, raw_fl)
        } else {
            (
                0.85 * self.metrics.f_drag + 0.15 * raw_fd,
                0.85 * self.metrics.f_lift + 0.15 * raw_fl,
            )
        };
        let c_d = compute_drag_coefficient(fd, LBM_RHO_0, LBM_U_IN, LBM_REF_DIAMETER);
        let visc = cfg.viscosity.clamp(0.018, 0.25);
        let cfl = if cfg.preset.is_gray_scott() {
            self.gray_scott.max_cfl_dt(CFL_SAFETY)
        } else {
            self.schnak.max_cfl_dt(CFL_SAFETY)
        };
        let (_, dt_eff) = cfl_substep_plan(cfg.dt, cfl);
        let max_vort = self
            .vorticity
            .iter()
            .copied()
            .fold(0.0_f64, |acc, v| acc.max(v.abs()));
        let field = self.active_morphogen(cfg.preset);
        let elongation = compute_wake_plume_elongation(field, &self.mask);

        self.metrics.f_drag = fd;
        self.metrics.f_lift = fl;
        self.metrics.c_d = c_d;
        self.metrics.reynolds = (LBM_U_IN * LBM_REF_DIAMETER) / visc;
        self.metrics.max_vorticity = max_vort;
        self.metrics.cfl_limit = cfl;
        self.metrics.effective_dt = dt_eff;
        self.metrics.cfl_clamped = cfg.dt > cfl + 1e-9;
        self.metrics.plume_elongation = elongation;
        self.metrics.solid_cells = self.mask.iter().filter(|&&b| b).count();
    }

    fn pin_solid_cells(&mut self, cfg: &AdvectiveToolState) {
        let u0 = (cfg.a + cfg.b).max(0.05);
        let v0 = cfg.b / (u0 * u0);
        self.schnak.pin_solid_species([u0, v0]);
        self.gray_scott.pin_solid_species([1.0, 0.0]);
    }
}

fn cfl_substep_plan(requested_dt: f64, cfl: f64) -> (usize, f64) {
    let clean_req = if requested_dt.is_finite() {
        requested_dt.max(1e-4)
    } else {
        0.05
    };
    let target_total = (BASE_RD_SUBSTEPS as f64) * clean_req;
    let substeps =
        ((target_total / cfl.max(1e-6)).ceil() as usize).clamp(BASE_RD_SUBSTEPS, MAX_RD_SUBSTEPS);
    let dt_eff = (target_total / (substeps as f64)).clamp(1e-4, cfl);
    (substeps, dt_eff)
}

fn clamp_coupled_velocities(ux: &mut [f64], uy: &mut [f64]) {
    for v in ux.iter_mut() {
        *v = if v.is_finite() {
            v.clamp(-MAX_COUPLED_SPEED, MAX_COUPLED_SPEED)
        } else {
            0.0
        };
    }
    for v in uy.iter_mut() {
        *v = if v.is_finite() {
            v.clamp(-MAX_COUPLED_SPEED, MAX_COUPLED_SPEED)
        } else {
            0.0
        };
    }
}

fn clamp_species_slice(slice: &mut [f64], fallback: f64) {
    for val in slice.iter_mut() {
        *val = if val.is_finite() {
            val.clamp(0.0, 1.2)
        } else {
            fallback
        };
    }
}

fn gray_scott_wake_seed(x: usize, y: usize, noise: f64) -> f64 {
    const NUCLEI: &[(i32, i32)] = &[
        (16, 25),
        (22, 18),
        (22, 32),
        (39, 21),
        (39, 29),
        (49, 25),
        (60, 19),
        (60, 31),
        (72, 25),
    ];
    for &(nx, ny) in NUCLEI {
        let dx = x as i32 - nx;
        let dy = y as i32 - ny;
        if dx * dx + dy * dy <= 7 {
            return (0.92 + 0.08 * noise).clamp(0.0, 1.0);
        }
    }
    0.0
}

fn compute_wake_plume_elongation(field: &[f64], mask: &[bool; LBM_CELL_COUNT]) -> f64 {
    let mut sum_gx_sq = 0.0_f64;
    let mut sum_gy_sq = 0.0_f64;
    for y in 6..(LBM_HEIGHT - 6) {
        for x in 36..(LBM_WIDTH - 4) {
            let idx = flatten_2d_index(x, y, LBM_WIDTH);
            if mask[idx]
                || mask[idx - 1]
                || mask[idx + 1]
                || mask[idx - LBM_WIDTH]
                || mask[idx + LBM_WIDTH]
            {
                continue;
            }
            let gx = 0.5 * (field[idx + 1] - field[idx - 1]);
            let gy = 0.5 * (field[idx + LBM_WIDTH] - field[idx - LBM_WIDTH]);
            sum_gx_sq += gx * gx;
            sum_gy_sq += gy * gy;
        }
    }
    if sum_gx_sq <= 1e-15 {
        1.0
    } else {
        (sum_gy_sq / sum_gx_sq).sqrt()
    }
}

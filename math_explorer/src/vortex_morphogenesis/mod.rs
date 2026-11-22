//! Coupled D2Q9 Lattice Boltzmann and 2D Reaction-Advection-Diffusion Morphogenesis Engine.
//!
//! Bridges column-major `nalgebra::DMatrix` fluid states from [`domain_physics`] into preallocated
//! row-major velocity and obstacle masks consumed by [`domain_biology::biology::diffusion::FiniteDifference2D`]
//! with zero heap allocations on the simulation hot path.

pub mod presets;

pub use presets::{
    DimensionlessRegimeMetrics, MAX_ADVECTIVE_CFL, MAX_DIFFUSIVE_CFL, MAX_TRANSPORT_SUBSTEPS,
    MorphogenKinetics, VELOCITY_SAFETY_CAP, VortexMorphogenesisConfig, VortexRegimePreset,
};

use domain_biology::biology::diffusion::{AdvectionField2D, FiniteDifference2D};
use domain_biology::biology::morphogenesis::TuringState;
use domain_physics::physics::fluid_dynamics::lattice_boltzmann::{
    BgkCollision, LatticeBoltzmannD2Q9,
};
use math_commons::math_kernel::types::{Dimension, StepSize, flatten_2d_index};

/// Zero-allocation coupled fluid-morphogenesis engine pairing `LatticeBoltzmannD2Q9` with 2D upwind
/// reaction-diffusion transport.
pub struct CoupledVortexMorphogenesis {
    /// Active simulation parameters.
    pub config: VortexMorphogenesisConfig,
    /// Underlying D2Q9 BGK Lattice Boltzmann fluid solver.
    pub lbm: LatticeBoltzmannD2Q9<BgkCollision>,
    /// 2D finite-difference diffusion/advection operator.
    pub fd2d: FiniteDifference2D,
    /// Active two-species morphogen concentrations (`u` = species 0, `v` = species 1).
    pub morphogens: TuringState<2>,
    next_morphogens: TuringState<2>,
    /// Preallocated row-major horizontal velocity buffer (`y * width + x`).
    ux_row: Vec<f64>,
    /// Preallocated row-major vertical velocity buffer (`y * width + x`).
    uy_row: Vec<f64>,
    /// Preallocated row-major solid obstacle mask (`true` = bounce-back & zero-flux solid).
    obstacles_row: Vec<bool>,
    /// Preallocated row-major chemical emitter mask (`true` = active morphogen source).
    emitters_row: Vec<bool>,
    /// Preallocated row-major scalar curl ($\omega_z = \partial_x u_y - \partial_y u_x$) buffer.
    vorticity_row: Vec<f64>,
    last_metrics: DimensionlessRegimeMetrics,
}

impl Default for CoupledVortexMorphogenesis {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::from_preset(VortexRegimePreset::KarmanVortexSpots)
    }
}

impl CoupledVortexMorphogenesis {
    /// Constructs a new coupled engine with preallocated fluid and morphogen buffers.
    #[verified_engine::verified]
    pub fn new(config: VortexMorphogenesisConfig) -> Self {
        let width = config.width.max(8);
        let height = config.height.max(8);
        let n = width * height;
        let mut sanitized = config;
        sanitized.width = width;
        sanitized.height = height;
        let dx = sanitized.dx.max(0.05);
        sanitized.dx = dx;
        sanitized.inlet_ux = sanitized
            .inlet_ux
            .clamp(-VELOCITY_SAFETY_CAP, VELOCITY_SAFETY_CAP);

        let mut engine = Self {
            config: sanitized,
            lbm: LatticeBoltzmannD2Q9::new(width, height, sanitized.tau.max(0.51)),
            fd2d: FiniteDifference2D::new(
                Dimension(width),
                Dimension(height),
                StepSize(dx),
                StepSize(dx),
            ),
            morphogens: TuringState::new(n),
            next_morphogens: TuringState::new(n),
            ux_row: vec![0.0; n],
            uy_row: vec![0.0; n],
            obstacles_row: vec![false; n],
            emitters_row: vec![false; n],
            vorticity_row: vec![0.0; n],
            last_metrics: DimensionlessRegimeMetrics {
                peclet: 0.0,
                damkohler: 0.0,
                max_speed: 0.0,
                substeps: 1,
            },
        };
        engine.reset_homogeneous_morphogens();
        engine.transpose_lbm_to_row_major();
        engine
    }

    /// Constructs and seeds one of the four canonical physical-biological presets.
    #[verified_engine::verified]
    pub fn from_preset(preset: VortexRegimePreset) -> Self {
        let mut engine = Self::new(VortexMorphogenesisConfig::for_preset(preset));
        engine.apply_preset(preset);
        engine
    }

    /// Resets morphogen concentrations to the homogeneous equilibrium of the active kinetics.
    #[verified_engine::verified]
    pub fn reset_homogeneous_morphogens(&mut self) {
        let (u0, v0) = self.config.kinetics.homogeneous_feed();
        self.morphogens.u_mut().fill(u0);
        self.morphogens.v_mut().fill(v0);
        self.next_morphogens.u_mut().fill(u0);
        self.next_morphogens.v_mut().fill(v0);
    }

    /// Configures geometry, initial perturbations, and wake spin-up for `preset`.
    #[verified_engine::verified]
    pub fn apply_preset(&mut self, preset: VortexRegimePreset) {
        let cfg = VortexMorphogenesisConfig::for_preset(preset);
        if cfg.width != self.config.width || cfg.height != self.config.height {
            *self = Self::new(cfg);
        } else {
            let dx = cfg.dx.max(0.05);
            self.config = cfg;
            self.config.dx = dx;
            self.lbm = LatticeBoltzmannD2Q9::new(cfg.width, cfg.height, cfg.tau.max(0.51));
            self.fd2d = FiniteDifference2D::new(
                Dimension(cfg.width),
                Dimension(cfg.height),
                StepSize(dx),
                StepSize(dx),
            );
            self.obstacles_row.fill(false);
            self.emitters_row.fill(false);
        }
        self.reset_homogeneous_morphogens();

        match preset {
            VortexRegimePreset::KarmanVortexSpots => self.seed_karman_spots(),
            VortexRegimePreset::ShearAlignedStripes => self.seed_shear_stripes(),
            VortexRegimePreset::StagnationPlumes => self.seed_stagnation_plumes(),
            VortexRegimePreset::DamkohlerWashout => self.seed_damkohler_washout(),
        }
        self.transpose_lbm_to_row_major();
    }

    /// Paints a circular solid obstacle (`is_obstacle = true`) or clears it (`false`), immediately
    /// updating both LBM bounce-back and zero-flux morphogen masks.
    #[verified_engine::verified]
    pub fn paint_obstacle_disc(&mut self, cx: usize, cy: usize, radius: usize, is_obstacle: bool) {
        let w = self.config.width;
        let h = self.config.height;
        let r_sq = (radius * radius) as isize;
        for y in 0..h {
            for x in 0..w {
                let dx = x as isize - cx as isize;
                let dy = y as isize - cy as isize;
                if dx * dx + dy * dy <= r_sq {
                    self.lbm.set_obstacle(x, y, is_obstacle);
                    let idx = flatten_2d_index(x, y, w);
                    self.obstacles_row[idx] = is_obstacle;
                    if is_obstacle {
                        self.emitters_row[idx] = false;
                        self.morphogens.u_mut()[idx] = 0.0;
                        self.morphogens.v_mut()[idx] = 0.0;
                    }
                }
            }
        }
    }

    /// Paints or clears a circular morphogen emitter brush at `(cx, cy)` and seeds initial catalyst.
    #[verified_engine::verified]
    pub fn paint_emitter_disc(&mut self, cx: usize, cy: usize, radius: usize, is_emitter: bool) {
        let w = self.config.width;
        let h = self.config.height;
        let r_sq = (radius * radius) as isize;
        for y in 0..h {
            for x in 0..w {
                let dx = x as isize - cx as isize;
                let dy = y as isize - cy as isize;
                if dx * dx + dy * dy <= r_sq {
                    let idx = flatten_2d_index(x, y, w);
                    if !self.obstacles_row[idx] {
                        self.emitters_row[idx] = is_emitter;
                        if is_emitter {
                            self.seed_catalyst_cell(idx);
                        }
                    }
                }
            }
        }
    }

    /// Clears all obstacles and chemical emitters from the domain.
    #[verified_engine::verified]
    pub fn clear_geometry(&mut self) {
        self.lbm.clear_obstacles();
        self.obstacles_row.fill(false);
        self.emitters_row.fill(false);
    }

    /// Transposes column-major `nalgebra::DMatrix` velocity and obstacle fields from `LatticeState`
    /// into preallocated row-major slices while clamping $|\mathbf{u}| \le 0.15$ and computing $\omega_z$.
    #[verified_engine::verified]
    pub fn transpose_lbm_to_row_major(&mut self) {
        let w = self.config.width;
        let h = self.config.height;
        let ux_col = self.lbm.state.velocity_x();
        let uy_col = self.lbm.state.velocity_y();
        let obs_col = self.lbm.state.obstacles();
        let coupling = self.config.advection_coupling.clamp(0.0, 2.0);
        let mut max_speed = 0.0_f64;
        let mut obs_count = 0_usize;

        for y in 0..h {
            for x in 0..w {
                let col_idx = flatten_2d_index(y, x, h);
                let row_idx = flatten_2d_index(x, y, w);
                let is_obs = obs_col[col_idx];
                self.obstacles_row[row_idx] = is_obs;
                if is_obs {
                    obs_count += 1;
                    self.ux_row[row_idx] = 0.0;
                    self.uy_row[row_idx] = 0.0;
                    continue;
                }
                let raw_ux = ux_col[col_idx] * coupling;
                let raw_uy = uy_col[col_idx] * coupling;
                let speed = (raw_ux * raw_ux + raw_uy * raw_uy).sqrt();
                let (ux, uy, clamped_speed) = if speed > VELOCITY_SAFETY_CAP {
                    let scale = VELOCITY_SAFETY_CAP / speed;
                    (raw_ux * scale, raw_uy * scale, VELOCITY_SAFETY_CAP)
                } else {
                    (raw_ux, raw_uy, speed)
                };
                self.ux_row[row_idx] = ux;
                self.uy_row[row_idx] = uy;
                if clamped_speed > max_speed {
                    max_speed = clamped_speed;
                }
            }
        }
        self.compute_vorticity();
        self.update_metrics(max_speed, obs_count, self.last_metrics.substeps);
    }

    /// Advances coupled LBM fluid dynamics and CFL-substepped morphogen reaction-advection-diffusion
    /// with zero dynamic heap allocations.
    #[verified_engine::verified]
    pub fn step(&mut self) {
        let dx = self.config.dx.max(0.05);
        self.config.dx = dx;
        if (*self.fd2d.geometry.dx - dx).abs() > 1e-12 {
            self.fd2d = FiniteDifference2D::new(
                Dimension(self.config.width),
                Dimension(self.config.height),
                StepSize(dx),
                StepSize(dx),
            );
        }
        self.lbm.collision_model.tau = self.config.tau.max(0.51);
        let lbm_iters = self.config.lbm_steps_per_tick.clamp(1, 8);
        for _ in 0..lbm_iters {
            self.apply_inlet_drive();
            self.lbm.step();
            if !self.config.shear_profile {
                self.lbm.apply_open_outlet();
            }
        }

        self.transpose_lbm_to_row_major();
        let substeps = self.required_substeps();
        self.last_metrics.substeps = substeps;
        let dt_sub = self.config.dt.max(0.0) / (substeps as f64);
        let coeffs = [self.config.d_u.max(0.0), self.config.d_v.max(0.0)];

        for _ in 0..substeps {
            let field =
                AdvectionField2D::with_obstacles(&self.ux_row, &self.uy_row, &self.obstacles_row)
                    .with_periodic_x(self.config.shear_profile);
            let (u_src, v_src) = self.morphogens.u_and_v();
            let (u_dst, v_dst) = self.next_morphogens.u_and_v_mut();
            self.fd2d.step_advection_diffusion_masked(
                [u_src, v_src],
                [u_dst, v_dst],
                field,
                dt_sub,
                coeffs,
                &self.config.kinetics,
            );
            std::mem::swap(&mut self.morphogens, &mut self.next_morphogens);
            self.apply_morphogen_sources_and_inlet(dt_sub);
        }
    }

    /// Read-only row-major horizontal velocity slice (`width * height`).
    #[verified_engine::verified]
    pub fn ux_row(&self) -> &[f64] {
        &self.ux_row
    }

    /// Read-only row-major vertical velocity slice (`width * height`).
    #[verified_engine::verified]
    pub fn uy_row(&self) -> &[f64] {
        &self.uy_row
    }

    /// Read-only row-major solid obstacle mask (`width * height`).
    #[verified_engine::verified]
    pub fn obstacles_row(&self) -> &[bool] {
        &self.obstacles_row
    }

    /// Read-only row-major chemical emitter mask (`width * height`).
    #[verified_engine::verified]
    pub fn emitters_row(&self) -> &[bool] {
        &self.emitters_row
    }

    /// Read-only row-major fluid vorticity curl slice $\omega_z = \partial_x u_y - \partial_y u_x$.
    #[verified_engine::verified]
    pub fn vorticity_row(&self) -> &[f64] {
        &self.vorticity_row
    }

    /// Latest dimensionless Peclet and Damkohler transport metrics.
    #[verified_engine::verified]
    pub fn metrics(&self) -> DimensionlessRegimeMetrics {
        self.last_metrics
    }

    /// Returns `true` if every fluid velocity and morphogen concentration on the grid is finite.
    #[verified_engine::verified]
    pub fn is_finite(&self) -> bool {
        self.morphogens.u().iter().all(|v| v.is_finite())
            && self.morphogens.v().iter().all(|v| v.is_finite())
            && self.ux_row.iter().all(|v| v.is_finite())
            && self.uy_row.iter().all(|v| v.is_finite())
    }
}

mod internals;

#[cfg(test)]
mod tests;

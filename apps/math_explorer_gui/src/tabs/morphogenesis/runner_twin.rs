//! Constructors and counterfactual shadow twin controls for [`super::runner::MorphogenesisKineticsRunner`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::advection::{MorphoWindTunnel, MorphogenesisAdvectionState};
use super::counterfactual::{active_morphogen_slice, MorphogenesisTwinState};
use super::runner::{
    apply_uniform_noise, MorphogenesisKineticsAdapter, MorphogenesisKineticsRunner,
    MorphogenesisKineticsState, MorphogenesisSharedParams, DEFAULT_MORPHOGENESIS_SEED,
    MAX_CFL_SUBSTEPS,
};
use super::sculpt::enforce_system_mask;
use super::MorphogenesisTool;
use crate::framework::decode_rle_mask;
use crate::framework::recovery::FiniteStateGuard;
use math_explorer::biology::diffusion::{FiniteDifference2D, MaskedFiniteDifference2D};
use math_explorer::biology::morphogenesis::{DiffusionCoeff, FusedEulerSolver, TuringSystem};
use math_explorer::math_kernel::types::{Dimension, StepSize};
#[cfg(feature = "physics")]
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::LatticeBoltzmannD2Q9;
use math_explorer::pure_math::analysis::pde::advection_2d::advection_diffusion_cfl_limit;
use oxidize_core::rng::OxidizeRng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Clones a 2D masked `TuringSystem` (pre-allocating `state`, `next_state`, and `CellMask2D`).
pub(super) fn clone_turing_system(
    src: &TuringSystem<2, MorphogenesisKineticsAdapter, MaskedFiniteDifference2D>,
    width: usize,
    height: usize,
) -> TuringSystem<2, MorphogenesisKineticsAdapter, MaskedFiniteDifference2D> {
    let mut dst = TuringSystem::new_with_kinetics(
        Dimension(width * height),
        DiffusionCoeff(src.diffusion_coeffs[0]),
        DiffusionCoeff(src.diffusion_coeffs[1]),
        src.kinetics.clone(),
        src.diffusion.clone(),
    );
    dst.u_mut().copy_from_slice(src.u());
    dst.v_mut().copy_from_slice(src.v());
    dst
}

impl MorphoWindTunnel {
    pub(crate) fn clone_outside_step(&self) -> Self {
        #[cfg(feature = "physics")]
        let solver = {
            let mut s = LatticeBoltzmannD2Q9::new(
                self.solver.width(),
                self.solver.height(),
                self.solver.collision_model.tau,
            );
            s.state = self.solver.state.clone();
            s.collision_model = self.solver.collision_model;
            s
        };
        Self {
            #[cfg(feature = "physics")]
            solver,
            #[cfg(not(feature = "physics"))]
            dims: self.dims,
            ux_buf: self.ux_buf.clone(),
            uy_buf: self.uy_buf.clone(),
            obstacle_buf: self.obstacle_buf.clone(),
            max_abs_ux: self.max_abs_ux,
            max_abs_uy: self.max_abs_uy,
            twin_step_ready: false,
        }
    }
}

fn build_seeded_system(
    params: &MorphogenesisSharedParams,
    ks: MorphogenesisKineticsState,
    mask_rle: &[[u32; 2]],
    width: usize,
    height: usize,
) -> TuringSystem<2, MorphogenesisKineticsAdapter, MaskedFiniteDifference2D> {
    let mut kinetics =
        MorphogenesisKineticsAdapter::new(ks.mode, params.a, params.b, ks.f, ks.k, width * height);
    super::brush::apply_mask_rle_to_kinetics(
        &mut kinetics,
        width,
        height,
        params.mask_rle.as_deref(),
    );
    let diffusion = FiniteDifference2D::new(
        Dimension(width),
        Dimension(height),
        StepSize(1.0),
        StepSize(1.0),
    )
    .with_mask();
    let mut system = TuringSystem::new_with_kinetics(
        Dimension(width * height),
        DiffusionCoeff(params.d_u),
        DiffusionCoeff(params.d_v),
        kinetics,
        diffusion,
    );
    MorphogenesisTool::initialize_adapter_system(&mut system, width, height);
    if !mask_rle.is_empty() {
        if let Ok(bytes) = decode_rle_mask(mask_rle, width * height) {
            let _ = system.diffusion.mask.set_roles_slice(&bytes);
        }
    }
    system
}

impl MorphogenesisKineticsRunner {
    /// Creates a deterministically seeded runner sharing `auto_dt` and wind-tunnel advection state.
    pub fn new_with_advection_and_cfl(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        kinetics_state: &Arc<std::sync::RwLock<MorphogenesisKineticsState>>,
        advection_state: &Arc<std::sync::RwLock<MorphogenesisAdvectionState>>,
        auto_dt: &Arc<AtomicBool>,
        mask_rle: &Arc<std::sync::RwLock<Vec<[u32; 2]>>>,
        dims: (usize, usize),
    ) -> Self {
        let (width, height) = dims;
        let ks = *kinetics_state.read().unwrap_or_else(|e| e.into_inner());
        let visc = advection_state
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .viscosity;
        let system = {
            let p = params.read().unwrap_or_else(|e| e.into_inner());
            let rle = mask_rle.read().unwrap_or_else(|e| e.into_inner());
            build_seeded_system(&p, ks, &rle, width, height)
        };
        let mut wind_tunnel = MorphoWindTunnel::new(width, height, visc);
        wind_tunnel.sync_from_cell_mask(&system.diffusion.mask);
        let mut runner = Self {
            system,
            wind_tunnel,
            params: Arc::clone(params),
            kinetics_state: Arc::clone(kinetics_state),
            advection_state: Arc::clone(advection_state),
            auto_dt: Arc::clone(auto_dt),
            width,
            height,
            step_count: 0,
            rng: OxidizeRng::new(DEFAULT_MORPHOGENESIS_SEED),
            noise_amplitude: 0.0,
            twin: None,
        };
        runner.enforce_mask_concentrations();
        runner
    }

    /// Creates a deterministically seeded multi-kinetics runner with shared wind-tunnel advection state.
    #[cfg(test)]
    pub fn new_with_advection(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        kinetics_state: &Arc<std::sync::RwLock<MorphogenesisKineticsState>>,
        advection_state: &Arc<std::sync::RwLock<MorphogenesisAdvectionState>>,
        mask_rle: &Arc<std::sync::RwLock<Vec<[u32; 2]>>>,
        width: usize,
        height: usize,
    ) -> Self {
        let auto_dt = Arc::new(AtomicBool::new(true));
        Self::new_with_advection_and_cfl(
            params,
            kinetics_state,
            advection_state,
            &auto_dt,
            mask_rle,
            (width, height),
        )
    }

    /// Creates a deterministically seeded multi-kinetics runner at `step_count = 0`.
    #[cfg(test)]
    pub fn new_with_kinetics(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        kinetics_state: &Arc<std::sync::RwLock<MorphogenesisKineticsState>>,
        mask_rle: &Arc<std::sync::RwLock<Vec<[u32; 2]>>>,
        width: usize,
        height: usize,
    ) -> Self {
        let adv = Arc::new(std::sync::RwLock::new(
            MorphogenesisAdvectionState::default(),
        ));
        Self::new_with_advection(params, kinetics_state, &adv, mask_rle, width, height)
    }

    /// Creates a deterministically seeded Schnakenberg runner at `step_count = 0`.
    #[cfg(test)]
    pub fn new_seeded(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        width: usize,
        height: usize,
    ) -> Self {
        let ks = Arc::new(std::sync::RwLock::new(MorphogenesisKineticsState::default()));
        let mask = Arc::new(std::sync::RwLock::new(Vec::new()));
        Self::new_with_kinetics(params, &ks, &mask, width, height)
    }

    /// Clones this runner outside `VerifiedEngine::run_verified`.
    #[must_use]
    pub fn clone_outside_step(&self) -> Self {
        Self {
            system: clone_turing_system(&self.system, self.width, self.height),
            wind_tunnel: self.wind_tunnel.clone_outside_step(),
            params: Arc::clone(&self.params),
            kinetics_state: Arc::clone(&self.kinetics_state),
            advection_state: Arc::clone(&self.advection_state),
            auto_dt: Arc::clone(&self.auto_dt),
            width: self.width,
            height: self.height,
            step_count: self.step_count,
            rng: self.rng.clone(),
            noise_amplitude: self.noise_amplitude,
            twin: self
                .twin
                .as_ref()
                .map(|t| t.clone_with_dims(self.width, self.height)),
        }
    }

    pub(super) fn step_twin_branch_a(&mut self, dt: f64, noise_amp: f64) {
        let adv = self.read_advection();
        if !adv.enabled {
            self.wind_tunnel.twin_step_ready = false;
            let use_cfl = self.auto_dt.load(Ordering::Relaxed);
            if let Some(ref mut twin) = self.twin {
                enforce_system_mask(&mut twin.system_a, self.width, self.height);
                let (subs, sub_dt) = if use_cfl {
                    FusedEulerSolver::cfl_substeps_2d(
                        dt,
                        1.0,
                        twin.system_a.diffusion_coeffs[0],
                        twin.system_a.diffusion_coeffs[1],
                        MAX_CFL_SUBSTEPS,
                    )
                } else {
                    (1, dt)
                };
                twin.step_branch_a(subs, sub_dt, self.width, self.height, noise_amp);
            }
            return;
        }
        if self.twin.is_none() {
            return;
        }
        self.wind_tunnel
            .step_and_project(adv.inlet_u, adv.viscosity);
        self.wind_tunnel.twin_step_ready = true;
        if let Some(ref mut twin) = self.twin {
            enforce_system_mask(&mut twin.system_a, self.width, self.height);
            let d_max = twin.system_a.diffusion_coeffs[0]
                .max(twin.system_a.diffusion_coeffs[1])
                .max(0.0);
            let cfl = advection_diffusion_cfl_limit(
                StepSize(1.0),
                StepSize(1.0),
                d_max,
                self.wind_tunnel.max_abs_ux,
                self.wind_tunnel.max_abs_uy,
            );
            let safe_dt = if cfl.is_finite() && cfl > 0.0 {
                dt.min(0.98 * cfl)
            } else {
                dt
            };
            twin.system_a
                .step_advected(safe_dt, self.wind_tunnel.ux(), self.wind_tunnel.uy());
            apply_uniform_noise(
                &mut twin.system_a,
                self.width * self.height,
                &mut twin.rng_a,
                noise_amp,
            );
            if twin.system_a.diffusion.mask.has_active_cells() {
                enforce_system_mask(&mut twin.system_a, self.width, self.height);
            }
        }
    }

    /// Forks a shadow twin at `step_count > 0` with perturbation `delta_u`.
    pub fn fork_here(&mut self, delta_u: f64) -> bool {
        if self.step_count == 0 || !FiniteStateGuard::is_finite_state(self) {
            return false;
        }
        let dt = self.sync_system_params().unwrap_or(0.005);
        self.twin = Some(MorphogenesisTwinState::fork_from(
            self.step_count,
            dt,
            self.width,
            self.height,
            &mut self.system,
            &mut self.rng,
            delta_u,
        ));
        true
    }

    /// Swaps baseline `Branch A` and active `Branch B` fields, RNGs, and parameters.
    pub fn swap_branches(&mut self) -> bool {
        let Some(ref mut twin) = self.twin else {
            return false;
        };
        std::mem::swap(&mut self.system, &mut twin.system_a);
        std::mem::swap(&mut self.rng, &mut twin.rng_a);
        {
            let mut ks = self
                .kinetics_state
                .write()
                .unwrap_or_else(|e| e.into_inner());
            ks.mode = self.system.kinetics.mode;
            ks.f = self.system.kinetics.f;
            ks.k = self.system.kinetics.k;
        }
        let dt = {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.a = self.system.kinetics.a;
            p.b = self.system.kinetics.b;
            p.d_u = self.system.diffusion_coeffs[0];
            p.d_v = self.system.diffusion_coeffs[1];
            p.dt
        };
        twin.observe_lockstep(self.step_count, dt, active_morphogen_slice(&self.system));
        true
    }

    /// Clears the active counterfactual twin while preserving `Branch B`.
    pub fn clear_fork(&mut self) -> bool {
        self.twin.take().is_some()
    }

    pub(super) fn process_twin_custom_command(&mut self, s: &str) -> bool {
        if let Some(rest) = s.strip_prefix("fork:") {
            let delta = rest.parse::<f64>().unwrap_or(0.0);
            let _ = self.fork_here(delta);
            true
        } else if s == "swap" {
            let _ = self.swap_branches();
            true
        } else if s == "clear_fork" {
            let _ = self.clear_fork();
            true
        } else {
            false
        }
    }
}

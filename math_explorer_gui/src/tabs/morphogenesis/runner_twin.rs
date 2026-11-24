//! Constructors and counterfactual shadow twin controls for [`super::runner::MorphogenesisKineticsRunner`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::advection::{MorphoWindTunnel, MorphogenesisAdvectionState};
use super::counterfactual::{active_morphogen_slice, MorphogenesisTwinState};
use super::runner::{
    MorphogenesisKineticsAdapter, MorphogenesisKineticsRunner, MorphogenesisKineticsState,
    MorphogenesisSharedParams, DEFAULT_MORPHOGENESIS_SEED,
};
use super::sculpt::enforce_system_mask;
use super::MorphogenesisTool;
use crate::framework::decode_rle_mask;
use crate::framework::recovery::FiniteStateGuard;
use math_explorer::biology::diffusion::{
    FiniteDifference2D, MaskedFiniteDifference2D, SpatialDiffusion,
};
use math_explorer::biology::morphogenesis::{DiffusionCoeff, ReactionKinetics, TuringSystem};
use math_explorer::math_kernel::types::{Dimension, StepSize};
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::LatticeBoltzmannD2Q9;
use math_explorer::pure_math::analysis::pde::advection_2d::advection_diffusion_cfl_limit;
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;
use std::sync::Arc;

pub(crate) fn apply_uniform_noise<K: ReactionKinetics<2>, D: SpatialDiffusion<2>>(
    system: &mut TuringSystem<2, K, D>,
    count: usize,
    rng: &mut OxidizeRng,
    amp: f64,
) {
    if !amp.is_finite() || amp <= 0.0 {
        return;
    }
    const INV_2_POW_53: f64 = 1.0 / ((1u64 << 53) as f64);
    for idx in 0..count {
        let u_unit = ((rng.next_u64() >> 11) as f64) * INV_2_POW_53;
        let v_unit = ((rng.next_u64() >> 11) as f64) * INV_2_POW_53;
        system.u_mut()[idx] += (2.0 * u_unit - 1.0) * amp;
        system.v_mut()[idx] += (2.0 * v_unit - 1.0) * amp;
    }
}

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
        src.kinetics,
        src.diffusion.clone(),
    );
    dst.u_mut().copy_from_slice(src.u());
    dst.v_mut().copy_from_slice(src.v());
    dst
}

impl MorphoWindTunnel {
    pub(crate) fn clone_outside_step(&self) -> Self {
        let mut solver = LatticeBoltzmannD2Q9::new(
            self.solver.width(),
            self.solver.height(),
            self.solver.collision_model.tau,
        );
        solver.state = self.solver.state.clone();
        solver.collision_model = self.solver.collision_model;
        Self {
            solver,
            ux_buf: self.ux_buf.clone(),
            uy_buf: self.uy_buf.clone(),
            obstacle_buf: self.obstacle_buf.clone(),
            max_abs_ux: self.max_abs_ux,
            max_abs_uy: self.max_abs_uy,
            twin_step_ready: false,
        }
    }
}

impl MorphogenesisKineticsRunner {
    /// Creates a deterministically seeded multi-kinetics runner with shared wind-tunnel advection state.
    pub fn new_with_advection(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        kinetics_state: &Arc<std::sync::RwLock<MorphogenesisKineticsState>>,
        advection_state: &Arc<std::sync::RwLock<MorphogenesisAdvectionState>>,
        mask_rle: &Arc<std::sync::RwLock<Vec<[u32; 2]>>>,
        width: usize,
        height: usize,
    ) -> Self {
        let (a, b, d_u, d_v) = {
            let p = params.read().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b, p.d_u, p.d_v)
        };
        let (mode, f, k) = {
            let ks = kinetics_state.read().unwrap_or_else(|e| e.into_inner());
            (ks.mode, ks.f, ks.k)
        };
        let visc = advection_state
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .viscosity;
        let diffusion = FiniteDifference2D::new(
            Dimension(width),
            Dimension(height),
            StepSize(1.0),
            StepSize(1.0),
        )
        .with_mask();
        let mut system = TuringSystem::new_with_kinetics(
            Dimension(width * height),
            DiffusionCoeff(d_u),
            DiffusionCoeff(d_v),
            MorphogenesisKineticsAdapter { mode, a, b, f, k },
            diffusion,
        );
        MorphogenesisTool::initialize_adapter_system(&mut system, width, height);
        let rle = mask_rle.read().unwrap_or_else(|e| e.into_inner()).clone();
        if !rle.is_empty() {
            if let Ok(bytes) = decode_rle_mask(&rle, width * height) {
                let _ = system.diffusion.mask.set_roles_slice(&bytes);
            }
        }
        let mut wind_tunnel = MorphoWindTunnel::new(width, height, visc);
        wind_tunnel.sync_from_cell_mask(&system.diffusion.mask);
        let mut runner = Self {
            system,
            wind_tunnel,
            params: Arc::clone(params),
            kinetics_state: Arc::clone(kinetics_state),
            advection_state: Arc::clone(advection_state),
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
            if let Some(ref mut twin) = self.twin {
                enforce_system_mask(&mut twin.system_a, self.width, self.height);
                twin.step_branch_a(dt, self.width, self.height, noise_amp);
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

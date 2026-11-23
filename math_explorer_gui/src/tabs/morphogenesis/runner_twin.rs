//! Constructors and counterfactual shadow twin controls for [`super::runner::MorphogenesisKineticsRunner`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::counterfactual::{active_morphogen_slice, MorphogenesisTwinState};
use super::diffusion::CoupledMorphoDiffusion;
use super::runner::{
    MorphogenesisKineticsAdapter, MorphogenesisKineticsRunner, MorphogenesisKineticsState,
    MorphogenesisSharedParams, DEFAULT_MORPHOGENESIS_SEED,
};
use super::types::{MorphogenesisFlowMode, MorphogenesisFlowState};
use super::MorphogenesisTool;
use crate::framework::decode_rle_mask;
use crate::framework::recovery::FiniteStateGuard;
use math_explorer::biology::diffusion::SpatialDiffusion;
use math_explorer::biology::morphogenesis::{DiffusionCoeff, ReactionKinetics, TuringSystem};
use math_explorer::math_kernel::types::{Dimension, StepSize};
#[cfg(feature = "physics")]
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::LatticeBoltzmannD2Q9;
#[cfg(feature = "physics")]
use math_explorer::physics::fluid_dynamics::potential_flow::PotentialFlowField;
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

/// Clones a 2D coupled `TuringSystem` (pre-allocating `state`, `next_state`, `CellMask2D`, and `AdvectionDiffusion2D`).
pub(super) fn clone_turing_system(
    src: &TuringSystem<2, MorphogenesisKineticsAdapter, CoupledMorphoDiffusion>,
    width: usize,
    height: usize,
) -> TuringSystem<2, MorphogenesisKineticsAdapter, CoupledMorphoDiffusion> {
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

fn build_initial_coupled_system(
    p: MorphogenesisSharedParams,
    ks: MorphogenesisKineticsState,
    flow: MorphogenesisFlowState,
    mask_rle: &Arc<std::sync::RwLock<Vec<[u32; 2]>>>,
    width: usize,
    height: usize,
) -> TuringSystem<2, MorphogenesisKineticsAdapter, CoupledMorphoDiffusion> {
    let mut diffusion = CoupledMorphoDiffusion::new(
        Dimension(width),
        Dimension(height),
        StepSize(1.0),
        StepSize(1.0),
    );
    diffusion.flow_active = flow.flow_mode != MorphogenesisFlowMode::None;
    let mut system = TuringSystem::new_with_kinetics(
        Dimension(width * height),
        DiffusionCoeff(p.d_u),
        DiffusionCoeff(p.d_v),
        MorphogenesisKineticsAdapter {
            mode: ks.mode,
            a: p.a,
            b: p.b,
            f: ks.f,
            k: ks.k,
        },
        diffusion,
    );
    MorphogenesisTool::initialize_adapter_system(&mut system, width, height);
    let rle = mask_rle.read().unwrap_or_else(|e| e.into_inner()).clone();
    if !rle.is_empty() {
        if let Ok(bytes) = decode_rle_mask(&rle, width * height) {
            let _ = system.diffusion.mask.set_roles_slice(&bytes);
        }
    }
    system
}

impl MorphogenesisKineticsRunner {
    /// Creates a deterministically seeded multi-kinetics + hydrodynamic runner at `step_count = 0`.
    pub fn new_with_coupled_flow(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        kinetics_state: &Arc<std::sync::RwLock<MorphogenesisKineticsState>>,
        flow_state: &Arc<std::sync::RwLock<MorphogenesisFlowState>>,
        mask_rle: &Arc<std::sync::RwLock<Vec<[u32; 2]>>>,
        width: usize,
        height: usize,
    ) -> Self {
        let p = *params.read().unwrap_or_else(|e| e.into_inner());
        let ks = *kinetics_state.read().unwrap_or_else(|e| e.into_inner());
        let flow = *flow_state.read().unwrap_or_else(|e| e.into_inner());
        let system = build_initial_coupled_system(p, ks, flow, mask_rle, width, height);
        #[cfg(feature = "physics")]
        let tau = (3.0 * flow.viscosity + 0.5).max(0.51);
        let mut runner = Self {
            system,
            params: Arc::clone(params),
            kinetics_state: Arc::clone(kinetics_state),
            flow_state: Arc::clone(flow_state),
            width,
            height,
            step_count: 0,
            rng: OxidizeRng::new(DEFAULT_MORPHOGENESIS_SEED),
            noise_amplitude: 0.0,
            vorticity_buf: vec![0.0; width * height],
            peclet: 0.0,
            damkohler: 0.0,
            last_flow_mode: flow.flow_mode,
            #[cfg(feature = "physics")]
            lbm: LatticeBoltzmannD2Q9::new(width.max(1), height.max(1), tau),
            #[cfg(feature = "physics")]
            potential_flow: PotentialFlowField::new(),
            #[cfg(feature = "physics")]
            last_potential_key: (u64::MAX, u64::MAX),
            #[cfg(feature = "physics")]
            potential_dirty: true,
            twin: None,
        };
        runner.enforce_mask_concentrations();
        runner.sync_mask_to_hydrodynamics();
        if flow.flow_mode != MorphogenesisFlowMode::None
            && !runner.system.diffusion.obstacles().iter().any(|&b| b)
        {
            runner.seed_default_bluff_obstacle();
        }
        #[cfg(feature = "physics")]
        if flow.flow_mode == MorphogenesisFlowMode::PotentialFlow {
            runner.rebuild_potential_elements(flow.inlet_velocity, flow.circulation);
        }
        runner.refresh_diagnostics();
        runner
    }

    /// Creates a deterministically seeded zero-drift multi-kinetics runner at `step_count = 0`.
    #[cfg(test)]
    pub fn new_with_kinetics(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        kinetics_state: &Arc<std::sync::RwLock<MorphogenesisKineticsState>>,
        mask_rle: &Arc<std::sync::RwLock<Vec<[u32; 2]>>>,
        width: usize,
        height: usize,
    ) -> Self {
        let flow = Arc::new(std::sync::RwLock::new(MorphogenesisFlowState::default()));
        Self::new_with_coupled_flow(params, kinetics_state, &flow, mask_rle, width, height)
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
        #[cfg(feature = "physics")]
        let flow = self.read_flow();
        let cloned = Self {
            system: clone_turing_system(&self.system, self.width, self.height),
            params: Arc::clone(&self.params),
            kinetics_state: Arc::clone(&self.kinetics_state),
            flow_state: Arc::clone(&self.flow_state),
            width: self.width,
            height: self.height,
            step_count: self.step_count,
            rng: self.rng.clone(),
            noise_amplitude: self.noise_amplitude,
            vorticity_buf: self.vorticity_buf.clone(),
            peclet: self.peclet,
            damkohler: self.damkohler,
            last_flow_mode: self.last_flow_mode,
            #[cfg(feature = "physics")]
            lbm: LatticeBoltzmannD2Q9 {
                state: self.lbm.state.clone(),
                collision_model: self.lbm.collision_model,
                _marker: std::marker::PhantomData,
            },
            #[cfg(feature = "physics")]
            potential_flow: PotentialFlowField::new(),
            #[cfg(feature = "physics")]
            last_potential_key: (u64::MAX, u64::MAX),
            #[cfg(feature = "physics")]
            potential_dirty: true,
            twin: self
                .twin
                .as_ref()
                .map(|t| t.clone_with_dims(self.width, self.height)),
        };
        #[cfg(feature = "physics")]
        let cloned = {
            let mut c = cloned;
            if flow.flow_mode == MorphogenesisFlowMode::PotentialFlow {
                c.rebuild_potential_elements(flow.inlet_velocity, flow.circulation);
            }
            c
        };
        cloned
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

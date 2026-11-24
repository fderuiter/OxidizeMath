//! Constructors and counterfactual shadow twin controls for [`super::runner::MorphogenesisKineticsRunner`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::counterfactual::{active_morphogen_slice, MorphogenesisTwinState};
use super::runner::{
    MorphogenesisKineticsAdapter, MorphogenesisKineticsRunner, MorphogenesisKineticsState,
    MorphogenesisSharedParams, DEFAULT_MORPHOGENESIS_SEED,
};
use super::MorphogenesisTool;
use crate::framework::decode_rle_mask;
use crate::framework::recovery::FiniteStateGuard;
use math_explorer::biology::diffusion::{FiniteDifference2D, MaskedFiniteDifference2D};
use math_explorer::biology::morphogenesis::{DiffusionCoeff, TuringSystem};
use math_explorer::math_kernel::types::{Dimension, StepSize};
use oxidize_core::rng::OxidizeRng;
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

impl MorphogenesisKineticsRunner {
    /// Creates a deterministically seeded multi-kinetics runner at `step_count = 0`.
    pub fn new_with_kinetics(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        kinetics_state: &Arc<std::sync::RwLock<MorphogenesisKineticsState>>,
        mask_rle: &Arc<std::sync::RwLock<Vec<[u32; 2]>>>,
        width: usize,
        height: usize,
    ) -> Self {
        let (a, b, d_u, d_v, kinetics_rle) = {
            let p = params.read().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b, p.d_u, p.d_v, p.mask_rle.clone())
        };
        let (mode, f, k) = {
            let ks = kinetics_state.read().unwrap_or_else(|e| e.into_inner());
            (ks.mode, ks.f, ks.k)
        };
        let mut kinetics = MorphogenesisKineticsAdapter::new(mode, a, b, f, k, width * height);
        super::brush::apply_mask_rle_to_kinetics(
            &mut kinetics,
            width,
            height,
            kinetics_rle.as_deref(),
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
            DiffusionCoeff(d_u),
            DiffusionCoeff(d_v),
            kinetics,
            diffusion,
        );
        MorphogenesisTool::initialize_adapter_system(&mut system, width, height);
        let rle = mask_rle.read().unwrap_or_else(|e| e.into_inner()).clone();
        if !rle.is_empty() {
            if let Ok(bytes) = decode_rle_mask(&rle, width * height) {
                let _ = system.diffusion.mask.set_roles_slice(&bytes);
            }
        }
        let mut runner = Self {
            system,
            params: Arc::clone(params),
            kinetics_state: Arc::clone(kinetics_state),
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
            params: Arc::clone(&self.params),
            kinetics_state: Arc::clone(&self.kinetics_state),
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

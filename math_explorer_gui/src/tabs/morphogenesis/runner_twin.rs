//! Constructors, parameter sync, finite-state guard, and counterfactual shadow twin controls
//! for [`super::runner::MorphogenesisRunner`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::cartographer::{grid_u_variance, is_pattern_forming_regime, WASHOUT_VARIANCE_THRESHOLD};
use super::counterfactual::{
    active_morphogen_slice, MorphogenesisHeatmapMode, MorphogenesisTwinState,
};
use super::gray_scott_atlas::{is_gray_scott_pattern_viable, is_gray_scott_v_washed_out};
use super::runner::{
    MorphogenesisKineticsState, MorphogenesisRunner, MorphogenesisSharedParams, MAX_CFL_SUBSTEPS,
};
use super::sculpt::MorphoDiffusion;
use super::state::{
    initialize_for_kinetics, initialize_gray_scott_system, initialize_schnakenberg_system,
    MorphogenesisKinetics,
};
use super::teleport::reconcile_kinetics_state;
#[cfg(test)]
use super::MorphogenesisTool;
use crate::app::ThemeMode;
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
#[cfg(test)]
use math_explorer::biology::diffusion::MaskedFiniteDifference2D;
use math_explorer::biology::morphogenesis::{
    DiffusionCoeff, FusedEulerSolver, SchnakenbergKinetics, TuringSystem,
};
use math_explorer::math_kernel::types::Dimension;
use std::sync::Arc;

/// Clones a 2D `TuringSystem` (pre-allocating `state`, `next_state`, and diffusion operator).
pub(super) fn clone_turing_system<D: MorphoDiffusion>(
    src: &TuringSystem<2, SchnakenbergKinetics, D>,
    width: usize,
    height: usize,
) -> TuringSystem<2, SchnakenbergKinetics, D> {
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

#[cfg(test)]
impl MorphogenesisRunner<MaskedFiniteDifference2D> {
    /// Creates a deterministically seeded Schnakenberg masked runner at `step_count = 0`.
    pub fn new_seeded(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        width: usize,
        height: usize,
    ) -> Self {
        MorphogenesisTool::create_runner(params, width, height)
    }
}

impl<D: MorphoDiffusion> FiniteStateGuard for MorphogenesisRunner<D> {
    fn is_finite_state(&self) -> bool {
        let b_ok = self.system.u().iter().all(|&x| x.is_finite())
            && self.system.v().iter().all(|&x| x.is_finite());
        let a_ok = self.twin.as_ref().is_none_or(|t| {
            t.system_a.u().iter().all(|&x| x.is_finite())
                && t.system_a.v().iter().all(|&x| x.is_finite())
        });
        b_ok && a_ok
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        None
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }
}

impl<D: MorphoDiffusion> MorphogenesisRunner<D> {
    pub(super) const fn mode_to_scalar(mode: MorphogenesisKinetics) -> f64 {
        match mode {
            MorphogenesisKinetics::Schnakenberg => 0.0,
            MorphogenesisKinetics::GrayScott => 1.0,
        }
    }

    pub(super) fn scalar_to_mode(tag: f64) -> MorphogenesisKinetics {
        if tag >= 0.5 {
            MorphogenesisKinetics::GrayScott
        } else {
            MorphogenesisKinetics::Schnakenberg
        }
    }

    pub(super) fn sync_system_params(&mut self) -> Option<(f64, bool)> {
        let mut p = *self.params.read().unwrap_or_else(|e| e.into_inner());
        if let Some(ks_arc) = &self.kinetics_state {
            reconcile_kinetics_state(&self.params, ks_arc, &mut self.last_synced_ks, &mut p);
        }
        self.apply_shared_params(&p)
    }

    fn reseed_washed_out_if_viable(&mut self, p: &MorphogenesisSharedParams) {
        let eff_dt = if p.auto_dt {
            FusedEulerSolver::cfl_substeps_2d(p.dt, 1.0, p.d_u, p.d_v, MAX_CFL_SUBSTEPS).1
        } else {
            p.dt
        };
        match self.kinetics {
            MorphogenesisKinetics::Schnakenberg => {
                if is_pattern_forming_regime(p.a, p.b, p.d_u, p.d_v, eff_dt) {
                    let var_u = grid_u_variance(self.system.u());
                    if var_u.is_finite() && var_u < WASHOUT_VARIANCE_THRESHOLD {
                        initialize_schnakenberg_system(&mut self.system, self.width, self.height);
                        self.enforce_mask_concentrations();
                    }
                }
            }
            MorphogenesisKinetics::GrayScott => {
                if is_gray_scott_pattern_viable(p.f, p.k, p.d_u, p.d_v, eff_dt)
                    && is_gray_scott_v_washed_out(self.system.v())
                {
                    initialize_gray_scott_system(&mut self.system, self.width, self.height);
                    self.enforce_mask_concentrations();
                }
            }
        }
    }

    fn apply_shared_params(&mut self, p: &MorphogenesisSharedParams) -> Option<(f64, bool)> {
        if ![p.a, p.b, p.f, p.k, p.d_u, p.d_v]
            .iter()
            .all(|x| x.is_finite())
            || p.d_u <= 0.0
            || p.d_v <= 0.0
        {
            return None;
        }
        self.system.kinetics.a = p.a;
        self.system.kinetics.b = p.b;
        self.gray_scott.f = p.f;
        self.gray_scott.k = p.k;
        self.system.diffusion_coeffs = [p.d_u, p.d_v];
        if self.kinetics != p.kinetics {
            self.kinetics = p.kinetics;
            if self.twin.is_none() {
                initialize_for_kinetics(&mut self.system, self.kinetics, self.width, self.height);
                self.enforce_mask_concentrations();
            }
        } else if self.twin.is_none() && p.dt.is_finite() && p.dt > 0.0 {
            self.reseed_washed_out_if_viable(p);
        }
        (p.dt.is_finite() && p.dt > 0.0).then_some((p.dt, p.auto_dt))
    }

    pub(super) fn apply_scalar_params(&mut self, s: &[f64]) -> Option<(f64, bool)> {
        let [mode, a, b, f, k, d_u, d_v, dt, auto, amp, ..] = s else {
            return None;
        };
        self.noise_amplitude = *amp;
        self.apply_shared_params(&MorphogenesisSharedParams {
            kinetics: Self::scalar_to_mode(*mode),
            a: *a,
            b: *b,
            f: *f,
            k: *k,
            d_u: *d_u,
            d_v: *d_v,
            dt: *dt,
            auto_dt: *auto >= 0.5,
            high_contrast: false,
            theme_mode: ThemeMode::Dark,
            heatmap_mode: MorphogenesisHeatmapMode::ActiveUb,
        })
    }

    /// Clones this runner outside `VerifiedEngine::run_verified`.
    #[must_use]
    pub fn clone_outside_step(&self) -> Self {
        Self {
            system: clone_turing_system(&self.system, self.width, self.height),
            kinetics: self.kinetics,
            gray_scott: self.gray_scott,
            params: Arc::clone(&self.params),
            kinetics_state: self.kinetics_state.as_ref().map(Arc::clone),
            last_synced_ks: self.last_synced_ks,
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
        let dt = self.sync_system_params().map_or(0.005, |(d, _)| d);
        self.twin = Some(MorphogenesisTwinState::fork_from(
            self.step_count,
            dt,
            (self.width, self.height),
            self.kinetics,
            self.gray_scott,
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
        std::mem::swap(&mut self.kinetics, &mut twin.kinetics_a);
        std::mem::swap(&mut self.gray_scott, &mut twin.gray_scott_a);
        std::mem::swap(&mut self.rng, &mut twin.rng_a);
        let next_ks = MorphogenesisKineticsState {
            mode: self.kinetics,
            f: self.gray_scott.f,
            k: self.gray_scott.k,
        };
        if let Some(ref ks_arc) = self.kinetics_state {
            *ks_arc.write().unwrap_or_else(|e| e.into_inner()) = next_ks;
        }
        self.last_synced_ks = next_ks;
        let dt = {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.kinetics = self.kinetics;
            p.a = self.system.kinetics.a;
            p.b = self.system.kinetics.b;
            p.f = self.gray_scott.f;
            p.k = self.gray_scott.k;
            p.d_u = self.system.diffusion_coeffs[0];
            p.d_v = self.system.diffusion_coeffs[1];
            p.dt
        };
        let ub = active_morphogen_slice(&self.system, self.kinetics);
        twin.observe_lockstep(self.step_count, dt, ub);
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

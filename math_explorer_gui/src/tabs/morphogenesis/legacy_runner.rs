//! Legacy single-kinetics Schnakenberg runner retained for unit tests.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::{grid_u_variance, is_pattern_forming_regime, WASHOUT_VARIANCE_THRESHOLD};
use super::brush::{apply_brush_sim_command, apply_mask_rle_to_kinetics};
use super::runner::{
    apply_uniform_noise, MorphoSnapshotData, MorphogenesisSharedParams, DEFAULT_MORPHOGENESIS_SEED,
};
use super::MorphogenesisTool;
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use math_explorer::biology::diffusion::{BoundaryMode, FiniteDifference2D};
use math_explorer::biology::morphogenesis::{SpatialSchnakenbergKinetics, TuringSystem};
use oxidize_core::rng::OxidizeRng;
use std::sync::Arc;

/// Legacy Schnakenberg-only 2D reaction-diffusion runner used in baseline unit tests.
pub struct MorphogenesisRunner {
    pub(crate) system: TuringSystem<2, SpatialSchnakenbergKinetics, FiniteDifference2D>,
    pub(crate) params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) step_count: usize,
    pub(crate) rng: OxidizeRng,
    pub(crate) noise_amplitude: f64,
}

impl FiniteStateGuard for MorphogenesisRunner {
    fn is_finite_state(&self) -> bool {
        self.system.u().iter().all(|&x| x.is_finite())
            && self.system.v().iter().all(|&x| x.is_finite())
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        None
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }
}

impl MorphogenesisRunner {
    fn sync_system_params(&mut self) -> Option<f64> {
        let (a, b, d_u, d_v, dt, periodic) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b, p.d_u, p.d_v, p.dt, p.periodic_boundary)
        };
        if !a.is_finite() || !b.is_finite() || !d_u.is_finite() || !d_v.is_finite() {
            return None;
        }
        self.system.kinetics.a = a;
        self.system.kinetics.b = b;
        self.system.diffusion_coeffs[0] = d_u;
        self.system.diffusion_coeffs[1] = d_v;
        self.system.diffusion.boundary_mode = if periodic {
            BoundaryMode::Periodic
        } else {
            BoundaryMode::Neumann
        };

        if dt.is_finite() && dt > 0.0 && is_pattern_forming_regime(a, b, d_u, d_v, dt) {
            let var_u = grid_u_variance(self.system.u());
            if var_u.is_finite() && var_u < WASHOUT_VARIANCE_THRESHOLD {
                MorphogenesisTool::initialize_system(&mut self.system, self.width, self.height);
            }
        }
        (dt.is_finite() && dt > 0.0).then_some(dt)
    }
}

impl SimulationRunner for MorphogenesisRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::Reset => {
                let _ = self.sync_system_params();
                let mask_rle = self
                    .params
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .mask_rle
                    .clone();
                apply_mask_rle_to_kinetics(
                    &mut self.system.kinetics,
                    self.width,
                    self.height,
                    mask_rle.as_deref(),
                );
                MorphogenesisTool::initialize_system(&mut self.system, self.width, self.height);
                self.step_count = 0;
                self.rng = OxidizeRng::new(DEFAULT_MORPHOGENESIS_SEED);
            }
            SimCommand::Custom(ref s) if s == "reinit" => {
                let _ = self.sync_system_params();
                MorphogenesisTool::initialize_system(&mut self.system, self.width, self.height);
                self.step_count = 0;
                self.rng = OxidizeRng::new(DEFAULT_MORPHOGENESIS_SEED);
            }
            other => apply_brush_sim_command(&mut self.system, self.width, self.height, &other),
        }
    }

    fn is_finite_state(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        (!FiniteStateGuard::is_finite_state(self)).then(|| {
            DiagnosticReport::new(
                self.step_count as f64,
                self.step_count,
                "concentration",
                "Turing morphogenesis diverged: non-finite concentration encountered.",
            )
        })
    }

    fn step(&mut self) {
        if !FiniteStateGuard::is_finite_state(self) {
            return;
        }
        let amp = self.noise_amplitude;
        if let Some(dt) = self.sync_system_params() {
            self.system.step(dt);
            apply_uniform_noise(
                &mut self.system,
                self.width * self.height,
                &mut self.rng,
                amp,
            );
            self.step_count += 1;
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let high_contrast = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .high_contrast;
        let u_vec = self.system.u().to_vec();
        let image = MorphogenesisTool::plot_concentration_with_contrast(
            &u_vec,
            self.width,
            self.height,
            high_contrast,
        );
        StateSnapshot {
            width: self.width,
            height: self.height,
            pixels: Arc::new(std::sync::RwLock::new(image.pixels)),
            custom_data: Vec::new(),
            structured_data: Some(Box::new(MorphoSnapshotData {
                u_data: u_vec,
                u_a: None,
                mask_roles: Vec::new(),
                max_abs_ux: 0.0,
                max_abs_uy: 0.0,
                step_count: self.step_count,
                divergence: None,
            })),
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }
}

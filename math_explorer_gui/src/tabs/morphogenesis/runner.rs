#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::MorphogenesisTool;
use crate::async_sim::{
    CheckpointParams, CheckpointPayload, SimCommand, SimulationRunner, StateSnapshot,
    TimeTravelCheckpoint,
};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::{SchnakenbergKinetics, TuringSystem};
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;
use std::sync::Arc;

/// Default PRNG seed for deterministic morphogenesis perturbation streams.
pub const DEFAULT_MORPHOGENESIS_SEED: u64 = 0x4D4F_5250_484F_3244;

#[allow(missing_docs)]
pub struct MorphogenesisSharedParams {
    pub a: f64,
    pub b: f64,
    pub d_u: f64,
    pub d_v: f64,
    pub dt: f64,
    pub high_contrast: bool,
}

#[allow(missing_docs)]
pub struct MorphogenesisRunner {
    pub(crate) system: TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
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
        let (a, b, d_u, d_v, dt) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b, p.d_u, p.d_v, p.dt)
        };
        self.apply_explicit_params(a, b, d_u, d_v, dt)
    }

    fn apply_explicit_params(
        &mut self,
        a: f64,
        b: f64,
        d_u: f64,
        d_v: f64,
        dt: f64,
    ) -> Option<f64> {
        if !a.is_finite() || !b.is_finite() || !d_u.is_finite() || !d_v.is_finite() {
            return None;
        }
        self.system.kinetics.a = a;
        self.system.kinetics.b = b;
        self.system.diffusion_coeffs[0] = d_u;
        self.system.diffusion_coeffs[1] = d_v;
        (dt.is_finite() && dt > 0.0).then_some(dt)
    }

    fn advance_grid_step(&mut self, dt: f64, noise_amp: f64) {
        self.system.step(dt);
        if noise_amp.is_finite() && noise_amp > 0.0 {
            const INV_2_POW_53: f64 = 1.0 / ((1u64 << 53) as f64);
            let n = self.width * self.height;
            for idx in 0..n {
                let u_unit = ((self.rng.next_u64() >> 11) as f64) * INV_2_POW_53;
                let v_unit = ((self.rng.next_u64() >> 11) as f64) * INV_2_POW_53;
                self.system.u_mut()[idx] += (2.0 * u_unit - 1.0) * noise_amp;
                self.system.v_mut()[idx] += (2.0 * v_unit - 1.0) * noise_amp;
            }
        }
        self.step_count += 1;
    }

    fn reset_grid_and_counters(&mut self) {
        let _ = self.sync_system_params();
        MorphogenesisTool::initialize_system(&mut self.system, self.width, self.height);
        self.step_count = 0;
        self.rng = OxidizeRng::new(DEFAULT_MORPHOGENESIS_SEED);
    }
}

impl SimulationRunner for MorphogenesisRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::Reset => self.reset_grid_and_counters(),
            SimCommand::Custom(ref s) if s == "reinit" => self.reset_grid_and_counters(),
            _ => {}
        }
    }

    fn is_finite_state(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        if !FiniteStateGuard::is_finite_state(self) {
            Some(DiagnosticReport::new(
                self.step_count as f64,
                self.step_count,
                "concentration",
                "Turing morphogenesis diverged: non-finite concentration encountered.",
            ))
        } else {
            None
        }
    }

    fn step(&mut self) {
        if !FiniteStateGuard::is_finite_state(self) {
            return;
        }
        let noise_amp = self.noise_amplitude;
        if let Some(dt) = self.sync_system_params() {
            self.advance_grid_step(dt, noise_amp);
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
        let pixels: Vec<eframe::egui::Color32> = image.pixels;
        StateSnapshot {
            width: self.width,
            height: self.height,
            pixels: Arc::new(std::sync::RwLock::new(pixels)),
            custom_data: Vec::new(),
            structured_data: Some(Box::new(u_vec)),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn checkpoint_stride(&self) -> Option<usize> {
        Some(16)
    }

    fn current_step(&self) -> usize {
        self.step_count
    }

    fn capture_active_params(&self) -> Option<CheckpointParams> {
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        Some(CheckpointParams::scalars(&[
            p.a,
            p.b,
            p.d_u,
            p.d_v,
            p.dt,
            self.noise_amplitude,
        ]))
    }

    fn capture_checkpoint(&self) -> Option<TimeTravelCheckpoint> {
        let params = self.capture_active_params()?;
        Some(TimeTravelCheckpoint {
            step: self.step_count,
            rng: self.rng.clone(),
            params,
            state: CheckpointPayload::DualField2D {
                width: self.width,
                height: self.height,
                u: self.system.u().into(),
                v: self.system.v().into(),
            },
        })
    }

    fn restore_checkpoint(&mut self, checkpoint: &TimeTravelCheckpoint) {
        if let CheckpointPayload::DualField2D {
            width,
            height,
            ref u,
            ref v,
        } = checkpoint.state
        {
            if width == self.width
                && height == self.height
                && u.len() == self.system.u().len()
                && v.len() == self.system.v().len()
            {
                self.system.u_mut().copy_from_slice(u);
                self.system.v_mut().copy_from_slice(v);
            }
        }
        if let Some(p) = checkpoint.params.as_scalars() {
            if p.len() >= 6 {
                let _ = self.apply_explicit_params(p[0], p[1], p[2], p[3], p[4]);
                self.noise_amplitude = p[5];
            }
        }
        self.step_count = checkpoint.step;
        self.rng = checkpoint.rng.clone();
    }

    fn replay_step(&mut self, params: &CheckpointParams) {
        let Some(p) = params.as_scalars() else {
            return;
        };
        if p.len() < 6 {
            return;
        }
        self.noise_amplitude = p[5];
        if let Some(dt) = self.apply_explicit_params(p[0], p[1], p[2], p[3], p[4]) {
            self.advance_grid_step(dt, p[5]);
        }
    }
}

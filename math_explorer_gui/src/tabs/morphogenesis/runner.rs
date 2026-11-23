#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::{
    grid_u_variance, is_pattern_forming_regime, WASHOUT_VARIANCE_THRESHOLD,
};
use super::MorphogenesisTool;
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::{SchnakenbergKinetics, TuringSystem};
use std::sync::Arc;

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
        if !a.is_finite() || !b.is_finite() || !d_u.is_finite() || !d_v.is_finite() {
            return None;
        }
        self.system.kinetics.a = a;
        self.system.kinetics.b = b;
        self.system.diffusion_coeffs[0] = d_u;
        self.system.diffusion_coeffs[1] = d_v;

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
                MorphogenesisTool::initialize_system(&mut self.system, self.width, self.height);
            }
            SimCommand::Custom(ref s) if s == "reinit" => {
                let _ = self.sync_system_params();
                MorphogenesisTool::initialize_system(&mut self.system, self.width, self.height);
            }
            _ => {}
        }
    }

    fn is_finite_state(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        if !FiniteStateGuard::is_finite_state(self) {
            Some(DiagnosticReport::new(
                0.0,
                0,
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
        if let Some(dt) = self.sync_system_params() {
            self.system.step(dt);
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
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }
}

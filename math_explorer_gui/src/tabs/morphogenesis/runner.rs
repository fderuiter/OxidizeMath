#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::{grid_u_variance, is_pattern_forming_regime, WASHOUT_VARIANCE_THRESHOLD};
use super::tool::{MorphogenesisBoundaryMode, MorphogenesisKineticsMode};
use super::MorphogenesisTool;
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::{
    GrayScottKinetics, SchnakenbergKinetics, TuringSystem,
};
use std::sync::Arc;

#[allow(missing_docs)]
pub struct MorphogenesisSharedParams {
    pub a: f64,
    pub b: f64,
    pub d_u: f64,
    pub d_v: f64,
    pub dt: f64,
    pub kinetics: MorphogenesisKineticsMode,
    pub boundary: MorphogenesisBoundaryMode,
    pub feed_rate: f64,
    pub kill_rate: f64,
    pub high_contrast: bool,
}

impl Default for MorphogenesisSharedParams {
    fn default() -> Self {
        Self {
            a: 0.1,
            b: 0.9,
            d_u: 1.0,
            d_v: 100.0,
            dt: 0.05,
            kinetics: MorphogenesisKineticsMode::Schnakenberg,
            boundary: MorphogenesisBoundaryMode::Neumann,
            feed_rate: 0.037,
            kill_rate: 0.060,
            high_contrast: false,
        }
    }
}

#[allow(missing_docs)]
pub struct MorphogenesisRunner {
    pub(crate) system: TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
    pub(crate) gray_scott_system: TuringSystem<2, GrayScottKinetics, FiniteDifference2D>,
    pub(crate) active_kinetics: MorphogenesisKineticsMode,
    pub(crate) params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl MorphogenesisRunner {
    #[cfg(test)]
    pub(crate) fn from_schnakenberg(
        system: TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
        params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        width: usize,
        height: usize,
    ) -> Self {
        use math_explorer::biology::morphogenesis::DiffusionCoeff;
        use math_explorer::math_kernel::types::{Dimension, StepSize};
        let gs_diffusion = FiniteDifference2D::new(
            Dimension(width),
            Dimension(height),
            StepSize(1.0),
            StepSize(1.0),
        );
        let mut gray_scott_system = TuringSystem::new_with_kinetics(
            Dimension(width * height),
            DiffusionCoeff(0.16),
            DiffusionCoeff(0.08),
            GrayScottKinetics { f: 0.037, k: 0.060 },
            gs_diffusion,
        );
        MorphogenesisTool::initialize_gray_scott_system(&mut gray_scott_system, width, height);
        Self {
            system,
            gray_scott_system,
            active_kinetics: MorphogenesisKineticsMode::Schnakenberg,
            params,
            width,
            height,
        }
    }
}

impl FiniteStateGuard for MorphogenesisRunner {
    fn is_finite_state(&self) -> bool {
        match self.active_kinetics {
            MorphogenesisKineticsMode::Schnakenberg => {
                self.system.u().iter().all(|&x| x.is_finite())
                    && self.system.v().iter().all(|&x| x.is_finite())
            }
            MorphogenesisKineticsMode::GrayScott => {
                self.gray_scott_system.u().iter().all(|&x| x.is_finite())
                    && self.gray_scott_system.v().iter().all(|&x| x.is_finite())
            }
        }
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
        let (a, b, d_u, d_v, dt, kinetics, boundary, feed_rate, kill_rate) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (
                p.a,
                p.b,
                p.d_u,
                p.d_v,
                p.dt,
                p.kinetics,
                p.boundary,
                p.feed_rate,
                p.kill_rate,
            )
        };
        if !a.is_finite()
            || !b.is_finite()
            || !d_u.is_finite()
            || !d_v.is_finite()
            || !feed_rate.is_finite()
            || !kill_rate.is_finite()
        {
            return None;
        }
        let bmode = boundary.to_domain_mode();
        self.system.diffusion.boundary_mode = bmode;
        self.gray_scott_system.diffusion.boundary_mode = bmode;
        self.system.kinetics.a = a;
        self.system.kinetics.b = b;
        self.system.diffusion_coeffs[0] = d_u;
        self.system.diffusion_coeffs[1] = d_v;
        self.gray_scott_system.kinetics.f = feed_rate;
        self.gray_scott_system.kinetics.k = kill_rate;
        self.reinit_on_kinetics_or_washout(kinetics, a, b, d_u, d_v, dt);
        (dt.is_finite() && dt > 0.0).then_some(dt)
    }

    fn reinit_on_kinetics_or_washout(
        &mut self,
        kinetics: MorphogenesisKineticsMode,
        a: f64,
        b: f64,
        d_u: f64,
        d_v: f64,
        dt: f64,
    ) {
        if self.active_kinetics != kinetics {
            self.active_kinetics = kinetics;
            match kinetics {
                MorphogenesisKineticsMode::Schnakenberg => {
                    MorphogenesisTool::initialize_system(&mut self.system, self.width, self.height);
                }
                MorphogenesisKineticsMode::GrayScott => {
                    MorphogenesisTool::initialize_gray_scott_system(
                        &mut self.gray_scott_system,
                        self.width,
                        self.height,
                    );
                }
            }
            return;
        }
        if kinetics == MorphogenesisKineticsMode::Schnakenberg
            && dt.is_finite()
            && dt > 0.0
            && is_pattern_forming_regime(a, b, d_u, d_v, dt)
        {
            let var_u = grid_u_variance(self.system.u());
            if var_u.is_finite() && var_u < WASHOUT_VARIANCE_THRESHOLD {
                MorphogenesisTool::initialize_system(&mut self.system, self.width, self.height);
            }
        }
    }

    pub(crate) fn active_concentration_slice(&self) -> &[f64] {
        match self.active_kinetics {
            MorphogenesisKineticsMode::Schnakenberg => self.system.u(),
            MorphogenesisKineticsMode::GrayScott => self.gray_scott_system.v(),
        }
    }
}

impl SimulationRunner for MorphogenesisRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        if matches!(cmd, SimCommand::Reset)
            || matches!(cmd, SimCommand::Custom(ref s) if s == "reinit")
        {
            let _ = self.sync_system_params();
            MorphogenesisTool::initialize_system(&mut self.system, self.width, self.height);
            MorphogenesisTool::initialize_gray_scott_system(
                &mut self.gray_scott_system,
                self.width,
                self.height,
            );
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
            match self.active_kinetics {
                MorphogenesisKineticsMode::Schnakenberg => self.system.step(dt),
                MorphogenesisKineticsMode::GrayScott => {
                    for _ in 0..4 {
                        self.gray_scott_system.step(1.0);
                    }
                }
            }
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let high_contrast = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .high_contrast;
        let field_vec = self.active_concentration_slice().to_vec();
        let image = MorphogenesisTool::plot_concentration_with_contrast(
            &field_vec,
            self.width,
            self.height,
            high_contrast,
        );
        StateSnapshot {
            width: self.width,
            height: self.height,
            pixels: Arc::new(std::sync::RwLock::new(image.pixels)),
            custom_data: Vec::new(),
            structured_data: Some(Box::new(field_vec)),
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }
}

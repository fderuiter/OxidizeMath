#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::{grid_u_variance, is_pattern_forming_regime, WASHOUT_VARIANCE_THRESHOLD};
use super::presets::MorphogenesisModelKind;
use super::MorphogenesisTool;
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::{
    DiffusionCoeff, GrayScottKinetics, SchnakenbergKinetics, TuringSystem,
};
use math_explorer::biology::reaction_diffusion::ReactionDiffusionSystem;
use math_explorer::math_kernel::types::{Dimension, StepSize};
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

/// Extended reaction-diffusion model parameters for Gray-Scott and generic R-D modes.
pub struct MorphogenesisModelParams {
    /// Gray-Scott feed rate `f`.
    pub f: f64,
    /// Gray-Scott kill rate `k`.
    pub k: f64,
    /// Active reaction-diffusion backend.
    pub model: MorphogenesisModelKind,
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
    pub(crate) fn new(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        width: usize,
        height: usize,
    ) -> Self {
        let (a, b, d_u, d_v) = {
            let p = params.read().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b, p.d_u, p.d_v)
        };
        let diffusion = FiniteDifference2D::new(
            Dimension(width),
            Dimension(height),
            StepSize(1.0),
            StepSize(1.0),
        );
        let mut system = TuringSystem::new_with_kinetics(
            Dimension(width * height),
            DiffusionCoeff(d_u),
            DiffusionCoeff(d_v),
            SchnakenbergKinetics { a, b },
            diffusion,
        );
        MorphogenesisTool::initialize_system(&mut system, width, height);
        Self {
            system,
            params: Arc::clone(params),
            width,
            height,
        }
    }

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
        StateSnapshot {
            width: self.width,
            height: self.height,
            pixels: Arc::new(std::sync::RwLock::new(image.pixels)),
            custom_data: Vec::new(),
            structured_data: Some(Box::new(u_vec)),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }
}

/// Multi-model worker runner supporting `Schnakenberg`, `GrayScott`, and `ReactionDiffusionSystem`.
pub struct MorphogenesisMultiRunner {
    pub(crate) base: MorphogenesisRunner,
    pub(crate) gray_scott_system: TuringSystem<2, GrayScottKinetics, FiniteDifference2D>,
    pub(crate) generic_rd_system: ReactionDiffusionSystem<GrayScottKinetics, FiniteDifference2D>,
    pub(crate) active_model: MorphogenesisModelKind,
    pub(crate) model_params: Arc<std::sync::RwLock<MorphogenesisModelParams>>,
}

impl MorphogenesisMultiRunner {
    pub(crate) fn new(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        model_params: &Arc<std::sync::RwLock<MorphogenesisModelParams>>,
        width: usize,
        height: usize,
    ) -> Self {
        let (d_u, d_v) = {
            let p = params.read().unwrap_or_else(|e| e.into_inner());
            (p.d_u, p.d_v)
        };
        let (f, k, model) = {
            let mp = model_params.read().unwrap_or_else(|e| e.into_inner());
            (mp.f, mp.k, mp.model)
        };
        let make_fd = || {
            FiniteDifference2D::new(
                Dimension(width),
                Dimension(height),
                StepSize(1.0),
                StepSize(1.0),
            )
        };
        let base = MorphogenesisRunner::new(params, width, height);
        let mut gray_scott_system = TuringSystem::new_with_kinetics(
            Dimension(width * height),
            DiffusionCoeff(d_u),
            DiffusionCoeff(d_v),
            GrayScottKinetics::new(f, k),
            make_fd(),
        );
        MorphogenesisTool::initialize_gray_scott_system(&mut gray_scott_system, width, height);

        let mut generic_rd_system = ReactionDiffusionSystem::builder()
            .num_species(2)
            .grid_size(width * height)
            .reaction(GrayScottKinetics::new(f, k))
            .diffusion(make_fd())
            .diffusion_coeffs(vec![d_u, d_v])
            .build()
            .expect("valid 2-species ReactionDiffusionSystem");
        MorphogenesisTool::initialize_chemical_state(&mut generic_rd_system.state, width, height);

        Self {
            base,
            gray_scott_system,
            generic_rd_system,
            active_model: model,
            model_params: Arc::clone(model_params),
        }
    }

    fn reinit_active(&mut self) {
        let w = self.base.width;
        let h = self.base.height;
        match self.active_model {
            MorphogenesisModelKind::Schnakenberg => {
                MorphogenesisTool::initialize_system(&mut self.base.system, w, h);
            }
            MorphogenesisModelKind::GrayScott => {
                MorphogenesisTool::initialize_gray_scott_system(&mut self.gray_scott_system, w, h);
            }
            MorphogenesisModelKind::GenericReactionDiffusion => {
                MorphogenesisTool::initialize_chemical_state(
                    &mut self.generic_rd_system.state,
                    w,
                    h,
                );
            }
        }
    }

    fn sync_all_params(&mut self) -> Option<f64> {
        let dt = self.base.sync_system_params()?;
        let (f, k, model) = {
            let mp = self.model_params.read().unwrap_or_else(|e| e.into_inner());
            (mp.f, mp.k, mp.model)
        };
        if !f.is_finite() || !k.is_finite() {
            return None;
        }
        let coeffs = self.base.system.diffusion_coeffs;
        self.gray_scott_system.kinetics.f = f;
        self.gray_scott_system.kinetics.k = k;
        self.gray_scott_system.diffusion_coeffs = coeffs;

        self.generic_rd_system.model.reaction.f = f;
        self.generic_rd_system.model.reaction.k = k;
        if self.generic_rd_system.model.diffusion_coeffs.len() >= 2 {
            self.generic_rd_system.model.diffusion_coeffs[0] = coeffs[0];
            self.generic_rd_system.model.diffusion_coeffs[1] = coeffs[1];
        }
        if model != self.active_model {
            self.active_model = model;
            self.reinit_active();
        }
        Some(dt)
    }

    fn active_u(&self) -> &[f64] {
        match self.active_model {
            MorphogenesisModelKind::Schnakenberg => self.base.system.u(),
            MorphogenesisModelKind::GrayScott => self.gray_scott_system.u(),
            MorphogenesisModelKind::GenericReactionDiffusion => {
                self.generic_rd_system.state.species(0)
            }
        }
    }

    fn active_v(&self) -> &[f64] {
        match self.active_model {
            MorphogenesisModelKind::Schnakenberg => self.base.system.v(),
            MorphogenesisModelKind::GrayScott => self.gray_scott_system.v(),
            MorphogenesisModelKind::GenericReactionDiffusion => {
                self.generic_rd_system.state.species(1)
            }
        }
    }
}

impl FiniteStateGuard for MorphogenesisMultiRunner {
    fn is_finite_state(&self) -> bool {
        self.active_u().iter().all(|&x| x.is_finite())
            && self.active_v().iter().all(|&x| x.is_finite())
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        None
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }
}

impl SimulationRunner for MorphogenesisMultiRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        if matches!(cmd, SimCommand::Reset)
            || matches!(cmd, SimCommand::Custom(ref s) if s == "reinit")
        {
            let _ = self.sync_all_params();
            self.reinit_active();
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
        if let Some(dt) = self.sync_all_params() {
            match self.active_model {
                MorphogenesisModelKind::Schnakenberg => self.base.system.step(dt),
                MorphogenesisModelKind::GrayScott => self.gray_scott_system.step(dt),
                MorphogenesisModelKind::GenericReactionDiffusion => {
                    self.generic_rd_system.step(dt);
                }
            }
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let hc = self
            .base
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .high_contrast;
        let u_vec = self.active_u().to_vec();
        let image = MorphogenesisTool::plot_concentration_with_contrast(
            &u_vec,
            self.base.width,
            self.base.height,
            hc,
        );
        StateSnapshot {
            width: self.base.width,
            height: self.base.height,
            pixels: Arc::new(std::sync::RwLock::new(image.pixels)),
            custom_data: Vec::new(),
            structured_data: Some(Box::new(u_vec)),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        match self.active_model {
            MorphogenesisModelKind::Schnakenberg => 1,
            MorphogenesisModelKind::GrayScott
            | MorphogenesisModelKind::GenericReactionDiffusion => 4,
        }
    }
}

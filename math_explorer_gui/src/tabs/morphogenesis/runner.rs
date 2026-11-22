//! Background simulation runner for dual-kinetics Turing and Gray-Scott morphogenesis.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::{
    apply_chemical_brush, ensure_nucleated_for_kinetics, initialize_for_kinetics,
    plot_concentration_with_contrast, MorphogenesisKinetics, DEFAULT_GRAY_SCOTT_F,
    DEFAULT_GRAY_SCOTT_K,
};
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::{
    FusedEulerSolver, GrayScottKinetics, SchnakenbergKinetics, TuringSystem,
};
use std::sync::Arc;

/// Maximum CFL substeps executed per runner step when `auto_dt` is active.
pub const MAX_CFL_SUBSTEPS: usize = 32;

/// Thread-safe parameter block shared between [`super::MorphogenesisTool`] and [`MorphogenesisRunner`].
#[derive(Clone, Debug)]
pub struct MorphogenesisSharedParams {
    /// Active reaction kinetics family (`Schnakenberg` or `GrayScott`).
    pub kinetics: MorphogenesisKinetics,
    /// Schnakenberg activator production rate `a`.
    pub a: f64,
    /// Schnakenberg inhibitor production rate `b`.
    pub b: f64,
    /// Gray-Scott dimensionless substrate feed rate `F`.
    pub f: f64,
    /// Gray-Scott dimensionless autocatalyst kill rate `k`.
    pub k: f64,
    /// Species `u` diffusion coefficient `D_u`.
    pub d_u: f64,
    /// Species `v` diffusion coefficient `D_v`.
    pub d_v: f64,
    /// Target integration step `dt` (substepped within CFL bound when `auto_dt` is `true`).
    pub dt: f64,
    /// Enables automatic 2D CFL time-step clamping (`dt <= 0.8 * dx^2 / (4 * max(D_u, D_v))`).
    pub auto_dt: bool,
    /// Whether high-contrast accessible colormap tokens are active.
    pub high_contrast: bool,
}

impl Default for MorphogenesisSharedParams {
    fn default() -> Self {
        Self {
            kinetics: MorphogenesisKinetics::Schnakenberg,
            a: 0.1,
            b: 0.9,
            f: DEFAULT_GRAY_SCOTT_F,
            k: DEFAULT_GRAY_SCOTT_K,
            d_u: 1.0,
            d_v: 100.0,
            dt: 0.05,
            auto_dt: true,
            high_contrast: false,
        }
    }
}

/// Zero-allocation dual-kinetics simulation runner executing on the background worker.
pub struct MorphogenesisRunner {
    pub(crate) system: TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
    pub(crate) kinetics: MorphogenesisKinetics,
    pub(crate) gray_scott: GrayScottKinetics,
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
    fn sync_system_params(&mut self) -> Option<(f64, bool)> {
        let p = self.params.read().unwrap_or_else(|e| e.into_inner()).clone();
        if !p.a.is_finite()
            || !p.b.is_finite()
            || !p.f.is_finite()
            || !p.k.is_finite()
            || !p.d_u.is_finite()
            || !p.d_v.is_finite()
            || p.d_u <= 0.0
            || p.d_v <= 0.0
        {
            return None;
        }
        self.system.kinetics.a = p.a;
        self.system.kinetics.b = p.b;
        self.gray_scott.f = p.f;
        self.gray_scott.k = p.k;
        self.system.diffusion_coeffs[0] = p.d_u;
        self.system.diffusion_coeffs[1] = p.d_v;
        if self.kinetics != p.kinetics {
            self.kinetics = p.kinetics;
            initialize_for_kinetics(&mut self.system, self.kinetics, self.width, self.height);
        }
        (p.dt.is_finite() && p.dt > 0.0).then_some((p.dt, p.auto_dt))
    }

    #[inline]
    fn step_active_kinetics(&mut self, dt: f64) {
        match self.kinetics {
            MorphogenesisKinetics::Schnakenberg => self.system.step(dt),
            MorphogenesisKinetics::GrayScott => {
                self.system.step_with_kinetics(&self.gray_scott, dt);
            }
        }
    }
}

impl SimulationRunner for MorphogenesisRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::Reset => {
                let _ = self.sync_system_params();
                initialize_for_kinetics(&mut self.system, self.kinetics, self.width, self.height);
            }
            SimCommand::Custom(ref s) if s == "reinit" => {
                let _ = self.sync_system_params();
                initialize_for_kinetics(&mut self.system, self.kinetics, self.width, self.height);
            }
            SimCommand::Custom(ref s) if s == "ensure_nucleated" => {
                let _ = self.sync_system_params();
                ensure_nucleated_for_kinetics(
                    &mut self.system,
                    self.kinetics,
                    self.width,
                    self.height,
                );
            }
            SimCommand::ApplyBrush {
                cx,
                cy,
                r,
                is_obstacle,
            } => {
                let _ = self.sync_system_params();
                apply_chemical_brush(
                    &mut self.system,
                    self.kinetics,
                    (self.width, self.height),
                    (cx, cy, r),
                    is_obstacle,
                );
            }
            SimCommand::ClearObstacles => {
                let _ = self.sync_system_params();
                let center = ((self.width / 2) as i32, (self.height / 2) as i32);
                let max_r = self.width.max(self.height) as i32;
                apply_chemical_brush(
                    &mut self.system,
                    self.kinetics,
                    (self.width, self.height),
                    (center.0, center.1, max_r),
                    false,
                );
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
                "Reaction-diffusion solver diverged: non-finite concentration encountered.",
            ))
        } else {
            None
        }
    }

    fn step(&mut self) {
        if !FiniteStateGuard::is_finite_state(self) {
            return;
        }
        let Some((dt, auto_dt)) = self.sync_system_params() else {
            return;
        };
        if auto_dt {
            let [d_u, d_v] = self.system.diffusion_coeffs;
            let (substeps, sub_dt) =
                FusedEulerSolver::cfl_substeps_2d(dt, 1.0, d_u, d_v, MAX_CFL_SUBSTEPS);
            for _ in 0..substeps {
                self.step_active_kinetics(sub_dt);
                if !FiniteStateGuard::is_finite_state(self) {
                    break;
                }
            }
        } else {
            self.step_active_kinetics(dt);
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let high_contrast = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .high_contrast;
        let u_vec = self.system.u().to_vec();
        let image = plot_concentration_with_contrast(&u_vec, self.width, self.height, high_contrast);
        StateSnapshot {
            width: self.width,
            height: self.height,
            pixels: Arc::new(std::sync::RwLock::new(image.pixels)),
            custom_data: Vec::new(),
            structured_data: Some(Box::new(u_vec)),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        4
    }
}

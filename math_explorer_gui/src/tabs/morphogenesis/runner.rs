#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::{grid_u_variance, is_pattern_forming_regime, WASHOUT_VARIANCE_THRESHOLD};
use super::counterfactual::{
    clone_turing_system, render_morphogenesis_heatmap, MorphogenesisHeatmapMode,
    MorphogenesisSnapshotData, MorphogenesisTwinState,
};
use super::{MorphogenesisTool, SimpleRng};
use crate::app::ThemeMode;
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::{DiffusionCoeff, SchnakenbergKinetics, TuringSystem};
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
    pub theme_mode: ThemeMode,
    pub heatmap_mode: MorphogenesisHeatmapMode,
}

#[allow(missing_docs)]
pub struct MorphogenesisRunner {
    pub(crate) system: TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
    pub(crate) params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) rng: SimpleRng,
    pub(crate) step_count: usize,
    pub(crate) twin: Option<MorphogenesisTwinState>,
}

impl FiniteStateGuard for MorphogenesisRunner {
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

impl MorphogenesisRunner {
    /// Creates a deterministically seeded runner (`seed = 12345`) at `step_count = 0`.
    pub fn new_seeded(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        width: usize,
        height: usize,
    ) -> Self {
        let (a, b, d_u, d_v) = {
            let p = params.read().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b, p.d_u, p.d_v)
        };
        let mut system = TuringSystem::new_with_kinetics(
            Dimension(width * height),
            DiffusionCoeff(d_u),
            DiffusionCoeff(d_v),
            SchnakenbergKinetics { a, b },
            FiniteDifference2D::new(
                Dimension(width),
                Dimension(height),
                StepSize(1.0),
                StepSize(1.0),
            ),
        );
        let mut rng = SimpleRng::new(12345);
        MorphogenesisTool::initialize_system_with_rng(&mut system, width, height, &mut rng);
        Self {
            system,
            params: Arc::clone(params),
            width,
            height,
            rng,
            step_count: 0,
            twin: None,
        }
    }

    /// Clones this runner outside `VerifiedEngine::run_verified`.
    #[must_use]
    pub fn clone_outside_step(&self) -> Self {
        Self {
            system: clone_turing_system(&self.system, self.width, self.height),
            params: Arc::clone(&self.params),
            width: self.width,
            height: self.height,
            rng: self.rng,
            step_count: self.step_count,
            twin: self
                .twin
                .as_ref()
                .map(|t| t.clone_with_dims(self.width, self.height)),
        }
    }

    /// Forks a shadow twin at `step_count > 0` with activator perturbation `delta_u`.
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
        let dt = {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.a = self.system.kinetics.a;
            p.b = self.system.kinetics.b;
            p.d_u = self.system.diffusion_coeffs[0];
            p.d_v = self.system.diffusion_coeffs[1];
            p.dt
        };
        twin.observe_lockstep(self.step_count, dt, self.system.u());
        true
    }

    /// Clears the active counterfactual twin while preserving `Branch B`.
    pub fn clear_fork(&mut self) -> bool {
        self.twin.take().is_some()
    }

    fn reset_in_place(&mut self) {
        let _ = self.sync_system_params();
        self.rng = SimpleRng::new(12345);
        self.step_count = 0;
        self.twin = None;
        MorphogenesisTool::initialize_system_with_rng(
            &mut self.system,
            self.width,
            self.height,
            &mut self.rng,
        );
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

        if !self.has_counterfactual_twin()
            && dt.is_finite()
            && dt > 0.0
            && is_pattern_forming_regime(a, b, d_u, d_v, dt)
        {
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
            SimCommand::Reset => self.reset_in_place(),
            SimCommand::Custom(ref s) if s == "reinit" => self.reset_in_place(),
            SimCommand::Custom(ref s) => {
                if let Some(rest) = s.strip_prefix("fork:") {
                    let delta = rest.parse::<f64>().unwrap_or(0.0);
                    let _ = self.fork_here(delta);
                } else if s == "swap" {
                    let _ = self.swap_branches();
                } else if s == "clear_fork" {
                    let _ = self.clear_fork();
                }
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
                self.step_count as f64,
                self.step_count,
                "concentration",
                "Turing morphogenesis diverged: non-finite concentration encountered.",
            ))
        } else {
            None
        }
    }

    fn has_counterfactual_twin(&self) -> bool {
        self.twin.is_some()
    }

    fn step_branch_a(&mut self) {
        if !FiniteStateGuard::is_finite_state(self) {
            return;
        }
        if let Some(dt) = self.sync_system_params() {
            if let Some(ref mut twin) = self.twin {
                twin.step_branch_a(dt);
            }
        }
    }

    fn step_branch_b(&mut self) {
        if !FiniteStateGuard::is_finite_state(self) {
            return;
        }
        if let Some(dt) = self.sync_system_params() {
            self.system.step(dt);
            self.step_count += 1;
            if let Some(ref mut twin) = self.twin {
                twin.observe_lockstep(self.step_count, dt, self.system.u());
            }
        }
    }

    fn step(&mut self) {
        if !FiniteStateGuard::is_finite_state(self) {
            return;
        }
        if let Some(dt) = self.sync_system_params() {
            if let Some(ref mut twin) = self.twin {
                twin.step_branch_a(dt);
            }
            self.system.step(dt);
            self.step_count += 1;
            if let Some(ref mut twin) = self.twin {
                twin.observe_lockstep(self.step_count, dt, self.system.u());
            }
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let (high_contrast, mut theme_mode, heatmap_mode, dt) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.high_contrast, p.theme_mode, p.heatmap_mode, p.dt)
        };
        if high_contrast {
            theme_mode = ThemeMode::HighContrast;
        }
        let u_b = self.system.u().to_vec();
        let u_a = self.twin.as_ref().map(|t| t.system_a.u().to_vec());
        let image = render_morphogenesis_heatmap(
            &u_b,
            u_a.as_deref(),
            self.width,
            self.height,
            heatmap_mode,
            theme_mode,
        );
        let divergence = self.twin.as_ref().map(|t| t.snapshot(self.step_count, dt));
        StateSnapshot {
            width: self.width,
            height: self.height,
            pixels: Arc::new(std::sync::RwLock::new(image.pixels)),
            custom_data: Vec::new(),
            structured_data: Some(Box::new(MorphogenesisSnapshotData {
                u_b,
                u_a,
                step_count: self.step_count,
                divergence,
            })),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }
}

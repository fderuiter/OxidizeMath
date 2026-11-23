#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::{
    grid_u_variance, is_pattern_forming_regime, WASHOUT_VARIANCE_THRESHOLD,
};
use super::overlay::render_morphogenesis_overlay;
use super::types::{
    compute_peclet_damkohler, MorphogenesisFlowMode, MorphogenesisKinetics,
    MorphogenesisKineticsMode, MorphogenesisSharedParams, MorphogenesisSnapshotData,
};
use super::MorphogenesisTool;
use crate::app::ThemeMode;
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use math_commons::math_kernel::types::{Dimension, StepSize};
use math_explorer::biology::diffusion::AdvectionDiffusion2D;
use math_explorer::biology::morphogenesis::{DiffusionCoeff, TuringSystem};
#[cfg(feature = "physics")]
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    BgkCollision, LatticeBoltzmannD2Q9,
};
#[cfg(feature = "physics")]
use math_explorer::physics::fluid_dynamics::potential_flow::PotentialFlowField;
use std::sync::Arc;

const MAX_CFL_SUBSTEPS: usize = 64;
const TARGET_ADVECTIVE_CFL: f64 = 0.40;

#[allow(missing_docs)]
pub struct MorphogenesisRunner {
    pub(crate) system: TuringSystem<2, MorphogenesisKinetics, AdvectionDiffusion2D>,
    pub(crate) params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) vorticity_buf: Vec<f64>,
    pub(crate) peclet: f64,
    pub(crate) damkohler: f64,
    last_kinetics_mode: MorphogenesisKineticsMode,
    last_flow_mode: MorphogenesisFlowMode,
    #[cfg(feature = "physics")]
    pub(crate) lbm: LatticeBoltzmannD2Q9<BgkCollision>,
    #[cfg(feature = "physics")]
    pub(crate) potential_flow: PotentialFlowField,
    #[cfg(feature = "physics")]
    pub(crate) last_potential_key: (u64, u64),
    #[cfg(feature = "physics")]
    pub(crate) potential_dirty: bool,
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
        let snap = params.read().unwrap_or_else(|e| e.into_inner()).clone();
        let kinetics = MorphogenesisKinetics {
            a: snap.a,
            b: snap.b,
            mode: snap.kinetics,
        };
        let diffusion = AdvectionDiffusion2D::new(
            Dimension(width),
            Dimension(height),
            StepSize(1.0),
            StepSize(1.0),
        );
        let mut system = TuringSystem::new_with_kinetics(
            Dimension(width * height),
            DiffusionCoeff(snap.d_u),
            DiffusionCoeff(snap.d_v),
            kinetics,
            diffusion,
        );
        MorphogenesisTool::initialize_coupled_system(&mut system, width, height);

        #[cfg(feature = "physics")]
        let tau = (3.0 * snap.viscosity + 0.5).max(0.51);
        let mut runner = Self {
            system,
            params: Arc::clone(params),
            width,
            height,
            vorticity_buf: vec![0.0; width * height],
            peclet: 0.0,
            damkohler: 0.0,
            last_kinetics_mode: snap.kinetics,
            last_flow_mode: snap.flow_mode,
            #[cfg(feature = "physics")]
            lbm: LatticeBoltzmannD2Q9::new(width.max(1), height.max(1), tau),
            #[cfg(feature = "physics")]
            potential_flow: PotentialFlowField::new(),
            #[cfg(feature = "physics")]
            last_potential_key: (u64::MAX, u64::MAX),
            #[cfg(feature = "physics")]
            potential_dirty: true,
        };
        if snap.flow_mode != MorphogenesisFlowMode::None {
            runner.seed_default_bluff_obstacle();
        }
        #[cfg(feature = "physics")]
        runner.rebuild_potential_elements(snap.inlet_velocity, snap.circulation);
        runner.refresh_diagnostics(&snap);
        runner
    }

    pub(crate) fn read_params(&self) -> MorphogenesisSharedParams {
        self.params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn sync_flow_field(&mut self, snap: &MorphogenesisSharedParams) {
        if self.last_flow_mode != snap.flow_mode {
            if snap.flow_mode == MorphogenesisFlowMode::None {
                self.system.diffusion.clear_velocity();
            } else if !self.system.diffusion.obstacles().iter().any(|&b| b) {
                self.seed_default_bluff_obstacle();
            }
            self.last_flow_mode = snap.flow_mode;
            #[cfg(feature = "physics")]
            {
                self.potential_dirty = true;
            }
        }
        match snap.flow_mode {
            MorphogenesisFlowMode::None => {}
            MorphogenesisFlowMode::LbmVortexWake => {
                #[cfg(feature = "physics")]
                self.step_lbm_coupling(snap.inlet_velocity, snap.viscosity);
                #[cfg(not(feature = "physics"))]
                self.system
                    .diffusion
                    .set_uniform_velocity(snap.inlet_velocity, 0.0);
            }
            MorphogenesisFlowMode::PotentialFlow => {
                #[cfg(feature = "physics")]
                self.rebuild_potential_elements(snap.inlet_velocity, snap.circulation);
                #[cfg(not(feature = "physics"))]
                self.system
                    .diffusion
                    .set_uniform_velocity(snap.inlet_velocity, 0.0);
            }
        }
    }

    fn sync_flow_and_params(&mut self) -> Option<(f64, MorphogenesisSharedParams)> {
        let snap = self.read_params();
        if ![
            snap.a,
            snap.b,
            snap.d_u,
            snap.d_v,
            snap.inlet_velocity,
            snap.viscosity,
            snap.circulation,
        ]
        .into_iter()
        .all(f64::is_finite)
        {
            return None;
        }
        self.system.kinetics.a = snap.a;
        self.system.kinetics.b = snap.b;
        self.system.diffusion_coeffs = [snap.d_u, snap.d_v];

        if self.last_kinetics_mode != snap.kinetics {
            self.last_kinetics_mode = snap.kinetics;
            self.system.kinetics.mode = snap.kinetics;
            MorphogenesisTool::initialize_coupled_system(&mut self.system, self.width, self.height);
        } else if snap.kinetics == MorphogenesisKineticsMode::Schnakenberg
            && snap.dt.is_finite()
            && snap.dt > 0.0
            && is_pattern_forming_regime(snap.a, snap.b, snap.d_u, snap.d_v, snap.dt)
        {
            let var_u = grid_u_variance(self.system.u());
            if var_u.is_finite() && var_u < WASHOUT_VARIANCE_THRESHOLD {
                MorphogenesisTool::initialize_coupled_system(
                    &mut self.system,
                    self.width,
                    self.height,
                );
            }
        }

        self.sync_flow_field(&snap);
        self.refresh_diagnostics(&snap);
        (snap.dt.is_finite() && snap.dt > 0.0).then_some((snap.dt, snap))
    }

    pub(crate) fn refresh_diagnostics(&mut self, snap: &MorphogenesisSharedParams) {
        self.system
            .diffusion
            .compute_vorticity(&mut self.vorticity_buf);
        let u_char = self.system.diffusion.characteristic_speed();
        let length = self.system.diffusion.characteristic_length();
        let (pe, da) =
            compute_peclet_damkohler(u_char, length, snap.d_u, snap.a, snap.b, snap.kinetics);
        self.peclet = pe;
        self.damkohler = da;
    }
}

impl SimulationRunner for MorphogenesisRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::Reset => {
                let snap = self.read_params();
                self.system.kinetics.a = snap.a;
                self.system.kinetics.b = snap.b;
                self.system.kinetics.mode = snap.kinetics;
                self.last_kinetics_mode = snap.kinetics;
                #[cfg(feature = "physics")]
                self.lbm.init_equilibrium();
                self.system.diffusion.clear_velocity();
                MorphogenesisTool::initialize_coupled_system(
                    &mut self.system,
                    self.width,
                    self.height,
                );
                #[cfg(feature = "physics")]
                {
                    self.potential_dirty = true;
                }
                self.refresh_diagnostics(&snap);
            }
            SimCommand::Custom(ref s) if s == "reinit" => {
                MorphogenesisTool::initialize_coupled_system(
                    &mut self.system,
                    self.width,
                    self.height,
                );
            }
            SimCommand::ApplyBrush {
                cx,
                cy,
                r,
                is_obstacle,
            } => self.handle_brush_command(cx, cy, r, is_obstacle),
            SimCommand::ClearObstacles => self.handle_clear_obstacles(),
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
        if let Some((dt, snap)) = self.sync_flow_and_params() {
            if snap.flow_mode == MorphogenesisFlowMode::None {
                self.system.step(dt);
            } else {
                let d_max = snap.d_u.max(snap.d_v);
                let substeps = self.system.diffusion.cfl_substeps(
                    dt,
                    d_max,
                    TARGET_ADVECTIVE_CFL,
                    MAX_CFL_SUBSTEPS,
                );
                let sub_dt = dt / (substeps as f64);
                for _ in 0..substeps {
                    self.system.step(sub_dt);
                    if !FiniteStateGuard::is_finite_state(self) {
                        break;
                    }
                }
            }
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let snap_params = self.read_params();
        let theme_mode = if snap_params.high_contrast {
            ThemeMode::HighContrast
        } else {
            snap_params.theme_mode
        };
        let conc_slice = match self.system.kinetics.mode {
            MorphogenesisKineticsMode::Schnakenberg => self.system.u(),
            MorphogenesisKineticsMode::GrayScott => self.system.v(),
        };
        let conc_vec = conc_slice.to_vec();
        let vort_vec = self.vorticity_buf.clone();
        let obs_vec = self.system.diffusion.obstacles().to_vec();

        let image = render_morphogenesis_overlay(
            &conc_vec,
            &vort_vec,
            &obs_vec,
            self.width,
            self.height,
            snap_params.overlay_mode,
            theme_mode,
        );
        let structured = MorphogenesisSnapshotData {
            concentration: conc_vec,
            vorticity: vort_vec,
            obstacles: obs_vec,
            peclet: self.peclet,
            damkohler: self.damkohler,
        };
        StateSnapshot {
            width: self.width,
            height: self.height,
            pixels: Arc::new(std::sync::RwLock::new(image.pixels)),
            custom_data: vec![self.peclet, self.damkohler],
            structured_data: Some(Box::new(structured)),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }
}

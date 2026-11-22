#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::overlay::render_morphogenesis_overlay;
use super::types::{
    compute_peclet_damkohler, MorphogenesisBrushMode, MorphogenesisFlowMode, MorphogenesisKinetics,
    MorphogenesisKineticsMode, MorphogenesisSharedParams, MorphogenesisSnapshotData,
};
use super::MorphogenesisTool;
use crate::app::ThemeMode;
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use math_commons::math_kernel::types::{flatten_2d_index, Dimension, StepSize};
use math_explorer::biology::diffusion::AdvectionDiffusion2D;
use math_explorer::biology::morphogenesis::{DiffusionCoeff, TuringSystem};
#[cfg(feature = "physics")]
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    BgkCollision, LatticeBoltzmannD2Q9,
};
#[cfg(feature = "physics")]
use math_explorer::physics::fluid_dynamics::potential_flow::{
    Doublet, FlowElement, PotentialFlowField, UniformFlow, Vortex,
};
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
    last_potential_key: (u64, u64),
    #[cfg(feature = "physics")]
    potential_dirty: bool,
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

    fn seed_default_bluff_obstacle(&mut self) {
        if self.width < 8 || self.height < 8 {
            return;
        }
        let cx = self.width / 4;
        let cy = self.height / 2;
        let radius = (self.height / 12).clamp(2, 6);
        self.paint_obstacle_both(cx, cy, radius, true);
    }

    fn paint_obstacle_both(&mut self, cx: usize, cy: usize, radius: usize, is_obstacle: bool) {
        self.system
            .diffusion
            .paint_obstacle_disk(cx, cy, radius, is_obstacle);
        #[cfg(feature = "physics")]
        {
            let r = radius as isize;
            let y_min = cy.saturating_sub(radius);
            let y_max = (cy + radius).min(self.height.saturating_sub(1));
            let x_min = cx.saturating_sub(radius);
            let x_max = (cx + radius).min(self.width.saturating_sub(1));
            for y in y_min..=y_max {
                for x in x_min..=x_max {
                    let dx = x as isize - cx as isize;
                    let dy = y as isize - cy as isize;
                    if dx * dx + dy * dy <= r * r {
                        self.lbm.set_obstacle(x, y, is_obstacle);
                    }
                }
            }
            self.potential_dirty = true;
        }
    }

    fn apply_reagent_disk(&mut self, cx: usize, cy: usize, radius: usize, inject: bool) {
        let r = radius as isize;
        let y_min = cy.saturating_sub(radius);
        let y_max = (cy + radius).min(self.height.saturating_sub(1));
        let x_min = cx.saturating_sub(radius);
        let x_max = (cx + radius).min(self.width.saturating_sub(1));
        let mode = self.system.kinetics.mode;
        let a = self.system.kinetics.a;
        let b = self.system.kinetics.b;

        for y in y_min..=y_max {
            for x in x_min..=x_max {
                let dx = x as isize - cx as isize;
                let dy = y as isize - cy as isize;
                let idx = flatten_2d_index(x, y, self.width);
                if dx * dx + dy * dy <= r * r && !self.system.diffusion.obstacles()[idx] {
                    let (u_val, v_val) = match (mode, inject) {
                        (MorphogenesisKineticsMode::Schnakenberg, true) => {
                            let u_eq = (a + b).max(0.05);
                            (u_eq + 1.2, b / (u_eq * u_eq))
                        }
                        (MorphogenesisKineticsMode::Schnakenberg, false) => {
                            let u_eq = (a + b).max(0.05);
                            (u_eq, b / (u_eq * u_eq))
                        }
                        (MorphogenesisKineticsMode::GrayScott, true) => (0.50, 0.25),
                        (MorphogenesisKineticsMode::GrayScott, false) => (1.0, 0.0),
                    };
                    self.system.u_mut()[idx] = u_val;
                    self.system.v_mut()[idx] = v_val;
                }
            }
        }
    }

    #[cfg(feature = "physics")]
    fn rebuild_potential_elements(&mut self, inlet_velocity: f64, circulation: f64) {
        let key = (inlet_velocity.to_bits(), circulation.to_bits());
        if self.last_potential_key == key && !self.potential_dirty {
            return;
        }
        self.potential_flow.clear();
        let cx = (self.width as f64) * 0.35;
        let cy = (self.height as f64) * 0.50;
        let cyl_r = ((self.height as f64) / 10.0).max(2.0);
        let kappa = 2.0 * std::f64::consts::PI * inlet_velocity * cyl_r * cyl_r;
        self.potential_flow
            .add(Box::new(UniformFlow::new(inlet_velocity, 0.0)));
        self.potential_flow
            .add(Box::new(Doublet::new(kappa, cx, cy)));
        self.potential_flow
            .add(Box::new(Vortex::new(circulation, cx, cy)));
        self.last_potential_key = key;
        self.potential_dirty = false;
        self.evaluate_potential_grid();
    }

    #[cfg(feature = "physics")]
    fn evaluate_potential_grid(&mut self) {
        let w = self.width;
        let h = self.height;
        let (vx, vy, obs) = self.system.diffusion.buffers_mut();
        for y in 0..h {
            for x in 0..w {
                let idx = flatten_2d_index(x, y, w);
                if obs[idx] {
                    vx[idx] = 0.0;
                    vy[idx] = 0.0;
                } else {
                    let vel = self.potential_flow.velocity(x as f64 + 0.5, y as f64 + 0.5);
                    vx[idx] = vel.x.clamp(-1.5, 1.5);
                    vy[idx] = vel.y.clamp(-1.5, 1.5);
                }
            }
        }
    }

    #[cfg(feature = "physics")]
    fn step_lbm_coupling(&mut self, inlet_velocity: f64, viscosity: f64) {
        self.lbm.collision_model.tau = (3.0 * viscosity + 0.5).max(0.51);
        let u_in = inlet_velocity.clamp(0.0, 0.22);
        let jet_h = self.height.saturating_sub(2).max(1);
        let jet_y = usize::from(self.height > 2);
        self.lbm.set_inlet(0, jet_y, 1, jet_h, u_in, 0.0);
        self.lbm.step();
        let (vx, vy, obs) = self.system.diffusion.buffers_mut();
        self.lbm.state.copy_macroscopic_row_major(vx, vy, obs);
    }

    fn read_params(&self) -> MorphogenesisSharedParams {
        self.params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn sync_flow_and_params(&mut self) -> Option<(f64, MorphogenesisSharedParams)> {
        let snap = self.read_params();
        if !snap.a.is_finite()
            || !snap.b.is_finite()
            || !snap.d_u.is_finite()
            || !snap.d_v.is_finite()
            || !snap.inlet_velocity.is_finite()
            || !snap.viscosity.is_finite()
            || !snap.circulation.is_finite()
        {
            return None;
        }
        self.system.kinetics.a = snap.a;
        self.system.kinetics.b = snap.b;
        self.system.diffusion_coeffs[0] = snap.d_u;
        self.system.diffusion_coeffs[1] = snap.d_v;

        if self.last_kinetics_mode != snap.kinetics {
            self.last_kinetics_mode = snap.kinetics;
            self.system.kinetics.mode = snap.kinetics;
            MorphogenesisTool::initialize_coupled_system(&mut self.system, self.width, self.height);
        }

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

        self.refresh_diagnostics(&snap);
        (snap.dt.is_finite() && snap.dt > 0.0).then_some((snap.dt, snap))
    }

    fn refresh_diagnostics(&mut self, snap: &MorphogenesisSharedParams) {
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

    fn handle_brush_command(&mut self, cx: i32, cy: i32, r: i32, is_obstacle: bool) {
        if self.width == 0 || self.height == 0 {
            return;
        }
        let gx = cx.clamp(0, self.width as i32 - 1) as usize;
        let gy = cy.clamp(0, self.height as i32 - 1) as usize;
        let radius = r.max(0) as usize;
        let snap = self.read_params();
        match snap.brush_mode {
            MorphogenesisBrushMode::Obstacle => {
                self.paint_obstacle_both(gx, gy, radius, is_obstacle);
            }
            MorphogenesisBrushMode::EraseObstacle => {
                self.paint_obstacle_both(gx, gy, radius, false);
            }
            MorphogenesisBrushMode::Reagent => {
                self.apply_reagent_disk(gx, gy, radius, is_obstacle);
            }
            MorphogenesisBrushMode::PanZoom => {}
        }
        #[cfg(feature = "physics")]
        if snap.flow_mode == MorphogenesisFlowMode::PotentialFlow && self.potential_dirty {
            self.rebuild_potential_elements(snap.inlet_velocity, snap.circulation);
        }
        self.refresh_diagnostics(&snap);
    }

    fn handle_clear_obstacles(&mut self) {
        self.system.diffusion.clear_obstacles();
        #[cfg(feature = "physics")]
        {
            self.lbm.clear_obstacles();
            self.potential_dirty = true;
        }
        let snap = self.read_params();
        #[cfg(feature = "physics")]
        if snap.flow_mode == MorphogenesisFlowMode::PotentialFlow {
            self.rebuild_potential_elements(snap.inlet_velocity, snap.circulation);
        }
        self.refresh_diagnostics(&snap);
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

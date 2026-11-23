//! Multi-kinetics morphogenesis worker runner coupling LBM / PotentialFlow advection,
//! microfluidic `ReactionCellRole` masks, counterfactual shadow twins, and `K = 16` checkpoints.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub use super::counterfactual::MorphogenesisSnapshotData as MorphoSnapshotData;
use super::counterfactual::{active_morphogen_slice, MorphogenesisTwinState};
use super::diffusion::CoupledMorphoDiffusion;
#[cfg(test)]
pub use super::legacy_runner::MorphogenesisRunner;
pub(crate) use super::runner_twin::apply_uniform_noise;
use super::sculpt::{apply_stroke_to_morpho_system, enforce_system_mask};
pub use super::types::MorphogenesisSharedParams;
use super::types::{
    compute_peclet_damkohler, MorphogenesisFlowMode, MorphogenesisFlowState,
    MorphogenesisKineticsMode,
};
use super::MorphogenesisTool;
use crate::async_sim::{
    CheckpointParams, SimCommand, SimulationRunner, StateSnapshot, TimeTravelCheckpoint,
};
use crate::framework::decode_rle_mask;
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use math_explorer::biology::morphogenesis::{
    GrayScottKinetics, ReactionKinetics, SchnakenbergKinetics, TuringSystem,
};
#[cfg(feature = "physics")]
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    BgkCollision, LatticeBoltzmannD2Q9,
};
#[cfg(feature = "physics")]
use math_explorer::physics::fluid_dynamics::potential_flow::PotentialFlowField;
use oxidize_core::rng::OxidizeRng;
use std::sync::Arc;

pub(super) const MAX_CFL_SUBSTEPS: usize = 64;
pub(super) const TARGET_ADVECTIVE_CFL: f64 = 0.40;

/// Default PRNG seed for deterministic morphogenesis perturbation streams.
pub const DEFAULT_MORPHOGENESIS_SEED: u64 = 0x4D4F_5250_484F_3244;

/// Zero-allocation kinetics adapter switching cleanly between [`SchnakenbergKinetics`]
/// and [`GrayScottKinetics`] inside [`TuringSystem`].
#[derive(Clone, Copy, Debug)]
pub struct MorphogenesisKineticsAdapter {
    /// Active reaction kinetics family.
    pub mode: MorphogenesisKineticsMode,
    /// Schnakenberg activator feed `a`.
    pub a: f64,
    /// Schnakenberg inhibitor feed `b`.
    pub b: f64,
    /// Gray-Scott substrate feed rate `f`.
    pub f: f64,
    /// Gray-Scott catalyst kill rate `k`.
    pub k: f64,
}

impl ReactionKinetics<2> for MorphogenesisKineticsAdapter {
    #[inline(always)]
    fn reaction(&self, concentrations: [f64; 2]) -> [f64; 2] {
        match self.mode {
            MorphogenesisKineticsMode::Schnakenberg => {
                SchnakenbergKinetics::new(self.a, self.b).reaction(concentrations)
            }
            MorphogenesisKineticsMode::GrayScott => {
                GrayScottKinetics::new(self.f, self.k).reaction(concentrations)
            }
        }
    }
}

/// Shared Gray-Scott mode and `(f, k)` parameters paired with [`MorphogenesisSharedParams`].
#[derive(Clone, Copy, Debug)]
pub struct MorphogenesisKineticsState {
    /// Selected kinetics regime.
    pub mode: MorphogenesisKineticsMode,
    /// Gray-Scott substrate feed rate `f`.
    pub f: f64,
    /// Gray-Scott autocatalyst kill rate `k`.
    pub k: f64,
}

impl Default for MorphogenesisKineticsState {
    fn default() -> Self {
        let gs = GrayScottKinetics::mitosis();
        Self {
            mode: MorphogenesisKineticsMode::Schnakenberg,
            f: gs.f,
            k: gs.k,
        }
    }
}

/// Multi-kinetics worker runner supporting [`SchnakenbergKinetics`], [`GrayScottKinetics`],
/// microfluidic [`CoupledMorphoDiffusion`] sculpting, coupled LBM / PotentialFlow advection,
/// counterfactual shadow twins, and `K = 16` zero-allocation time-travel checkpoint replay.
pub struct MorphogenesisKineticsRunner {
    pub(crate) system: TuringSystem<2, MorphogenesisKineticsAdapter, CoupledMorphoDiffusion>,
    pub(crate) params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
    pub(crate) kinetics_state: Arc<std::sync::RwLock<MorphogenesisKineticsState>>,
    pub(crate) flow_state: Arc<std::sync::RwLock<MorphogenesisFlowState>>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) step_count: usize,
    pub(crate) rng: OxidizeRng,
    pub(crate) noise_amplitude: f64,
    pub(crate) vorticity_buf: Vec<f64>,
    pub(crate) peclet: f64,
    pub(crate) damkohler: f64,
    pub(crate) last_flow_mode: MorphogenesisFlowMode,
    #[cfg(feature = "physics")]
    pub(crate) lbm: LatticeBoltzmannD2Q9<BgkCollision>,
    #[cfg(feature = "physics")]
    pub(crate) potential_flow: PotentialFlowField,
    #[cfg(feature = "physics")]
    pub(crate) last_potential_key: (u64, u64),
    #[cfg(feature = "physics")]
    pub(crate) potential_dirty: bool,
    pub(crate) twin: Option<MorphogenesisTwinState>,
}

impl FiniteStateGuard for MorphogenesisKineticsRunner {
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

impl MorphogenesisKineticsRunner {
    pub(crate) fn mode_to_scalar(mode: MorphogenesisKineticsMode) -> f64 {
        match mode {
            MorphogenesisKineticsMode::Schnakenberg => 0.0,
            MorphogenesisKineticsMode::GrayScott => 1.0,
        }
    }

    pub(crate) fn scalar_to_mode(tag: f64) -> MorphogenesisKineticsMode {
        if tag >= 0.5 {
            MorphogenesisKineticsMode::GrayScott
        } else {
            MorphogenesisKineticsMode::Schnakenberg
        }
    }

    pub(crate) fn read_kinetics(&self) -> MorphogenesisKineticsState {
        *self
            .kinetics_state
            .read()
            .unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn read_flow(&self) -> MorphogenesisFlowState {
        *self.flow_state.read().unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn enforce_mask_concentrations(&mut self) {
        enforce_system_mask(&mut self.system, self.width, self.height);
    }

    pub(crate) fn refresh_diagnostics(&mut self) {
        self.system
            .diffusion
            .advection
            .compute_vorticity(&mut self.vorticity_buf);
        let u_char = self.system.diffusion.advection.characteristic_speed();
        let length = self.system.diffusion.advection.characteristic_length();
        let mode = self.system.kinetics.mode;
        let (r1, r2) = match mode {
            MorphogenesisKineticsMode::Schnakenberg => {
                (self.system.kinetics.a, self.system.kinetics.b)
            }
            MorphogenesisKineticsMode::GrayScott => {
                (self.system.kinetics.f, self.system.kinetics.k)
            }
        };
        let d_u = self.system.diffusion_coeffs[0];
        let (pe, da) = compute_peclet_damkohler(u_char, length, d_u, r1, r2, mode);
        self.peclet = pe;
        self.damkohler = da;
    }

    pub(super) fn sync_system_params(&mut self) -> Option<f64> {
        let p = *self.params.read().unwrap_or_else(|e| e.into_inner());
        let ks = self.read_kinetics();
        let flow = self.read_flow();
        if ![flow.inlet_velocity, flow.viscosity, flow.circulation]
            .into_iter()
            .all(f64::is_finite)
        {
            return None;
        }
        let dt = self.apply_explicit_params(ks.mode, p.a, p.b, ks.f, ks.k, p.d_u, p.d_v, p.dt)?;
        self.sync_flow_field(&flow);
        if flow.flow_mode != MorphogenesisFlowMode::None {
            self.refresh_diagnostics();
        }
        Some(dt)
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_explicit_params(
        &mut self,
        mode: MorphogenesisKineticsMode,
        a: f64,
        b: f64,
        f: f64,
        k: f64,
        d_u: f64,
        d_v: f64,
        dt: f64,
    ) -> Option<f64> {
        if ![a, b, f, k, d_u, d_v].into_iter().all(f64::is_finite) {
            return None;
        }
        let prev_mode = self.system.kinetics.mode;
        self.system.kinetics.mode = mode;
        self.system.kinetics.a = a;
        self.system.kinetics.b = b;
        self.system.kinetics.f = f;
        self.system.kinetics.k = k;
        self.system.diffusion_coeffs = [d_u, d_v];
        if !self.has_counterfactual_twin()
            && (prev_mode != mode
                || super::gray_scott_atlas::should_reseed_washed_out_system(
                    &self.system,
                    d_u,
                    d_v,
                    dt,
                ))
        {
            MorphogenesisTool::initialize_adapter_system(&mut self.system, self.width, self.height);
            self.enforce_mask_concentrations();
        }
        (dt.is_finite() && dt > 0.0).then_some(dt)
    }

    pub(crate) fn apply_scalar_params(&mut self, p: &[f64]) -> Option<f64> {
        if p.len() < 9 {
            return None;
        }
        self.noise_amplitude = p[8];
        let mode = Self::scalar_to_mode(p[0]);
        self.apply_explicit_params(mode, p[1], p[2], p[3], p[4], p[5], p[6], p[7])
    }

    pub(crate) fn advance_grid_step(&mut self, dt: f64, noise_amp: f64) {
        self.enforce_mask_concentrations();
        if self.system.diffusion.flow_active {
            let d_max = self.system.diffusion_coeffs[0].max(self.system.diffusion_coeffs[1]);
            let substeps = self.system.diffusion.advection.cfl_substeps(
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
        } else {
            self.system.step(dt);
        }
        apply_uniform_noise(
            &mut self.system,
            self.width * self.height,
            &mut self.rng,
            noise_amp,
        );
        if self.system.diffusion.mask.has_active_cells() {
            self.enforce_mask_concentrations();
        }
        self.step_count += 1;
    }

    fn reset_grid_and_counters(&mut self) {
        #[cfg(feature = "physics")]
        {
            self.lbm.init_equilibrium();
            self.potential_dirty = true;
        }
        self.system.diffusion.advection.clear_velocity();
        self.twin = None;
        let _ = self.sync_system_params();
        MorphogenesisTool::initialize_adapter_system(&mut self.system, self.width, self.height);
        self.enforce_mask_concentrations();
        self.refresh_diagnostics();
        self.step_count = 0;
        self.rng = OxidizeRng::new(DEFAULT_MORPHOGENESIS_SEED);
    }

    fn apply_sculpt_command(&mut self, cmd: &SimCommand) {
        match cmd {
            SimCommand::ApplyBrush {
                cx,
                cy,
                r,
                is_obstacle,
            } => self.handle_brush_command(*cx, *cy, *r, *is_obstacle),
            SimCommand::ApplyBrushStroke(stroke) => {
                apply_stroke_to_morpho_system(&mut self.system, self.width, self.height, stroke);
                self.sync_mask_to_hydrodynamics();
            }
            SimCommand::SyncSculptState(payload) => {
                if payload.rle_mask.is_empty() {
                    self.system.diffusion.mask.clear();
                } else if let Ok(bytes) =
                    decode_rle_mask(&payload.rle_mask, self.width * self.height)
                {
                    let _ = self.system.diffusion.mask.set_roles_slice(&bytes);
                }
                self.enforce_mask_concentrations();
                self.sync_mask_to_hydrodynamics();
                if payload.rle_mask.is_empty()
                    && self.read_flow().flow_mode != MorphogenesisFlowMode::None
                {
                    self.seed_default_bluff_obstacle();
                }
            }
            SimCommand::ClearObstacles => self.handle_clear_obstacles(),
            SimCommand::Custom(s) if s == "clear_mask" => self.handle_clear_obstacles(),
            SimCommand::Custom(s) if s == "stamp_maze" => {
                self.system.diffusion.mask.stamp_microfluidic_maze();
                self.enforce_mask_concentrations();
                self.sync_mask_to_hydrodynamics();
            }
            _ => {}
        }
    }
}

impl SimulationRunner for MorphogenesisKineticsRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::Reset => self.reset_grid_and_counters(),
            SimCommand::Custom(ref s) if s == "reinit" => self.reset_grid_and_counters(),
            SimCommand::Custom(ref s) if self.process_twin_custom_command(s) => {}
            _ => self.apply_sculpt_command(&cmd),
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
                "Morphogenesis reaction-diffusion diverged: non-finite concentration encountered.",
            )
        })
    }

    fn has_counterfactual_twin(&self) -> bool {
        self.twin.is_some()
    }

    fn step_branch_a(&mut self) {
        if !FiniteStateGuard::is_finite_state(self) {
            return;
        }
        let noise_amp = self.noise_amplitude;
        if let Some(dt) = self.sync_system_params() {
            if let Some(ref mut twin) = self.twin {
                enforce_system_mask(&mut twin.system_a, self.width, self.height);
                twin.step_branch_a(dt, self.width, self.height, noise_amp);
            }
        }
    }

    fn step_branch_b(&mut self) {
        if !FiniteStateGuard::is_finite_state(self) {
            return;
        }
        let noise_amp = self.noise_amplitude;
        if let Some(dt) = self.sync_system_params() {
            self.advance_grid_step(dt, noise_amp);
            if let Some(ref mut twin) = self.twin {
                twin.observe_lockstep(self.step_count, dt, active_morphogen_slice(&self.system));
            }
        }
    }

    fn step(&mut self) {
        if !FiniteStateGuard::is_finite_state(self) {
            return;
        }
        let noise_amp = self.noise_amplitude;
        if let Some(dt) = self.sync_system_params() {
            if let Some(ref mut twin) = self.twin {
                enforce_system_mask(&mut twin.system_a, self.width, self.height);
                twin.step_branch_a(dt, self.width, self.height, noise_amp);
            }
            self.advance_grid_step(dt, noise_amp);
            if let Some(ref mut twin) = self.twin {
                twin.observe_lockstep(self.step_count, dt, active_morphogen_slice(&self.system));
            }
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        self.build_snapshot()
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn running_substeps_per_tick(&self) -> usize {
        if self.read_flow().flow_mode != MorphogenesisFlowMode::None {
            1
        } else {
            match self.read_kinetics().mode {
                MorphogenesisKineticsMode::Schnakenberg => 1,
                MorphogenesisKineticsMode::GrayScott => 4,
            }
        }
    }

    fn checkpoint_stride(&self) -> Option<usize> {
        Some(16)
    }

    fn current_step(&self) -> usize {
        self.step_count
    }

    fn capture_active_params(&self) -> Option<CheckpointParams> {
        self.capture_active_params_impl()
    }

    fn capture_checkpoint(&self) -> Option<TimeTravelCheckpoint> {
        self.capture_checkpoint_impl()
    }

    fn restore_checkpoint(&mut self, checkpoint: &TimeTravelCheckpoint) {
        self.restore_checkpoint_impl(checkpoint);
    }

    fn replay_step(&mut self, params: &CheckpointParams) {
        self.replay_step_impl(params);
    }
}

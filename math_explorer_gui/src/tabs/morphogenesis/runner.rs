//! Background simulation runner for dual-kinetics Turing, Gray-Scott, microfluidic
//! sculpting, and `K = 16` zero-allocation time-travel checkpoint replay.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::cartographer::{grid_u_variance, is_pattern_forming_regime, WASHOUT_VARIANCE_THRESHOLD};
use super::gray_scott_atlas::{is_gray_scott_pattern_viable, is_gray_scott_v_washed_out};
use super::sculpt::overlay_cell_mask_colors;
pub use super::sculpt::MorphoDiffusion;
pub use super::state::MorphoSnapshotData;
use super::state::{
    apply_chemical_brush, apply_uniform_noise, ensure_nucleated_for_kinetics,
    initialize_for_kinetics, initialize_gray_scott_system, initialize_schnakenberg_system,
    plot_concentration_with_contrast, MorphogenesisKinetics,
};
use super::teleport::reconcile_kinetics_state;
pub use super::teleport::{MorphogenesisKineticsState, MorphogenesisSharedParams};
use crate::async_sim::{
    CheckpointParams, CheckpointPayload, SimCommand, SimulationRunner, StateSnapshot,
    TimeTravelCheckpoint,
};
use crate::framework::decode_rle_mask;
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use math_explorer::biology::diffusion::{
    MaskedFiniteDifference2D, ReactionCellRole, SpatialDiffusion,
};
use math_explorer::biology::morphogenesis::{
    FusedEulerSolver, GrayScottKinetics, SchnakenbergKinetics, TuringSystem,
};
use oxidize_core::rng::OxidizeRng;
use std::sync::Arc;

/// Maximum CFL substeps executed per runner step when `auto_dt` is active.
pub const MAX_CFL_SUBSTEPS: usize = 32;

/// Default PRNG seed for deterministic morphogenesis perturbation streams.
pub const DEFAULT_MORPHOGENESIS_SEED: u64 = 0x4D4F_5250_484F_3244;

/// Zero-allocation dual-kinetics simulation runner with `K = 16` time-travel checkpointing.
pub struct MorphogenesisRunner<D: SpatialDiffusion<2> = MaskedFiniteDifference2D> {
    pub(crate) system: TuringSystem<2, SchnakenbergKinetics, D>,
    pub(crate) kinetics: MorphogenesisKinetics,
    pub(crate) gray_scott: GrayScottKinetics,
    pub(crate) params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
    pub(crate) kinetics_state: Option<Arc<std::sync::RwLock<MorphogenesisKineticsState>>>,
    pub(crate) last_synced_ks: MorphogenesisKineticsState,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) step_count: usize,
    pub(crate) rng: OxidizeRng,
    pub(crate) noise_amplitude: f64,
}

/// Test alias matching masked dual-kinetics time-travel runner signatures.
#[cfg(test)]
pub type MorphogenesisKineticsRunner = MorphogenesisRunner<MaskedFiniteDifference2D>;

impl<D: MorphoDiffusion> FiniteStateGuard for MorphogenesisRunner<D> {
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

impl<D: MorphoDiffusion> MorphogenesisRunner<D> {
    const fn mode_to_scalar(mode: MorphogenesisKinetics) -> f64 {
        match mode {
            MorphogenesisKinetics::Schnakenberg => 0.0,
            MorphogenesisKinetics::GrayScott => 1.0,
        }
    }

    fn scalar_to_mode(tag: f64) -> MorphogenesisKinetics {
        if tag >= 0.5 {
            MorphogenesisKinetics::GrayScott
        } else {
            MorphogenesisKinetics::Schnakenberg
        }
    }

    pub(crate) fn enforce_mask_concentrations(&mut self) {
        self.system.diffusion.sync_emitter_levels(self.kinetics);
        if !self.system.diffusion.has_active_mask() {
            return;
        }
        let (emit_u, emit_v) = self.system.diffusion.emitter_levels();
        let roles_len = self
            .system
            .diffusion
            .mask_roles()
            .len()
            .min(self.width * self.height);
        for idx in 0..roles_len {
            let role = ReactionCellRole::from_u8(self.system.diffusion.mask_roles()[idx]);
            match role {
                ReactionCellRole::ZeroFluxWall | ReactionCellRole::Sink => {
                    self.system.u_mut()[idx] = 0.0;
                    self.system.v_mut()[idx] = 0.0;
                }
                ReactionCellRole::EmitterU => self.system.u_mut()[idx] = emit_u,
                ReactionCellRole::EmitterV => self.system.v_mut()[idx] = emit_v,
                ReactionCellRole::Free => {}
            }
        }
    }

    fn sync_system_params(&mut self) -> Option<(f64, bool)> {
        let mut p = *self.params.read().unwrap_or_else(|e| e.into_inner());
        if let Some(ks_arc) = &self.kinetics_state {
            reconcile_kinetics_state(&self.params, ks_arc, &mut self.last_synced_ks, &mut p);
        }
        self.apply_shared_params(&p)
    }

    fn apply_shared_params(&mut self, p: &MorphogenesisSharedParams) -> Option<(f64, bool)> {
        if ![p.a, p.b, p.f, p.k, p.d_u, p.d_v]
            .iter()
            .all(|x| x.is_finite())
            || p.d_u <= 0.0
            || p.d_v <= 0.0
        {
            return None;
        }
        self.system.kinetics.a = p.a;
        self.system.kinetics.b = p.b;
        self.gray_scott.f = p.f;
        self.gray_scott.k = p.k;
        self.system.diffusion_coeffs = [p.d_u, p.d_v];
        if self.kinetics != p.kinetics {
            self.kinetics = p.kinetics;
            initialize_for_kinetics(&mut self.system, self.kinetics, self.width, self.height);
            self.enforce_mask_concentrations();
        } else if p.dt.is_finite() && p.dt > 0.0 {
            let eff_dt = if p.auto_dt {
                FusedEulerSolver::cfl_substeps_2d(p.dt, 1.0, p.d_u, p.d_v, MAX_CFL_SUBSTEPS).1
            } else {
                p.dt
            };
            match self.kinetics {
                MorphogenesisKinetics::Schnakenberg => {
                    if is_pattern_forming_regime(p.a, p.b, p.d_u, p.d_v, eff_dt) {
                        let var_u = grid_u_variance(self.system.u());
                        if var_u.is_finite() && var_u < WASHOUT_VARIANCE_THRESHOLD {
                            initialize_schnakenberg_system(
                                &mut self.system,
                                self.width,
                                self.height,
                            );
                            self.enforce_mask_concentrations();
                        }
                    }
                }
                MorphogenesisKinetics::GrayScott => {
                    if is_gray_scott_pattern_viable(p.f, p.k, p.d_u, p.d_v, eff_dt)
                        && is_gray_scott_v_washed_out(self.system.v())
                    {
                        initialize_gray_scott_system(&mut self.system, self.width, self.height);
                        self.enforce_mask_concentrations();
                    }
                }
            }
        }
        (p.dt.is_finite() && p.dt > 0.0).then_some((p.dt, p.auto_dt))
    }

    fn apply_scalar_params(&mut self, s: &[f64]) -> Option<(f64, bool)> {
        let [mode, a, b, f, k, d_u, d_v, dt, auto, amp, ..] = s else {
            return None;
        };
        self.noise_amplitude = *amp;
        self.apply_shared_params(&MorphogenesisSharedParams {
            kinetics: Self::scalar_to_mode(*mode),
            a: *a,
            b: *b,
            f: *f,
            k: *k,
            d_u: *d_u,
            d_v: *d_v,
            dt: *dt,
            auto_dt: *auto >= 0.5,
            high_contrast: false,
        })
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

    fn advance_grid_step(&mut self, dt: f64, auto_dt: bool, noise_amp: f64) {
        self.enforce_mask_concentrations();
        if auto_dt {
            let [d_u, d_v] = self.system.diffusion_coeffs;
            let (substeps, sub_dt) =
                FusedEulerSolver::cfl_substeps_2d(dt, 1.0, d_u, d_v, MAX_CFL_SUBSTEPS);
            for _ in 0..substeps {
                self.step_active_kinetics(sub_dt);
                if !self.system.u().first().is_none_or(|x| x.is_finite()) {
                    break;
                }
            }
        } else {
            self.step_active_kinetics(dt);
        }
        apply_uniform_noise(
            &mut self.system,
            self.width * self.height,
            &mut self.rng,
            noise_amp,
        );
        if self.system.diffusion.has_active_mask() {
            self.enforce_mask_concentrations();
        }
        self.step_count += 1;
    }

    fn process_custom_command(&mut self, name: &str) {
        match name {
            "reinit" => {
                let _ = self.sync_system_params();
                initialize_for_kinetics(&mut self.system, self.kinetics, self.width, self.height);
                self.enforce_mask_concentrations();
                self.step_count = 0;
                self.rng = OxidizeRng::new(DEFAULT_MORPHOGENESIS_SEED);
            }
            "ensure_nucleated" => {
                let _ = self.sync_system_params();
                ensure_nucleated_for_kinetics(
                    &mut self.system,
                    self.kinetics,
                    self.width,
                    self.height,
                );
                self.enforce_mask_concentrations();
            }
            "clear_mask" => self.system.diffusion.clear_mask(),
            "stamp_maze" => {
                let _ = self.sync_system_params();
                self.system.diffusion.sync_emitter_levels(self.kinetics);
                self.system.diffusion.stamp_maze();
                self.enforce_mask_concentrations();
            }
            _ => {}
        }
    }
}

impl<D: MorphoDiffusion> SimulationRunner for MorphogenesisRunner<D> {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::Reset => self.process_custom_command("reinit"),
            SimCommand::Custom(ref s) => self.process_custom_command(s),
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
                self.enforce_mask_concentrations();
            }
            SimCommand::ApplyBrushStroke(ref stroke) => {
                let _ = self.sync_system_params();
                self.system.diffusion.sync_emitter_levels(self.kinetics);
                D::apply_morpho_stroke(&mut self.system, (self.width, self.height), stroke);
            }
            SimCommand::SyncSculptState(payload) => {
                if payload.rle_mask.is_empty() {
                    self.system.diffusion.clear_mask();
                } else if let Ok(bytes) =
                    decode_rle_mask(&payload.rle_mask, self.width * self.height)
                {
                    self.system.diffusion.set_mask_slice(&bytes);
                }
                self.enforce_mask_concentrations();
            }
            SimCommand::ClearObstacles => self.system.diffusion.clear_mask(),
            _ => {}
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
                "Reaction-diffusion solver diverged: non-finite concentration encountered.",
            )
        })
    }

    fn step(&mut self) {
        if !FiniteStateGuard::is_finite_state(self) {
            return;
        }
        let noise_amp = self.noise_amplitude;
        if let Some((dt, auto_dt)) = self.sync_system_params() {
            self.advance_grid_step(dt, auto_dt, noise_amp);
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let high_contrast = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .high_contrast;
        let field_vec = match self.kinetics {
            MorphogenesisKinetics::Schnakenberg => self.system.u().to_vec(),
            MorphogenesisKinetics::GrayScott => self.system.v().to_vec(),
        };
        let mut image =
            plot_concentration_with_contrast(&field_vec, self.width, self.height, high_contrast);
        let mask_roles = if self.system.diffusion.has_active_mask() {
            let roles = self.system.diffusion.mask_roles().to_vec();
            overlay_cell_mask_colors(&mut image, &roles, high_contrast);
            roles
        } else {
            Vec::new()
        };
        StateSnapshot {
            width: self.width,
            height: self.height,
            pixels: Arc::new(std::sync::RwLock::new(image.pixels)),
            custom_data: Vec::new(),
            structured_data: Some(Box::new(MorphoSnapshotData {
                u_data: field_vec,
                mask_roles,
            })),
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn running_substeps_per_tick(&self) -> usize {
        match self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .kinetics
        {
            MorphogenesisKinetics::Schnakenberg => 1,
            MorphogenesisKinetics::GrayScott => 4,
        }
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
            Self::mode_to_scalar(p.kinetics),
            p.a,
            p.b,
            p.f,
            p.k,
            p.d_u,
            p.d_v,
            p.dt,
            if p.auto_dt { 1.0 } else { 0.0 },
            self.noise_amplitude,
        ]))
    }

    fn capture_checkpoint(&self) -> Option<TimeTravelCheckpoint> {
        let params = self.capture_active_params()?;
        let mask = self
            .system
            .diffusion
            .has_active_mask()
            .then(|| self.system.diffusion.mask_roles().into());
        Some(TimeTravelCheckpoint {
            step: self.step_count,
            rng: self.rng.clone(),
            params,
            state: CheckpointPayload::DualField2D {
                width: self.width,
                height: self.height,
                u: self.system.u().into(),
                v: self.system.v().into(),
                mask,
            },
        })
    }

    fn restore_checkpoint(&mut self, checkpoint: &TimeTravelCheckpoint) {
        if let Some(s) = checkpoint.params.as_scalars() {
            if s.len() >= 10 {
                self.kinetics = Self::scalar_to_mode(s[0]);
                self.system.kinetics.a = s[1];
                self.system.kinetics.b = s[2];
                self.gray_scott.f = s[3];
                self.gray_scott.k = s[4];
                self.system.diffusion_coeffs = [s[5], s[6]];
                self.noise_amplitude = s[9];
            }
        }
        if let CheckpointPayload::DualField2D {
            width,
            height,
            ref u,
            ref v,
            ref mask,
        } = checkpoint.state
        {
            if width == self.width
                && height == self.height
                && u.len() == self.system.u().len()
                && v.len() == self.system.v().len()
            {
                self.system.u_mut().copy_from_slice(u);
                self.system.v_mut().copy_from_slice(v);
                match mask {
                    Some(bytes) if bytes.len() == width * height => {
                        self.system.diffusion.set_mask_slice(bytes);
                    }
                    _ => self.system.diffusion.clear_mask(),
                }
            }
        }
        self.step_count = checkpoint.step;
        self.rng = checkpoint.rng.clone();
    }

    fn replay_step(&mut self, params: &CheckpointParams) {
        if let Some((dt, auto_dt)) = params
            .as_scalars()
            .and_then(|s| self.apply_scalar_params(s))
        {
            let amp = self.noise_amplitude;
            self.advance_grid_step(dt, auto_dt, amp);
        }
    }
}

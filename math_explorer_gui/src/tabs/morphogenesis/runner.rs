//! Background simulation runner for dual-kinetics Turing, Gray-Scott, microfluidic
//! sculpting, counterfactual shadow twins, and `K = 16` time-travel checkpoint replay.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub use super::counterfactual::MorphogenesisSnapshotData as MorphoSnapshotData;
use super::counterfactual::{
    active_morphogen_slice, render_morphogenesis_heatmap, MorphogenesisHeatmapMode,
    MorphogenesisTwinState,
};
pub use super::sculpt::MorphoDiffusion;
use super::sculpt::{enforce_system_mask_generic, overlay_cell_mask_colors};
use super::state::{
    apply_chemical_brush, apply_uniform_noise, ensure_nucleated_for_kinetics,
    initialize_for_kinetics, MorphogenesisKinetics,
};
pub use super::teleport::{MorphogenesisKineticsState, MorphogenesisSharedParams};
use crate::app::ThemeMode;
use crate::async_sim::{
    CheckpointParams, CheckpointPayload, SimCommand, SimulationRunner, StateSnapshot,
    TimeTravelCheckpoint,
};
use crate::framework::decode_rle_mask;
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use math_explorer::biology::diffusion::{MaskedFiniteDifference2D, SpatialDiffusion};
use math_explorer::biology::morphogenesis::{
    FusedEulerSolver, GrayScottKinetics, SchnakenbergKinetics, TuringSystem,
};
use oxidize_core::rng::OxidizeRng;
use std::sync::Arc;

/// Maximum CFL substeps executed per runner step when `auto_dt` is active.
pub const MAX_CFL_SUBSTEPS: usize = 32;

/// Default PRNG seed for deterministic morphogenesis perturbation streams.
pub const DEFAULT_MORPHOGENESIS_SEED: u64 = 0x4D4F_5250_484F_3244;

/// Zero-allocation dual-kinetics simulation runner with `K = 16` time-travel checkpointing
/// and counterfactual shadow twins.
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
    pub(crate) twin: Option<MorphogenesisTwinState<D>>,
}

/// Type alias for the masked 2D dual-kinetics morphogenesis worker runner.
#[cfg(test)]
pub type MorphogenesisKineticsRunner = MorphogenesisRunner<MaskedFiniteDifference2D>;

#[inline]
pub(crate) fn step_system_kinetics<D: MorphoDiffusion>(
    system: &mut TuringSystem<2, SchnakenbergKinetics, D>,
    kinetics: MorphogenesisKinetics,
    gray_scott: &GrayScottKinetics,
    dt: f64,
    auto_dt: bool,
) {
    if auto_dt {
        let [d_u, d_v] = system.diffusion_coeffs;
        let (substeps, sub_dt) =
            FusedEulerSolver::cfl_substeps_2d(dt, 1.0, d_u, d_v, MAX_CFL_SUBSTEPS);
        for _ in 0..substeps {
            match kinetics {
                MorphogenesisKinetics::Schnakenberg => system.step(sub_dt),
                MorphogenesisKinetics::GrayScott => system.step_with_kinetics(gray_scott, sub_dt),
            }
            if !system.u().first().is_none_or(|x| x.is_finite()) {
                break;
            }
        }
    } else {
        match kinetics {
            MorphogenesisKinetics::Schnakenberg => system.step(dt),
            MorphogenesisKinetics::GrayScott => system.step_with_kinetics(gray_scott, dt),
        }
    }
}

impl<D: MorphoDiffusion> MorphogenesisRunner<D> {
    pub(crate) fn enforce_mask_concentrations(&mut self) {
        enforce_system_mask_generic(&mut self.system, self.kinetics, self.width, self.height);
    }

    fn advance_grid_step(&mut self, dt: f64, auto_dt: bool, noise_amp: f64) {
        self.enforce_mask_concentrations();
        step_system_kinetics(
            &mut self.system,
            self.kinetics,
            &self.gray_scott,
            dt,
            auto_dt,
        );
        apply_uniform_noise(
            &mut self.system,
            self.width * self.height,
            &mut self.rng,
            noise_amp,
        );
        self.enforce_mask_concentrations();
        self.step_count += 1;
    }

    fn process_custom_command(&mut self, name: &str) {
        match name {
            "reinit" => {
                self.twin = None;
                let _ = self.sync_system_params();
                initialize_for_kinetics(&mut self.system, self.kinetics, self.width, self.height);
                self.enforce_mask_concentrations();
                self.step_count = 0;
                self.rng = OxidizeRng::new(DEFAULT_MORPHOGENESIS_SEED);
            }
            "ensure_nucleated" => {
                if self.twin.is_some() {
                    return;
                }
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
            SimCommand::Custom(ref s) if self.process_twin_custom_command(s) => {}
            SimCommand::Custom(ref s) => self.process_custom_command(s),
            SimCommand::ApplyBrush {
                cx,
                cy,
                r,
                is_obstacle,
            } => {
                let dt = self.sync_system_params().map_or(0.005, |(d, _)| d);
                apply_chemical_brush(
                    &mut self.system,
                    self.kinetics,
                    (self.width, self.height),
                    (cx, cy, r),
                    is_obstacle,
                );
                self.enforce_mask_concentrations();
                if let Some(ref mut twin) = self.twin {
                    let ub = active_morphogen_slice(&self.system, self.kinetics);
                    twin.observe_lockstep(self.step_count, dt, ub);
                }
            }
            SimCommand::ApplyBrushStroke(ref stroke) => {
                let dt = self.sync_system_params().map_or(0.005, |(d, _)| d);
                self.system.diffusion.sync_emitter_levels(self.kinetics);
                D::apply_morpho_stroke(&mut self.system, (self.width, self.height), stroke);
                if let Some(ref mut twin) = self.twin {
                    let ub = active_morphogen_slice(&self.system, self.kinetics);
                    twin.observe_lockstep(self.step_count, dt, ub);
                }
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

    fn has_counterfactual_twin(&self) -> bool {
        self.twin.is_some()
    }

    fn step_branch_a(&mut self) {
        if !FiniteStateGuard::is_finite_state(self) {
            return;
        }
        let noise_amp = self.noise_amplitude;
        if let Some((dt, auto_dt)) = self.sync_system_params() {
            if let Some(ref mut twin) = self.twin {
                twin.step_branch_a(dt, auto_dt, self.width, self.height, noise_amp);
            }
        }
    }

    fn step_branch_b(&mut self) {
        if !FiniteStateGuard::is_finite_state(self) {
            return;
        }
        let noise_amp = self.noise_amplitude;
        if let Some((dt, auto_dt)) = self.sync_system_params() {
            self.advance_grid_step(dt, auto_dt, noise_amp);
            if let Some(ref mut twin) = self.twin {
                let ub = active_morphogen_slice(&self.system, self.kinetics);
                twin.observe_lockstep(self.step_count, dt, ub);
            }
        }
    }

    fn step(&mut self) {
        if !FiniteStateGuard::is_finite_state(self) {
            return;
        }
        let noise_amp = self.noise_amplitude;
        if let Some((dt, auto_dt)) = self.sync_system_params() {
            if let Some(ref mut twin) = self.twin {
                twin.step_branch_a(dt, auto_dt, self.width, self.height, noise_amp);
            }
            self.advance_grid_step(dt, auto_dt, noise_amp);
            if let Some(ref mut twin) = self.twin {
                let ub = active_morphogen_slice(&self.system, self.kinetics);
                twin.observe_lockstep(self.step_count, dt, ub);
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
        let u_data = active_morphogen_slice(&self.system, self.kinetics).to_vec();
        let u_a = self
            .twin
            .as_ref()
            .map(|t| active_morphogen_slice(&t.system_a, t.kinetics_a).to_vec());
        let mut image = render_morphogenesis_heatmap(
            &u_data,
            u_a.as_deref(),
            self.width,
            self.height,
            heatmap_mode,
            theme_mode,
        );
        let mask_roles = if self.system.diffusion.has_active_mask() {
            let roles = self.system.diffusion.mask_roles().to_vec();
            if heatmap_mode == MorphogenesisHeatmapMode::ActiveUb {
                overlay_cell_mask_colors(&mut image, &roles, theme_mode == ThemeMode::HighContrast);
            }
            roles
        } else {
            Vec::new()
        };
        let divergence = self.twin.as_ref().map(|t| t.snapshot(self.step_count, dt));
        StateSnapshot {
            width: self.width,
            height: self.height,
            pixels: Arc::new(std::sync::RwLock::new(image.pixels)),
            custom_data: Vec::new(),
            structured_data: Some(Box::new(MorphoSnapshotData {
                u_data,
                u_a,
                mask_roles,
                step_count: self.step_count,
                divergence,
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
        self.twin = None;
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

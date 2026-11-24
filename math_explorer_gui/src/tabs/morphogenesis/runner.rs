#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::advection::{MorphoWindTunnel, MorphogenesisAdvectionState};
use super::brush::{apply_mask_rle_to_kinetics, apply_morpho_cells_to_system, MorphoBrushMode};
pub use super::counterfactual::MorphogenesisSnapshotData as MorphoSnapshotData;
use super::counterfactual::{
    active_morphogen_slice, render_morphogenesis_heatmap, MorphogenesisHeatmapMode,
    MorphogenesisTwinState,
};
pub(crate) use super::kinetics::apply_uniform_noise;
pub use super::kinetics::{
    MorphogenesisKineticsAdapter, MorphogenesisKineticsState, MorphogenesisSharedParams,
};
#[cfg(test)]
pub use super::legacy_runner::MorphogenesisRunner;
use super::sculpt::{apply_sculpt_sim_command, overlay_cell_mask_colors};
use super::MorphogenesisTool;
use crate::app::ThemeMode;
use crate::async_sim::{
    CheckpointParams, CheckpointPayload, SimCommand, SimulationRunner, StateSnapshot,
    TimeTravelCheckpoint,
};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::presets::MorphogenesisKineticsMode;
use math_explorer::biology::diffusion::MaskedFiniteDifference2D;
use math_explorer::biology::morphogenesis::TuringSystem;
use oxidize_core::rng::OxidizeRng;
use std::sync::Arc;

/// Default PRNG seed for deterministic morphogenesis perturbation streams.
pub const DEFAULT_MORPHOGENESIS_SEED: u64 = 0x4D4F_5250_484F_3244;

/// Multi-kinetics worker runner supporting [`super::kinetics::MorphogenesisKineticsAdapter`],
/// [`MaskedFiniteDifference2D`] microfluidic field sculpting, `K = 16` zero-allocation
/// time-travel checkpoint replay, and counterfactual shadow twins.
pub struct MorphogenesisKineticsRunner {
    pub(crate) system: TuringSystem<2, MorphogenesisKineticsAdapter, MaskedFiniteDifference2D>,
    pub(crate) wind_tunnel: MorphoWindTunnel,
    pub(crate) params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
    pub(crate) kinetics_state: Arc<std::sync::RwLock<MorphogenesisKineticsState>>,
    pub(crate) advection_state: Arc<std::sync::RwLock<MorphogenesisAdvectionState>>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) step_count: usize,
    pub(crate) rng: OxidizeRng,
    pub(crate) noise_amplitude: f64,
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
        b_ok && a_ok && self.wind_tunnel.is_finite_state()
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        None
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }
}

impl MorphogenesisKineticsRunner {
    fn mode_to_scalar(mode: MorphogenesisKineticsMode) -> f64 {
        match mode {
            MorphogenesisKineticsMode::Schnakenberg => 0.0,
            MorphogenesisKineticsMode::GrayScott => 1.0,
        }
    }

    fn scalar_to_mode(tag: f64) -> MorphogenesisKineticsMode {
        if tag >= 0.5 {
            MorphogenesisKineticsMode::GrayScott
        } else {
            MorphogenesisKineticsMode::Schnakenberg
        }
    }

    pub(super) fn read_kinetics(&self) -> MorphogenesisKineticsState {
        *self
            .kinetics_state
            .read()
            .unwrap_or_else(|e| e.into_inner())
    }

    pub(super) fn sync_system_params(&mut self) -> Option<f64> {
        let (a, b, d_u, d_v, dt) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b, p.d_u, p.d_v, p.dt)
        };
        let ks = self.read_kinetics();
        self.apply_explicit_params(ks.mode, a, b, ks.f, ks.k, d_u, d_v, dt)
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
        if !a.is_finite()
            || !b.is_finite()
            || !f.is_finite()
            || !k.is_finite()
            || !d_u.is_finite()
            || !d_v.is_finite()
        {
            return None;
        }
        self.system.kinetics.mode = mode;
        self.system.kinetics.a = a;
        self.system.kinetics.b = b;
        self.system.kinetics.spatial.a = a;
        self.system.kinetics.spatial.b = b;
        self.system.kinetics.f = f;
        self.system.kinetics.k = k;
        self.system.diffusion_coeffs[0] = d_u;
        self.system.diffusion_coeffs[1] = d_v;
        if !self.has_counterfactual_twin()
            && super::gray_scott_atlas::should_reseed_washed_out_system(&self.system, d_u, d_v, dt)
        {
            MorphogenesisTool::initialize_adapter_system(&mut self.system, self.width, self.height);
            self.enforce_mask_concentrations();
        }
        (dt.is_finite() && dt > 0.0).then_some(dt)
    }

    fn reset_with_optional_kinetics_mask(&mut self, reload_kinetics_mask: bool) {
        let _ = self.sync_system_params();
        if reload_kinetics_mask {
            let mask_rle = self
                .params
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .mask_rle
                .clone();
            apply_mask_rle_to_kinetics(
                &mut self.system.kinetics,
                self.width,
                self.height,
                mask_rle.as_deref(),
            );
        }
        MorphogenesisTool::initialize_adapter_system(&mut self.system, self.width, self.height);
        self.enforce_mask_concentrations();
        let visc = self.read_advection().viscosity;
        self.wind_tunnel
            .reset_preserving_mask(&self.system.diffusion.mask, visc);
        self.step_count = 0;
        self.rng = OxidizeRng::new(DEFAULT_MORPHOGENESIS_SEED);
        self.twin = None;
    }

    fn apply_scalar_params(&mut self, p: &[f64]) -> Option<f64> {
        if p.len() < 9 {
            return None;
        }
        self.noise_amplitude = p[8];
        let mode = Self::scalar_to_mode(p[0]);
        self.apply_explicit_params(mode, p[1], p[2], p[3], p[4], p[5], p[6], p[7])
    }

    fn advance_grid_step(&mut self, dt: f64, noise_amp: f64) {
        self.enforce_mask_concentrations();
        self.step_coupled_or_isotropic(dt);
        apply_uniform_noise(
            &mut self.system,
            self.width * self.height,
            &mut self.rng,
            noise_amp,
        );
        self.enforce_mask_concentrations();
        self.step_count += 1;
    }

    pub(crate) fn apply_sculpt_command(&mut self, cmd: &SimCommand) {
        if apply_sculpt_sim_command(&mut self.system, self.width, self.height, cmd) {
            self.wind_tunnel
                .sync_from_cell_mask(&self.system.diffusion.mask);
            return;
        }
        match cmd {
            SimCommand::ApplyBrushCells { ref cells, mode } => {
                apply_morpho_cells_to_system(
                    &mut self.system,
                    self.width,
                    self.height,
                    cells,
                    MorphoBrushMode::from_code(*mode),
                );
                self.enforce_mask_concentrations();
            }
            SimCommand::Custom(s) if s == "clear_mask" => {
                self.system.diffusion.mask.clear();
                self.wind_tunnel
                    .sync_from_cell_mask(&self.system.diffusion.mask);
            }
            SimCommand::Custom(s) if s == "clear_kinetics_mask" => {
                self.system.kinetics.clear_mask();
            }
            SimCommand::Custom(s) if s == "stamp_maze" => {
                self.system.diffusion.mask.stamp_microfluidic_maze();
                self.enforce_mask_concentrations();
                self.wind_tunnel
                    .sync_from_cell_mask(&self.system.diffusion.mask);
            }
            _ => {}
        }
    }
}

impl SimulationRunner for MorphogenesisKineticsRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::Reset => self.reset_with_optional_kinetics_mask(true),
            SimCommand::Custom(ref s) if s == "reinit" => {
                self.reset_with_optional_kinetics_mask(false);
            }
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
            self.step_twin_branch_a(dt, noise_amp);
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
            self.step_twin_branch_a(dt, noise_amp);
            self.advance_grid_step(dt, noise_amp);
            if let Some(ref mut twin) = self.twin {
                twin.observe_lockstep(self.step_count, dt, active_morphogen_slice(&self.system));
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
        let u_data = active_morphogen_slice(&self.system).to_vec();
        let u_a = self
            .twin
            .as_ref()
            .map(|t| active_morphogen_slice(&t.system_a).to_vec());
        let mut image = render_morphogenesis_heatmap(
            &u_data,
            u_a.as_deref(),
            self.width,
            self.height,
            heatmap_mode,
            theme_mode,
        );
        let mask_roles = if self.system.diffusion.mask.has_active_cells() {
            let roles = self.system.diffusion.mask.roles().to_vec();
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
                max_abs_ux: self.wind_tunnel.max_abs_ux,
                max_abs_uy: self.wind_tunnel.max_abs_uy,
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
        match self.read_kinetics().mode {
            MorphogenesisKineticsMode::Schnakenberg => 1,
            MorphogenesisKineticsMode::GrayScott => 4,
        }
    }

    fn checkpoint_stride(&self) -> Option<usize> {
        (!self.read_advection().enabled).then_some(16)
    }

    fn current_step(&self) -> usize {
        self.step_count
    }

    fn capture_active_params(&self) -> Option<CheckpointParams> {
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        let ks = self.read_kinetics();
        Some(CheckpointParams::scalars(&[
            Self::mode_to_scalar(ks.mode),
            p.a,
            p.b,
            ks.f,
            ks.k,
            p.d_u,
            p.d_v,
            p.dt,
            self.noise_amplitude,
        ]))
    }

    fn capture_checkpoint(&self) -> Option<TimeTravelCheckpoint> {
        let params = self.capture_active_params()?;
        let mask = self
            .system
            .diffusion
            .mask
            .has_active_cells()
            .then(|| self.system.diffusion.mask.roles().into());
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
                        let _ = self.system.diffusion.mask.set_roles_slice(bytes);
                    }
                    _ => self.system.diffusion.mask.clear(),
                }
                self.wind_tunnel
                    .sync_from_cell_mask(&self.system.diffusion.mask);
            }
        }
        if let Some(p) = checkpoint.params.as_scalars() {
            let _ = self.apply_scalar_params(p);
        }
        self.step_count = checkpoint.step;
        self.rng = checkpoint.rng.clone();
        self.twin = None;
    }

    fn replay_step(&mut self, params: &CheckpointParams) {
        if let Some(dt) = params
            .as_scalars()
            .and_then(|p| self.apply_scalar_params(p))
        {
            self.advance_grid_step(dt, self.noise_amplitude);
        }
    }
}

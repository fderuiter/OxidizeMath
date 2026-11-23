#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub use super::counterfactual::MorphogenesisSnapshotData as MorphoSnapshotData;
use super::counterfactual::{
    active_morphogen_slice, render_morphogenesis_heatmap, MorphogenesisHeatmapMode,
    MorphogenesisTwinState,
};
#[cfg(test)]
pub use super::legacy_runner::MorphogenesisRunner;
pub(crate) use super::runner_twin::apply_uniform_noise;
use super::sculpt::{apply_sculpt_sim_command, enforce_system_mask, overlay_cell_mask_colors};
use super::MorphogenesisTool;
use crate::app::ThemeMode;
use crate::async_sim::{
    CheckpointParams, CheckpointPayload, SimCommand, SimulationRunner, StateSnapshot,
    TimeTravelCheckpoint,
};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::presets::MorphogenesisKineticsMode;
use math_explorer::biology::diffusion::{BoundaryMode, MaskedFiniteDifference2D};
use math_explorer::biology::morphogenesis::{
    GrayScottKinetics, ReactionKinetics, SchnakenbergKinetics, TuringSystem,
};
use oxidize_core::rng::OxidizeRng;
use std::sync::Arc;

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

/// Shared reaction-diffusion parameters synchronized between the UI thread and worker runner.
pub struct MorphogenesisSharedParams {
    /// Schnakenberg activator feed rate `a`.
    pub a: f64,
    /// Schnakenberg inhibitor feed rate `b`.
    pub b: f64,
    /// Primary species diffusion coefficient `D_u`.
    pub d_u: f64,
    /// Secondary species diffusion coefficient `D_v`.
    pub d_v: f64,
    /// Integration step size `dt`.
    pub dt: f64,
    /// Whether high-contrast palette rendering is active.
    pub high_contrast: bool,
    /// Whether toroidal periodic boundaries are active.
    pub periodic_boundary: bool,
    /// Active visual theme mode (`Dark`, `Light`, or `HighContrast`).
    pub theme_mode: ThemeMode,
    /// Active counterfactual spatial heatmap mode (`ActiveUb`, `SplitUaUb`, or `DifferenceAbs`).
    pub heatmap_mode: MorphogenesisHeatmapMode,
}

/// Multi-kinetics worker runner supporting [`SchnakenbergKinetics`], [`GrayScottKinetics`],
/// [`MaskedFiniteDifference2D`] microfluidic field sculpting, `K = 16` zero-allocation
/// time-travel checkpoint replay, and counterfactual shadow twins.
pub struct MorphogenesisKineticsRunner {
    pub(crate) system: TuringSystem<2, MorphogenesisKineticsAdapter, MaskedFiniteDifference2D>,
    pub(crate) params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
    pub(crate) kinetics_state: Arc<std::sync::RwLock<MorphogenesisKineticsState>>,
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
    fn mode_to_scalar(mode: MorphogenesisKineticsMode) -> f64 {
        f64::from(u8::from(mode == MorphogenesisKineticsMode::GrayScott))
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

    pub(crate) fn enforce_mask_concentrations(&mut self) {
        enforce_system_mask(&mut self.system, self.width, self.height);
    }

    pub(super) fn sync_system_params(&mut self) -> Option<f64> {
        let (a, b, d_u, d_v, dt, periodic) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b, p.d_u, p.d_v, p.dt, p.periodic_boundary)
        };
        let ks = self.read_kinetics();
        self.apply_explicit_params(ks.mode, a, b, ks.f, ks.k, d_u, d_v, dt, periodic)
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
        periodic: bool,
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
        self.system.kinetics.f = f;
        self.system.kinetics.k = k;
        self.system.diffusion_coeffs[0] = d_u;
        self.system.diffusion_coeffs[1] = d_v;
        self.system.diffusion.base.boundary_mode = if periodic {
            BoundaryMode::Periodic
        } else {
            BoundaryMode::Neumann
        };
        if !self.has_counterfactual_twin()
            && super::gray_scott_atlas::should_reseed_washed_out_system(&self.system, d_u, d_v, dt)
        {
            MorphogenesisTool::initialize_adapter_system(&mut self.system, self.width, self.height);
            self.enforce_mask_concentrations();
        }
        (dt.is_finite() && dt > 0.0).then_some(dt)
    }

    fn apply_scalar_params(&mut self, p: &[f64]) -> Option<f64> {
        if p.len() < 9 {
            return None;
        }
        self.noise_amplitude = p[8];
        let mode = Self::scalar_to_mode(p[0]);
        let periodic = if p.len() >= 10 {
            p[9] >= 0.5
        } else {
            self.params
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .periodic_boundary
        };
        self.apply_explicit_params(mode, p[1], p[2], p[3], p[4], p[5], p[6], p[7], periodic)
    }

    fn advance_grid_step(&mut self, dt: f64, noise_amp: f64) {
        self.enforce_mask_concentrations();
        self.system.step(dt);
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
        let _ = self.sync_system_params();
        MorphogenesisTool::initialize_adapter_system(&mut self.system, self.width, self.height);
        self.enforce_mask_concentrations();
        self.step_count = 0;
        self.rng = OxidizeRng::new(DEFAULT_MORPHOGENESIS_SEED);
        self.twin = None;
    }

    fn apply_sculpt_command(&mut self, cmd: &SimCommand) {
        if apply_sculpt_sim_command(&mut self.system, self.width, self.height, cmd) {
            return;
        }
        match cmd {
            SimCommand::Custom(s) if s == "clear_mask" => self.system.diffusion.mask.clear(),
            SimCommand::Custom(s) if s == "stamp_maze" => {
                self.system.diffusion.mask.stamp_microfluidic_maze();
                self.enforce_mask_concentrations();
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
        Some(16)
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
            f64::from(u8::from(p.periodic_boundary)),
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

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::{grid_u_variance, is_pattern_forming_regime, WASHOUT_VARIANCE_THRESHOLD};
use super::MorphogenesisTool;
use crate::async_sim::{
    CheckpointParams, CheckpointPayload, SimCommand, SimulationRunner, StateSnapshot,
    TimeTravelCheckpoint,
};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::presets::MorphogenesisKineticsMode;
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::{
    GrayScottKinetics, ReactionKinetics, SchnakenbergKinetics, TuringSystem,
};
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;
use std::sync::Arc;

/// Default PRNG seed for deterministic morphogenesis perturbation streams.
pub const DEFAULT_MORPHOGENESIS_SEED: u64 = 0x4D4F_5250_484F_3244;

fn build_field_snapshot(
    field_vec: Vec<f64>,
    width: usize,
    height: usize,
    high_contrast: bool,
) -> StateSnapshot {
    let image = MorphogenesisTool::plot_concentration_with_contrast(
        &field_vec,
        width,
        height,
        high_contrast,
    );
    StateSnapshot {
        width,
        height,
        pixels: Arc::new(std::sync::RwLock::new(image.pixels)),
        custom_data: Vec::new(),
        structured_data: Some(Box::new(field_vec)),
        sonification: crate::audio::SonificationFrame::silent(),
    }
}

fn apply_uniform_noise<K: ReactionKinetics<2>>(
    system: &mut TuringSystem<2, K, FiniteDifference2D>,
    count: usize,
    rng: &mut OxidizeRng,
    amp: f64,
) {
    if !amp.is_finite() || amp <= 0.0 {
        return;
    }
    const INV_2_POW_53: f64 = 1.0 / ((1u64 << 53) as f64);
    for idx in 0..count {
        let u_unit = ((rng.next_u64() >> 11) as f64) * INV_2_POW_53;
        let v_unit = ((rng.next_u64() >> 11) as f64) * INV_2_POW_53;
        system.u_mut()[idx] += (2.0 * u_unit - 1.0) * amp;
        system.v_mut()[idx] += (2.0 * v_unit - 1.0) * amp;
    }
}

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

#[allow(missing_docs)]
pub struct MorphogenesisSharedParams {
    pub a: f64,
    pub b: f64,
    pub d_u: f64,
    pub d_v: f64,
    pub dt: f64,
    pub high_contrast: bool,
}

#[cfg(test)]
#[allow(missing_docs)]
pub struct MorphogenesisRunner {
    pub(crate) system: TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
    pub(crate) params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) step_count: usize,
    pub(crate) rng: OxidizeRng,
    pub(crate) noise_amplitude: f64,
}

#[cfg(test)]
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

#[cfg(test)]
impl MorphogenesisRunner {
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

#[cfg(test)]
impl SimulationRunner for MorphogenesisRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        if matches!(cmd, SimCommand::Reset)
            || matches!(cmd, SimCommand::Custom(ref s) if s == "reinit")
        {
            let _ = self.sync_system_params();
            MorphogenesisTool::initialize_schnakenberg(&mut self.system, self.width, self.height);
            self.step_count = 0;
            self.rng = OxidizeRng::new(DEFAULT_MORPHOGENESIS_SEED);
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
                "Turing morphogenesis diverged: non-finite concentration encountered.",
            )
        })
    }

    fn step(&mut self) {
        if !FiniteStateGuard::is_finite_state(self) {
            return;
        }
        let amp = self.noise_amplitude;
        if let Some(dt) = self.sync_system_params() {
            self.system.step(dt);
            apply_uniform_noise(
                &mut self.system,
                self.width * self.height,
                &mut self.rng,
                amp,
            );
            self.step_count += 1;
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let hc = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .high_contrast;
        build_field_snapshot(self.system.u().to_vec(), self.width, self.height, hc)
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }
}

/// Multi-kinetics worker runner supporting both [`SchnakenbergKinetics`] and [`GrayScottKinetics`]
/// with `K = 16` zero-allocation time-travel checkpoint replay.
pub struct MorphogenesisKineticsRunner {
    pub(crate) system: TuringSystem<2, MorphogenesisKineticsAdapter, FiniteDifference2D>,
    pub(crate) params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
    pub(crate) kinetics_state: Arc<std::sync::RwLock<MorphogenesisKineticsState>>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) step_count: usize,
    pub(crate) rng: OxidizeRng,
    pub(crate) noise_amplitude: f64,
}

impl FiniteStateGuard for MorphogenesisKineticsRunner {
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

    fn read_kinetics(&self) -> MorphogenesisKineticsState {
        *self
            .kinetics_state
            .read()
            .unwrap_or_else(|e| e.into_inner())
    }

    fn sync_system_params(&mut self) -> Option<f64> {
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
        self.system.kinetics.f = f;
        self.system.kinetics.k = k;
        self.system.diffusion_coeffs[0] = d_u;
        self.system.diffusion_coeffs[1] = d_v;
        if mode == MorphogenesisKineticsMode::Schnakenberg
            && dt.is_finite()
            && dt > 0.0
            && is_pattern_forming_regime(a, b, d_u, d_v, dt)
        {
            let var_u = grid_u_variance(self.system.u());
            if var_u.is_finite() && var_u < WASHOUT_VARIANCE_THRESHOLD {
                MorphogenesisTool::initialize_adapter_system(
                    &mut self.system,
                    self.width,
                    self.height,
                );
            }
        }
        (dt.is_finite() && dt > 0.0).then_some(dt)
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
        self.system.step(dt);
        apply_uniform_noise(
            &mut self.system,
            self.width * self.height,
            &mut self.rng,
            noise_amp,
        );
        self.step_count += 1;
    }

    fn reset_grid_and_counters(&mut self) {
        let _ = self.sync_system_params();
        MorphogenesisTool::initialize_adapter_system(&mut self.system, self.width, self.height);
        self.step_count = 0;
        self.rng = OxidizeRng::new(DEFAULT_MORPHOGENESIS_SEED);
    }
}

impl SimulationRunner for MorphogenesisKineticsRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        if matches!(cmd, SimCommand::Reset)
            || matches!(cmd, SimCommand::Custom(ref s) if s == "reinit")
        {
            self.reset_grid_and_counters();
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

    fn step(&mut self) {
        if !FiniteStateGuard::is_finite_state(self) {
            return;
        }
        let noise_amp = self.noise_amplitude;
        if let Some(dt) = self.sync_system_params() {
            self.advance_grid_step(dt, noise_amp);
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let hc = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .high_contrast;
        let field_vec = match self.system.kinetics.mode {
            MorphogenesisKineticsMode::Schnakenberg => self.system.u().to_vec(),
            MorphogenesisKineticsMode::GrayScott => self.system.v().to_vec(),
        };
        build_field_snapshot(field_vec, self.width, self.height, hc)
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
        ]))
    }

    fn capture_checkpoint(&self) -> Option<TimeTravelCheckpoint> {
        let params = self.capture_active_params()?;
        Some(TimeTravelCheckpoint {
            step: self.step_count,
            rng: self.rng.clone(),
            params,
            state: CheckpointPayload::DualField2D {
                width: self.width,
                height: self.height,
                u: self.system.u().into(),
                v: self.system.v().into(),
            },
        })
    }

    fn restore_checkpoint(&mut self, checkpoint: &TimeTravelCheckpoint) {
        if let CheckpointPayload::DualField2D {
            width,
            height,
            ref u,
            ref v,
        } = checkpoint.state
        {
            if width == self.width
                && height == self.height
                && u.len() == self.system.u().len()
                && v.len() == self.system.v().len()
            {
                self.system.u_mut().copy_from_slice(u);
                self.system.v_mut().copy_from_slice(v);
            }
        }
        if let Some(p) = checkpoint.params.as_scalars() {
            let _ = self.apply_scalar_params(p);
        }
        self.step_count = checkpoint.step;
        self.rng = checkpoint.rng.clone();
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

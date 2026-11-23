#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::MorphogenesisTool;
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use math_explorer::biology::lenia::{LeniaKernelCore, LeniaPreset, LeniaSystem, LeniaTelemetry};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Serializable organism preset choice for [`super::lenia_tool::LeniaTool`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum LeniaPresetChoice {
    /// `Orbium unicaudatus` single-ring smooth glider.
    #[default]
    Orbium,
    /// `Hydrogeminium natans` three-ring self-dividing organism.
    Hydrogeminium,
}

impl LeniaPresetChoice {
    pub(crate) fn to_domain(self) -> LeniaPreset {
        match self {
            Self::Orbium => LeniaPreset::Orbium,
            Self::Hydrogeminium => LeniaPreset::Hydrogeminium,
        }
    }
}

/// Shared worker parameters for [`LeniaRunner`].
pub struct LeniaSharedParams {
    pub(crate) mu: f64,
    pub(crate) sigma: f64,
    pub(crate) dt: f64,
    pub(crate) kernel_radius: usize,
    pub(crate) preset: LeniaPresetChoice,
    pub(crate) high_contrast: bool,
}

/// Snapshot payload bundling cell activations and live telemetry.
#[derive(Clone, Debug)]
pub struct LeniaSnapshotPayload {
    /// Flattened cell state activations $A(x, y) \in [0, 1]$.
    pub grid: Vec<f64>,
    /// Live toroidal kinematic and symmetry invariants.
    pub telemetry: LeniaTelemetry,
}

/// Background worker runner executing zero-allocation spectral [`LeniaSystem`] steps.
pub struct LeniaRunner {
    pub(crate) system: LeniaSystem,
    pub(crate) params: Arc<std::sync::RwLock<LeniaSharedParams>>,
    pub(crate) active_preset: LeniaPresetChoice,
}

impl LeniaRunner {
    pub(crate) fn new(
        params: &Arc<std::sync::RwLock<LeniaSharedParams>>,
        width: usize,
        height: usize,
    ) -> Self {
        let (mu, sigma, dt, kernel_radius, preset) = {
            let p = params.read().unwrap_or_else(|e| e.into_inner());
            (p.mu, p.sigma, p.dt, p.kernel_radius, p.preset)
        };
        let mut domain_params = preset.to_domain().params();
        domain_params.mu = mu;
        domain_params.sigma = sigma;
        domain_params.dt = dt;
        domain_params.kernel_radius = kernel_radius.min(width.min(height) / 2 - 1).max(2);
        domain_params.kernel_core = LeniaKernelCore::ExponentialBump;

        let mut system =
            LeniaSystem::new(width, height, domain_params.clone()).unwrap_or_else(|_| {
                LeniaSystem::from_preset(width, height, preset.to_domain()).unwrap()
            });
        preset.to_domain().stamp_into_grid(
            system.grid_mut(),
            width,
            height,
            domain_params.kernel_radius,
            width / 2,
            height / 2,
        );
        system.recompute_telemetry();

        Self {
            system,
            params: Arc::clone(params),
            active_preset: preset,
        }
    }

    fn sync_params(&mut self) -> bool {
        let (mu, sigma, dt, kernel_radius, preset) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.mu, p.sigma, p.dt, p.kernel_radius, p.preset)
        };
        let max_r = (self.system.width().min(self.system.height()) / 2)
            .saturating_sub(1)
            .max(2);
        let clamped_r = kernel_radius.clamp(2, max_r);

        if preset != self.active_preset {
            self.active_preset = preset;
            self.system.apply_preset(preset.to_domain());
        }
        self.system
            .update_scalar_params(mu, sigma, dt, clamped_r)
            .is_ok()
    }
}

impl FiniteStateGuard for LeniaRunner {
    fn is_finite_state(&self) -> bool {
        self.system.grid().iter().all(|&v| v.is_finite())
            && self.system.telemetry().total_mass.is_finite()
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        None
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }
}

impl SimulationRunner for LeniaRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        if matches!(cmd, SimCommand::Reset) {
            let preset = self.params.read().unwrap_or_else(|e| e.into_inner()).preset;
            self.active_preset = preset;
            self.system.apply_preset(preset.to_domain());
            let _ = self.sync_params();
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
                "activation",
                "Lenia spectral solver encountered non-finite activation values.",
            ))
        } else {
            None
        }
    }

    fn step(&mut self) {
        if !FiniteStateGuard::is_finite_state(self) {
            return;
        }
        if self.sync_params() {
            self.system.step();
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let high_contrast = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .high_contrast;
        let w = self.system.width();
        let h = self.system.height();
        let grid = self.system.grid().to_vec();
        let image = MorphogenesisTool::plot_concentration_with_contrast(&grid, w, h, high_contrast);
        let payload = LeniaSnapshotPayload {
            grid,
            telemetry: self.system.telemetry(),
        };
        StateSnapshot {
            width: w,
            height: h,
            pixels: Arc::new(std::sync::RwLock::new(image.pixels)),
            custom_data: Vec::new(),
            structured_data: Some(Box::new(payload)),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        2
    }
}

//! Interactive 2D Model A (Allen-Cahn) and Model B (Cahn-Hilliard) spinodal decomposition studio.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::hash::Hash;

use eframe::egui;
use math_explorer::physics::stat_mech::phase_field::{
    PhaseFieldModel, PhaseFieldParams, PhaseFieldSolver2D,
};
use oxidize_core::rng::OxidizeRng;
use serde::{Deserialize, Serialize};

use super::presets::{default_brush_radius, SpinodalTheory, SpinodalToolState};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};

const COOPERATIVE_BUDGET_SECS: f64 = 0.008;

/// Serializable Hohenberg-Halperin dynamics selector for [`SpinodalDecompositionTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PhaseFieldModelChoice {
    /// Non-conserved Ginzburg-Landau relaxation (Model A / Allen-Cahn).
    AllenCahn,
    /// Conserved diffusive phase separation (Model B / Cahn-Hilliard).
    #[default]
    CahnHilliard,
}

impl PhaseFieldModelChoice {
    /// Converts to the `domain_physics` [`PhaseFieldModel`].
    pub const fn to_domain(self) -> PhaseFieldModel {
        match self {
            Self::AllenCahn => PhaseFieldModel::AllenCahn,
            Self::CahnHilliard => PhaseFieldModel::CahnHilliard,
        }
    }

    /// Human-readable UI label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::AllenCahn => "Model A (Allen-Cahn)",
            Self::CahnHilliard => "Model B (Cahn-Hilliard)",
        }
    }
}

/// Telemetry plot mode for [`SpinodalDecompositionTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpinodalPlotMode {
    /// Instantaneous circularly averaged radial structure factor $S(k, t)$ and peak $k_{\max}(t)$.
    #[default]
    StructureFactor,
    /// Time evolution of characteristic coarsening wavenumber $k_{\max}(t)$ and $k_1(t)$.
    PeakWavenumber,
}

impl SpinodalPlotMode {
    /// Human-readable UI label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::StructureFactor => "Radial S(k, t)",
            Self::PeakWavenumber => "Coarsening k_max(t)",
        }
    }
}

/// Interactive droplet brush polarity on the 2D order-parameter canvas.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpinodalBrushAction {
    /// Enriches the $+\phi_{\text{eq}}$ phase (`delta > 0`).
    #[default]
    EnrichPositive,
    /// Enriches the $-\phi_{\text{eq}}$ phase (`delta < 0`).
    EnrichNegative,
}

/// Interactive 2D stochastic Allen-Cahn and Cahn-Hilliard phase-field playground.
pub struct SpinodalDecompositionTool {
    /// Selected dynamics (`AllenCahn` vs `CahnHilliard`).
    pub model: PhaseFieldModelChoice,
    /// Bath quench temperature $T$.
    pub temperature: f64,
    /// Critical spinodal temperature $T_c$.
    pub critical_temperature: f64,
    /// Kinetic mobility $M$.
    pub mobility: f64,
    /// Interfacial gradient penalty $\kappa$.
    pub kappa: f64,
    /// Thermal fluctuation strength $\sigma$.
    pub noise_sigma: f64,
    /// Initial spatial mean concentration $\bar{\phi}$.
    pub mean_phi: f64,
    /// Semi-implicit time step $\Delta t$.
    pub dt: f64,
    /// Periodic square grid side length (`32..=128`).
    pub grid_size: usize,
    /// Substeps executed per animation frame.
    pub steps_per_frame: usize,
    /// Deterministic `OxidizeRng` seed.
    pub rng_seed: u64,
    /// Active spectral plot view.
    pub plot_mode: SpinodalPlotMode,
    /// Active canvas brush polarity.
    pub brush_action: SpinodalBrushAction,
    /// Canvas droplet radius in cells.
    pub brush_radius: usize,
    /// Whether continuous simulation playback is running.
    pub playing: bool,
    pub(crate) solver: PhaseFieldSolver2D,
    pub(crate) rng: OxidizeRng,
    pub(crate) cursor: [usize; 2],
    pub(crate) focused_canvas: Option<egui::Id>,
    pub(crate) pixels: Vec<egui::Color32>,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) texture_dirty: bool,
    pub(crate) high_contrast: bool,
    pub(crate) last_step_ms: f64,
    pub(crate) diagnostic: Option<DiagnosticReport>,
    pub(crate) theory: SpinodalTheory,
}

impl Default for SpinodalDecompositionTool {
    fn default() -> Self {
        let grid_size = 64;
        let rng_seed = 42;
        let mut rng = OxidizeRng::from_seed(rng_seed);
        let params = PhaseFieldParams {
            model: PhaseFieldModel::CahnHilliard,
            temperature: 0.25,
            critical_temperature: 1.0,
            mobility: 1.0,
            kappa: 0.5,
            noise_sigma: 0.04,
            dx: 1.0,
            dt: 0.05,
        };
        let solver =
            PhaseFieldSolver2D::new_spinodal(grid_size, grid_size, 0.0, 0.12, &mut rng, params);
        let mut tool = Self {
            model: PhaseFieldModelChoice::CahnHilliard,
            temperature: params.temperature,
            critical_temperature: params.critical_temperature,
            mobility: params.mobility,
            kappa: params.kappa,
            noise_sigma: params.noise_sigma,
            mean_phi: 0.0,
            dt: params.dt,
            grid_size,
            steps_per_frame: 3,
            rng_seed,
            plot_mode: SpinodalPlotMode::StructureFactor,
            brush_action: SpinodalBrushAction::EnrichPositive,
            brush_radius: default_brush_radius(),
            playing: false,
            solver,
            rng,
            cursor: [grid_size / 2, grid_size / 2],
            focused_canvas: None,
            pixels: vec![egui::Color32::BLACK; grid_size * grid_size],
            texture: None,
            texture_dirty: true,
            high_contrast: false,
            last_step_ms: 0.0,
            diagnostic: None,
            theory: SpinodalTheory,
        };
        tool.solver.step_n(&mut tool.rng, 2);
        let _ = tool.solver.refresh_telemetry();
        tool.refresh_pixels_in_place();
        tool
    }
}

impl SpinodalDecompositionTool {
    pub(crate) fn current_domain_params(&self) -> PhaseFieldParams {
        PhaseFieldParams {
            model: self.model.to_domain(),
            temperature: self.temperature,
            critical_temperature: self.critical_temperature,
            mobility: self.mobility,
            kappa: self.kappa,
            noise_sigma: self.noise_sigma,
            dx: 1.0,
            dt: self.dt,
        }
    }

    /// Rebuilds/reseeds the 2D quenched spinodal field from `rng_seed` and refreshes $S(k, t)$.
    pub fn reseed_spinodal_state(&mut self) {
        if !self.parameters_within_safe_bounds() {
            return;
        }
        let params = self.current_domain_params();
        self.rng = OxidizeRng::from_seed(self.rng_seed);
        self.diagnostic = None;
        if self.solver.width() != self.grid_size || self.solver.height() != self.grid_size {
            self.solver = PhaseFieldSolver2D::new_spinodal(
                self.grid_size,
                self.grid_size,
                self.mean_phi,
                0.12,
                &mut self.rng,
                params,
            );
            self.cursor = [self.grid_size / 2, self.grid_size / 2];
        } else {
            self.solver.set_params(params);
            self.solver
                .reseed_spinodal(self.mean_phi, 0.12, &mut self.rng);
        }
        self.solver.step_n(&mut self.rng, 2);
        let _ = self.solver.refresh_telemetry();
        self.refresh_pixels_in_place();
    }

    pub(crate) fn sync_solver_params(&mut self) {
        if self.parameters_within_safe_bounds() {
            self.solver.set_params(self.current_domain_params());
        }
    }

    pub(crate) fn parameters_within_safe_bounds(&self) -> bool {
        self.temperature.is_finite()
            && (0.0..=2.0).contains(&self.temperature)
            && self.critical_temperature.is_finite()
            && (0.2..=2.5).contains(&self.critical_temperature)
            && self.mobility.is_finite()
            && (0.05..=5.0).contains(&self.mobility)
            && self.kappa.is_finite()
            && (0.05..=4.0).contains(&self.kappa)
            && self.noise_sigma.is_finite()
            && (0.0..=0.50).contains(&self.noise_sigma)
            && self.mean_phi.is_finite()
            && (-0.65..=0.65).contains(&self.mean_phi)
            && self.dt.is_finite()
            && (0.005..=0.25).contains(&self.dt)
            && (32..=128).contains(&self.grid_size)
            && (1..=12).contains(&self.steps_per_frame)
            && (1..=16).contains(&self.brush_radius)
    }

    fn check_and_latch_instability(&mut self) -> bool {
        let field_ok = self.solver.field().iter().all(|v| v.is_finite());
        if !self.parameters_within_safe_bounds() || !field_ok {
            self.playing = false;
            self.diagnostic = Some(DiagnosticReport::new(
                self.solver.time(),
                self.solver.steps() as usize,
                "spinodal_phase_field",
                "Out-of-bounds parameter or non-finite order parameter detected; halted by FiniteStateGuard.",
            ));
            return false;
        }
        true
    }

    /// Advances the 2D phase field by one guarded semi-implicit step and refreshes $S(k, t)$.
    pub fn step_once(&mut self) {
        if !self.check_and_latch_instability() {
            return;
        }
        self.sync_solver_params();
        let t0 = crate::web_io::PlatformInstant::now();
        self.solver.step(&mut self.rng);
        let _ = self.solver.refresh_telemetry();
        self.last_step_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if self.check_and_latch_instability() {
            self.refresh_pixels_in_place();
        }
    }

    pub(crate) fn step_frame_cooperative(&mut self) {
        let start = crate::web_io::PlatformInstant::now();
        let n = self.steps_per_frame.clamp(1, 12);
        if !self.check_and_latch_instability() {
            return;
        }
        self.sync_solver_params();
        for _ in 0..n {
            let t0 = crate::web_io::PlatformInstant::now();
            self.solver.step(&mut self.rng);
            self.last_step_ms = t0.elapsed().as_secs_f64() * 1000.0;
            if start.elapsed().as_secs_f64() >= COOPERATIVE_BUDGET_SECS {
                break;
            }
        }
        if self.check_and_latch_instability() {
            let _ = self.solver.refresh_telemetry();
            self.refresh_pixels_in_place();
        }
    }

    pub(crate) fn snapshot_state(&self) -> SpinodalToolState {
        SpinodalToolState {
            model: self.model,
            temperature: self.temperature,
            critical_temperature: self.critical_temperature,
            mobility: self.mobility,
            kappa: self.kappa,
            noise_sigma: self.noise_sigma,
            mean_phi: self.mean_phi,
            dt: self.dt,
            grid_size: self.grid_size,
            steps_per_frame: self.steps_per_frame,
            rng_seed: self.rng_seed,
            plot_mode: self.plot_mode,
            brush_action: self.brush_action,
            brush_radius: self.brush_radius,
            playing: self.playing,
        }
    }
}

impl FiniteStateGuard for SpinodalDecompositionTool {
    fn is_finite_state(&self) -> bool {
        self.diagnostic.is_none()
            && self.parameters_within_safe_bounds()
            && self.solver.field().iter().all(|v| v.is_finite())
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.diagnostic.as_ref()
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }

    fn auto_scale_step_size(&mut self) {
        let next_dt = if self.dt.is_finite() && self.dt > 0.0 {
            self.dt * 0.5
        } else {
            0.05
        };
        self.dt = next_dt.clamp(0.005, 0.12);
        FiniteStateGuard::clamp_parameters(self);
    }

    fn clamp_parameters(&mut self) {
        self.temperature = sanitize_f64(self.temperature, 0.0, 2.0, 0.25);
        self.critical_temperature = sanitize_f64(self.critical_temperature, 0.2, 2.5, 1.0);
        self.mobility = sanitize_f64(self.mobility, 0.05, 5.0, 1.0);
        self.kappa = sanitize_f64(self.kappa, 0.05, 4.0, 0.5);
        self.noise_sigma = sanitize_f64(self.noise_sigma, 0.0, 0.50, 0.04);
        self.mean_phi = sanitize_f64(self.mean_phi, -0.65, 0.65, 0.0);
        self.dt = sanitize_f64(self.dt, 0.005, 0.25, 0.05);
        self.grid_size = self.grid_size.clamp(32, 128);
        self.steps_per_frame = self.steps_per_frame.clamp(1, 12);
        self.brush_radius = self.brush_radius.clamp(1, 16);

        if !self.solver.field().iter().all(|v| v.is_finite()) {
            self.reseed_spinodal_state();
        } else {
            self.diagnostic = None;
            self.sync_solver_params();
            self.refresh_pixels_in_place();
        }
    }

    fn reset_parameters(&mut self) {
        *self = Self::default();
    }
}

fn sanitize_f64(val: f64, min: f64, max: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val.clamp(min, max)
    } else {
        fallback
    }
}

//! Interactive 1D and 2D Edwards-Wilkinson and KPZ surface growth playground (`KpzInterfaceTool`).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::hash::Hash;

use eframe::egui;
use math_explorer::physics::stat_mech::interface_growth::{
    InterfaceGrowth1D, InterfaceGrowth2D, InterfaceGrowthKind, InterfaceGrowthParams,
    InterfaceScalingTelemetry,
};
use oxidize_core::rng::OxidizeRng;
use serde::{Deserialize, Serialize};

use super::presets::{default_brush_radius, KpzTheory, KpzToolState};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};

const COOPERATIVE_BUDGET_SECS: f64 = 0.008;

/// Spatial lattice dimensionality for [`KpzInterfaceTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KpzDimensionMode {
    /// 1+1D periodic interface profile $h(x, t)$.
    #[default]
    Dim1D,
    /// 2+1D periodic heightfield heatmap $h(x, y, t)$.
    Dim2D,
}

impl KpzDimensionMode {
    /// Human-readable UI label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Dim1D => "1D Profile",
            Self::Dim2D => "2D Heightfield",
        }
    }
}

/// Scaling plot display mode for [`KpzInterfaceTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KpzPlotView {
    /// Log-log plot of $\ln W(L, t)$ vs $\ln t$ with OLS $\beta$ slope fit and EW/KPZ guides.
    #[default]
    RawLogLog,
    /// Family-Vicsek dynamic scaling collapse $W(L, t) / L^\alpha$ vs $t / L^z$.
    FamilyVicsek,
}

impl KpzPlotView {
    /// Human-readable UI label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::RawLogLog => "Log-Log W(L, t)",
            Self::FamilyVicsek => "Family-Vicsek Collapse",
        }
    }
}

/// Interactive 1D & 2D Edwards-Wilkinson and KPZ interface roughening studio.
pub struct KpzInterfaceTool {
    /// Active spatial dimension (`Dim1D` or `Dim2D`).
    pub dimension: KpzDimensionMode,
    /// Surface tension $\nu$.
    pub nu: f64,
    /// Nonlinear KPZ coupling $\lambda$ (`0.0` = Edwards-Wilkinson).
    pub lambda: f64,
    /// White-noise strength $\sigma$.
    pub noise_sigma: f64,
    /// Spatial lattice step $\Delta x$.
    pub dx: f64,
    /// Time step $\Delta t$.
    pub dt: f64,
    /// Lattice side length $L$.
    pub system_size: usize,
    /// Substeps executed per animation frame.
    pub steps_per_frame: usize,
    /// Deterministic `OxidizeRng` seed.
    pub rng_seed: u64,
    /// Selected scaling telemetry chart mode.
    pub plot_view: KpzPlotView,
    /// Interactive mound deposition radius.
    pub brush_radius: usize,
    /// Whether continuous simulation playback is running.
    pub playing: bool,
    pub(crate) solver_1d: InterfaceGrowth1D,
    pub(crate) solver_2d: InterfaceGrowth2D,
    pub(crate) telemetry: InterfaceScalingTelemetry,
    pub(crate) rng: OxidizeRng,
    pub(crate) cursor: [usize; 2],
    pub(crate) focused_canvas: Option<egui::Id>,
    pub(crate) pixels: Vec<egui::Color32>,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) texture_dirty: bool,
    pub(crate) high_contrast: bool,
    pub(crate) last_step_ms: f64,
    pub(crate) diagnostic: Option<DiagnosticReport>,
    pub(crate) theory: KpzTheory,
}

impl Default for KpzInterfaceTool {
    fn default() -> Self {
        let params = InterfaceGrowthParams {
            nu: 0.5,
            lambda: 3.2,
            noise_sigma: 1.0,
            dx: 1.0,
            dt: 0.02,
        };
        let system_size = 128;
        let rng_seed = 42;
        let rng = OxidizeRng::from_seed(rng_seed);
        let mut tool = Self {
            dimension: KpzDimensionMode::Dim1D,
            nu: params.nu,
            lambda: params.lambda,
            noise_sigma: params.noise_sigma,
            dx: params.dx,
            dt: params.dt,
            system_size,
            steps_per_frame: 4,
            rng_seed,
            plot_view: KpzPlotView::RawLogLog,
            brush_radius: default_brush_radius(),
            playing: false,
            solver_1d: InterfaceGrowth1D::new(system_size, params),
            solver_2d: InterfaceGrowth2D::new(64, 64, params),
            telemetry: InterfaceScalingTelemetry::new(256),
            rng,
            cursor: [system_size / 2, 32],
            focused_canvas: None,
            pixels: vec![egui::Color32::BLACK; 64 * 64],
            texture: None,
            texture_dirty: true,
            high_contrast: false,
            last_step_ms: 0.0,
            diagnostic: None,
            theory: KpzTheory,
        };
        tool.rebuild_and_seed();
        tool
    }
}

impl KpzInterfaceTool {
    pub(crate) fn current_growth_kind(&self) -> InterfaceGrowthKind {
        if self.lambda.abs() < 1e-6 {
            InterfaceGrowthKind::EdwardsWilkinson
        } else {
            InterfaceGrowthKind::Kpz
        }
    }

    pub(crate) fn current_dim_num(&self) -> usize {
        match self.dimension {
            KpzDimensionMode::Dim1D => 1,
            KpzDimensionMode::Dim2D => 2,
        }
    }

    pub(crate) fn current_params(&self) -> InterfaceGrowthParams {
        InterfaceGrowthParams {
            nu: self.nu,
            lambda: self.lambda,
            noise_sigma: self.noise_sigma,
            dx: self.dx,
            dt: self.dt,
        }
    }

    /// Re-initializes the active 1D or 2D lattice from `rng_seed` and seeds initial scaling points.
    pub fn rebuild_and_seed(&mut self) {
        if !self.parameters_within_safe_bounds() {
            return;
        }
        let params = self.current_params();
        self.rng = OxidizeRng::from_seed(self.rng_seed);
        self.telemetry.clear();
        self.diagnostic = None;

        match self.dimension {
            KpzDimensionMode::Dim1D => {
                if self.solver_1d.len() != self.system_size {
                    self.solver_1d = InterfaceGrowth1D::new(self.system_size, params);
                } else {
                    self.solver_1d.reset_flat();
                    self.solver_1d.set_params(params);
                }
                self.cursor[0] = self.system_size / 2;
            }
            KpzDimensionMode::Dim2D => {
                let side = self.system_size.clamp(32, 128);
                self.system_size = side;
                if self.solver_2d.width() != side || self.solver_2d.height() != side {
                    self.solver_2d = InterfaceGrowth2D::new(side, side, params);
                } else {
                    self.solver_2d.reset_flat();
                    self.solver_2d.set_params(params);
                }
                self.cursor = [side / 2, side / 2];
            }
        }
        // Advance 4 lightweight initial steps so frame-0 plots and exports have valid log-log series.
        for _ in 0..4 {
            self.step_active_once_unchecked();
        }
        self.refresh_2d_pixels();
    }

    pub(crate) fn sync_solver_params(&mut self) {
        if !self.parameters_within_safe_bounds() {
            return;
        }
        let params = self.current_params();
        self.solver_1d.set_params(params);
        self.solver_2d.set_params(params);
        let (alpha, _, z) = self
            .current_growth_kind()
            .scaling_exponents(self.current_dim_num());
        self.telemetry
            .recompute_family_vicsek(self.system_size as f64, alpha, z);
    }

    pub(crate) fn parameters_within_safe_bounds(&self) -> bool {
        let max_l = match self.dimension {
            KpzDimensionMode::Dim1D => 512,
            KpzDimensionMode::Dim2D => 128,
        };
        self.nu.is_finite()
            && (0.05..=5.0).contains(&self.nu)
            && self.lambda.is_finite()
            && (-8.0..=8.0).contains(&self.lambda)
            && self.noise_sigma.is_finite()
            && (0.0..=4.0).contains(&self.noise_sigma)
            && self.dx.is_finite()
            && (0.25..=4.0).contains(&self.dx)
            && self.dt.is_finite()
            && (0.002..=0.10).contains(&self.dt)
            && (32..=max_l).contains(&self.system_size)
            && (1..=16).contains(&self.steps_per_frame)
            && (1..=24).contains(&self.brush_radius)
    }

    fn step_active_once_unchecked(&mut self) {
        let (alpha, _, z) = self
            .current_growth_kind()
            .scaling_exponents(self.current_dim_num());
        match self.dimension {
            KpzDimensionMode::Dim1D => {
                self.solver_1d.step(&mut self.rng);
                self.telemetry.record_sample(
                    self.solver_1d.time(),
                    self.solver_1d.interface_width(),
                    self.solver_1d.len() as f64,
                    alpha,
                    z,
                );
            }
            KpzDimensionMode::Dim2D => {
                self.solver_2d.step(&mut self.rng);
                self.telemetry.record_sample(
                    self.solver_2d.time(),
                    self.solver_2d.interface_width(),
                    self.solver_2d.width() as f64,
                    alpha,
                    z,
                );
            }
        }
    }

    /// Executes one guarded simulation step using [`crate::web_io::PlatformInstant`].
    pub fn step_once(&mut self) {
        if !self.check_and_latch_instability() {
            return;
        }
        self.sync_solver_params();
        let t0 = crate::web_io::PlatformInstant::now();
        self.step_active_once_unchecked();
        self.last_step_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if self.check_and_latch_instability() && self.dimension == KpzDimensionMode::Dim2D {
            self.refresh_2d_pixels();
        }
    }

    pub(crate) fn step_frame_cooperative(&mut self) {
        let start = crate::web_io::PlatformInstant::now();
        for _ in 0..self.steps_per_frame.clamp(1, 16) {
            self.step_once();
            if !self.playing || start.elapsed().as_secs_f64() >= COOPERATIVE_BUDGET_SECS {
                break;
            }
        }
    }

    pub(crate) fn active_heights_finite(&self) -> bool {
        match self.dimension {
            KpzDimensionMode::Dim1D => self.solver_1d.heights().iter().all(|v| v.is_finite()),
            KpzDimensionMode::Dim2D => self.solver_2d.heights().iter().all(|v| v.is_finite()),
        }
    }

    fn check_and_latch_instability(&mut self) -> bool {
        if !self.parameters_within_safe_bounds() || !self.active_heights_finite() {
            self.playing = false;
            let (t, s) = match self.dimension {
                KpzDimensionMode::Dim1D => (self.solver_1d.time(), self.solver_1d.steps() as usize),
                KpzDimensionMode::Dim2D => (self.solver_2d.time(), self.solver_2d.steps() as usize),
            };
            self.diagnostic = Some(DiagnosticReport::new(
                t,
                s,
                "kpz_interface",
                "Out-of-bounds parameter or non-finite height detected; halted by FiniteStateGuard.",
            ));
            return false;
        }
        true
    }

    /// Current fitted growth exponent $\beta$ over recorded scaling history.
    pub fn fitted_beta(&self) -> Option<(f64, f64)> {
        let t_min = self.dt * 2.0;
        let t_max = self.telemetry.times().last().copied().unwrap_or(1.0);
        self.telemetry.fit_growth_exponent_beta(t_min, t_max)
    }

    pub(crate) fn snapshot_state(&self) -> KpzToolState {
        KpzToolState {
            dimension: self.dimension,
            nu: self.nu,
            lambda: self.lambda,
            noise_sigma: self.noise_sigma,
            dx: self.dx,
            dt: self.dt,
            system_size: self.system_size,
            steps_per_frame: self.steps_per_frame,
            rng_seed: self.rng_seed,
            plot_view: self.plot_view,
            brush_radius: self.brush_radius,
            playing: self.playing,
        }
    }
}

impl FiniteStateGuard for KpzInterfaceTool {
    fn is_finite_state(&self) -> bool {
        self.diagnostic.is_none()
            && self.parameters_within_safe_bounds()
            && self.active_heights_finite()
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
            0.02
        };
        self.dt = next_dt.clamp(0.002, 0.05);
        FiniteStateGuard::clamp_parameters(self);
    }

    fn clamp_parameters(&mut self) {
        self.nu = if self.nu.is_finite() {
            self.nu.clamp(0.05, 5.0)
        } else {
            0.5
        };
        self.lambda = if self.lambda.is_finite() {
            self.lambda.clamp(-8.0, 8.0)
        } else {
            3.2
        };
        self.noise_sigma = if self.noise_sigma.is_finite() {
            self.noise_sigma.clamp(0.0, 4.0)
        } else {
            1.0
        };
        self.dx = if self.dx.is_finite() {
            self.dx.clamp(0.25, 4.0)
        } else {
            1.0
        };
        self.dt = if self.dt.is_finite() {
            self.dt.clamp(0.002, 0.10)
        } else {
            0.02
        };
        let max_l = match self.dimension {
            KpzDimensionMode::Dim1D => 512,
            KpzDimensionMode::Dim2D => 128,
        };
        self.system_size = self.system_size.clamp(32, max_l);
        self.steps_per_frame = self.steps_per_frame.clamp(1, 16);
        self.brush_radius = self.brush_radius.clamp(1, 24);
        if !self.active_heights_finite() {
            self.rebuild_and_seed();
        } else {
            self.diagnostic = None;
            self.sync_solver_params();
        }
    }

    fn reset_parameters(&mut self) {
        *self = Self::default();
    }
}

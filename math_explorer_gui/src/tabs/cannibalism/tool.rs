#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::{PopulationStructureState, PopulationViewMode, CANNIBALISM_STATE_BOUNDS};
use super::theory::CannibalismTheory;
use crate::framework::{
    DiagnosticReport, ExportCapabilities, FiniteStateGuard, InteractiveTool, ToolStateBounds,
};
use eframe::egui;
use egui::{Color32, ColorImage};
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::applied::cannibalism::{
    CannibalismModel, JuvenileAdultSolver, McKendrickSolver,
};
use nalgebra::Vector2;
use scientific_metadata::theory::TheoryDescribable;
use std::collections::VecDeque;
use std::hash::Hasher;

/// Number of historical time columns retained in the Lexis `(a, t)` heatmap buffer.
pub const LEXIS_TIME_COLS: usize = 120;

/// Maximum number of trajectory samples retained for phase and time-series plots.
pub const HISTORY_CAP: usize = 320;

/// Deterministic warmup steps executed on reset so plots and exports start populated.
const WARMUP_STEPS: usize = 160;

/// Maximum wall-clock budget (4 ms) allowed per cooperative GUI frame on native and `wasm32`.
const FRAME_BUDGET_SECS: f64 = 0.004;

/// Maximum substeps executed during a single continuous playback frame.
const MAX_FRAME_SUBSTEPS: usize = 6;

/// Interactive simulation tool coupling McKendrick-von Foerster PDE, van den Bosch delay dynamics, and 2D ODE.
pub struct PopulationStructureTool {
    /// Validated configuration parameters persisted across undo, presets, and URL share links.
    pub config: PopulationStructureState,
    /// Full age-structured McKendrick-von Foerster upwind PDE solver.
    pub mvf_solver: McKendrickSolver,
    /// Juvenile-adult delay cohort PDE solver generating Hopf cycles.
    pub ja_solver: JuvenileAdultSolver,
    /// Lumped two-compartment cannibal-normal ODE model for side-by-side comparison.
    pub ode_model: CannibalismModel,
    /// Current `[C(t), N(t)]` state vector of the lumped 2D ODE model.
    pub ode_state: Vector2<f64>,
    /// Whether continuous playback is paused.
    pub paused: bool,
    /// Monotonic count of completed simulation steps since last reset.
    pub step_index: usize,
    /// Ring buffer of `n(t, a)` age-density columns for the Lexis `(a, t)` heatmap.
    pub lexis_columns: VecDeque<Vec<f64>>,
    /// Ring buffer of `[t, J(t), A(t)]` samples from [`JuvenileAdultSolver`].
    pub ja_history: VecDeque<[f64; 3]>,
    /// Ring buffer of `[t, C(t), N(t)]` samples from [`CannibalismModel`].
    pub ode_history: VecDeque<[f64; 3]>,
    /// Active WCAG high-contrast colormap toggle synchronized from `ThemeMode`.
    pub high_contrast: bool,
    /// Marks whether `lexis_texture` needs uploading on the next UI frame.
    pub lexis_dirty: bool,
    /// Cached GPU texture handle for the Lexis `(a, t)` density raster.
    pub lexis_texture: Option<egui::TextureHandle>,
    /// Structured diagnostic report populated if non-finite values occur.
    pub diagnostic: Option<DiagnosticReport>,
    theory: CannibalismTheory,
}

impl Default for PopulationStructureTool {
    fn default() -> Self {
        let config = PopulationStructureState::default();
        let mvf_solver = McKendrickSolver::new(
            config.age_bins,
            config.max_age,
            config.to_mvf_kernel(),
        );
        let ja_solver = JuvenileAdultSolver::new(
            config.age_bins,
            4.5,
            config.to_juvenile_adult_params(),
        );
        let ode_model = build_ode_model(&config);
        let mut tool = Self {
            config,
            mvf_solver,
            ja_solver,
            ode_model,
            ode_state: Vector2::new(7.5, 3.2),
            paused: true,
            step_index: 0,
            lexis_columns: VecDeque::with_capacity(LEXIS_TIME_COLS),
            ja_history: VecDeque::with_capacity(HISTORY_CAP),
            ode_history: VecDeque::with_capacity(HISTORY_CAP),
            high_contrast: false,
            lexis_dirty: true,
            lexis_texture: None,
            diagnostic: None,
            theory: CannibalismTheory,
        };
        tool.rebuild_and_warmup();
        tool
    }
}

fn build_ode_model(cfg: &PopulationStructureState) -> CannibalismModel {
    CannibalismModel::new(
        (cfg.fecundity_rate * 0.032).clamp(0.08, 2.0),
        (cfg.fecundity_rate * 0.022).clamp(0.05, 1.5),
        (cfg.attack_rate * 0.65).clamp(0.02, 1.2),
        cfg.ode_phi,
        cfg.base_mortality,
        (cfg.base_mortality * 1.15).max(0.05),
    )
}

impl PopulationStructureTool {
    /// Returns the concrete [`CannibalismTheory`] descriptor for hover accessibility.
    #[must_use]
    pub const fn theory_model(&self) -> &CannibalismTheory {
        &self.theory
    }

    /// Re-initializes all three solvers from `self.config` and runs deterministic warmup steps.
    pub fn rebuild_and_warmup(&mut self) {
        let cfg = self.config.clamped();
        self.mvf_solver =
            McKendrickSolver::new(cfg.age_bins, cfg.max_age, cfg.to_mvf_kernel());
        self.mvf_solver.set_initial_density(|a| {
            let pulse = 7.5 * (-0.5 * ((a - 1.1) / 0.42).powi(2)).exp();
            let tail = 1.2 * (-0.35 * a).exp();
            pulse + tail
        });
        self.ja_solver =
            JuvenileAdultSolver::new(cfg.age_bins, 4.5, cfg.to_juvenile_adult_params());
        let center = 0.32 * cfg.maturation_age;
        let width = (0.14 * cfg.maturation_age).max(0.08);
        self.ja_solver.set_initial_state(4.5, |a| {
            5.5 * (-0.5 * ((a - center) / width).powi(2)).exp()
        });
        self.ode_model = build_ode_model(&cfg);
        self.ode_state = Vector2::new(7.5, 3.2);
        self.step_index = 0;
        self.diagnostic = None;
        self.lexis_columns.clear();
        self.ja_history.clear();
        self.ode_history.clear();
        self.record_snapshot();
        for _ in 0..WARMUP_STEPS {
            self.advance_one_step();
        }
    }

    fn record_snapshot(&mut self) {
        if self.lexis_columns.len() >= LEXIS_TIME_COLS {
            self.lexis_columns.pop_front();
        }
        self.lexis_columns
            .push_back(self.mvf_solver.density().to_vec());

        if self.ja_history.len() >= HISTORY_CAP {
            self.ja_history.pop_front();
        }
        self.ja_history.push_back([
            self.ja_solver.time(),
            self.ja_solver.total_juveniles(),
            self.ja_solver.adults(),
        ]);

        if self.ode_history.len() >= HISTORY_CAP {
            self.ode_history.pop_front();
        }
        self.ode_history.push_back([
            self.mvf_solver.time(),
            self.ode_state[1],
            self.ode_state[0],
        ]);
        self.lexis_dirty = true;
    }

    /// Advances all coupled solvers by one `config.dt` interval and validates finiteness.
    pub fn advance_one_step(&mut self) {
        if self.diverged() {
            self.paused = true;
            return;
        }
        let dt = self.config.dt;
        if !dt.is_finite() || dt <= 0.0 {
            self.diagnostic = Some(DiagnosticReport::new(
                self.mvf_solver.time(),
                self.step_index,
                "dt",
                "Non-positive or non-finite time step dt encountered.",
            ));
            self.paused = true;
            return;
        }
        let t = self.mvf_solver.time();
        self.mvf_solver.step_upwind(dt);
        self.ja_solver.step_upwind(dt);
        self.ode_state = self.ode_model.step_rk4(t, &self.ode_state, dt);
        self.step_index = self.step_index.saturating_add(1);
        if !self.solvers_finite() {
            self.diagnostic = Some(DiagnosticReport::new(
                self.mvf_solver.time(),
                self.step_index,
                "density",
                "Non-finite population density encountered during PDE/ODE step.",
            ));
            self.paused = true;
            return;
        }
        self.record_snapshot();
    }

    /// Steps continuous playback within a cooperative `PlatformInstant` frame budget.
    pub fn advance_frame_budgeted(&mut self) {
        let start = crate::web_io::PlatformInstant::now();
        for _ in 0..MAX_FRAME_SUBSTEPS {
            if self.paused || self.diverged() {
                break;
            }
            self.advance_one_step();
            if start.elapsed().as_secs_f64() >= FRAME_BUDGET_SECS {
                break;
            }
        }
    }

    /// Courant ratio `lambda = dt / da` on the McKendrick-von Foerster age grid.
    #[must_use]
    pub fn courant_number(&self) -> f64 {
        self.config.dt / self.mvf_solver.da().max(1e-12)
    }

    fn solvers_finite(&self) -> bool {
        self.mvf_solver.time().is_finite()
            && self.mvf_solver.density().iter().all(|v| v.is_finite())
            && self.ja_solver.juvenile_density().iter().all(|v| v.is_finite())
            && self.ja_solver.adults().is_finite()
            && self.ode_state[0].is_finite()
            && self.ode_state[1].is_finite()
    }

    /// Renders the `(a, t)` Lexis density history into an `egui::ColorImage` guarded for zero density.
    #[must_use]
    pub fn compute_lexis_image(&self, high_contrast: bool) -> ColorImage {
        let width = self.lexis_columns.len().max(1);
        let height = self.mvf_solver.ages().len().max(1);
        let mut pixels = vec![Color32::BLACK; width * height];
        let max_density = self
            .lexis_columns
            .iter()
            .flat_map(|col| col.iter().copied())
            .filter(|v| v.is_finite() && *v > 0.0)
            .fold(0.0_f64, f64::max);

        for (x, col) in self.lexis_columns.iter().enumerate() {
            for age_idx in 0..height {
                let val = col.get(age_idx).copied().unwrap_or(0.0);
                let intensity = if max_density > 0.0 && val.is_finite() && val > 0.0 {
                    (val / max_density).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let rgb = if high_contrast {
                    heatmap_color_high_contrast(intensity)
                } else {
                    heatmap_color(intensity)
                };
                let y = height - 1 - age_idx;
                pixels[flatten_2d_index(x, y, width)] = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
            }
        }
        ColorImage::new([width, height], pixels)
    }

    fn age_pyramid_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mvf_pts: Vec<[f64; 2]> = self
            .mvf_solver
            .ages()
            .iter()
            .zip(self.mvf_solver.density().iter())
            .filter(|(a, n)| a.is_finite() && n.is_finite())
            .map(|(&a, &n)| [a, n])
            .collect();
        let ja_pts: Vec<[f64; 2]> = self
            .ja_solver
            .ages()
            .iter()
            .zip(self.ja_solver.juvenile_density().iter())
            .filter(|(a, n)| a.is_finite() && n.is_finite())
            .map(|(&a, &n)| [a, n])
            .collect();
        vec![
            ("Age Density n(t, a)".to_string(), mvf_pts),
            ("Juvenile Cohort n_J(t, a)".to_string(), ja_pts),
        ]
    }

    fn phase_and_ode_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let phase: Vec<[f64; 2]> = self
            .ja_history
            .iter()
            .filter(|s| s[1].is_finite() && s[2].is_finite())
            .map(|s| [s[1], s[2]])
            .collect();
        let ode_c: Vec<[f64; 2]> = self
            .ode_history
            .iter()
            .filter(|s| s[0].is_finite() && s[1].is_finite())
            .map(|s| [s[0], s[1]])
            .collect();
        let ode_n: Vec<[f64; 2]> = self
            .ode_history
            .iter()
            .filter(|s| s[0].is_finite() && s[2].is_finite())
            .map(|s| [s[0], s[2]])
            .collect();
        vec![
            ("Hopf Phase (J, A)".to_string(), phase),
            ("2D ODE Cannibals C(t)".to_string(), ode_c),
            ("2D ODE Normals N(t)".to_string(), ode_n),
        ]
    }
}

impl FiniteStateGuard for PopulationStructureTool {
    fn is_finite_state(&self) -> bool {
        self.config.validate().is_ok() && self.diagnostic.is_none() && self.solvers_finite()
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.diagnostic.as_ref()
    }

    fn auto_scale_step_size(&mut self) {
        let halved = if self.config.dt.is_finite() && self.config.dt > 0.0 {
            (self.config.dt * 0.5).clamp(0.002, 0.5)
        } else {
            0.02
        };
        let mut next = self.config.clamped();
        next.dt = halved;
        self.config = next;
        self.rebuild_and_warmup();
    }

    fn clamp_parameters(&mut self) {
        self.config = self.config.clamped();
        self.rebuild_and_warmup();
    }

    fn reset_parameters(&mut self) {
        self.config = PopulationStructureState::default();
        self.rebuild_and_warmup();
    }
}

impl InteractiveTool for PopulationStructureTool {
    fn name(&self) -> &'static str {
        "Population Structure & Cannibalism"
    }

    fn domain(&self) -> &'static str {
        "cannibalism"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory
    }

    fn state_bounds(&self) -> ToolStateBounds {
        CANNIBALISM_STATE_BOUNDS
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.config.hash_into(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        self.is_finite_state()
    }

    fn clamp_parameters(&mut self) -> bool {
        <Self as FiniteStateGuard>::clamp_parameters(self);
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.config).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: PopulationStructureState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid PopulationStructureState: {e}"))?;
        parsed.validate()?;
        if self.config != parsed || self.diverged() {
            self.config = parsed;
            self.rebuild_and_warmup();
        }
        Ok(())
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        match self.config.view_mode {
            PopulationViewMode::AgePyramid => self.age_pyramid_datasets(),
            PopulationViewMode::PhasePortrait => {
                self.phase_and_ode_datasets().into_iter().take(1).collect()
            }
            PopulationViewMode::OdeComparison => {
                self.phase_and_ode_datasets().into_iter().skip(1).collect()
            }
            PopulationViewMode::Dashboard | PopulationViewMode::LexisHeatmap => {
                let mut all = self.age_pyramid_datasets();
                all.extend(self.phase_and_ode_datasets());
                all
            }
        }
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        if self.config.view_mode == PopulationViewMode::LexisHeatmap {
            Vec::new()
        } else {
            egui_plot::export_primitives::datasets_to_primitives(&self.export_datasets())
        }
    }

    fn export_raster_image(&self) -> Option<ColorImage> {
        if self.config.view_mode == PopulationViewMode::LexisHeatmap {
            Some(self.compute_lexis_image(self.high_contrast))
        } else {
            None
        }
    }

    fn step_simulation(&mut self) {
        self.advance_one_step();
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        super::views::render_tool_inside(self, ui);
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

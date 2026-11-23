// [cite:quantum_mechanics]
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "wave_sim/canvas.rs"]
mod canvas;
#[path = "wave_sim/observables.rs"]
mod observables;
#[path = "wave_sim/persistence.rs"]
mod persistence;
#[path = "wave_sim/presets.rs"]
pub mod presets;
#[path = "wave_sim/runner.rs"]
pub mod runner;
#[cfg(test)]
#[path = "wave_sim/tests.rs"]
mod tests;
#[path = "wave_sim/ui.rs"]
mod ui;
#[cfg(all(test, target_arch = "wasm32"))]
#[path = "wave_sim/wasm_tests.rs"]
mod wasm_tests;

#[cfg(test)]
pub use presets::wave_parameter_label;
pub use presets::{wave_parameter_constraints, WavePotentialPreset};
pub use runner::{WaveData, WaveRunner, WaveSimConfig};

use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::framework::recovery::RecoveryAction;
use crate::framework::{ExportCapabilities, InteractiveTool, MathExpressionInputBar};
use crate::url_state::quantize_url_centi;
use eframe::egui;
use math_commons::ast::Expr;
use math_explorer::physics::quantum::{
    apply_bresenham_potential_stroke, MAX_SCULPT_POTENTIAL, MIN_SCULPT_POTENTIAL,
};
use nalgebra::DVector;
#[cfg(test)]
use observables::STARTER_PRESET_JSON;
use presets::{
    build_wave_datasets, clamp_wavepacket_params, custom_formula_presets, default_compiled_expr,
    DEFAULT_CUSTOM_EXPR, DEFAULT_DT, DEFAULT_K0, DEFAULT_SIGMA, DEFAULT_STEPS_PER_FRAME,
    DEFAULT_V0, DEFAULT_WIDTH, DEFAULT_X0, DT_RECOVERY_MIN, GRID_POINTS,
};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Dedicated 1D Schrödinger wavepacket tunneling and potential sculpting sandbox.
pub struct WaveSimulatorTool {
    pub(crate) preset: WavePotentialPreset,
    pub(crate) x0: f64,
    pub(crate) k0: f64,
    pub(crate) sigma: f64,
    pub(crate) v0: f64,
    pub(crate) width: f64,
    pub(crate) dt: f64,
    pub(crate) steps_per_frame: usize,
    pub(crate) math_input: MathExpressionInputBar,
    pub(crate) last_valid_expr: Expr,
    pub(crate) custom_potential: Option<Vec<f64>>,
    pub(crate) shared_config: Arc<RwLock<WaveSimConfig>>,
    pub(crate) worker: BackgroundWorker<WaveRunner>,
    pub(crate) export_runner: Option<WaveRunner>,
    pub(crate) export_data: Option<WaveData>,
    pub(crate) latest_sonification: crate::audio::SonificationFrame,
    pub(crate) revision: u64,
}

impl Default for WaveSimulatorTool {
    fn default() -> Self {
        Self::new()
    }
}

impl WaveSimulatorTool {
    /// Creates a new `WaveSimulatorTool` initialized to the Infinite Potential Well baseline.
    #[must_use]
    pub fn new() -> Self {
        let math_input =
            MathExpressionInputBar::new("V(x)", DEFAULT_CUSTOM_EXPR, custom_formula_presets());
        let last_valid_expr = math_input
            .expr()
            .cloned()
            .unwrap_or_else(default_compiled_expr);
        let initial_cfg = WaveSimConfig {
            preset: WavePotentialPreset::InfiniteWell,
            x0: DEFAULT_X0,
            k0: DEFAULT_K0,
            sigma: DEFAULT_SIGMA,
            v0: DEFAULT_V0,
            width: DEFAULT_WIDTH,
            dt: DEFAULT_DT,
            steps_per_frame: DEFAULT_STEPS_PER_FRAME,
            custom_expr: last_valid_expr.clone(),
            custom_potential: None,
        };
        let shared_config = Arc::new(RwLock::new(initial_cfg));
        let worker = BackgroundWorker::new(WaveRunner::new(Arc::clone(&shared_config)));
        let latest_sonification = worker
            .latest_snapshot()
            .map_or_else(crate::audio::SonificationFrame::silent, |s| s.sonification);
        Self {
            preset: WavePotentialPreset::InfiniteWell,
            x0: DEFAULT_X0,
            k0: DEFAULT_K0,
            sigma: DEFAULT_SIGMA,
            v0: DEFAULT_V0,
            width: DEFAULT_WIDTH,
            dt: DEFAULT_DT,
            steps_per_frame: DEFAULT_STEPS_PER_FRAME,
            math_input,
            last_valid_expr,
            custom_potential: None,
            shared_config,
            worker,
            export_runner: None,
            export_data: None,
            latest_sonification,
            revision: 1,
        }
    }

    /// Current worker lifecycle state.
    #[cfg(test)]
    #[must_use]
    pub fn state(&self) -> SimWorkerState {
        self.worker.state()
    }

    pub(crate) fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    /// Applies a Bresenham potential stroke across `i0..=i1`, centi-quantizing persisted values.
    pub(crate) fn apply_sculpt_stroke(
        &mut self,
        base_potential: &[f64],
        i0: usize,
        v0: f64,
        i1: usize,
        v1: f64,
    ) {
        let entry = self.custom_potential.get_or_insert_with(|| {
            if base_potential.len() == GRID_POINTS {
                base_potential
                    .iter()
                    .map(|&v| quantize_url_centi(v, MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL))
                    .collect()
            } else {
                vec![0.0; GRID_POINTS]
            }
        });
        if entry.len() != GRID_POINTS {
            *entry = vec![0.0; GRID_POINTS];
        }
        let mut dvec = DVector::from_column_slice(entry);
        apply_bresenham_potential_stroke(&mut dvec, i0, v0, i1, v1);
        for (dst, &src) in entry.iter_mut().zip(dvec.iter()) {
            *dst = quantize_url_centi(src, MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL);
        }
        if let Ok(mut cfg) = self.shared_config.write() {
            cfg.custom_potential = self.custom_potential.clone();
        }
        self.bump_revision();
        self.export_runner = None;
        self.export_data = None;
        let qv0 = quantize_url_centi(v0, MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL);
        let qv1 = quantize_url_centi(v1, MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL);
        let _ = self.worker.send_command(SimCommand::SculptStroke1D {
            i0,
            v0: qv0,
            i1,
            v1: qv1,
        });
    }

    /// Updates wavepacket emitter parameters without erasing any freehand sculpted potential.
    pub(crate) fn update_wavepacket_live(
        &mut self,
        x0: f64,
        k0: f64,
        sigma: f64,
        ctx: Option<&egui::Context>,
    ) {
        let (sx0, sk0, ssigma) = clamp_wavepacket_params(x0, k0, sigma);
        self.x0 = sx0;
        self.k0 = sk0;
        self.sigma = ssigma;
        if let Ok(mut cfg) = self.shared_config.write() {
            cfg.x0 = sx0;
            cfg.k0 = sk0;
            cfg.sigma = ssigma;
        }
        self.bump_revision();
        self.export_runner = None;
        self.export_data = None;
        let _ = self.worker.send_command(SimCommand::SetWavepacket {
            x0: sx0,
            k0: sk0,
            sigma: ssigma,
        });
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    /// Clears freehand sculpted strokes and restores the active preset or custom AST potential.
    pub fn clear_custom_potential(&mut self, ctx: Option<&egui::Context>) {
        self.custom_potential = None;
        if let Ok(mut cfg) = self.shared_config.write() {
            cfg.custom_potential = None;
        }
        self.bump_revision();
        self.export_runner = None;
        self.export_data = None;
        let _ = self.worker.send_command(SimCommand::ClearObstacles);
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    /// Configures the playable 50% sub-barrier tunneling split challenge (`E0 = 12.5`, `V0 = 11.35`, `w = 0.5`).
    pub fn apply_tunneling_split_challenge(&mut self, ctx: Option<&egui::Context>) {
        self.preset = WavePotentialPreset::RectangularBarrier;
        self.x0 = DEFAULT_X0;
        self.k0 = DEFAULT_K0;
        self.sigma = DEFAULT_SIGMA;
        self.v0 = DEFAULT_V0;
        self.width = DEFAULT_WIDTH;
        self.dt = DEFAULT_DT;
        self.custom_potential = None;
        self.sync_config_and_reset(ctx);
    }

    /// Synchronizes shared worker config, clears stateful export caches, and resets or respawns `WaveRunner`.
    pub fn sync_config_and_reset(&mut self, ctx: Option<&egui::Context>) {
        self.bump_revision();
        self.export_runner = None;
        self.export_data = None;
        let next_cfg = WaveSimConfig {
            preset: self.preset,
            x0: self.x0,
            k0: self.k0,
            sigma: self.sigma,
            v0: self.v0,
            width: self.width,
            dt: self.dt,
            steps_per_frame: self.steps_per_frame,
            custom_expr: self.last_valid_expr.clone(),
            custom_potential: self.custom_potential.clone(),
        };
        if let Ok(mut guard) = self.shared_config.write() {
            *guard = next_cfg;
        }
        let shared = Arc::clone(&self.shared_config);
        self.worker.reset_or_respawn(|| WaveRunner::new(shared));
        let _ = self
            .worker
            .send_command(SimCommand::SetSpeed(self.steps_per_frame));
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    pub(crate) fn reset_to_defaults(&mut self, ctx: Option<&egui::Context>) {
        self.preset = WavePotentialPreset::InfiniteWell;
        self.x0 = DEFAULT_X0;
        self.k0 = DEFAULT_K0;
        self.sigma = DEFAULT_SIGMA;
        self.v0 = DEFAULT_V0;
        self.width = DEFAULT_WIDTH;
        self.dt = DEFAULT_DT;
        self.steps_per_frame = DEFAULT_STEPS_PER_FRAME;
        self.custom_potential = None;
        self.math_input.set_expression(DEFAULT_CUSTOM_EXPR);
        if let Some(expr) = self.math_input.expr() {
            self.last_valid_expr = expr.clone();
        }
        self.sync_config_and_reset(ctx);
    }

    pub(crate) fn handle_recovery_action(&mut self, action: RecoveryAction, ctx: &egui::Context) {
        match action {
            RecoveryAction::AutoScaleStepSize => {
                self.dt = if self.dt.is_finite() && self.dt > 0.0 {
                    (self.dt * 0.5).max(DT_RECOVERY_MIN)
                } else {
                    DEFAULT_DT
                };
                self.sync_config_and_reset(Some(ctx));
            }
            RecoveryAction::ClampParameters => {
                self.clamp_wave_parameters();
                ctx.request_repaint();
            }
            RecoveryAction::ResetParameters => {
                self.reset_to_defaults(Some(ctx));
                let _ = self.worker.send_command(SimCommand::Start);
            }
        }
    }
}

impl InteractiveTool for WaveSimulatorTool {
    fn name(&self) -> &'static str {
        "Wave Simulator"
    }

    fn domain(&self) -> &'static str {
        "quantum"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn step_simulation(&mut self) {
        let shared = Arc::clone(&self.shared_config);
        let runner = self
            .export_runner
            .get_or_insert_with(|| WaveRunner::new(shared));
        for _ in 0..self.steps_per_frame.max(1) {
            runner.step_with_dt(self.dt);
        }
        self.export_data = Some(runner.build_wave_data());
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        if let Some(ref stepped) = self.export_data {
            return build_wave_datasets(stepped);
        }
        if let Some(wave_data) = self
            .worker
            .latest_snapshot()
            .and_then(|s| s.structured_data.as_ref())
            .and_then(|d| d.downcast_ref::<WaveData>())
        {
            return build_wave_datasets(wave_data);
        }
        egui_plot::get_last_plot_datasets()
    }

    fn state_revision(&self) -> u64 {
        self.compute_state_revision()
    }

    fn has_finite_parameters(&self) -> bool {
        self.parameters_are_valid()
    }

    fn save_state(&self) -> Option<String> {
        self.serialize_state()
    }

    fn save_recovery_state(&self) -> Option<String> {
        self.serialize_state()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.deserialize_state(state)
    }

    fn try_load_recovery_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        self.deserialize_state(state)
    }

    fn clamp_parameters(&mut self) -> bool {
        self.clamp_wave_parameters();
        true
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let mut has_update = false;
        if let Some(snapshot) = self.worker.update() {
            self.latest_sonification = snapshot.sonification;
            has_update = true;
        }
        if has_update {
            self.export_runner = None;
            self.export_data = None;
        }
        if self.worker.state().is_faulted() {
            self.latest_sonification = crate::audio::SonificationFrame::silent();
            crate::audio::notify_solver_diverged();
        }
        self.draw_left_panel(ui);
        self.draw_central_panel(ui);
        if has_update
            || matches!(
                self.worker.state(),
                SimWorkerState::Running | SimWorkerState::Stepping
            )
        {
            ui.ctx().request_repaint();
        }
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

impl TheoryDescribable for WaveSimulatorTool {
    fn theory_description(&self) -> String {
        "Time-dependent 1D Schrödinger wavepacket evolution $i\\hbar \\partial_t \\psi(x,t) = [-\\frac{\\hbar^2}{2m}\\partial_x^2 + V(x)]\\psi(x,t)$ with unitary propagation, freehand Bresenham potential sculpting, custom AST barriers, and Born tunneling integrals.".to_string()
    }

    fn phonetic_description(&self) -> String {
        "Time-dependent one-dimensional Schrodinger wavepacket tunneling and potential sculpting simulation".to_string()
    }

    fn theory_citation(&self) -> String {
        "Schrödinger, E. (1926). An Undulatory Theory of the Mechanics of Atoms and Molecules. Physical Review, 28(6), 1049-1070; Griffiths, D. J., & Schroeter, D. F. (2018). Introduction to Quantum Mechanics (3rd ed.). Cambridge University Press.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::from([
            (
                "Potential Profile".to_string(),
                "Selects built-in confining wells, tunneling barriers, or a custom V(x) AST equation.".to_string(),
            ),
            (
                "Barrier Height (V0)".to_string(),
                "Sets the potential barrier height V0 relative to incident kinetic energy E0 = k0²/2.".to_string(),
            ),
            (
                "Time Step (Δt)".to_string(),
                "Integration time step Δt controlling unitary matrix-exponential wavepacket propagation.".to_string(),
            ),
        ])
    }

    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        wave_parameter_constraints()
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "WaveSimulator",
        display_name: "Wave Simulator",
        domain: "quantum",
        tags: &["tunneling", "sculpting", "schrodinger", "wavepacket"],
        build: || Box::new(WaveSimulatorTool::new()),
    }
}

// [cite:quantum_mechanics]
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "wave_sim/atlas.rs"]
pub mod atlas;
#[cfg(test)]
#[path = "wave_sim/atlas_tests.rs"]
mod atlas_tests;
#[path = "wave_sim/challenge.rs"]
pub mod challenge;
#[path = "wave_sim/persistence.rs"]
mod persistence;
#[path = "wave_sim/plot_ui.rs"]
mod plot_ui;
#[path = "wave_sim/presets.rs"]
pub mod presets;
#[path = "wave_sim/runner.rs"]
pub mod runner;
#[path = "wave_sim/sculpt.rs"]
pub mod sculpt;
#[path = "wave_sim/sonification.rs"]
mod sonification;
#[cfg(test)]
#[path = "wave_sim/tests.rs"]
mod tests;
#[path = "wave_sim/ui.rs"]
mod ui;
#[cfg(all(test, target_arch = "wasm32"))]
#[path = "wave_sim/wasm_tests.rs"]
mod wasm_tests;

pub use atlas::WAVE_ATLAS_AXIS_PAIRS;
#[cfg(test)]
pub use atlas::{QuantumTunnelingCartographer, QuantumTunnelingRegime};
#[cfg(test)]
pub use challenge::CHALLENGE_FABRY_PEROT_RESONANCE;
#[cfg(test)]
use persistence::STARTER_PRESET_JSON;
#[cfg(test)]
pub use presets::wave_parameter_label;
pub use presets::{wave_parameter_constraints, WavePotentialPreset};
pub use runner::{WaveData, WaveRunner, WaveSimConfig};

use crate::async_sim::atlas::ParameterAtlasController;
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::framework::recovery::RecoveryAction;
use crate::framework::{
    CalloutManager, CalloutSpec, ChallengeEvaluation, ExportCapabilities, InteractionContext,
    InteractiveTool, MathExpressionInputBar, SpatialBrushKit,
};
use eframe::egui;
use math_commons::ast::Expr;
use presets::{
    build_wave_datasets, custom_formula_presets, default_compiled_expr, DEFAULT_CUSTOM_EXPR,
    DEFAULT_DT, DEFAULT_K0, DEFAULT_SIGMA, DEFAULT_STEPS_PER_FRAME, DEFAULT_V0, DEFAULT_WIDTH,
    DEFAULT_X0, DT_RECOVERY_MIN,
};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use sculpt::default_wave_brush_kit;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Dedicated 1D Schrödinger wavepacket tunneling sandbox backed by [`BackgroundWorker<WaveRunner>`].
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
    pub(crate) brush_kit: SpatialBrushKit,
    pub(crate) sparse_potential: Vec<(u16, f64)>,
    pub(crate) shared_config: Arc<RwLock<WaveSimConfig>>,
    pub(crate) worker: BackgroundWorker<WaveRunner>,
    pub(crate) export_runner: Option<WaveRunner>,
    pub(crate) export_data: Option<WaveData>,
    pub(crate) audio_settings: crate::audio::AcousticSettings,
    pub(crate) latest_sonification: crate::audio::SonificationFrame,
    pub(crate) atlas: ParameterAtlasController,
    pub(crate) pending_slider_reset: bool,
    pub(crate) propagator_rebuilds: Arc<std::sync::atomic::AtomicUsize>,
    pub(crate) active_challenge: Option<&'static str>,
    pub(crate) callout_manager: CalloutManager,
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
            MathExpressionInputBar::new("V(x)", DEFAULT_CUSTOM_EXPR, custom_formula_presets())
                .with_allowed_vars(&["x", "v0", "width"]);
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
            sparse_potential: Vec::new(),
        };
        let shared_config = Arc::new(RwLock::new(initial_cfg));
        let propagator_rebuilds = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let worker = BackgroundWorker::new(WaveRunner::new_with_counter(
            Arc::clone(&shared_config),
            Arc::clone(&propagator_rebuilds),
        ));
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
            brush_kit: default_wave_brush_kit(),
            sparse_potential: Vec::new(),
            shared_config,
            worker,
            export_runner: None,
            export_data: None,
            audio_settings: crate::audio::AcousticSettings::default(),
            latest_sonification,
            atlas: ParameterAtlasController::new(WAVE_ATLAS_AXIS_PAIRS.to_vec()),
            pending_slider_reset: false,
            propagator_rebuilds,
            active_challenge: None,
            callout_manager: CalloutManager::new(),
            revision: 1,
        }
    }

    /// Current worker lifecycle state.
    #[cfg(test)]
    #[must_use]
    pub fn state(&self) -> SimWorkerState {
        self.worker.state()
    }

    /// Total number of `time_evolution_operator` matrix-exponential rebuilds triggered on runners.
    #[cfg(test)]
    #[must_use]
    pub fn propagator_rebuild_count(&self) -> usize {
        self.propagator_rebuilds
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
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
            sparse_potential: self.sparse_potential.clone(),
        };
        if let Ok(mut guard) = self.shared_config.write() {
            *guard = next_cfg;
        }
        let shared = Arc::clone(&self.shared_config);
        let rebuilds = Arc::clone(&self.propagator_rebuilds);
        self.worker
            .reset_or_respawn(|| WaveRunner::new_with_counter(shared, rebuilds));
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
        self.sparse_potential.clear();
        self.math_input.set_expression(DEFAULT_CUSTOM_EXPR);
        if self.math_input.expr().is_some() {
            self.last_valid_expr = self.math_input.expr().unwrap().clone();
        }
        self.enforce_challenge_locks();
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
        ExportCapabilities::plot().with_audio_score()
    }

    fn export_wav_audio(&self, sample_rate: u32, duration_secs: f32) -> Result<Vec<u8>, String> {
        Ok(self.render_wave_wav(sample_rate, duration_secs))
    }

    fn export_midi_score(&self, duration_secs: f32) -> Result<Vec<u8>, String> {
        Ok(self.render_wave_midi(duration_secs))
    }

    fn on_brush(&mut self, ctx: &InteractionContext) {
        if !self.is_challenge_locked() {
            self.apply_ui_brush_context(ctx);
        }
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.brush_kit.focused_plot_id()
    }

    fn step_simulation(&mut self) {
        if self.pending_slider_reset {
            self.pending_slider_reset = false;
            self.sync_config_and_reset(None);
        }
        while self.worker.update().is_some() {}
        let shared = Arc::clone(&self.shared_config);
        let rebuilds = Arc::clone(&self.propagator_rebuilds);
        let runner = self
            .export_runner
            .get_or_insert_with(|| WaveRunner::new_with_counter(shared, rebuilds));
        for _ in 0..self.steps_per_frame.max(1) {
            runner.step_with_dt(self.dt);
        }
        self.latest_sonification = runner.sonification_frame();
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

    fn apply_challenge(&mut self, challenge_id: Option<&str>) -> Result<(), String> {
        self.set_wave_challenge(challenge_id)
    }

    fn active_challenge_id(&self) -> Option<&'static str> {
        self.active_challenge
    }

    fn evaluate_challenge(&self) -> Option<ChallengeEvaluation> {
        self.eval_wave_challenge()
    }

    fn sync_callouts(&mut self, callouts: &[CalloutSpec]) {
        self.callout_manager.sync_specs(callouts);
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let mut has_update = false;
        let mut clear_export = false;
        if let Some(snapshot) = self.worker.update() {
            let snap_time = snapshot
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<WaveData>())
                .map(|wd| wd.time);
            clear_export = match (snap_time, self.export_data.as_ref()) {
                (Some(st), Some(ed)) => st > ed.time,
                _ => true,
            };
            self.latest_sonification = snapshot.sonification;
            has_update = true;
        }
        if clear_export {
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

    fn sonification_frame(&self) -> Option<crate::audio::SonificationFrame> {
        if !self.has_finite_parameters() {
            return Some(crate::audio::SonificationFrame::silent());
        }
        Some(self.latest_sonification.sanitize())
    }

    fn acoustic_settings(&self) -> crate::audio::AcousticSettings {
        self.audio_settings
    }

    fn set_acoustic_settings(&mut self, settings: crate::audio::AcousticSettings) {
        self.audio_settings = settings.sanitized();
    }

    fn sample_sonification_frames(
        &mut self,
        num_frames: usize,
    ) -> Vec<crate::audio::SonificationFrame> {
        self.sample_wave_sonification_frames(num_frames)
    }
}

impl TheoryDescribable for WaveSimulatorTool {
    fn theory_description(&self) -> String {
        "Time-dependent 1D Schrödinger wavepacket evolution $i\\hbar \\partial_t \\psi(x,t) = [-\\frac{\\hbar^2}{2m}\\partial_x^2 + V(x)]\\psi(x,t)$ with unitary propagation, custom AST barriers, live potential sculpting, and Born tunneling integrals.".to_string()
    }

    fn phonetic_description(&self) -> String {
        "Time-dependent one-dimensional Schrodinger wavepacket tunneling simulation".to_string()
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
        tags: &["tunneling", "schrodinger", "wavepacket", "sculpt"],
        build: || Box::new(WaveSimulatorTool::new()),
    }
}

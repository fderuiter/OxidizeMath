#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::plots::{
    compute_bifurcation_slice, render_lexis_plot, render_secondary_plots, LexisCarpetBuffer,
    LEXIS_HISTORY_COLS,
};
use super::state::{CannibalismRegime, CannibalismToolState};
use super::theory::CannibalismTheory;
use crate::accessibility::{announce_live_status, AccessibleHoverText};
use crate::framework::{ExportCapabilities, InteractiveTool, PaneSidePanel, ToolStateBounds};
use eframe::egui;
use math_explorer::applied::cannibalism::AgeStructuredSolver;
use scientific_metadata::theory::TheoryDescribable;
use std::hash::Hasher;

const WARMUP_SNAPSHOT_STRIDE: usize = 3;
const DEFERRED_SLICES_PER_FRAME: usize = 8;

/// Interactive McKendrick-von Foerster PDE Lexis carpet and *Tribolium* bifurcation tool.
pub struct CannibalismTool {
    /// Validated persisted configuration parameters.
    pub state: CannibalismToolState,
    /// Preallocated zero-allocation characteristic-shift PDE solver.
    pub solver: AgeStructuredSolver,
    /// Preallocated rolling 2D Lexis `(t, a)` density buffer.
    pub lexis: LexisCarpetBuffer,
    /// Asymptotic `(parameter, A)` points accumulated across the bifurcation sweep.
    pub bifurcation_points: Vec<[f64; 2]>,
    /// Next slice index in `0..=state.sweep_points` for deferred bifurcation computation.
    pub sweep_cursor: usize,
    lexis_dirty: bool,
    last_high_contrast: bool,
    lexis_texture: Option<egui::TextureHandle>,
    theory: CannibalismTheory,
}

impl Default for CannibalismTool {
    fn default() -> Self {
        let state = CannibalismToolState::default();
        let solver = AgeStructuredSolver::new(state.to_pde_config());
        Self {
            state,
            solver,
            lexis: LexisCarpetBuffer::default(),
            bifurcation_points: Vec::with_capacity(256),
            sweep_cursor: 0,
            lexis_dirty: true,
            last_high_contrast: false,
            lexis_texture: None,
            theory: CannibalismTheory,
        }
    }
}

impl CannibalismTool {
    /// Marks the PDE Lexis buffer and deferred bifurcation sweep for lazy recomputation,
    /// optionally completing both immediately.
    pub fn rebuild_simulation(&mut self, eager: bool) {
        let cfg = self.state.to_pde_config();
        self.solver.reconfigure(cfg);
        self.solver.seed_perturbed_equilibrium(0.35);
        self.lexis.reset(self.solver.config().grid_size);
        self.lexis_texture = None;
        self.bifurcation_points.clear();
        self.sweep_cursor = 0;
        self.lexis_dirty = true;
        if eager {
            self.ensure_lexis_seeded();
            self.complete_bifurcation_sweep();
        }
    }

    /// Populates the initial rolling Lexis carpet columns if invalidated since the last parameter change.
    pub fn ensure_lexis_seeded(&mut self) {
        if !self.lexis_dirty {
            return;
        }
        self.lexis_dirty = false;
        self.lexis.reset(self.solver.config().grid_size);
        for _ in 0..LEXIS_HISTORY_COLS {
            self.solver.step_n(WARMUP_SNAPSHOT_STRIDE);
            self.lexis.push_snapshot(&self.solver);
        }
        self.lexis_texture = None;
    }

    /// Advances the PDE solver by `steps` characteristic shifts and records a Lexis column.
    pub fn advance_pde_steps(&mut self, steps: usize) {
        self.ensure_lexis_seeded();
        self.solver.step_n(steps.max(1));
        self.lexis.push_snapshot(&self.solver);
        self.lexis_texture = None;
    }

    /// Completes any remaining parameter slices in the deferred bifurcation sweep.
    pub fn complete_bifurcation_sweep(&mut self) {
        while self.sweep_cursor < self.state.sweep_points {
            compute_bifurcation_slice(&self.state, self.sweep_cursor, &mut self.bifurcation_points);
            self.sweep_cursor += 1;
        }
    }

    fn step_deferred_bifurcation(&mut self, is_dragging: bool) {
        if is_dragging || self.sweep_cursor >= self.state.sweep_points {
            return;
        }
        let end = (self.sweep_cursor + DEFERRED_SLICES_PER_FRAME).min(self.state.sweep_points);
        while self.sweep_cursor < end {
            compute_bifurcation_slice(&self.state, self.sweep_cursor, &mut self.bifurcation_points);
            self.sweep_cursor += 1;
        }
    }

    fn render_regime_and_playback_group(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.group(|ui| {
            ui.strong("Ecological Regime");
            for candidate in [
                CannibalismRegime::DiekmannHopf,
                CannibalismRegime::TriboliumChaos,
            ] {
                if ui
                    .radio_value(&mut self.state.regime, candidate, candidate.label())
                    .changed()
                {
                    changed = true;
                }
            }
        });
        ui.group(|ui| {
            ui.strong("Transport Playback");
            ui.horizontal(|ui| {
                let play_lbl = if self.state.is_playing {
                    "⏸ Pause Waves"
                } else {
                    "▶ Play Waves"
                };
                if ui
                    .button(play_lbl)
                    .accessible_hover_text("Toggle real-time cohort characteristic stepping")
                    .clicked()
                {
                    self.state.is_playing = !self.state.is_playing;
                    let status = if self.state.is_playing {
                        "Cannibalism wave simulation playing"
                    } else {
                        "Cannibalism wave simulation paused"
                    };
                    announce_live_status(ui.ctx(), status);
                }
                if ui
                    .button("⏭ Step Cohort")
                    .accessible_hover_text("Advance one characteristic wave burst")
                    .clicked()
                {
                    self.advance_pde_steps(self.state.steps_per_frame);
                }
            });
            let _ = ui.add(
                egui::Slider::new(&mut self.state.steps_per_frame, 1..=20).text("Steps per Frame"),
            );
        });
        changed
    }

    fn render_parameter_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.group(|ui| {
            ui.strong("Vital & Cannibalism Rates");
            changed |= ui
                .add(egui::Slider::new(&mut self.state.beta0, 5.0..=200.0).text("Fecundity β₀"))
                .changed();
            if self.state.regime == CannibalismRegime::DiekmannHopf {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.state.c_cannibalism, 0.05..=3.0)
                            .text("Cannibalism Attack Rate c"),
                    )
                    .changed();
            } else {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.state.c_pa, 0.0..=2.0)
                            .text("Callow Predation c_pa"),
                    )
                    .changed();
            }
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.state.mu0, 0.02..=1.5).text("Natural Mortality μ₀"),
                )
                .changed();
        });
        ui.group(|ui| {
            ui.strong("Discretization & Bifurcation");
            changed |= ui
                .add(egui::Slider::new(&mut self.state.grid_size, 32..=256).text("Age Bins"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.state.sweep_points, 16..=80).text("Sweep Slices"))
                .changed();
        });
        changed
    }

    fn render_readout_summary(&self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.strong("Live Cohort Diagnostics");
            ui.label(format!("Time t = {:.2}", self.solver.time()));
            ui.label(format!(
                "Adults A(t) = {:.3} | Juveniles J(t) = {:.3}",
                self.solver.adult_population(),
                self.solver.juvenile_population()
            ));
            ui.label(format!(
                "Boundary Renewal n(t, 0) = {:.3}",
                self.solver.last_birth_rate()
            ));
            if let Some(eq) = self.solver.diekmann_equilibrium() {
                ui.label(format!("Diekmann Equilibrium A* = {eq:.4}"));
            } else {
                let lpa = self.state.to_lpa_model();
                let mut s = [1.5, 0.5, 1.0];
                for _ in 0..64 {
                    s = lpa.step(s);
                }
                let s_next = lpa.step(s);
                ui.label(format!(
                    "LPA Orbit Pair A = [{:.3}, {:.3}]",
                    s[2].min(s_next[2]),
                    s[2].max(s_next[2])
                ));
            }
            if self.sweep_cursor < self.state.sweep_points {
                ui.label(format!(
                    "Deferred sweep: {}/{}",
                    self.sweep_cursor, self.state.sweep_points
                ));
            }
        });
    }
}

impl InteractiveTool for CannibalismTool {
    fn name(&self) -> &'static str {
        "Cannibalism"
    }

    fn domain(&self) -> &'static str {
        "cannibalism"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let (mut adults, mut juveniles) = self.lexis.ordered_trajectories();
        if adults.is_empty() {
            let mut preview = AgeStructuredSolver::new(self.state.to_pde_config());
            for _ in 0..32 {
                preview.step_n(WARMUP_SNAPSHOT_STRIDE);
                adults.push([preview.time(), preview.adult_population()]);
                juveniles.push([preview.time(), preview.juvenile_population()]);
            }
        }
        let profile: Vec<[f64; 2]> = self
            .solver
            .ages()
            .iter()
            .zip(self.solver.density().iter())
            .map(|(&a, &n)| [a, n])
            .collect();
        let mut datasets = vec![
            ("Adult Population A(t)".to_string(), adults),
            ("Juvenile Population J(t)".to_string(), juveniles),
            ("Age Profile n(t, a)".to_string(), profile),
        ];
        if !self.bifurcation_points.is_empty() {
            datasets.push((
                "Bifurcation Orbit Branches".to_string(),
                self.bifurcation_points.clone(),
            ));
        }
        datasets
    }

    fn step_simulation(&mut self) {
        self.advance_pde_steps(self.state.steps_per_frame);
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let left_id = crate::framework::scoped_panel_id(ui, "cannibalism_controls_panel");
        let mut params_changed = false;
        PaneSidePanel::left(left_id)
            .min_width(265.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Cannibalism PDE Lab");
                    ui.separator();
                    params_changed |= self.render_regime_and_playback_group(ui);
                    params_changed |= self.render_parameter_sliders(ui);
                    self.render_readout_summary(ui);
                });
            });

        let is_dragging = ui.ctx().is_using_pointer();
        if params_changed {
            self.rebuild_simulation(false);
        }
        self.ensure_lexis_seeded();
        if self.state.is_playing {
            self.advance_pde_steps(self.state.steps_per_frame);
            ui.ctx().request_repaint();
        }
        self.step_deferred_bifurcation(is_dragging);
        if self.sweep_cursor < self.state.sweep_points && !is_dragging {
            ui.ctx().request_repaint();
        }

        let high_contrast = ui.visuals().override_text_color.is_some();
        if high_contrast != self.last_high_contrast {
            self.last_high_contrast = high_contrast;
            self.lexis_texture = None;
        }

        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.strong(
                    "2D Lexis Carpet ln(1 + n(t, a)) — Diagonal Missing-Generation Cohort Waves",
                );
                render_lexis_plot(
                    ui,
                    &mut self.lexis,
                    &mut self.lexis_texture,
                    self.solver.config().max_age,
                    self.solver.config().maturation_age,
                    high_contrast,
                );
                ui.add_space(6.0);
                render_secondary_plots(
                    ui,
                    &self.state,
                    &self.solver,
                    &self.lexis,
                    &self.bifurcation_points,
                );
            });
        });
    }

    fn has_finite_parameters(&self) -> bool {
        self.state.validate().is_ok()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.state.clamp_in_place();
        self.rebuild_simulation(false);
        true
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        let mut obs = vec![
            ("adult_population", self.solver.adult_population()),
            ("juvenile_population", self.solver.juvenile_population()),
            ("birth_rate", self.solver.last_birth_rate()),
        ];
        if let Some(eq) = self.solver.diekmann_equilibrium() {
            obs.push(("diekmann_equilibrium", eq));
        }
        obs
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.state.hash_into(&mut h);
        h.finish().max(1)
    }

    fn state_bounds(&self) -> ToolStateBounds {
        crate::url_state::URL_JSON_BOUNDS
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: CannibalismToolState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Cannibalism state: {e}"))?;
        parsed.validate()?;
        let changed = self.state != parsed;
        self.state = parsed;
        if changed {
            self.rebuild_simulation(false);
        }
        Ok(())
    }
}

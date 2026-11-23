#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "sir/atlas.rs"]
pub mod atlas;
mod metadata;
mod metrics;

use crate::async_sim::atlas::ParameterAtlasController;
use crate::framework::{InteractiveTool, TraceSeries, TraceSnapshotManager};
use crate::reflective_ui::AnimatedSlider;
use atlas::{default_sir_atlas_axes, is_default_sir_atlas_axes, SIR_ATLAS_AXIS_PAIRS};
#[cfg(test)]
pub use atlas::{SirCartographer, SirRegime};
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::epidemiology::compartmental::SIRModel;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::hash::{Hash, Hasher};

#[path = "sir/counterfactual.rs"]
pub mod counterfactual;
pub use counterfactual::*;

pub(super) const MIN_SIR_DURATION: f64 = 10.0;
pub(super) const MAX_SIR_DURATION: f64 = 365.0;
const MAX_SIR_HARD_DURATION: f64 = 10_000.0;
const MAX_SIR_STEPS: usize = 100_000;

#[derive(serde::Serialize)]
struct SirStateRef<'a> {
    n: f64,
    i0: f64,
    beta: f64,
    gamma: f64,
    duration: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    fork: Option<SirForkDescriptor>,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot_manager: Option<&'a TraceSnapshotManager>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    atlas_visible: bool,
    #[serde(skip_serializing_if = "is_default_sir_atlas_axes")]
    atlas_axes: [String; 2],
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SirState {
    n: f64,
    i0: f64,
    beta: f64,
    gamma: f64,
    duration: f64,
    #[serde(default)]
    fork: Option<SirForkDescriptor>,
    #[serde(default)]
    snapshot_manager: TraceSnapshotManager,
    #[serde(default)]
    atlas_visible: bool,
    #[serde(default = "default_sir_atlas_axes")]
    atlas_axes: [String; 2],
}

#[allow(missing_docs)]
pub struct SirTool {
    pub(crate) n: f64,
    pub(crate) i0: f64,
    pub(crate) beta: f64,
    pub(crate) gamma: f64,
    pub(crate) duration: f64,
    pub(crate) atlas: ParameterAtlasController,

    // Cached plot data (full-resolution trajectories for metrics and exports)
    pub(crate) s_points: Vec<PlotPoint>,
    pub(crate) i_points: Vec<PlotPoint>,
    pub(crate) r_points: Vec<PlotPoint>,
    // Decimated display buffers capped at MAX_DISPLAY_POLYLINE_POINTS
    pub(crate) display_s: std::sync::Arc<Vec<PlotPoint>>,
    pub(crate) display_i: std::sync::Arc<Vec<PlotPoint>>,
    pub(crate) display_r: std::sync::Arc<Vec<PlotPoint>>,

    pub(crate) snapshot_manager: TraceSnapshotManager,
    pub(super) sensitivity_engine: crate::sensitivity::SensitivityEngine,
    pub(super) last_param_summary: String,
    revision: u64,
    pub cf: SirCounterfactualState,
    series_key: (usize, usize, usize, usize, usize, usize),
    series_finite: bool,
    cached_peak: Option<(f64, f64)>,
}

impl Default for SirTool {
    fn default() -> Self {
        let mut tool = Self {
            n: 1000.0,
            i0: 10.0,
            beta: 0.5,
            gamma: 0.1,
            duration: 100.0,
            atlas: ParameterAtlasController::new(SIR_ATLAS_AXIS_PAIRS.to_vec()),
            s_points: vec![],
            i_points: vec![],
            r_points: vec![],
            display_s: std::sync::Arc::new(Vec::new()),
            display_i: std::sync::Arc::new(Vec::new()),
            display_r: std::sync::Arc::new(Vec::new()),
            snapshot_manager: TraceSnapshotManager::new(),
            sensitivity_engine: crate::sensitivity::SensitivityEngine::new(),
            last_param_summary: String::new(),
            revision: 1,
            cf: SirCounterfactualState::default(),
            series_key: (0, 0, 0, 0, 0, 0),
            series_finite: true,
            cached_peak: None,
        };
        tool.recalculate();
        tool
    }
}

impl SirTool {
    #[allow(missing_docs)]
    pub fn param_summary(&self) -> String {
        format!(
            r"\beta={:.2}, \gamma={:.2}, N={:.0}",
            self.beta, self.gamma, self.n
        )
    }

    fn record_history_and_recalculate(&mut self, sweep_active: bool) {
        if !sweep_active {
            let to_pairs = |pts: &[PlotPoint]| pts.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>();
            let series = vec![
                TraceSeries::new("Susceptible", to_pairs(&self.s_points)),
                TraceSeries::new("Infected", to_pairs(&self.i_points)),
                TraceSeries::new("Recovered", to_pairs(&self.r_points)),
            ];
            self.snapshot_manager
                .push_history_run(&self.last_param_summary, series);
        }
        self.recalculate();
    }

    fn render_parameter_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let bounds = self.theory_parameters();
        let get_b = |k: &str, min: f64, max: f64, step: f64| {
            bounds
                .get(k)
                .cloned()
                .unwrap_or(ParameterConstraint { min, max, step })
        };
        let cn = get_b("N", 100.0, 100_000.0, 100.0);
        let cb = get_b("beta", 0.0, 5.0, 0.02);
        let cg = get_b("gamma", 0.0, 1.0, 0.01);
        let cd = get_b("duration", MIN_SIR_DURATION, MAX_SIR_DURATION, 1.0);

        let mut changed = false;
        changed |= ui
            .add(
                AnimatedSlider::new(&mut self.n, cn.min..=cn.max)
                    .hard_bounds(cn.min..=cn.max)
                    .text("Population (N)")
                    .id_salt("sir_population_n"),
            )
            .changed();
        let max_i0_soft = (self.n * 0.5).max(1.0);
        let max_i0_hard = self.n.max(1.0);
        self.i0 = self.i0.clamp(1.0, max_i0_hard);
        changed |= ui
            .add(
                AnimatedSlider::new(&mut self.i0, 1.0..=max_i0_soft)
                    .hard_bounds(1.0..=max_i0_hard)
                    .text("Initial Infected (I0)")
                    .id_salt("sir_initial_i0"),
            )
            .changed();
        changed |= ui
            .add(
                AnimatedSlider::new(&mut self.beta, cb.min..=cb.max)
                    .hard_bounds(cb.min..=cb.max)
                    .text("Transmission Rate (beta)")
                    .id_salt("sir_beta"),
            )
            .changed();
        changed |= ui
            .add(
                AnimatedSlider::new(&mut self.gamma, cg.min..=cg.max)
                    .hard_bounds(cg.min..=cg.max)
                    .text("Recovery Rate (gamma)")
                    .id_salt("sir_gamma"),
            )
            .changed();
        changed |= ui
            .add(
                AnimatedSlider::new(&mut self.duration, cd.min..=cd.max)
                    .hard_bounds(cd.min..=MAX_SIR_HARD_DURATION)
                    .text("Duration (days)")
                    .id_salt("sir_duration"),
            )
            .changed();
        changed
    }

    pub(crate) fn recalculate(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
        self.s_points.clear();
        self.i_points.clear();
        self.r_points.clear();
        self.display_s = std::sync::Arc::new(Vec::new());
        self.display_i = std::sync::Arc::new(Vec::new());
        self.display_r = std::sync::Arc::new(Vec::new());
        self.series_key = (0, 0, 0, 0, 0, 0);
        self.last_param_summary = self.param_summary();

        if !self.duration.is_finite() || self.duration <= 0.0 {
            self.cf.branch_engine.clear();
            return;
        }
        let end_step = self.effective_end_step();
        if let Some(desc) = self.cf.fork {
            self.recalculate_forked(desc, end_step);
            self.sync_series_cache();
            return;
        }
        self.cf.reset_branch_a_points();
        self.cf.branch_engine.clear();

        // Ensure parameters are valid before passing to model
        let n = self.n.max(1.0);
        let i0 = self.i0.clamp(0.0, n);
        let beta = self.beta.max(0.0);
        let gamma = self.gamma.max(0.0);

        if let Ok(mut model) = SIRModel::new(n, i0, beta, gamma) {
            let dt = 0.1;
            let Some(point_cap) = end_step.checked_add(1) else {
                return;
            };
            if self.s_points.try_reserve(point_cap).is_err()
                || self.i_points.try_reserve(point_cap).is_err()
                || self.r_points.try_reserve(point_cap).is_err()
            {
                return;
            }
            for i in 0..=end_step {
                let t = i as f64 * dt;
                let state = model.state();
                self.s_points.push(PlotPoint::new(t, state.s));
                self.i_points.push(PlotPoint::new(t, state.i));
                self.r_points.push(PlotPoint::new(t, state.r));
                model.step(dt);
            }
        }
        self.sync_series_cache();
    }

    fn load_sir_state(&mut self, state: &str, preserve_snapshots: bool) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let s: SirState =
            serde_json::from_str(state).map_err(|e| format!("Invalid SIR state: {e}"))?;
        if ![s.n, s.i0, s.beta, s.gamma, s.duration]
            .iter()
            .all(|v| v.is_finite())
            || !(100.0..=100_000.0).contains(&s.n)
            || !(1.0..=s.n).contains(&s.i0)
            || !(0.0..=5.0).contains(&s.beta)
            || !(0.0..=1.0).contains(&s.gamma)
            || s.duration < 0.0
        {
            return Err("SIR parameters out of bounds".into());
        }
        if let Some(ref f) = s.fork {
            metrics::validate_sir_fork_descriptor(f, MAX_SIR_STEPS)?;
        }
        self.atlas
            .set_axis_keys(&s.atlas_axes[0], &s.atlas_axes[1])?;
        self.atlas.set_visible(s.atlas_visible);
        self.n = s.n;
        self.i0 = s.i0;
        self.beta = s.beta;
        self.gamma = s.gamma;
        self.duration = s.duration.clamp(MIN_SIR_DURATION, MAX_SIR_DURATION);
        self.cf.is_playing = false;
        self.cf.playhead_step = None;
        self.cf.fork = s.fork;
        if let Some(f) = s.fork {
            self.cf.target_fork_day = f.fork_step as f64 * SIR_STEP_DT;
            self.cf.branch_engine.set_effective_delta(f.delta_x);
        } else {
            self.cf.branch_engine.clear();
        }
        if !preserve_snapshots {
            self.snapshot_manager = s.snapshot_manager;
        }
        self.recalculate();
        Ok(())
    }

    fn hash_shared_parameters(&self, h: &mut impl Hasher) {
        for val in [self.n, self.i0, self.beta, self.gamma, self.duration] {
            val.to_bits().hash(h);
        }
        if let Some(ref f) = self.cf.fork {
            true.hash(h);
            f.fork_step.hash(h);
            f.baseline_n.to_bits().hash(h);
            f.baseline_i0.to_bits().hash(h);
            f.baseline_beta.to_bits().hash(h);
            f.baseline_gamma.to_bits().hash(h);
            f.delta_x.to_bits().hash(h);
        } else {
            false.hash(h);
        }
        self.atlas.is_visible().hash(h);
        self.atlas.axis_keys().hash(h);
    }

    fn serialize_sir_state(&self, include_snapshots: bool) -> Option<String> {
        let (xk, yk) = self.atlas.axis_keys();
        let snapshot_manager = (include_snapshots && !self.snapshot_manager.is_empty())
            .then_some(&self.snapshot_manager);
        serde_json::to_string(&SirStateRef {
            n: self.n,
            i0: self.i0,
            beta: self.beta,
            gamma: self.gamma,
            duration: self.duration,
            fork: self.cf.fork,
            snapshot_manager,
            atlas_visible: self.atlas.is_visible(),
            atlas_axes: [xk.to_string(), yk.to_string()],
        })
        .ok()
    }
}

impl InteractiveTool for SirTool {
    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "SIR Model"
    }
    fn domain(&self) -> &'static str {
        "epidemiology"
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.revision.hash(&mut h);
        self.hash_shared_parameters(&mut h);
        self.snapshot_manager.len().hash(&mut h);
        self.snapshot_manager.history_len().hash(&mut h);
        h.finish().max(1)
    }

    fn url_state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.hash_shared_parameters(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        let scalars_ok = [self.n, self.i0, self.beta, self.gamma, self.duration]
            .into_iter()
            .all(f64::is_finite)
            && self.n > 0.0
            && self.duration > 0.0;
        if !scalars_ok {
            return false;
        }
        if self.series_key == self.current_series_key() {
            self.series_finite
        } else {
            self.s_points
                .iter()
                .all(|p| p.x.is_finite() && p.y.is_finite())
                && self
                    .i_points
                    .iter()
                    .all(|p| p.x.is_finite() && p.y.is_finite())
                && self
                    .r_points
                    .iter()
                    .all(|p| p.x.is_finite() && p.y.is_finite())
        }
    }

    fn counterfactual_snapshot(&self) -> Option<crate::framework::DivergenceSnapshot> {
        self.cf.branch_engine.active_snapshot.clone()
    }

    fn apply_counterfactual_action(
        &mut self,
        action: crate::framework::BranchControlAction,
    ) -> bool {
        self.apply_sir_branch_action(action)
    }

    fn step_simulation(&mut self) {
        self.step_lockstep(5);
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.collect_sir_export_datasets()
    }

    fn save_state(&self) -> Option<String> {
        self.serialize_sir_state(true)
    }

    fn save_recovery_state(&self) -> Option<String> {
        self.serialize_sir_state(false)
    }

    fn try_load_recovery_state(&mut self, state: &str) -> Result<(), String> {
        self.load_sir_state(state, true)
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.load_sir_state(state, false)
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.sync_series_cache();
            ui.vertical(|ui| {
                ui.heading("Parameters");
                let sweeps_before = crate::reflective_ui::slider::live_active_sweep_count(ui.ctx());
                let changed = self.render_parameter_controls(ui);
                let sweep_active =
                    crate::reflective_ui::slider::live_active_sweep_count(ui.ctx()) > sweeps_before;
                if changed {
                    self.record_history_and_recalculate(sweep_active);
                }
                metrics::render_epidemic_metrics(ui, self);
                self.render_sir_readouts_and_atlas(ui);
                self.render_counterfactual_bar(ui);
                ui.separator();
                metrics::render_simulation_plot(ui, self);
                ui.add_space(6.0);
                self.render_snapshot_bar(ui);
                self.evaluate_sensitivity(ui);
            });
        });
    }
}

// [cite:graph_parameters_rust]

#[cfg(test)]
mod tests;

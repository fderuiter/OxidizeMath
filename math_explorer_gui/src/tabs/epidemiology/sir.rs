mod metrics;

use crate::framework::{InteractiveTool, TraceSeries, TraceSnapshotManager};
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::epidemiology::compartmental::SIRModel;

#[path = "sir/counterfactual.rs"]
pub mod counterfactual;
#[path = "sir/presets.rs"]
mod presets;
pub use counterfactual::*;

const MIN_SIR_DURATION: f64 = 10.0;
const MAX_SIR_DURATION: f64 = 365.0;
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
    #[serde(skip_serializing_if = "TraceSnapshotManager::is_empty")]
    snapshot_manager: &'a TraceSnapshotManager,
}

#[derive(serde::Serialize)]
struct SirRecoveryState {
    n: f64,
    i0: f64,
    beta: f64,
    gamma: f64,
    duration: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    fork: Option<SirForkDescriptor>,
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
}

pub struct SirTool {
    pub(crate) n: f64,
    pub(crate) i0: f64,
    pub(crate) beta: f64,
    pub(crate) gamma: f64,
    pub(crate) duration: f64,

    // Cached plot data (full-resolution trajectories for metrics and exports)
    pub(crate) s_points: Vec<PlotPoint>,
    pub(crate) i_points: Vec<PlotPoint>,
    pub(crate) r_points: Vec<PlotPoint>,
    // Decimated display buffers capped at MAX_DISPLAY_POLYLINE_POINTS
    pub(crate) display_s: std::sync::Arc<Vec<PlotPoint>>,
    pub(crate) display_i: std::sync::Arc<Vec<PlotPoint>>,
    pub(crate) display_r: std::sync::Arc<Vec<PlotPoint>>,

    pub(crate) snapshot_manager: TraceSnapshotManager,
    sensitivity_engine: crate::sensitivity::SensitivityEngine,
    last_param_summary: String,
    revision: u64,
    pub cf: SirCounterfactualState,
}

impl Default for SirTool {
    fn default() -> Self {
        let mut tool = Self {
            n: 1000.0,
            i0: 10.0,
            beta: 0.5,
            gamma: 0.1,
            duration: 100.0,
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
        };
        tool.recalculate();
        tool
    }
}

impl SirTool {
    pub fn param_summary(&self) -> String {
        format!(
            r"\beta={:.2}, \gamma={:.2}, N={:.0}",
            self.beta, self.gamma, self.n
        )
    }

    fn render_parameter_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        changed |= ui
            .add(egui::Slider::new(&mut self.n, 100.0..=100_000.0).text("Population (N)"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.i0, 1.0..=self.n / 2.0).text("Initial Infected (I0)"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.beta, 0.0..=5.0).text("Transmission Rate (beta)"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.gamma, 0.0..=1.0).text("Recovery Rate (gamma)"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.duration, 10.0..=365.0).text("Duration (days)"))
            .changed();
        changed
    }

    fn evaluate_sensitivity(&mut self, ui: &mut egui::Ui) {
        let params = [
            ("N", self.n),
            ("I0", self.i0),
            ("beta", self.beta),
            ("gamma", self.gamma),
        ];
        let duration_bits = crate::sensitivity::canonical_f64_bits(self.duration);

        self.sensitivity_engine
            .compute_with_context(&params, &duration_bits, |p_map| {
                let n = p_map.get("N").copied().unwrap_or(1000.0).max(1.0);
                let i0 = p_map.get("I0").copied().unwrap_or(10.0).clamp(0.0, n);
                let beta = p_map.get("beta").copied().unwrap_or(0.5).max(0.0);
                let gamma = p_map.get("gamma").copied().unwrap_or(0.1).max(0.0001);

                if let Ok(mut model) = SIRModel::new(n, i0, beta, gamma) {
                    let dt = 0.2;
                    let steps = 250;
                    let mut max_i: f64 = 0.0;
                    for _ in 0..steps {
                        max_i = max_i.max(model.state().i);
                        model.step(dt);
                    }
                    max_i
                } else {
                    0.0
                }
            });

        crate::sensitivity::render_sensitivity_panel_ui(ui, &self.sensitivity_engine, "sir_tool");
    }

    pub(crate) fn recalculate(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
        self.s_points.clear();
        self.i_points.clear();
        self.r_points.clear();
        self.display_s = std::sync::Arc::new(Vec::new());
        self.display_i = std::sync::Arc::new(Vec::new());
        self.display_r = std::sync::Arc::new(Vec::new());
        self.last_param_summary = self.param_summary();

        if !self.duration.is_finite() || self.duration <= 0.0 {
            self.cf.branch_engine.clear();
            return;
        }

        let end_step = self.effective_end_step();
        if let Some(desc) = self.cf.fork {
            self.recalculate_forked(desc, end_step);
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
            self.refresh_display_polylines();
        }
    }

    fn load_sir_state(&mut self, state: &str, preserve_snapshots: bool) -> Result<(), String> {
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
            if f.fork_step == 0
                || f.fork_step > MAX_SIR_STEPS
                || ![f.baseline_n, f.baseline_i0, f.baseline_beta, f.baseline_gamma, f.delta_x]
                    .iter()
                    .all(|v| v.is_finite())
                || !(100.0..=100_000.0).contains(&f.baseline_n)
                || !(1.0..=f.baseline_n).contains(&f.baseline_i0)
                || !(0.0..=5.0).contains(&f.baseline_beta)
                || !(0.0..=1.0).contains(&f.baseline_gamma)
            {
                return Err("SIR fork descriptor out of bounds".into());
            }
        }
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
}

impl InteractiveTool for SirTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "SIR Model"
    }
    fn domain(&self) -> &'static str {
        "epidemiology"
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.revision.hash(&mut h);
        self.n.to_bits().hash(&mut h);
        self.i0.to_bits().hash(&mut h);
        self.beta.to_bits().hash(&mut h);
        self.gamma.to_bits().hash(&mut h);
        self.duration.to_bits().hash(&mut h);
        self.snapshot_manager.len().hash(&mut h);
        self.snapshot_manager.history_len().hash(&mut h);
        if let Some(ref f) = self.cf.fork {
            true.hash(&mut h);
            f.fork_step.hash(&mut h);
            f.baseline_beta.to_bits().hash(&mut h);
            f.baseline_gamma.to_bits().hash(&mut h);
            f.delta_x.to_bits().hash(&mut h);
        } else {
            false.hash(&mut h);
        }
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        [self.n, self.i0, self.beta, self.gamma, self.duration]
            .into_iter()
            .all(f64::is_finite)
            && self.n > 0.0
            && self.duration > 0.0
            && self
                .s_points
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

    fn counterfactual_snapshot(&self) -> Option<crate::framework::DivergenceSnapshot> {
        self.cf.branch_engine.active_snapshot.clone()
    }

    fn apply_counterfactual_action(&mut self, action: crate::framework::BranchControlAction) -> bool {
        self.apply_sir_branch_action(action)
    }

    fn step_simulation(&mut self) {
        self.step_lockstep(5);
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut datasets = Vec::new();
        if !self.s_points.is_empty() {
            let to_pairs = |pts: &[PlotPoint]| pts.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>();
            if self.cf.fork.is_some() {
                datasets.push(("Susceptible (Branch B)".to_string(), to_pairs(&self.s_points)));
                datasets.push(("Infected (Branch B)".to_string(), to_pairs(&self.i_points)));
                datasets.push(("Recovered (Branch B)".to_string(), to_pairs(&self.r_points)));
                datasets.push(("Susceptible (Branch A)".to_string(), to_pairs(&self.cf.s_points_a)));
                datasets.push(("Infected (Branch A)".to_string(), to_pairs(&self.cf.i_points_a)));
                datasets.push(("Recovered (Branch A)".to_string(), to_pairs(&self.cf.r_points_a)));
            } else {
                datasets.push(("Susceptible".to_string(), to_pairs(&self.s_points)));
                datasets.push(("Infected".to_string(), to_pairs(&self.i_points)));
                datasets.push(("Recovered".to_string(), to_pairs(&self.r_points)));
            }
        }
        datasets.extend(self.snapshot_manager.export_datasets());
        for (name, pts) in egui_plot::get_last_plot_datasets() {
            if !datasets.iter().any(|(existing, _)| existing == &name) {
                datasets.push((name, pts));
            }
        }
        datasets
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&SirStateRef {
            n: self.n,
            i0: self.i0,
            beta: self.beta,
            gamma: self.gamma,
            duration: self.duration,
            fork: self.cf.fork,
            snapshot_manager: &self.snapshot_manager,
        })
        .ok()
    }

    fn save_recovery_state(&self) -> Option<String> {
        serde_json::to_string(&SirRecoveryState {
            n: self.n,
            i0: self.i0,
            beta: self.beta,
            gamma: self.gamma,
            duration: self.duration,
            fork: self.cf.fork,
        })
        .ok()
    }

    fn try_load_recovery_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        self.load_sir_state(state, true)
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.load_sir_state(state, false)
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        eframe::egui::CentralPanel::default().show_inside(ui, |ui| {
            self.show_ui(ui);
        });
    }

    fn show_ui(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            ui.heading("Parameters");
            if self.render_parameter_controls(ui) {
                let to_pairs =
                    |pts: &[PlotPoint]| pts.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>();
                let series = vec![
                    TraceSeries::new("Susceptible", to_pairs(&self.s_points)),
                    TraceSeries::new("Infected", to_pairs(&self.i_points)),
                    TraceSeries::new("Recovered", to_pairs(&self.r_points)),
                ];
                self.snapshot_manager
                    .push_history_run(&self.last_param_summary, series);
                self.recalculate();
            }

            metrics::render_epidemic_metrics(ui, self);
            self.render_counterfactual_bar(ui);

            ui.add_space(6.0);
            self.render_snapshot_bar(ui);
            self.evaluate_sensitivity(ui);

            ui.separator();
            metrics::render_simulation_plot(ui, self);
        });
    }
}

impl SirTool {
    fn render_snapshot_bar(&mut self, ui: &mut egui::Ui) {
        let param_summary = self.param_summary();
        let current_len = self.snapshot_manager.len();
        let s_points = &self.s_points;
        let i_points = &self.i_points;
        let r_points = &self.r_points;
        if self.snapshot_manager.render_ui(ui, move || {
            let label = format!("Baseline {}", current_len + 1);
            let to_pairs = |pts: &[PlotPoint]| pts.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>();
            let series = vec![
                TraceSeries::new("Susceptible", to_pairs(s_points)),
                TraceSeries::new("Infected", to_pairs(i_points)),
                TraceSeries::new("Recovered", to_pairs(r_points)),
            ];
            Some((label, param_summary, series))
        }) {
            self.revision = self.revision.wrapping_add(1).max(1);
        }
    }
}

// [cite:graph_parameters_rust]

#[cfg(test)]
mod tests;


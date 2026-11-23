#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "sir/atlas.rs"]
pub mod atlas;
mod metrics;

use crate::async_sim::atlas::ParameterAtlasController;
use crate::framework::{InteractiveTool, TraceSeries, TraceSnapshotManager};
use atlas::{default_sir_atlas_axes, is_default_sir_atlas_axes, SIR_ATLAS_AXIS_PAIRS};
#[cfg(test)]
pub use atlas::{SirCartographer, SirRegime};
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::epidemiology::compartmental::SIRModel;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

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
    #[serde(skip_serializing_if = "TraceSnapshotManager::is_empty")]
    snapshot_manager: &'a TraceSnapshotManager,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    atlas_visible: bool,
    #[serde(skip_serializing_if = "is_default_sir_atlas_axes")]
    atlas_axes: [String; 2],
}

#[derive(serde::Serialize)]
struct SirRecoveryState {
    n: f64,
    i0: f64,
    beta: f64,
    gamma: f64,
    duration: f64,
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
    sensitivity_engine: crate::sensitivity::SensitivityEngine,
    pub(super) last_param_summary: String,
    revision: u64,
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

    fn record_history_and_recalculate(&mut self) {
        let to_pairs = |pts: &[PlotPoint]| pts.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>();
        let series = vec![
            TraceSeries::new("Susceptible", to_pairs(&self.s_points)),
            TraceSeries::new("Infected", to_pairs(&self.i_points)),
            TraceSeries::new("Recovered", to_pairs(&self.r_points)),
        ];
        self.snapshot_manager
            .push_history_run(&self.last_param_summary, series);
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
            .add(egui::Slider::new(&mut self.n, cn.min..=cn.max).text("Population (N)"))
            .changed();
        let i0_max = (self.n * 0.5).max(1.0);
        self.i0 = self.i0.min(i0_max);
        changed |= ui
            .add(egui::Slider::new(&mut self.i0, 1.0..=i0_max).text("Initial Infected (I0)"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.beta, cb.min..=cb.max).text("Transmission Rate (beta)"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.gamma, cg.min..=cg.max).text("Recovery Rate (gamma)"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.duration, cd.min..=cd.max).text("Duration (days)"))
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
            return;
        }
        let n = self.n.max(1.0);
        let i0 = self.i0.clamp(0.0, n);
        let beta = self.beta.max(0.0);
        let gamma = self.gamma.max(0.0);

        if let Ok(mut model) = SIRModel::new(n, i0, beta, gamma) {
            let dt = 0.1;
            let raw_steps = self.duration / dt;
            if !raw_steps.is_finite() || raw_steps < 0.0 {
                return;
            }
            let steps = raw_steps.min(MAX_SIR_STEPS as f64) as usize;
            let Some(point_cap) = steps.checked_add(1) else {
                return;
            };
            if self.s_points.try_reserve(point_cap).is_err()
                || self.i_points.try_reserve(point_cap).is_err()
                || self.r_points.try_reserve(point_cap).is_err()
            {
                return;
            }
            for i in 0..=steps {
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
        self.atlas
            .set_axis_keys(&s.atlas_axes[0], &s.atlas_axes[1])?;
        self.atlas.set_visible(s.atlas_visible);
        self.n = s.n;
        self.i0 = s.i0;
        self.beta = s.beta;
        self.gamma = s.gamma;
        self.duration = s.duration.clamp(MIN_SIR_DURATION, MAX_SIR_DURATION);
        if !preserve_snapshots {
            self.snapshot_manager = s.snapshot_manager;
        }
        self.recalculate();
        Ok(())
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
        self.n.to_bits().hash(&mut h);
        self.i0.to_bits().hash(&mut h);
        self.beta.to_bits().hash(&mut h);
        self.gamma.to_bits().hash(&mut h);
        self.duration.to_bits().hash(&mut h);
        self.snapshot_manager.len().hash(&mut h);
        self.snapshot_manager.history_len().hash(&mut h);
        self.atlas.is_visible().hash(&mut h);
        self.atlas.axis_keys().hash(&mut h);
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

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut datasets = Vec::new();
        if !self.s_points.is_empty() {
            let to_pairs = |pts: &[PlotPoint]| pts.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>();
            datasets.push(("Susceptible".to_string(), to_pairs(&self.s_points)));
            datasets.push(("Infected".to_string(), to_pairs(&self.i_points)));
            datasets.push(("Recovered".to_string(), to_pairs(&self.r_points)));
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
        let (xk, yk) = self.atlas.axis_keys();
        serde_json::to_string(&SirStateRef {
            n: self.n,
            i0: self.i0,
            beta: self.beta,
            gamma: self.gamma,
            duration: self.duration,
            snapshot_manager: &self.snapshot_manager,
            atlas_visible: self.atlas.is_visible(),
            atlas_axes: [xk.to_string(), yk.to_string()],
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
        })
        .ok()
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
                if self.render_parameter_controls(ui) {
                    self.record_history_and_recalculate();
                }
                metrics::render_epidemic_metrics(ui, self);
                self.render_sir_readouts_and_atlas(ui);
                ui.separator();
                metrics::render_simulation_plot(ui, self);
                ui.add_space(8.0);
                let param_summary = self.param_summary();
                let current_len = self.snapshot_manager.len();
                let (s_pts, i_pts, r_pts) = (&self.s_points, &self.i_points, &self.r_points);
                if self.snapshot_manager.render_ui(ui, move || {
                    let label = format!("Baseline {}", current_len + 1);
                    let to_pairs =
                        |pts: &[PlotPoint]| pts.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>();
                    let series = vec![
                        TraceSeries::new("Susceptible", to_pairs(s_pts)),
                        TraceSeries::new("Infected", to_pairs(i_pts)),
                        TraceSeries::new("Recovered", to_pairs(r_pts)),
                    ];
                    Some((label, param_summary, series))
                }) {
                    self.revision = self.revision.wrapping_add(1).max(1);
                }
                self.evaluate_sensitivity(ui);
            });
        });
    }
}

// [cite:graph_parameters_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "SirTool",
        display_name: "SIR Model",
        domain: "epidemiology",
        tags: &[],
        build: || Box::new(SirTool::default()),
    }
}

impl TheoryDescribable for SirTool {
    fn theory_description(&self) -> String {
        let r0 = self.beta / self.gamma;
        format!(
            "Kermack-McKendrick SIR compartmental epidemic model: $\\frac{{dS}}{{dt}} = -\\frac{{\\beta S I}}{{N}}$, $\\frac{{dI}}{{dt}} = \\frac{{\\beta S I}}{{N}} - \\gamma I$, $\\frac{{dR}}{{dt}} = \\gamma I$. Transmission rate $\\beta = {:.3}$, recovery rate $\\gamma = {:.3}$, basic reproduction number $R_0 = {:.2}$.",
            self.beta, self.gamma, r0
        )
    }
    fn phonetic_description(&self) -> String {
        "Kermack-McKendrick SIR compartmental epidemic model theory".into()
    }

    fn theory_citation(&self) -> String {
        "A Contribution to the Mathematical Theory of Epidemics (Kermack & McKendrick, 1927)".into()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        [
            ("N", 100.0, 100_000.0, 100.0),
            ("I0", 1.0, 50_000.0, 1.0),
            ("beta", 0.0, 5.0, 0.02),
            ("gamma", 0.0, 1.0, 0.01),
            ("duration", MIN_SIR_DURATION, MAX_SIR_DURATION, 1.0),
        ]
        .into_iter()
        .map(|(k, min, max, step)| (k.to_string(), ParameterConstraint { min, max, step }))
        .collect()
    }
}

#[cfg(test)]
mod tests;

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": { "schema_version": "1.0", "title": "Kermack-McKendrick SIR Model", "domain": "epidemiology", "timestamp": "2026-11-16T00:00:00Z" },
  "config": { "n": 10000.0, "i0": 25.0, "beta": 0.3, "gamma": 0.1, "duration": 120.0 }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "sir_starter",
        title: "Kermack-McKendrick SIR Model",
        description: "dS/dt = -βSI/N, dI/dt = βSI/N - γI: Compartmental epidemic outbreak dynamics and herd immunity.",
        category: "Compartmental Models",
        icon: "🦠",
        domain: "epidemiology",
        tool_name: Some("SirTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

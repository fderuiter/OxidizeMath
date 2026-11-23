#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Cooperative frame-budgeted sensitivity, tornado, and Monte Carlo hooks for [`super::InteractiveTool`].

use eframe::egui;

use super::{scoped_panel_id, InteractiveTool, PaneSidePanel, SimulationFramework};
use crate::accessibility::AccessibleHoverText;
use crate::sensitivity::{
    render_labeled_sensitivity_panel_ui, ParameterUncertainty, SensitivityEngine, TornadoBounds,
    DEFAULT_MONTE_CARLO_SAMPLES, DEFAULT_MONTE_CARLO_SEED, MAX_SENSITIVITY_BATCH_EVALS,
};

pub(super) fn sample_tool_animation_frames<T: InteractiveTool + ?Sized>(
    tool: &mut T,
    num_frames: usize,
) -> Vec<egui_plot::export::DatasetFrame> {
    let count = num_frames.max(1);
    let mut frames = Vec::with_capacity(count);
    for _ in 0..count {
        tool.step_simulation();
        let ds = tool.export_datasets();
        if !ds.is_empty() {
            frames.push(ds);
        }
    }
    if frames.is_empty() {
        let ds = tool.export_datasets();
        if !ds.is_empty() {
            frames = egui_plot::export::sample_progressive_frames(&ds, count);
        }
    }
    frames
}

/// Metadata describing a single scalar parameter exposed to cooperative sensitivity analysis.
#[derive(Clone, Debug, PartialEq)]
pub struct SensitivityParamSpec {
    /// Canonical parameter key passed to [`InteractiveTool::evaluate_sensitivity_observable`].
    pub name: &'static str,
    /// Human-readable display label rendered on sensitivity cards.
    pub label: &'static str,
    /// Current baseline value of the parameter.
    pub value: f64,
    /// Lower physical bound used for wide-range tornado excursions and Monte Carlo clamping.
    pub min: f64,
    /// Upper physical bound used for wide-range tornado excursions and Monte Carlo clamping.
    pub max: f64,
    /// Gaussian perturbation standard deviation `σ` for Monte Carlo uncertainty propagation.
    pub sigma: f64,
}

impl SensitivityParamSpec {
    /// Builds a parameter specification with default `σ = 5%` of `(max - min)`.
    #[must_use]
    pub fn new(name: &'static str, label: &'static str, value: f64, min: f64, max: f64) -> Self {
        let sigma = ((max - min).abs() * 0.05).max(1.0e-6);
        Self {
            name,
            label,
            value,
            min,
            max,
            sigma,
        }
    }

    /// Overrides the Gaussian perturbation standard deviation `σ`.
    #[must_use]
    pub const fn with_sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }
}

/// Framework-managed cooperative sensitivity & uncertainty state (`<= 8` model evaluations/frame).
#[derive(Clone, Debug)]
pub struct CooperativeSensitivityState {
    /// Underlying progressive finite-difference, tornado, and Monte Carlo engine.
    pub engine: SensitivityEngine,
    /// Active tool identifier whose cache is currently loaded in `engine`.
    pub cached_tool_name: Option<&'static str>,
    /// Cached observable metric label from the active tool.
    pub cached_label: Option<&'static str>,
    /// Whether cooperative tornado & Monte Carlo sweeps are enabled after local elasticity converges.
    pub uncertainty_enabled: bool,
    /// Explicit `u64` `SplitMix64` seed for deterministic Monte Carlo propagation.
    pub seed: u64,
    /// Target Monte Carlo sample count.
    pub target_samples: usize,
}

impl Default for CooperativeSensitivityState {
    fn default() -> Self {
        Self {
            engine: SensitivityEngine::new(),
            cached_tool_name: None,
            cached_label: None,
            uncertainty_enabled: false,
            seed: DEFAULT_MONTE_CARLO_SEED,
            target_samples: DEFAULT_MONTE_CARLO_SAMPLES,
        }
    }
}

fn eval_tool_from_slice(
    tool: &dyn InteractiveTool,
    specs: &[SensitivityParamSpec],
    vals: &[f64],
) -> Option<(&'static str, f64)> {
    const STACK_CAP: usize = 16;
    if specs.len() <= STACK_CAP {
        let mut buf: [(&str, f64); STACK_CAP] = [("", 0.0); STACK_CAP];
        let n = specs.len().min(vals.len());
        for i in 0..n {
            buf[i] = (specs[i].name, vals[i]);
        }
        tool.evaluate_sensitivity_observable(&buf[..n])
    } else {
        let pairs: Vec<(&str, f64)> = specs
            .iter()
            .zip(vals.iter())
            .map(|(s, &v)| (s.name, v))
            .collect();
        tool.evaluate_sensitivity_observable(&pairs)
    }
}

impl CooperativeSensitivityState {
    /// Advances at most one cooperative batch (`<= 8` model evaluations) against `tool`.
    pub fn step_for_tool(&mut self, tool: &dyn InteractiveTool) -> Option<&'static str> {
        let specs = tool.sensitivity_parameters();
        if specs.is_empty() {
            self.cached_label = None;
            return None;
        }
        if self.cached_tool_name != Some(tool.name()) {
            self.engine = SensitivityEngine::new();
            self.cached_tool_name = Some(tool.name());
            self.cached_label = None;
        }
        let pairs: Vec<(&str, f64)> = specs.iter().map(|s| (s.name, s.value)).collect();
        let is_hit = self.engine.matches_cache(&pairs, &());
        let mut observed_label = if is_hit { self.cached_label } else { None };

        if !is_hit || !self.engine.is_converged() {
            self.engine.compute_slice(&pairs, |vals| {
                if let Some((lbl, val)) = eval_tool_from_slice(tool, &specs, vals) {
                    observed_label = Some(lbl);
                    val
                } else {
                    0.0
                }
            });
            self.cached_label = observed_label;
            if observed_label.is_some() {
                for (metric, spec) in self.engine.metrics.iter_mut().zip(&specs) {
                    if metric.param_name != spec.label {
                        metric.param_name = spec.label.to_owned();
                    }
                }
            }
            return self.cached_label;
        }

        if self.uncertainty_enabled && self.cached_label.is_some() {
            self.step_uncertainty_slices(tool, &specs, &pairs);
        }
        self.cached_label
    }

    fn step_uncertainty_slices(
        &mut self,
        tool: &dyn InteractiveTool,
        specs: &[SensitivityParamSpec],
        pairs: &[(&str, f64)],
    ) {
        if !self.engine.is_tornado_converged() {
            let bounds: Vec<TornadoBounds> = specs
                .iter()
                .map(|s| TornadoBounds::new(s.min, s.max))
                .collect();
            self.engine
                .step_tornado_slice(pairs, &bounds, MAX_SENSITIVITY_BATCH_EVALS, |vals| {
                    eval_tool_from_slice(tool, specs, vals).map_or(0.0, |(_, v)| v)
                });
        } else if !self.engine.is_monte_carlo_converged() {
            let uncertainties: Vec<ParameterUncertainty> = specs
                .iter()
                .map(|s| ParameterUncertainty::new(s.sigma, s.min, s.max))
                .collect();
            self.engine.step_monte_carlo_slice(
                pairs,
                &uncertainties,
                self.seed,
                self.target_samples,
                MAX_SENSITIVITY_BATCH_EVALS,
                |vals| eval_tool_from_slice(tool, specs, vals).map_or(0.0, |(_, v)| v),
            );
        }
    }

    /// Renders uncertainty controls and the labeled sensitivity & perturbation panel.
    pub fn render_ui(&mut self, ui: &mut egui::Ui, id_source: &str, metric_label: &str) {
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(
                &mut self.uncertainty_enabled,
                "Tornado & Monte Carlo Sweep",
            )
            .accessible_hover_text(
                "Cooperatively step wide-range tornado excursions and SplitMix64 Monte Carlo quantiles (<= 8 evals/frame)",
            );
            if self.uncertainty_enabled {
                ui.label("Seed:");
                ui.add(egui::DragValue::new(&mut self.seed).speed(1.0))
                    .accessible_hover_text("Explicit u64 SplitMix64 PRNG seed");
            }
        });
        if self.uncertainty_enabled
            && (!self.engine.is_tornado_converged() || !self.engine.is_monte_carlo_converged())
        {
            ui.ctx().request_repaint();
        }
        render_labeled_sensitivity_panel_ui(ui, &self.engine, id_source, metric_label);
    }
}

impl SimulationFramework {
    /// Returns `true` if the active tool exposes cooperative sensitivity parameters.
    #[must_use]
    pub fn active_tool_supports_sensitivity(&self) -> bool {
        self.active_tool
            .as_ref()
            .is_some_and(|t| !t.sensitivity_parameters().is_empty())
    }

    /// Advances one cooperative sensitivity batch (`<= 8` model evaluations) for the active tool.
    pub fn step_active_tool_sensitivity(&mut self) -> Option<&'static str> {
        let tool = self.active_tool.as_deref()?;
        self.cooperative_sensitivity.step_for_tool(tool)
    }

    pub(super) fn render_sensitivity_bar_toggle(&mut self, ui: &mut egui::Ui) {
        if !self.active_tool_supports_sensitivity() {
            return;
        }
        if ui
            .selectable_label(self.show_sensitivity_panel, "📊 Sensitivity")
            .accessible_hover_text(
                "Toggle cooperative Sensitivity, Tornado & Monte Carlo uncertainty panel",
            )
            .clicked()
        {
            self.show_sensitivity_panel = !self.show_sensitivity_panel;
        }
    }

    pub(super) fn show_cooperative_sensitivity_panel(
        &mut self,
        ui: &mut egui::Ui,
        id_source: &str,
    ) {
        if !self.show_sensitivity_panel || self.is_diverged {
            return;
        }
        let Some(label) = self.step_active_tool_sensitivity() else {
            return;
        };
        let tool_name = self.active_tool.as_ref().map_or("tool", |t| t.name());
        let panel_id = scoped_panel_id(ui, format!("{id_source}_cooperative_sensitivity"));
        PaneSidePanel::right(panel_id)
            .resizable(true)
            .default_width(300.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.cooperative_sensitivity.render_ui(ui, tool_name, label);
                });
            });
    }
}

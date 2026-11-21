#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Parameter sensitivity and perturbation analysis panel module.
//!
//! Provides central finite difference perturbation analysis (+/- 1% delta)
//! and normalized sensitivity elasticity metrics for active parameters.

use eframe::egui;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

/// Normalizes floating-point values to canonical IEEE-754 bit patterns
/// (`-0.0` maps to `+0.0` and all `NaN` payloads map to a single quiet `NaN`).
#[must_use]
pub fn canonical_f64_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0_f64.to_bits()
    } else if value.is_nan() {
        f64::NAN.to_bits()
    } else {
        value.to_bits()
    }
}

fn hash_context<C: Hash>(context: &C) -> u64 {
    let mut hasher = DefaultHasher::new();
    context.hash(&mut hasher);
    hasher.finish()
}

fn set_map_value(map: &mut HashMap<String, f64>, key: &str, value: f64) {
    if let Some(slot) = map.get_mut(key) {
        *slot = value;
    } else {
        map.insert(key.to_string(), value);
    }
}

/// Quantitative sensitivity metrics for a single input parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct SensitivityMetric {
    /// Parameter identifier name.
    pub param_name: String,
    /// Baseline value of parameter.
    pub baseline_value: f64,
    /// Approximated partial derivative dY/dP using central finite differences.
    pub partial_derivative: f64,
    /// Normalized sensitivity index (elasticity = |(dY/dP) * (P / Y)|).
    pub elasticity: f64,
    /// Flag indicating if elasticity exceeds the high-sensitivity threshold (> 2.0).
    pub is_high_sensitivity: bool,
}

/// Sensitivity analysis engine for parameter perturbation calculations.
#[derive(Debug, Clone, Default)]
pub struct SensitivityEngine {
    /// Cached calculated sensitivity metrics.
    pub metrics: Vec<SensitivityMetric>,
    /// Last calculated baseline model output Y.
    pub baseline_output: f64,
    /// Current batch index for round-robin parameter evaluation bounding (max 5 sample runs).
    pub batch_offset: usize,
    cached_params: Vec<(String, u64)>,
    cached_context_hash: Option<u64>,
    evaluated_count: usize,
    baseline_map: HashMap<String, f64>,
}

impl SensitivityEngine {
    /// Creates a new SensitivityEngine instance.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` when all active parameters have been evaluated for the current inputs.
    #[must_use]
    pub fn is_converged(&self) -> bool {
        self.evaluated_count >= self.cached_params.len()
    }

    /// Returns `true` when `params` and `context` match the memoized parameter and context state.
    #[must_use]
    pub fn matches_cache<K: AsRef<str>, C: Hash>(&self, params: &[(K, f64)], context: &C) -> bool {
        self.cached_context_hash == Some(hash_context(context))
            && self.schema_matches(params)
            && self.values_match(params)
    }

    fn schema_matches<K: AsRef<str>>(&self, params: &[(K, f64)]) -> bool {
        self.cached_params.len() == params.len()
            && self
                .cached_params
                .iter()
                .zip(params)
                .all(|((cached_key, _), (key, _))| cached_key == key.as_ref())
    }

    fn values_match<K: AsRef<str>>(&self, params: &[(K, f64)]) -> bool {
        self.cached_params
            .iter()
            .zip(params)
            .all(|((_, cached_bits), (_, val))| *cached_bits == canonical_f64_bits(*val))
    }

    fn reset_empty(&mut self) {
        self.metrics.clear();
        self.cached_params.clear();
        self.baseline_map.clear();
        self.cached_context_hash = None;
        self.evaluated_count = 0;
        self.batch_offset = 0;
    }

    fn sync_cache_on_miss<K: AsRef<str>>(&mut self, params: &[(K, f64)], context_hash: u64) {
        if self.schema_matches(params) {
            for (idx, (key, val)) in params.iter().enumerate() {
                self.cached_params[idx].1 = canonical_f64_bits(*val);
                set_map_value(&mut self.baseline_map, key.as_ref(), *val);
                self.metrics[idx].baseline_value = *val;
            }
        } else {
            self.batch_offset = 0;
            self.cached_params.clear();
            self.baseline_map.clear();
            self.metrics.clear();
            self.cached_params.reserve(params.len());
            self.baseline_map.reserve(params.len());
            self.metrics.reserve(params.len());
            for (key, val) in params {
                let name = key.as_ref();
                self.cached_params
                    .push((name.to_string(), canonical_f64_bits(*val)));
                self.baseline_map.insert(name.to_string(), *val);
                self.metrics.push(SensitivityMetric {
                    param_name: name.to_string(),
                    baseline_value: *val,
                    partial_derivative: 0.0,
                    elasticity: 0.0,
                    is_high_sensitivity: false,
                });
            }
        }
        self.cached_context_hash = Some(context_hash);
        self.evaluated_count = 0;
    }

    /// Evaluates normalized sensitivity indices using central finite differences (+/- 1% delta).
    ///
    /// Respects the constraint bounding perturbation evaluations to at most 5 sample runs
    /// (1 baseline + max 2 parameters * 2 perturbations per step = 5 runs max) to prevent UI slowdown.
    pub fn compute<K, F>(&mut self, params: &[(K, f64)], eval_fn: F) -> &Vec<SensitivityMetric>
    where
        K: AsRef<str>,
        F: FnMut(&HashMap<String, f64>) -> f64,
    {
        self.compute_with_context(params, &(), eval_fn)
    }

    /// Evaluates normalized sensitivity indices memoized by canonical parameter bits and `context`.
    pub fn compute_with_context<K, C, F>(
        &mut self,
        params: &[(K, f64)],
        context: &C,
        mut eval_fn: F,
    ) -> &Vec<SensitivityMetric>
    where
        K: AsRef<str>,
        C: Hash,
        F: FnMut(&HashMap<String, f64>) -> f64,
    {
        if params.is_empty() {
            self.reset_empty();
            return &self.metrics;
        }

        let context_hash = hash_context(context);
        let is_hit = self.cached_context_hash == Some(context_hash)
            && self.schema_matches(params)
            && self.values_match(params);

        if is_hit {
            if self.evaluated_count >= params.len() {
                return &self.metrics;
            }
        } else {
            self.sync_cache_on_miss(params, context_hash);
            self.baseline_output = eval_fn(&self.baseline_map);
        }

        let num_params = params.len();
        let batch_count = (num_params - self.evaluated_count).min(2);
        let y0 = self.baseline_output;

        for step in 0..batch_count {
            let idx = (self.batch_offset + step) % num_params;
            let (ref name, p_val) = params[idx];
            self.metrics[idx] = eval_single_param(
                name.as_ref(),
                p_val,
                &mut self.baseline_map,
                y0,
                &mut eval_fn,
            );
        }

        self.batch_offset = (self.batch_offset + batch_count) % num_params;
        self.evaluated_count += batch_count;
        &self.metrics
    }
}

fn eval_single_param<F>(
    name: &str,
    p_val: f64,
    baseline_map: &mut HashMap<String, f64>,
    y0: f64,
    eval_fn: &mut F,
) -> SensitivityMetric
where
    F: FnMut(&HashMap<String, f64>) -> f64,
{
    let delta = if p_val.abs() > 1e-12 {
        0.01 * p_val.abs()
    } else {
        0.01
    };

    // Evaluate upper perturbation P+ = P + delta
    set_map_value(baseline_map, name, p_val + delta);
    let y_plus = eval_fn(baseline_map);

    // Evaluate lower perturbation P- = P - delta
    set_map_value(baseline_map, name, p_val - delta);
    let y_minus = eval_fn(baseline_map);

    // Restore baseline
    set_map_value(baseline_map, name, p_val);

    let dy_dp = if delta > 0.0 {
        (y_plus - y_minus) / (2.0 * delta)
    } else {
        0.0
    };

    let elasticity = if y0.abs() > 1e-12 && p_val.abs() > 1e-12 {
        (dy_dp * p_val / y0).abs()
    } else if y0.abs() > 1e-12 {
        (dy_dp / y0).abs()
    } else {
        dy_dp.abs()
    };

    let dy_dp_clean = if dy_dp.is_finite() { dy_dp } else { 0.0 };
    let elasticity_clean = if elasticity.is_finite() {
        elasticity
    } else {
        0.0
    };

    SensitivityMetric {
        param_name: name.to_string(),
        baseline_value: p_val,
        partial_derivative: dy_dp_clean,
        elasticity: elasticity_clean,
        is_high_sensitivity: elasticity_clean > 2.0,
    }
}

/// Renders the collapsible Sensitivity & Perturbation analysis panel in the UI sidebar.
pub fn render_sensitivity_panel_ui(ui: &mut egui::Ui, engine: &SensitivityEngine, id_source: &str) {
    render_labeled_sensitivity_panel_ui(ui, engine, id_source, "Baseline Metric");
}

/// Renders the collapsible Sensitivity & Perturbation analysis panel with an explicit observable label.
pub fn render_labeled_sensitivity_panel_ui(
    ui: &mut egui::Ui,
    engine: &SensitivityEngine,
    id_source: &str,
    metric_label: &str,
) {
    let baseline_heading = if metric_label.is_empty() || metric_label == "Baseline Metric" {
        format!("Baseline Metric: {:.4}", engine.baseline_output)
    } else if metric_label.starts_with("Baseline ") {
        format!("{}: {:.4}", metric_label, engine.baseline_output)
    } else {
        format!("Baseline {}: {:.4}", metric_label, engine.baseline_output)
    };

    if !engine.is_converged() {
        ui.ctx().request_repaint();
    }

    ui.add_space(6.0);
    egui::CollapsingHeader::new("Sensitivity & Perturbation Analysis")
        .id_salt(format!("{}_sensitivity_panel", id_source))
        .default_open(true)
        .show(ui, |ui| {
            ui.label(baseline_heading);
            ui.label(
                egui::RichText::new(
                    "Normalized elasticity |(dY/dP) * (P/Y)| via central finite diff (±1% delta)",
                )
                .weak()
                .small(),
            );
            ui.add_space(4.0);

            if engine.metrics.is_empty() {
                ui.label("No active parameters to analyze.");
                return;
            }

            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            for metric in &engine.metrics {
                render_sensitivity_metric_card(ui, metric, &palette);
                ui.add_space(2.0);
            }
        });
}

fn render_sensitivity_metric_card(
    ui: &mut egui::Ui,
    metric: &SensitivityMetric,
    palette: &crate::app::SemanticPalette,
) {
    ui.group(|ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(&metric.param_name).strong());
            if metric.is_high_sensitivity {
                ui.colored_label(palette.error, "⚠️ High Sensitivity (> 2.0 elasticity)");
            }
        });

        ui.horizontal(|ui| {
            ui.label(format!("dY/dP: {:.4}", metric.partial_derivative));
            ui.add_space(10.0);
            ui.label(format!("Elasticity: {:.2}", metric.elasticity));
        });

        let progress = (metric.elasticity / 3.0).clamp(0.0, 1.0) as f32;
        let color = crate::app::SemanticPalette::progress_bar_fill(ui.visuals(), metric.elasticity);

        let progress_bar = egui::ProgressBar::new(progress)
            .fill(color)
            .text(format!("{:.2} elasticity", metric.elasticity));
        ui.add(progress_bar);
    });
}

#[cfg(test)]
#[path = "sensitivity_tests.rs"]
pub(crate) mod tests;

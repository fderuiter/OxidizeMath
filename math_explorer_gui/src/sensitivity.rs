#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Parameter sensitivity and perturbation analysis panel module.
//!
//! Provides central finite difference perturbation analysis (+/- 1% delta)
//! and normalized sensitivity elasticity metrics for active parameters.

use eframe::egui;
use std::collections::HashMap;

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
}

impl SensitivityEngine {
    /// Creates a new SensitivityEngine instance.
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            baseline_output: 0.0,
            batch_offset: 0,
        }
    }

    /// Evaluates normalized sensitivity indices using central finite differences (+/- 1% delta).
    ///
    /// Respects the constraint bounding perturbation evaluations to at most 5 sample runs
    /// (1 baseline + max 2 parameters * 2 perturbations per step = 5 runs max) to prevent UI slowdown.
    pub fn compute<F>(
        &mut self,
        params: &[(String, f64)],
        mut eval_fn: F,
    ) -> &Vec<SensitivityMetric>
    where
        F: FnMut(&HashMap<String, f64>) -> f64,
    {
        if params.is_empty() {
            self.metrics.clear();
            return &self.metrics;
        }

        // Convert slice to parameter map
        let mut baseline_map: HashMap<String, f64> = params.iter().cloned().collect();
        let y0 = eval_fn(&baseline_map);
        self.baseline_output = y0;

        // Ensure metrics vector matches param count
        if self.metrics.len() != params.len() {
            self.metrics = params
                .iter()
                .map(|(name, val)| SensitivityMetric {
                    param_name: name.clone(),
                    baseline_value: *val,
                    partial_derivative: 0.0,
                    elasticity: 0.0,
                    is_high_sensitivity: false,
                })
                .collect();
        }

        // Bounded perturbation evaluations: Max 5 sample runs per frame (1 baseline + 2 params * 2 runs = 5 runs)
        let num_params = params.len();
        let max_params_per_frame = 2;

        for step in 0..num_params.min(max_params_per_frame) {
            let idx = (self.batch_offset + step) % num_params;
            let (ref name, p_val) = params[idx];
            self.metrics[idx] = eval_single_param(name, p_val, &mut baseline_map, y0, &mut eval_fn);
        }

        self.batch_offset = (self.batch_offset + max_params_per_frame) % num_params.max(1);
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
    baseline_map.insert(name.to_string(), p_val + delta);
    let y_plus = eval_fn(baseline_map);

    // Evaluate lower perturbation P- = P - delta
    baseline_map.insert(name.to_string(), p_val - delta);
    let y_minus = eval_fn(baseline_map);

    // Restore baseline
    baseline_map.insert(name.to_string(), p_val);

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
        let color = if metric.elasticity > 2.0 {
            palette.error
        } else if metric.elasticity > 1.0 {
            palette.warning
        } else {
            palette.success
        };

        let progress_bar = egui::ProgressBar::new(progress)
            .fill(color)
            .text(format!("{:.2} elasticity", metric.elasticity));
        ui.add(progress_bar);
    });
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn collect_shape_texts(shapes: &[egui::epaint::Shape], out: &mut Vec<String>) {
        for shape in shapes {
            match shape {
                egui::epaint::Shape::Text(text_shape) => {
                    out.push(text_shape.galley.text().to_string());
                }
                egui::epaint::Shape::Vec(nested) => collect_shape_texts(nested, out),
                _ => {}
            }
        }
    }

    #[test]
    fn test_sensitivity_linear_function() {
        let mut engine = SensitivityEngine::new();
        let params = vec![("a".to_string(), 5.0), ("b".to_string(), 10.0)];

        // Y = 2*a + b -> dY/da = 2, dY/db = 1
        let eval_fn = |p: &HashMap<String, f64>| -> f64 {
            let a = p.get("a").copied().unwrap_or(0.0);
            let b = p.get("b").copied().unwrap_or(0.0);
            2.0 * a + b
        };

        let metrics = engine.compute(&params, eval_fn);
        assert_eq!(metrics.len(), 2);
        assert!((metrics[0].partial_derivative - 2.0).abs() < 1e-4);
        assert!(!metrics[0].is_high_sensitivity);
    }

    #[test]
    fn test_high_sensitivity_warning() {
        let mut engine = SensitivityEngine::new();
        let params = vec![("x".to_string(), 3.0)];

        // Y = x^3 -> dY/dx = 3*x^2 = 27. Elasticity = 27 * 3 / 27 = 3.0 > 2.0
        let eval_fn = |p: &HashMap<String, f64>| -> f64 {
            let x = p.get("x").copied().unwrap_or(1.0);
            x.powi(3)
        };

        let metrics = engine.compute(&params, eval_fn);
        assert_eq!(metrics.len(), 1);
        assert!((metrics[0].elasticity - 3.0).abs() < 1e-2);
        assert!(metrics[0].is_high_sensitivity);
    }

    #[test]
    fn test_bounded_evaluations_run_count() {
        let mut engine = SensitivityEngine::new();
        let params = vec![
            ("p1".to_string(), 1.0),
            ("p2".to_string(), 2.0),
            ("p3".to_string(), 3.0),
            ("p4".to_string(), 4.0),
        ];

        let mut call_count = 0;
        let eval_fn = |p: &HashMap<String, f64>| -> f64 {
            call_count += 1;
            p.values().sum()
        };

        engine.compute(&params, eval_fn);
        // 1 baseline evaluation + max 2 params * 2 perturbations = 5 evaluations max
        assert!(call_count <= 5);
        assert_eq!(call_count, 5);
    }

    #[test]
    fn test_render_labeled_sensitivity_panel_ui_renders_custom_label() {
        let ctx = egui::Context::default();
        let mut engine = SensitivityEngine::new();
        let params = vec![("g_na".to_string(), 120.0)];
        engine.compute(&params, |p| p.get("g_na").copied().unwrap_or(0.0) * 0.25);

        let out = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_labeled_sensitivity_panel_ui(
                    ui,
                    &engine,
                    "hh_test",
                    "Peak Membrane Potential (mV)",
                );
            });
        });

        let raw_shapes: Vec<egui::epaint::Shape> =
            out.shapes.into_iter().map(|cs| cs.shape).collect();
        let mut texts = Vec::new();
        collect_shape_texts(&raw_shapes, &mut texts);

        assert!(
            texts
                .iter()
                .any(|t| t.contains("Baseline Peak Membrane Potential (mV): 30.0000")),
            "Expected labeled baseline metric in rendered UI texts, got: {:?}",
            texts
        );
    }

    #[test]
    fn test_sensitivity_metric_card_uses_wcag_semantic_palette_across_themes() {
        let mut engine = SensitivityEngine::new();
        engine.metrics = vec![
            SensitivityMetric {
                param_name: "high_p".to_string(),
                baseline_value: 3.0,
                partial_derivative: 27.0,
                elasticity: 3.0,
                is_high_sensitivity: true,
            },
            SensitivityMetric {
                param_name: "mid_p".to_string(),
                baseline_value: 2.0,
                partial_derivative: 1.5,
                elasticity: 1.5,
                is_high_sensitivity: false,
            },
            SensitivityMetric {
                param_name: "low_p".to_string(),
                baseline_value: 1.0,
                partial_derivative: 0.5,
                elasticity: 0.5,
                is_high_sensitivity: false,
            },
        ];

        for mode in [
            crate::app::ThemeMode::Dark,
            crate::app::ThemeMode::Light,
            crate::app::ThemeMode::HighContrast,
        ] {
            let ctx = egui::Context::default();
            ctx.set_visuals(mode.visuals());
            let palette = mode.palette();
            let panel_bg = mode.visuals().panel_fill;
            assert!(crate::app::contrast_ratio(palette.error, panel_bg) >= 4.5);
            assert!(crate::app::contrast_ratio(palette.warning, panel_bg) >= 4.5);
            assert!(crate::app::contrast_ratio(palette.success, panel_bg) >= 4.5);

            let out = ctx.run(egui::RawInput::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    render_sensitivity_panel_ui(ui, &engine, "wcag_sens");
                });
            });
            let raw_shapes: Vec<egui::epaint::Shape> =
                out.shapes.into_iter().map(|cs| cs.shape).collect();
            let mut texts = Vec::new();
            collect_shape_texts(&raw_shapes, &mut texts);
            assert!(texts.iter().any(|t| t.contains("High Sensitivity")));
        }
    }
}

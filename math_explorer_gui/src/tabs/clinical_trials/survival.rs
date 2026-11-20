use crate::framework::InteractiveTool;
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::applied::clinical_trials::survival_analysis::{
    kaplan_meier, Observation, TimePoint,
};
use math_explorer::applied::clinical_trials::types::SurvivalTime;

pub struct SurvivalAnalysisTool {
    input_text: String,
    observations: Vec<Observation>,
    curve: Vec<TimePoint>,
    error_message: Option<String>,
}

impl Default for SurvivalAnalysisTool {
    fn default() -> Self {
        let default_text = "6.0, 1\n6.0, 1\n6.0, 1\n7.0, 1\n10.0, 1\n13.0, 1\n16.0, 1\n22.0, 1\n23.0, 1\n6.0, 0\n9.0, 0\n10.0, 0\n11.0, 0\n17.0, 0\n19.0, 0\n20.0, 0\n25.0, 0\n32.0, 0\n32.0, 0\n34.0, 0\n35.0, 0";
        let mut tool = Self {
            input_text: default_text.to_string(),
            observations: Vec::new(),
            curve: Vec::new(),
            error_message: None,
        };
        tool.recalculate();
        tool
    }
}

impl SurvivalAnalysisTool {
    fn parse_observation_line(line_idx: usize, line: &str) -> Result<Observation, String> {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 2 {
            return Err(format!("Line {}: Expected 'time, event'", line_idx + 1));
        }

        let time_str = parts[0].trim();
        let event_str = parts[1].trim();

        let time_val = time_str
            .parse::<f64>()
            .map_err(|_| format!("Line {}: Invalid time '{}'", line_idx + 1, time_str))?;

        let event_occurred = match event_str.parse::<u8>() {
            Ok(1) => true,
            Ok(0) => false,
            _ if event_str.eq_ignore_ascii_case("true") => true,
            _ if event_str.eq_ignore_ascii_case("false") => false,
            _ => {
                return Err(format!(
                    "Line {}: Event must be 1 (event) or 0 (censored)",
                    line_idx + 1
                ));
            }
        };

        let time =
            SurvivalTime::new(time_val).map_err(|e| format!("Line {}: {}", line_idx + 1, e))?;

        Ok(Observation {
            time,
            event_occurred,
        })
    }

    fn recalculate(&mut self) {
        self.observations.clear();
        self.curve.clear();
        self.error_message = None;

        for (line_idx, raw_line) in self.input_text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }

            if self.observations.len() >= math_commons::registry::MAX_ITERATIONS {
                self.observations.clear();
                self.error_message = Some(format!(
                    "Line {}: Maximum of {} observations exceeded",
                    line_idx + 1,
                    math_commons::registry::MAX_ITERATIONS
                ));
                return;
            }

            match Self::parse_observation_line(line_idx, line) {
                Ok(obs) => self.observations.push(obs),
                Err(err) => {
                    self.observations.clear();
                    self.error_message = Some(err);
                    return;
                }
            }
        }

        self.curve = kaplan_meier(&self.observations);
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SurvivalState {
    input_text: String,
}

impl InteractiveTool for SurvivalAnalysisTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Survival Curves (Kaplan-Meier)"
    }
    fn domain(&self) -> &'static str {
        "clinical_trials"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&SurvivalState {
            input_text: self.input_text.clone(),
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: SurvivalState =
            serde_json::from_str(state).map_err(|e| format!("Invalid survival state: {e}"))?;
        let mut candidate = Self {
            input_text: s.input_text,
            observations: Vec::new(),
            curve: Vec::new(),
            error_message: None,
        };
        candidate.recalculate();
        if let Some(err) = candidate.error_message {
            return Err(err);
        }
        if candidate.observations.is_empty() {
            return Err("Survival observations cannot be empty".into());
        }
        *self = candidate;
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "survival_input_panel");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            ui.heading("Input Data");
            ui.label("Format: Time, Event (1=Death, 0=Censored)");
            ui.add_space(5.0);

            let text_edit = egui::TextEdit::multiline(&mut self.input_text)
                .desired_width(f32::INFINITY)
                .desired_rows(20);

            if ui.add(text_edit).changed() {
                self.recalculate();
            }

            if let Some(err) = &self.error_message {
                let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
                ui.colored_label(palette.error, err);
            }

            ui.separator();
            render_copyable_metric(ui, "Observations", &self.observations.len().to_string());
        });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Kaplan-Meier Survival Curve");

            let mut plot_points: Vec<[f64; 2]> = Vec::new();

            // Start at (0, 1.0)
            plot_points.push([0.0, 1.0]);

            // Add steps
            for point in &self.curve {
                // Step logic: Horizontal to new time, then vertical down to new probability
                if let Some(last) = plot_points.last() {
                    plot_points.push([point.time, last[1]]);
                }
                plot_points.push([point.time, point.survival_probability]);
            }

            // Extend to max time if needed (usually handled by plot bounds, but nice to visualize)
            if let Some(last) = plot_points.last() {
                // Extend a bit further for visualization
                plot_points.push([last[0] * 1.1, last[1]]);
            }

            Plot::new("kaplan_meier_plot")
                .view_aspect(2.0)
                .x_axis_label("Time")
                .y_axis_label("Survival Probability")
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new("Survival Probability", PlotPoints::new(plot_points))
                            .name("Survival Probability"),
                    );
                });
        });
    }
}

// [cite:clinical_trials_statistics]


inventory::submit! {
    crate::framework::ToolMetadata {
        id: "SurvivalAnalysisTool",
        display_name: "Survival Curves (Kaplan-Meier)",
        domain: "clinical_trials",
        tags: &[],
        build: || Box::new(SurvivalAnalysisTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for SurvivalAnalysisTool {
    fn theory_description(&self) -> String {
        format!(
            "Kaplan-Meier non-parametric maximum likelihood estimator of survival function $S(t) = \\prod_{{t_i \\le t}} \\left(1 - \\frac{{d_i}}{{n_i}}\\right)$ accounting for right-censored time-to-event data across {} observations.",
            self.observations.len()
        )
    }
    fn phonetic_description(&self) -> String { "Kaplan-Meier non-parametric survival analysis theory".into() }
    fn theory_citation(&self) -> String { "Nonparametric Estimation from Incomplete Observations (Kaplan & Meier, 1958)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Kaplan-Meier Survival",
    "domain": "clinical_trials",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "input_text": "3.0, 1\n5.0, 1\n6.0, 0\n8.0, 1\n11.0, 0\n14.0, 1\n18.0, 1\n21.0, 0\n24.0, 1\n30.0, 0"
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "survival_analysis_starter",
        title: "Kaplan-Meier Survival",
        description: "Ŝ(t) = ∏(1 - d_i/n_i): Non-parametric product-limit estimator for time-to-event censorship data.",
        category: "Survival Models",
        icon: "⏱️",
        domain: "clinical_trials",
        tool_name: Some("SurvivalAnalysisTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reject_infinite_and_non_finite_presets() {
        let inf_preset = r#"{"metadata":{"schema_version":"1.0","title":"KM","domain":"clinical_trials","timestamp":"2026-11-19T00:00:00Z"},"config":{"input_text":"inf, 1"}}"#;
        let mut tool = SurvivalAnalysisTool::default();
        let baseline_len = tool.observations.len();
        assert!(tool.import_preset_json(inf_preset).is_err());
        assert_eq!(tool.observations.len(), baseline_len);

        for invalid_input in ["-inf, 1", "Infinity, 0", "NaN, 1", "-1.0, 1"] {
            let payload = format!(
                r#"{{"metadata":{{"schema_version":"1.0","title":"KM","domain":"clinical_trials","timestamp":"2026-11-19T00:00:00Z"}},"config":{{"input_text":"{invalid_input}"}}}}"#
            );
            assert!(
                tool.import_preset_json(&payload).is_err(),
                "Expected error for preset input '{invalid_input}'"
            );
            assert_eq!(tool.observations.len(), baseline_len);
        }
    }

    #[test]
    fn test_observation_cap_enforced_in_recalculate_and_load_state() {
        let max_obs = math_commons::registry::MAX_ITERATIONS;
        let at_cap_input: String = (0..max_obs)
            .map(|idx| format!("{}.0, 1", idx + 1))
            .collect::<Vec<_>>()
            .join("\n");

        let mut tool = SurvivalAnalysisTool::default();
        let state_ok = serde_json::to_string(&SurvivalState {
            input_text: at_cap_input,
        })
        .unwrap();
        assert!(tool.try_load_state(&state_ok).is_ok());
        assert_eq!(tool.observations.len(), max_obs);
        assert!(tool.error_message.is_none());

        let over_cap_input: String = (0..=max_obs)
            .map(|idx| format!("{}.0, 1", idx + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let state_over = serde_json::to_string(&SurvivalState {
            input_text: over_cap_input.clone(),
        })
        .unwrap();
        let err = tool.try_load_state(&state_over).unwrap_err();
        assert!(err.contains(&max_obs.to_string()));
        assert_eq!(tool.observations.len(), max_obs);

        tool.input_text = over_cap_input;
        tool.recalculate();
        assert!(tool.error_message.is_some());
        assert!(tool.observations.is_empty());
        assert!(tool.curve.is_empty());
    }

    #[test]
    fn test_starter_preset_preserves_survival_probabilities() {
        let mut tool = SurvivalAnalysisTool::default();
        tool.import_preset_json(STARTER_PRESET_JSON).unwrap();
        assert_eq!(tool.observations.len(), 10);
        assert_eq!(tool.curve.len(), 10);
        assert!((tool.curve[0].survival_probability - 0.9).abs() < 1e-12);
        assert!((tool.curve[1].survival_probability - 0.8).abs() < 1e-12);
        assert!((tool.curve[2].survival_probability - 0.8).abs() < 1e-12);
    }
}

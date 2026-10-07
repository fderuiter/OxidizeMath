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
    fn recalculate(&mut self) {
        self.observations.clear();
        self.error_message = None;

        for (line_idx, line) in self.input_text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() != 2 {
                self.error_message = Some(format!("Line {}: Expected 'time, event'", line_idx + 1));
                return;
            }

            let time_str = parts[0].trim();
            let event_str = parts[1].trim();

            let time_val = match time_str.parse::<f64>() {
                Ok(t) => t,
                Err(_) => {
                    self.error_message = Some(format!(
                        "Line {}: Invalid time '{}'",
                        line_idx + 1,
                        time_str
                    ));
                    return;
                }
            };

            let event_occurred = match event_str.parse::<u8>() {
                Ok(1) => true,
                Ok(0) => false,
                _ => {
                    // Try parsing as boolean "true"/"false" just in case
                    if event_str.eq_ignore_ascii_case("true") {
                        true
                    } else if event_str.eq_ignore_ascii_case("false") {
                        false
                    } else {
                        self.error_message = Some(format!(
                            "Line {}: Event must be 1 (event) or 0 (censored)",
                            line_idx + 1
                        ));
                        return;
                    }
                }
            };

            match SurvivalTime::new(time_val) {
                Ok(t) => {
                    self.observations.push(Observation {
                        time: t,
                        event_occurred,
                    });
                }
                Err(e) => {
                    self.error_message = Some(format!("Line {}: {}", line_idx + 1, e));
                    return;
                }
            }
        }

        self.curve = kaplan_meier(&self.observations);
    }
}

impl InteractiveTool for SurvivalAnalysisTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Survival Curves (Kaplan-Meier)"
    }

    fn show(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("survival_input_panel").show(ctx, |ui| {
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
                ui.colored_label(egui::Color32::RED, err);
            }

            ui.separator();
            render_copyable_metric(ui, "Observations", &self.observations.len().to_string());
        });

        egui::CentralPanel::default().show(ctx, |ui| {
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
        name: "SurvivalAnalysisTool",
        domain: "clinical_trials",
        tags: &[],
        build: || Box::new(SurvivalAnalysisTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for SurvivalAnalysisTool {
    fn theory_description(&self) -> String { "Theoretical context not available.".into() }
    fn phonetic_description(&self) -> String { "Theoretical context not available.".into() }
    fn theory_citation(&self) -> String { "Uncited".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}

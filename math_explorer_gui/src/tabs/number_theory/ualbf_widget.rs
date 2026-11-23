use crate::accessibility::AccessibleHoverText;
use eframe::egui;
use math_explorer::pure_math::number_theory::ualbf::{ualbf_search, UalbfSearchResult};

pub struct UalbfWidget {
    limit_p: u64,
    max_exponent: u32,
    stop_threshold_log: f64,
    result: Option<UalbfSearchResult>,
    status_message: String,
}

impl Default for UalbfWidget {
    fn default() -> Self {
        Self {
            limit_p: 200,
            max_exponent: 1,
            stop_threshold_log: 10.0,
            result: None,
            status_message: String::from("Ready. Configure parameters and click Run."),
        }
    }
}

use crate::framework::InteractiveTool;

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct UalbfState {
    limit_p: u64,
    max_exponent: u32,
    stop_threshold_log: f64,
}

impl InteractiveTool for UalbfWidget {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "UALBF"
    }
    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::none()
    }
    fn domain(&self) -> &'static str {
        "number_theory"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&UalbfState {
            limit_p: self.limit_p,
            max_exponent: self.max_exponent,
            stop_threshold_log: self.stop_threshold_log,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: UalbfState =
            serde_json::from_str(state).map_err(|e| format!("Invalid UALBF state: {e}"))?;
        if !(100..=10_000).contains(&parsed.limit_p)
            || !(1..=4).contains(&parsed.max_exponent)
            || !parsed.stop_threshold_log.is_finite()
            || !(10.0..=15.0).contains(&parsed.stop_threshold_log)
        {
            return Err("UALBF parameters out of bounds".into());
        }
        self.limit_p = parsed.limit_p;
        self.max_exponent = parsed.max_exponent;
        self.stop_threshold_log = parsed.stop_threshold_log;
        self.result = None;
        self.status_message = String::from("Ready. Configure parameters and click Run.");
        Ok(())
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show_inside(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("UALBF: Unified Algebraic-Lattice Bipartition Framework");
            ui.label(
                "A synthesis of ALCF and AMBS for proving lower bounds on quasiperfect numbers.",
            );
            ui.add_space(10.0);

            ui.group(|ui| {
                ui.label(egui::RichText::new("UALBF Parameters").strong());

                let mut limit_p_f = self.limit_p as f64;
                ui.add(egui::Slider::new(&mut limit_p_f, 100.0..=10000.0).text("Prime limit"));
                self.limit_p = limit_p_f as u64;

                let mut max_exp_f = self.max_exponent as f64;
                ui.add(
                    egui::Slider::new(&mut max_exp_f, 1.0..=4.0)
                        .text("Max exponent")
                        .step_by(1.0),
                );
                self.max_exponent = max_exp_f as u32;

                ui.add(
                    egui::Slider::new(&mut self.stop_threshold_log, 10.0..=15.0)
                        .text("Stop threshold (10^x)"),
                );

                if ui
                    .button("▶ Run UALBF Pipeline")
                    .accessible_hover_text("Execute the Unified Algebraic-Lattice Bipartition Framework search pipeline")
                    .clicked()
                {
                    let threshold_str = format!("{:.0}", 10f64.powf(self.stop_threshold_log));
                    let target_max = "1000000000000";
                    match ualbf_search(self.limit_p, self.max_exponent, &threshold_str, target_max) {
                        Ok(result) => {
                            self.status_message = format!(
                                "Done. Valid: {}, Pruned: {}, Prefixes: {}, Rejected: {}, Candidates: {}",
                                result.valid_components,
                                result.pruned_components,
                                result.prefix_count,
                                result.rejected_by_lattice,
                                result.candidates_checked,
                            );
                            self.result = Some(result);
                        }
                        Err(e) => {
                            self.status_message = format!("Error: {}", e);
                            self.result = None;
                        }
                    }
                }
            });

            ui.label(&self.status_message);

            if let Some(ref result) = self.result {
                ui.separator();
                ui.label(egui::RichText::new("Results").strong());
                ui.label(format!(
                    "Phase 1 — Valid components: {}",
                    result.valid_components
                ));
                ui.label(format!(
                    "Phase 1 — Pruned components: {}",
                    result.pruned_components
                ));
                ui.label(format!("Phase 2 — Prefix count: {}", result.prefix_count));
                ui.label(format!(
                    "Phase 3 — Rejected by lattice oracle: {}",
                    result.rejected_by_lattice
                ));
                ui.label(format!(
                    "Phase 4 — Candidates checked: {}",
                    result.candidates_checked
                ));
            }
        });
    }
}

// [cite:ualbf_quasiperfect]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "UalbfWidget",
        display_name: "UALBF",
        domain: "number_theory",
        tags: &["quasiperfect", "abundant", "divisor sum", "sigma", "arithmetic"],
        build: || Box::new(UalbfWidget::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for UalbfWidget {
    fn theory_description(&self) -> String {
        format!(
            "Unified Algebraic-Lattice Bipartition Framework (UALBF) for establishing lower bounds on hypothetical quasiperfect numbers, searching prime base $p \\le {}$ with maximum exponent {}.",
            self.limit_p, self.max_exponent
        )
    }
    fn phonetic_description(&self) -> String {
        "Unified Algebraic-Lattice Bipartition Framework for quasiperfect numbers theory".into()
    }
    fn theory_citation(&self) -> String {
        "Unified Algebraic-Lattice Bipartition Framework for Quasiperfect Numbers (de Ruiter, 2025)"
            .into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "UALBF Quasiperfect Search",
    "domain": "number_theory",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "limit_p": 500,
    "max_exponent": 2,
    "stop_threshold_log": 12.0
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "ualbf_starter",
        title: "UALBF Quasiperfect Search",
        description: "Unified Algebraic-Lattice Bipartition Framework search for quasiperfect number lower bounds.",
        category: "Number Theory",
        icon: "🧮",
        domain: "number_theory",
        tool_name: Some("UalbfWidget"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

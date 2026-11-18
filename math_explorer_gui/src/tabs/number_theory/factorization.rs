use crate::accessibility::AccessibleHoverText;
use crate::framework::InteractiveTool;
use eframe::egui;
use math_explorer::pure_math::number_theory::ambs::prime_factors;
use math_explorer::pure_math::number_theory::primes::is_prime;

type FactorizationResult = Result<(u64, bool, Vec<(u64, u32)>), String>;

#[derive(Default)]
pub struct FactorizationTool {
    input_text: String,
    results: Option<FactorizationResult>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct FactorizationState {
    input_text: String,
}

impl InteractiveTool for FactorizationTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Factorization Tool"
    }
    fn domain(&self) -> &'static str {
        "number_theory"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&FactorizationState {
            input_text: self.input_text.clone(),
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: FactorizationState =
            serde_json::from_str(state).map_err(|e| format!("Invalid factorization state: {e}"))?;
        let trimmed = parsed.input_text.trim();
        if trimmed.is_empty() {
            self.input_text = parsed.input_text;
            self.results = None;
            return Ok(());
        }
        let num = trimmed
            .parse::<u64>()
            .map_err(|_| "input_text must be an integer >= 2 or empty".to_string())?;
        if num < 2 {
            return Err("Number must be at least 2".into());
        }
        self.input_text = parsed.input_text;
        self.results = Some(Ok((num, is_prime(num), prime_factors(num))));
        Ok(())
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Large Number Factorization and Primality Testing");
            ui.label("Enter an integer to test for primality and find its prime factors.");
            ui.add_space(10.0);

            ui.horizontal(|ui| {
                ui.label("Number:");
                let response =
                    ui.add(egui::TextEdit::singleline(&mut self.input_text).hint_text("e.g. 42"));
                let enter_shortcut_triggered = egui_plot::commands::CommandRegistryData::register_and_check(
                    ui.ctx(),
                    "Analyze Factorization",
                    "Analyze the number for primality and calculate its prime factors",
                    egui_plot::commands::CommandTrigger::Key(egui::Key::Enter),
                    false,
                    "Factorization Tool",
                    Some(ui),
                    None,
                );
                let enter_pressed = response.lost_focus() && enter_shortcut_triggered;

                if ui
                    .button("▶ Analyze")
                    .accessible_hover_text(
                        "Analyze the number for primality and calculate its prime factors",
                    )
                    .clicked()
                    || enter_pressed
                {
                    self.results = Some(match self.input_text.trim().parse::<u64>() {
                        Ok(num) => {
                            if num < 2 {
                                Err("Number must be at least 2.".to_string())
                            } else {
                                let prime_check = is_prime(num);
                                let factors = prime_factors(num);
                                Ok((num, prime_check, factors))
                            }
                        }
                        Err(_) => {
                            Err("Please enter a valid positive integer (max 64-bit).".to_string())
                        }
                    });
                }
            });

            ui.add_space(15.0);

            if let Some(ref res) = self.results {
                match res {
                    Ok((num, prime_check, factors)) => {
                        ui.group(|ui| {
                            ui.label(egui::RichText::new(format!("Analysis for {}", num)).strong());
                            ui.add_space(5.0);

                            ui.horizontal(|ui| {
                                ui.label("Is Prime:");
                                if *prime_check {
                                    ui.label(
                                        egui::RichText::new("Yes").color(egui::Color32::GREEN),
                                    );
                                } else {
                                    ui.label(egui::RichText::new("No").color(egui::Color32::RED));
                                }
                            });

                            ui.add_space(5.0);
                            ui.label("Prime Factors:");
                            if factors.is_empty() {
                                ui.label("None (less than 2)");
                            } else {
                                let mut factor_strings = Vec::new();
                                for (p, e) in factors {
                                    if *e == 1 {
                                        factor_strings.push(format!("{}", p));
                                    } else {
                                        factor_strings.push(format!("{}^{}", p, e));
                                    }
                                }
                                ui.label(factor_strings.join(" × "));
                            }
                        });
                    }
                    Err(err_msg) => {
                        ui.label(egui::RichText::new(err_msg).color(egui::Color32::RED));
                    }
                }
            }
        });
    }
}

// [cite:algorithmic_information_rust]


inventory::submit! {
    crate::framework::ToolMetadata {
        id: "FactorizationTool",
        display_name: "Factorization Tool",
        domain: "number_theory",
        tags: &[],
        build: || Box::new(FactorizationTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for FactorizationTool {
    fn theory_description(&self) -> String {
        "Integer factorization into unique prime powers by the Fundamental Theorem of Arithmetic: $n = \\prod_{i=1}^k p_i^{a_i}$, with deterministic trial division and probabilistic Miller-Rabin primality certification.".into()
    }
    fn phonetic_description(&self) -> String { "Fundamental Theorem of Arithmetic prime factorization theory".into() }
    fn theory_citation(&self) -> String { "An Introduction to the Theory of Numbers (Hardy & Wright, 2008)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Prime Factorization",
    "domain": "number_theory",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "input_text": "104729"
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "factorization_starter",
        title: "Prime Factorization",
        description: "Fundamental theorem of arithmetic prime power decomposition and deterministic primality testing.",
        category: "Arithmetic",
        icon: "🔢",
        domain: "number_theory",
        tool_name: Some("FactorizationTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

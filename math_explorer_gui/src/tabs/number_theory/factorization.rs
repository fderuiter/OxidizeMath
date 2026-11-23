use crate::accessibility::AccessibleHoverText;
use crate::framework::InteractiveTool;
use eframe::egui;
use math_explorer::pure_math::number_theory::ambs::prime_factors;
use math_explorer::pure_math::number_theory::primes::is_prime;

/// Maximum integer accepted for synchronous trial-division primality and factorization.
///
/// Prevents UI-thread stalls and avoids `i * i` `u64` multiplication overflow in trial division.
pub const MAX_SYNC_FACTORIZATION_INPUT: u64 = 10_000_000_000_000;

type FactorizationResult = Result<(u64, bool, Vec<(u64, u32)>), String>;

fn factorize_bounded(num: u64) -> FactorizationResult {
    if num < 2 {
        return Err("Number must be at least 2.".to_string());
    }
    if num > MAX_SYNC_FACTORIZATION_INPUT {
        return Err(format!(
            "Number exceeds maximum synchronous limit ({MAX_SYNC_FACTORIZATION_INPUT})."
        ));
    }
    Ok((num, is_prime(num), prime_factors(num)))
}

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
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Factorization Tool"
    }
    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::none()
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
        self.state_bounds().validate_json(state)?;
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
        let computed = factorize_bounded(num)?;
        self.input_text = parsed.input_text;
        self.results = Some(Ok(computed));
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Large Number Factorization and Primality Testing");
            ui.label("Enter an integer to test for primality and find its prime factors.");
            ui.add_space(10.0);

            self.render_input_row(ui);
            self.render_chord_presets_row(ui);
            ui.add_space(12.0);
            self.render_results_and_sonification(ui);
        });
    }
}

impl FactorizationTool {
    fn analyze_current_input(&mut self) {
        self.results = Some(match self.input_text.trim().parse::<u64>() {
            Ok(num) => factorize_bounded(num),
            Err(_) => Err("Please enter a valid positive integer (max 64-bit).".to_string()),
        });
    }

    fn render_input_row(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Number:");
            let response =
                ui.add(egui::TextEdit::singleline(&mut self.input_text).hint_text("e.g. 42"));
            egui_plot::commands::CommandRegistryData::register(
                ui.ctx(),
                "Analyze Factorization",
                "Analyze the number for primality and calculate its prime factors",
                egui_plot::commands::CommandTrigger::Key(egui::Key::Enter),
                false,
                "Factorization Tool",
            );
            let enter_pressed =
                response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if enter_pressed {
                egui_plot::emit_plot_live_message(
                    ui.ctx(),
                    "Action executed: Analyze Factorization".to_string(),
                    false,
                );
            }
            if ui
                .button("▶ Analyze")
                .accessible_hover_text(
                    "Analyze the number for primality and calculate its prime factors",
                )
                .clicked()
                || enter_pressed
            {
                self.analyze_current_input();
            }
        });
    }

    fn render_chord_presets_row(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label("Just-Intonation Chords:");
            for (label, n_str) in [
                ("12 (2²×3)", "12"),
                ("60 (2²×3×5)", "60"),
                ("180 (2²×3²×5)", "180"),
                ("210 (2×3×5×7)", "210"),
            ] {
                if ui
                    .small_button(label)
                    .accessible_hover_text(format!("Factorize {n_str} into a prime harmonic chord"))
                    .clicked()
                {
                    self.input_text = n_str.to_string();
                    self.analyze_current_input();
                }
            }
        });
    }

    fn render_results_and_sonification(&mut self, ui: &mut egui::Ui) {
        let mut chord_frame = crate::audio::SonificationFrame::silent();
        if let Some(ref res) = self.results {
            match res {
                Ok((num, prime_check, factors)) => {
                    chord_frame = crate::audio::mappings::prime_chord_frame(
                        factors,
                        crate::audio::WaveformPreset::JustHarmonic,
                    );
                    render_factorization_summary(ui, *num, *prime_check, factors);
                }
                Err(err_msg) => {
                    let err_color = crate::app::SemanticPalette::from_visuals(ui.visuals()).error;
                    ui.label(egui::RichText::new(err_msg).color(err_color));
                }
            }
            ui.add_space(10.0);
        }
        crate::audio::render_sonification_panel(ui, "factorization", &chord_frame, false);
    }
}

fn render_factorization_summary(
    ui: &mut egui::Ui,
    num: u64,
    prime_check: bool,
    factors: &[(u64, u32)],
) {
    ui.group(|ui| {
        ui.label(egui::RichText::new(format!("Analysis for {num}")).strong());
        ui.add_space(5.0);

        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.horizontal(|ui| {
            ui.label("Is Prime:");
            if prime_check {
                ui.label(egui::RichText::new("Yes").color(palette.success));
            } else {
                ui.label(egui::RichText::new("No").color(palette.error));
            }
        });

        ui.add_space(5.0);
        ui.label("Prime Factors & Just-Intonation Ratios:");
        if factors.is_empty() {
            ui.label("None (less than 2)");
        } else {
            let parts: Vec<String> = factors
                .iter()
                .map(|&(p, e)| {
                    let ratio = crate::audio::mappings::just_intonation_ratio(p);
                    if e == 1 {
                        format!("{p} ({ratio:.3}×)")
                    } else {
                        format!("{p}^{e} ({ratio:.3}×)")
                    }
                })
                .collect();
            ui.label(parts.join(" × "));
        }
    });
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
    fn phonetic_description(&self) -> String {
        "Fundamental Theorem of Arithmetic prime factorization theory".into()
    }
    fn theory_citation(&self) -> String {
        "An Introduction to the Theory of Numbers (Hardy & Wright, 2008)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn click_analyze_button(ctx: &egui::Context, tool: &mut FactorizationTool) {
        let rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(800.0, 600.0),
        ));
        let out = ctx.run(
            egui::RawInput {
                screen_rect: rect,
                ..Default::default()
            },
            |ctx| tool.show(ctx),
        );
        let btn_bounds = out
            .platform_output
            .accesskit_update
            .as_ref()
            .and_then(|ak| {
                ak.nodes.iter().find_map(|(_, n)| {
                    (n.role() == egui::accesskit::Role::Button && n.label() == Some("▶ Analyze"))
                        .then(|| n.bounds())
                        .flatten()
                })
            })
            .expect("Analyze button must exist");
        let pos = egui::pos2(
            ((btn_bounds.x0 + btn_bounds.x1) * 0.5) as f32,
            ((btn_bounds.y0 + btn_bounds.y1) * 0.5) as f32,
        );
        let events = vec![
            egui::Event::PointerMoved(pos),
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ];
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: rect,
                events,
                ..Default::default()
            },
            |ctx| tool.show(ctx),
        );
    }

    #[test]
    fn test_factorization_synchronous_cap_and_ui_analyze() {
        assert!(factorize_bounded(1).is_err());
        assert!(factorize_bounded(18_446_744_073_709_551_557).is_err());
        assert_eq!(
            factorize_bounded(42).unwrap(),
            (42, false, vec![(2, 1), (3, 1), (7, 1)])
        );

        let mut tool = FactorizationTool {
            input_text: "18446744073709551557".to_string(),
            results: None,
        };
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        click_analyze_button(&ctx, &mut tool);
        assert!(matches!(tool.results, Some(Err(_))));
    }
}

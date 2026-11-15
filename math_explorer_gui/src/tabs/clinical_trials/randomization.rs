use crate::accessibility::AccessibleHoverText;
use crate::framework::InteractiveTool;
use eframe::egui;
use math_explorer::applied::clinical_trials::design::{
    AllocationStrategy, BlockRandomizer, Group, SimpleRandomizer,
};

#[derive(PartialEq, Clone, Copy, serde::Serialize, serde::Deserialize)]
enum RandomizationType {
    Simple,
    Block,
}

pub struct RandomizationTool {
    randomization_type: RandomizationType,
    n_subjects: usize,
    block_size: usize,
    assignments: Vec<Group>,
    error_message: Option<String>,
}

impl Default for RandomizationTool {
    fn default() -> Self {
        Self {
            randomization_type: RandomizationType::Simple,
            n_subjects: 10,
            block_size: 4,
            assignments: Vec::new(),
            error_message: None,
        }
    }
}

impl InteractiveTool for RandomizationTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Randomization"
    }
    fn domain(&self) -> &'static str {
        "clinical_trials"
    }

    fn save_state(&self) -> Option<String> {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct RandomizationState {
            randomization_type: RandomizationType,
            n_subjects: usize,
            block_size: usize,
        }
        serde_json::to_string(&RandomizationState {
            randomization_type: self.randomization_type,
            n_subjects: self.n_subjects,
            block_size: self.block_size,
        }).ok()
    }

    fn load_state(&mut self, state: &str) {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct RandomizationState {
            randomization_type: RandomizationType,
            n_subjects: usize,
            block_size: usize,
        }
        if let Ok(s) = serde_json::from_str::<RandomizationState>(state) {
            self.randomization_type = s.randomization_type;
            self.n_subjects = s.n_subjects;
            self.block_size = s.block_size;
            self.assignments.clear();
            self.error_message = None;
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Interactive Subject Allocation");

            ui.add_space(10.0);

            ui.horizontal(|ui| {
                ui.label("Randomization Type:");
                ui.radio_value(
                    &mut self.randomization_type,
                    RandomizationType::Simple,
                    "Simple",
                );
                ui.radio_value(
                    &mut self.randomization_type,
                    RandomizationType::Block,
                    "Block",
                );
            });

            ui.add_space(5.0);

            ui.horizontal(|ui| {
                ui.label("Number of Subjects:");
                ui.add(egui::DragValue::new(&mut self.n_subjects).range(1..=1000));
            });

            if self.randomization_type == RandomizationType::Block {
                ui.horizontal(|ui| {
                    ui.label("Block Size:");
                    ui.add(egui::DragValue::new(&mut self.block_size).range(2..=100));
                });
            }

            ui.add_space(10.0);

            let enter_pressed = ui.input(|i| i.key_pressed(egui::Key::Enter));
            if ui
                .button("▶ Allocate Subjects")
                .accessible_hover_text("Randomly assign subjects to treatment and control groups")
                .clicked()
                || enter_pressed
            {
                self.error_message = None;
                self.assignments.clear();

                let mut rng = oxidize_core::rng::OxidizeRng::default();

                match self.randomization_type {
                    RandomizationType::Simple => {
                        let randomizer = SimpleRandomizer;
                        match randomizer.assign(&mut rng, self.n_subjects) {
                            Ok(assignments) => self.assignments = assignments,
                            Err(e) => self.error_message = Some(e.to_string()),
                        }
                    }
                    RandomizationType::Block => match BlockRandomizer::new(self.block_size) {
                        Ok(randomizer) => match randomizer.assign(&mut rng, self.n_subjects) {
                            Ok(assignments) => self.assignments = assignments,
                            Err(e) => self.error_message = Some(e.to_string()),
                        },
                        Err(e) => self.error_message = Some(e.to_string()),
                    },
                }
            }

            ui.add_space(10.0);

            if let Some(ref err) = self.error_message {
                ui.colored_label(egui::Color32::RED, format!("Error: {}", err));
            } else if self.assignments.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.add_space(20.0);
                    ui.label(
                        egui::RichText::new("No subjects allocated yet.")
                            .strong()
                            .color(egui::Color32::DARK_GRAY),
                    );
                    ui.label(
                        egui::RichText::new(
                            "Configure parameters and click 'Allocate Subjects' to begin.",
                        )
                        .color(egui::Color32::GRAY),
                    );
                });
            } else {
                ui.heading("Assignments:");

                let mut treatment_count = 0;
                let mut control_count = 0;
                for group in &self.assignments {
                    match group {
                        Group::Treatment => treatment_count += 1,
                        Group::Control => control_count += 1,
                    }
                }

                ui.label(format!("Total Treatment: {}", treatment_count));
                ui.label(format!("Total Control: {}", control_count));

                egui::ScrollArea::vertical()
                    .max_height(300.0)
                    .show(ui, |ui| {
                        for (i, group) in self.assignments.iter().enumerate() {
                            let group_str = match group {
                                Group::Treatment => "Treatment",
                                Group::Control => "Control",
                            };
                            ui.label(format!("Subject {}: {}", i + 1, group_str));
                        }
                    });
            }
        });
    }
}

// [cite:clinical_trials_statistics]


inventory::submit! {
    crate::framework::ToolMetadata {
        name: "RandomizationTool",
        domain: "clinical_trials",
        tags: &[],
        build: || Box::new(RandomizationTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for RandomizationTool {
    fn theory_description(&self) -> String { "Theoretical context not available.".into() }
    fn phonetic_description(&self) -> String { "Theoretical context not available.".into() }
    fn theory_citation(&self) -> String { "Uncited".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}

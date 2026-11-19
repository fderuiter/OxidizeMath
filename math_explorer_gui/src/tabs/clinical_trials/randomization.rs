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

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RandomizationState {
    randomization_type: RandomizationType,
    n_subjects: usize,
    block_size: usize,
}

impl InteractiveTool for RandomizationTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Randomization"
    }
    fn domain(&self) -> &'static str {
        "clinical_trials"
    }
    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::none()
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&RandomizationState {
            randomization_type: self.randomization_type,
            n_subjects: self.n_subjects,
            block_size: self.block_size,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: RandomizationState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid randomization state: {e}"))?;
        if !(1..=1000).contains(&s.n_subjects) || !(2..=100).contains(&s.block_size) {
            return Err("Randomization subject count or block_size out of bounds".into());
        }
        if s.randomization_type == RandomizationType::Block {
            BlockRandomizer::new(s.block_size).map_err(|e| e.to_string())?;
        }
        self.randomization_type = s.randomization_type;
        self.n_subjects = s.n_subjects;
        self.block_size = s.block_size;
        self.assignments.clear();
        self.error_message = None;
        Ok(())
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

            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            if let Some(ref err) = self.error_message {
                ui.colored_label(palette.error, format!("Error: {}", err));
            } else if self.assignments.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.add_space(20.0);
                    ui.label(
                        egui::RichText::new("No subjects allocated yet.")
                            .strong()
                            .color(palette.muted_text),
                    );
                    ui.label(
                        egui::RichText::new(
                            "Configure parameters and click 'Allocate Subjects' to begin.",
                        )
                        .color(palette.muted_text),
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
        id: "RandomizationTool",
        display_name: "Randomization",
        domain: "clinical_trials",
        tags: &[],
        build: || Box::new(RandomizationTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for RandomizationTool {
    fn theory_description(&self) -> String {
        let scheme = match self.randomization_type {
            RandomizationType::Simple => "simple Bernoulli randomization with equal allocation probability $P(\\text{Treatment}) = 0.5$",
            RandomizationType::Block => "permuted block randomization with block size guaranteeing periodic balance",
        };
        format!(
            "Clinical trial subject allocation using {} for $N = {}$ subjects to eliminate selection bias and balance prognostic factors.",
            scheme, self.n_subjects
        )
    }
    fn phonetic_description(&self) -> String {
        "Clinical trial randomization allocation theory".into()
    }
    fn theory_citation(&self) -> String {
        "Randomization in Clinical Trials: Theory and Practice (Rosenberger & Lachin, 2015)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Simple".into(),
            "Independent coin-toss allocation with potential small-sample imbalance".into(),
        );
        map.insert(
            "Block".into(),
            "Permuted block allocation ensuring exact balance at block boundaries".into(),
        );
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Permuted Block Randomization",
    "domain": "clinical_trials",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "randomization_type": "Block",
    "n_subjects": 24,
    "block_size": 4
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "randomization_starter",
        title: "Permuted Block Randomization",
        description: "Simple and permuted block patient allocation maintaining treatment balance across trial arms.",
        category: "Biostatistics",
        icon: "🎲",
        domain: "clinical_trials",
        tool_name: Some("RandomizationTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

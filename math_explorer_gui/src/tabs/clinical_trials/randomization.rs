//! Interactive Clinical Trial Subject Allocation and Simpson's Paradox Tool.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::simpson_paradox::{SimpsonParadoxConfig, SimpsonParadoxLab};
use crate::accessibility::AccessibleHoverText;
use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use math_explorer::applied::clinical_trials::design::{
    AllocationStrategy, BlockRandomizer, Group, SimpleRandomizer,
};
use oxidize_core::rng::{OxidizeRng, BASE_SEED};

#[derive(PartialEq, Eq, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
enum RandomizationType {
    Simple,
    Block,
    Stratified,
}

/// Interactive clinical trial randomization and stratified Simpson's paradox explorer.
pub struct RandomizationTool {
    randomization_type: RandomizationType,
    n_subjects: usize,
    block_size: usize,
    seed_counter: u64,
    assignments: Vec<Group>,
    error_message: Option<String>,
    /// Hosted stratified confounding and Simpson's paradox laboratory.
    pub simpson_lab: SimpsonParadoxLab,
}

impl Default for RandomizationTool {
    fn default() -> Self {
        Self {
            randomization_type: RandomizationType::Simple,
            n_subjects: 10,
            block_size: 4,
            seed_counter: 0,
            assignments: Vec::new(),
            error_message: None,
            simpson_lab: SimpsonParadoxLab::default(),
        }
    }
}

impl RandomizationTool {
    /// Increments `seed_counter` and draws a fresh deterministic subject allocation.
    pub fn allocate_subjects(&mut self) {
        self.error_message = None;
        self.assignments.clear();
        self.seed_counter = self.seed_counter.wrapping_add(1);
        let mut rng = OxidizeRng::from_seed(BASE_SEED ^ self.seed_counter);

        match self.randomization_type {
            RandomizationType::Simple => match SimpleRandomizer.assign(&mut rng, self.n_subjects) {
                Ok(assignments) => self.assignments = assignments,
                Err(e) => self.error_message = Some(e.to_string()),
            },
            RandomizationType::Block => match BlockRandomizer::new(self.block_size) {
                Ok(randomizer) => match randomizer.assign(&mut rng, self.n_subjects) {
                    Ok(assignments) => self.assignments = assignments,
                    Err(e) => self.error_message = Some(e.to_string()),
                },
                Err(e) => self.error_message = Some(e.to_string()),
            },
            RandomizationType::Stratified => {
                self.simpson_lab
                    .balance_via_stratified_randomizer(&mut self.seed_counter);
            }
        }
    }

    fn render_mode_selector(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
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
            ui.radio_value(
                &mut self.randomization_type,
                RandomizationType::Stratified,
                "Stratified (Simpson's Paradox Lab)",
            );
        });
    }

    fn render_unstratified_controls(&mut self, ui: &mut egui::Ui) {
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

        let enter_pressed = crate::framework::is_pane_activation_shortcut_allowed(ui)
            && ui.input(|i| i.key_pressed(egui::Key::Enter));
        ui.horizontal(|ui| {
            if ui
                .button("▶ Allocate Subjects")
                .accessible_hover_text("Randomly assign subjects to treatment and control groups")
                .clicked()
                || enter_pressed
            {
                self.allocate_subjects();
            }
            ui.label(format!("(Seed #{})", self.seed_counter));
        });
    }

    fn render_unstratified_results(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        if let Some(ref err) = self.error_message {
            ui.colored_label(palette.error, format!("Error: {err}"));
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
            self.render_assignment_list(ui);
        }
    }

    fn render_assignment_list(&self, ui: &mut egui::Ui) {
        ui.heading("Assignments:");
        let treatment_count = self
            .assignments
            .iter()
            .filter(|&&g| g == Group::Treatment)
            .count();
        let control_count = self.assignments.len() - treatment_count;

        ui.label(format!("Total Treatment: {treatment_count}"));
        ui.label(format!("Total Control: {control_count}"));

            egui::ScrollArea::vertical()
                .max_height(300.0)
                .show(ui, |ui| {
                    for (i, group) in self.assignments.iter().enumerate() {
                        let group_str = match group {
                            Group::Treatment => "Treatment",
                            Group::Control => "Control",
                        };
                        ui.label(format!("Subject {}: {group_str}", i + 1));
                    }
                });
    }

    fn build_export_tables(&self) -> Vec<crate::framework::ExportTable> {
        use crate::framework::export_table::{ExportCell, ExportTable};
        if self.randomization_type == RandomizationType::Stratified || self.assignments.is_empty() {
            return Vec::new();
        }
        let method = match self.randomization_type {
            RandomizationType::Simple => "Simple",
            RandomizationType::Block => "Block",
            RandomizationType::Stratified => "Stratified",
        };
        let block = match self.randomization_type {
            RandomizationType::Block => self.block_size as u64,
            _ => 0,
        };
        let rows = self
            .assignments
            .iter()
            .enumerate()
            .map(|(idx, grp)| {
                let g_name = match grp {
                    Group::Treatment => "Treatment",
                    Group::Control => "Control",
                };
                vec![
                    ExportCell::UInt((idx + 1) as u64),
                    ExportCell::Text(g_name.into()),
                    ExportCell::Text(method.into()),
                    ExportCell::UInt(block),
                ]
            })
            .collect();
        vec![ExportTable::new(
            "Subject Allocations",
            vec![
                "subject_number".into(),
                "group".into(),
                "method".into(),
                "block_size".into(),
            ],
            rows,
        )]
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RandomizationState {
    randomization_type: RandomizationType,
    n_subjects: usize,
    block_size: usize,
    #[serde(default, skip_serializing_if = "SimpsonParadoxConfig::is_default")]
    simpson_config: SimpsonParadoxConfig,
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
    fn export_capabilities(&self) -> ExportCapabilities {
        if self.randomization_type == RandomizationType::Stratified {
            ExportCapabilities::plot()
        } else if !self.assignments.is_empty() {
            ExportCapabilities::dataset_only()
        } else {
            ExportCapabilities::unpopulated(
                "Click 'Allocate Subjects' to generate a trial randomization schedule before exporting.",
            )
        }
    }
    fn step_simulation(&mut self) {
        if self.randomization_type != RandomizationType::Stratified && self.assignments.is_empty() {
            self.allocate_subjects();
        }
    }
    fn export_tables(&self) -> Vec<crate::framework::ExportTable> {
        self.build_export_tables()
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&RandomizationState {
            randomization_type: self.randomization_type,
            n_subjects: self.n_subjects,
            block_size: self.block_size,
            simpson_config: self.simpson_lab.config.clone(),
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: RandomizationState =
            serde_json::from_str(state).map_err(|e| format!("Invalid randomization state: {e}"))?;
        if !(1..=1000).contains(&s.n_subjects) || !(2..=100).contains(&s.block_size) {
            return Err("Randomization subject count or block_size out of bounds".into());
        }
        if s.randomization_type == RandomizationType::Block {
            BlockRandomizer::new(s.block_size).map_err(|e| e.to_string())?;
        }
        s.simpson_config.validate()?;
        self.randomization_type = s.randomization_type;
        self.n_subjects = s.n_subjects;
        self.block_size = s.block_size;
        self.simpson_lab.config = s.simpson_config;
        self.simpson_lab.recalculate();
        self.assignments.clear();
        self.error_message = None;
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Interactive Subject Allocation");
                ui.add_space(8.0);
                self.render_mode_selector(ui);
                ui.add_space(6.0);
                if self.randomization_type == RandomizationType::Stratified {
                    self.simpson_lab.show_ui(ui, &mut self.seed_counter);
                } else {
                    self.render_unstratified_controls(ui);
                    ui.add_space(10.0);
                    self.render_unstratified_results(ui);
                }
            });
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
            RandomizationType::Stratified => "stratified block randomization balancing prognostic strata to prevent Simpson's paradox confounding",
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
        map.insert(
            "Stratified".into(),
            "Within-stratum block randomization preventing Simpson's paradox reversals".into(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_randomization_seed_counter_produces_distinct_allocations() {
        let mut tool = RandomizationTool {
            randomization_type: RandomizationType::Simple,
            n_subjects: 40,
            ..Default::default()
        };
        tool.allocate_subjects();
        let first = tool.assignments.clone();
        assert_eq!(tool.seed_counter, 1);
        assert_eq!(first.len(), 40);

        tool.allocate_subjects();
        let second = tool.assignments.clone();
        assert_eq!(tool.seed_counter, 2);
        assert_ne!(
            first, second,
            "Consecutive Allocate Subjects clicks must advance seed_counter"
        );
    }

    #[test]
    fn test_randomization_legacy_and_stratified_state_roundtrip() {
        let mut tool = RandomizationTool::default();
        let legacy = r#"{"randomization_type":"Block","n_subjects":24,"block_size":6}"#;
        tool.try_load_state(legacy).unwrap();
        assert_eq!(tool.save_state().as_deref(), Some(legacy));

        tool.randomization_type = RandomizationType::Stratified;
        tool.simpson_lab.config.mild_treatment_share = 0.35;
        tool.simpson_lab.apply_allocation_sliders();
        let saved = tool.save_state().unwrap();

        let mut restored = RandomizationTool::default();
        restored.try_load_state(&saved).unwrap();
        assert_eq!(restored.randomization_type, RandomizationType::Stratified);
        assert!((restored.simpson_lab.config.mild_treatment_share - 0.35).abs() < 1e-9);
    }
}

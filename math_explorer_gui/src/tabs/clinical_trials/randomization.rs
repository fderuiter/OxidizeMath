//! Interactive Clinical Trial Subject Allocation and Simpson's Paradox Tool.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::clinical_io::ClinicalIoState;
use super::randomization_schedule::{
    build_cumulative_balance_datasets, build_deterministic_stratum_schedule,
    build_unstratified_schedule, render_cumulative_balance_plot, render_schedule_export_bar,
    SubjectAllocationRecord,
};
use super::simpson_paradox::{SimpsonParadoxConfig, SimpsonParadoxLab};
use crate::accessibility::AccessibleHoverText;
use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use eframe::egui;
use math_explorer::applied::clinical_trials::design::{
    AllocationStrategy, BlockRandomizer, Group, SimpleRandomizer,
};
use oxidize_core::rng::{OxidizeRng, BASE_SEED};

/// Clinical trial subject allocation scheme.
#[derive(PartialEq, Eq, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum RandomizationType {
    /// Independent coin-flip Bernoulli allocation (`SimpleRandomizer`).
    Simple,
    /// Permuted block randomization (`BlockRandomizer`).
    Block,
    /// Stratified permuted block randomization (`StratifiedRandomizer` + Simpson's Paradox Lab).
    Stratified,
}

impl RandomizationType {
    /// Short mode label for `.csv` and `.json` allocation schedule exports.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Simple => "Simple",
            Self::Block => "Block",
            Self::Stratified => "Stratified",
        }
    }
}

/// Interactive clinical trial randomization and stratified Simpson's paradox explorer.
pub struct RandomizationTool {
    /// Active randomization strategy (`Simple`, `Block`, or `Stratified`).
    pub randomization_type: RandomizationType,
    /// Number of subjects to enroll in `Simple` or `Block` modes (`1..=1000`).
    pub n_subjects: usize,
    /// Permuted block size in `Block` mode (`2..=100`, even).
    pub block_size: usize,
    /// Monotonic deterministic RNG seed counter incremented on each explicit allocation click.
    pub seed_counter: u64,
    /// Latest unstratified `Group` vector (`Simple` / `Block`).
    pub assignments: Vec<Group>,
    /// Retained per-subject allocation records (`Simple` / `Block`).
    pub schedule: Vec<SubjectAllocationRecord>,
    /// Validation or domain error message, if any.
    pub error_message: Option<String>,
    /// Hosted stratified confounding and Simpson's paradox laboratory.
    pub simpson_lab: SimpsonParadoxLab,
    /// Non-blocking save/status helper for `.csv` and `.json` schedule downloads.
    pub(crate) io_state: ClinicalIoState,
}

impl Default for RandomizationTool {
    fn default() -> Self {
        let mut tool = Self {
            randomization_type: RandomizationType::Simple,
            n_subjects: 10,
            block_size: 4,
            seed_counter: 0,
            assignments: Vec::new(),
            schedule: Vec::new(),
            error_message: None,
            simpson_lab: SimpsonParadoxLab::default(),
            io_state: ClinicalIoState::default(),
        };
        tool.refresh_preview_without_seed_advance();
        tool
    }
}

impl RandomizationTool {
    /// Draws a deterministic baseline preview at `BASE_SEED ^ seed_counter` without advancing `seed_counter`.
    pub(crate) fn refresh_preview_without_seed_advance(&mut self) {
        self.error_message = None;
        if self.randomization_type == RandomizationType::Stratified {
            return;
        }
        let mut rng = OxidizeRng::from_seed(BASE_SEED ^ self.seed_counter);
        self.run_unstratified_draw(&mut rng);
    }

    fn run_unstratified_draw(&mut self, rng: &mut OxidizeRng) {
        self.assignments.clear();
        self.schedule.clear();
        match self.randomization_type {
            RandomizationType::Simple => match SimpleRandomizer.assign(rng, self.n_subjects) {
                Ok(assignments) => {
                    self.schedule = build_unstratified_schedule(&assignments, None);
                    self.assignments = assignments;
                }
                Err(e) => self.error_message = Some(e.to_string()),
            },
            RandomizationType::Block => match BlockRandomizer::new(self.block_size) {
                Ok(randomizer) => match randomizer.assign(rng, self.n_subjects) {
                    Ok(assignments) => {
                        self.schedule =
                            build_unstratified_schedule(&assignments, Some(self.block_size));
                        self.assignments = assignments;
                    }
                    Err(e) => self.error_message = Some(e.to_string()),
                },
                Err(e) => self.error_message = Some(e.to_string()),
            },
            RandomizationType::Stratified => {}
        }
    }

    /// Increments `seed_counter` and draws a fresh deterministic subject allocation.
    pub fn allocate_subjects(&mut self) {
        self.error_message = None;
        if self.randomization_type == RandomizationType::Stratified {
            self.simpson_lab
                .balance_via_stratified_randomizer(&mut self.seed_counter);
            return;
        }
        self.seed_counter = self.seed_counter.wrapping_add(1);
        let mut rng = OxidizeRng::from_seed(BASE_SEED ^ self.seed_counter);
        self.run_unstratified_draw(&mut rng);
    }

    /// Returns the active subject allocation schedule across `Simple`, `Block`, or `Stratified`.
    #[must_use]
    pub fn active_schedule(&self) -> &[SubjectAllocationRecord] {
        if self.randomization_type == RandomizationType::Stratified {
            &self.simpson_lab.allocated_schedule
        } else {
            &self.schedule
        }
    }

    fn render_mode_selector(&mut self, ui: &mut egui::Ui) {
        let prev_mode = self.randomization_type;
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
        if self.randomization_type != prev_mode
            && self.randomization_type != RandomizationType::Stratified
            && self.schedule.is_empty()
        {
            self.refresh_preview_without_seed_advance();
        }
    }

    fn render_unstratified_controls(&mut self, ui: &mut egui::Ui) {
        let mut params_changed = false;
        ui.horizontal(|ui| {
            ui.label("Number of Subjects:");
            params_changed |= ui
                .add(egui::DragValue::new(&mut self.n_subjects).range(1..=1000))
                .changed();
        });

        if self.randomization_type == RandomizationType::Block {
            ui.horizontal(|ui| {
                ui.label("Block Size (even):");
                params_changed |= ui
                    .add(egui::DragValue::new(&mut self.block_size).range(2..=100))
                    .changed();
            });
        }
        if params_changed {
            self.refresh_preview_without_seed_advance();
        }

        ui.add_space(6.0);
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

    fn render_unstratified_results(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        if let Some(ref err) = self.error_message {
            ui.colored_label(palette.error, format!("Error: {err}"));
            return;
        }
        render_cumulative_balance_plot(ui, "randomization_balance_plot", &self.schedule);
        ui.add_space(6.0);
        render_schedule_export_bar(
            &mut self.io_state,
            ui,
            self.randomization_type.label(),
            self.seed_counter,
            &self.schedule,
        );
        self.io_state.render_status(ui);
        ui.add_space(6.0);
        self.render_assignment_list(ui);
    }

    fn render_assignment_list(&self, ui: &mut egui::Ui) {
        ui.heading("Subject Allocation Roster:");
        let treatment_count = self
            .assignments
            .iter()
            .filter(|&&g| g == Group::Treatment)
            .count();
        let control_count = self.assignments.len().saturating_sub(treatment_count);
        ui.label(format!(
            "Total Treatment: {treatment_count} | Total Control: {control_count}"
        ));

        egui::ScrollArea::vertical()
            .id_salt("unstratified_roster_scroll")
            .max_height(220.0)
            .show(ui, |ui| {
                for rec in &self.schedule {
                    ui.label(format!(
                        "#{} | {} ({}): {}",
                        rec.sequence, rec.subject_id, rec.stratum, rec.arm
                    ));
                }
            });
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
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        build_cumulative_balance_datasets(self.active_schedule())
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::DEFAULT
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
        self.state_bounds().validate_json(state)?;
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
        self.simpson_lab.allocated_schedule = build_deterministic_stratum_schedule(
            self.simpson_lab.config.mild_counts,
            self.simpson_lab.config.severe_counts,
        );
        self.simpson_lab.recalculate();
        self.refresh_preview_without_seed_advance();
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.io_state.poll_save("allocation schedule");
        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Interactive Subject Allocation");
                ui.add_space(6.0);
                self.render_mode_selector(ui);
                ui.add_space(6.0);
                if self.randomization_type == RandomizationType::Stratified {
                    self.simpson_lab.show_ui(ui, &mut self.seed_counter);
                    ui.add_space(6.0);
                    render_schedule_export_bar(
                        &mut self.io_state,
                        ui,
                        self.randomization_type.label(),
                        self.seed_counter,
                        &self.simpson_lab.allocated_schedule,
                    );
                    self.io_state.render_status(ui);
                } else {
                    self.render_unstratified_controls(ui);
                    ui.add_space(8.0);
                    self.render_unstratified_results(ui);
                }
            });
        });
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "RandomizationTool",
        display_name: "Randomization",
        domain: "clinical_trials",
        tags: &[
            "permuted block",
            "stratified",
            "bernoulli",
            "allocation",
            "confounding",
            "simpson",
            "rct",
        ],
        build: || Box::new(RandomizationTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for RandomizationTool {
    fn theory_description(&self) -> String {
        let scheme = match self.randomization_type {
            RandomizationType::Simple => {
                "simple Bernoulli randomization with equal allocation probability $P(\\text{Treatment}) = 0.5$"
            }
            RandomizationType::Block => {
                "permuted block randomization with block size guaranteeing periodic balance"
            }
            RandomizationType::Stratified => {
                "stratified block randomization balancing prognostic strata to prevent Simpson's paradox confounding"
            }
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

/// Starter preset JSON payload for permuted block randomization.
pub const STARTER_PRESET_JSON: &str = r#"{
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
#[path = "randomization_tests.rs"]
mod tests;

// [cite:clinical_trials_statistics]

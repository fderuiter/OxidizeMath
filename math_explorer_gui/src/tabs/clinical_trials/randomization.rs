//! Interactive Clinical Trial Subject Allocation and Simpson's Paradox Tool.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::clinical_io::ClinicalIoState;
pub use super::randomization_schedule::STARTER_PRESET_JSON;
use super::randomization_schedule::{
    build_cumulative_balance_datasets, build_deterministic_stratum_schedule,
    build_stratified_schedule_from_map, build_unstratified_schedule, default_seed,
    default_strata_spec, is_default_seed, is_default_strata_spec, parse_strata_patients,
    randomization_available_descriptions, render_cumulative_balance_plot, render_roster_summary,
    render_schedule_export_bar, SubjectAllocationRecord, DEFAULT_RNG_SEED,
};
use super::simpson_paradox::{SimpsonParadoxConfig, SimpsonParadoxLab};
use crate::accessibility::AccessibleHoverText;
use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use eframe::egui;
use math_explorer::applied::clinical_trials::design::{
    AllocationStrategy, BlockRandomizer, Group, SimpleRandomizer, StratifiedRandomizer,
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

/// Active view within `RandomizationType::Stratified`.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
pub enum StratifiedView {
    /// Cumulative arm-balance plot across custom `'StratumName:Count'` cohorts.
    ScheduleBalance,
    /// Interactive confounding & Simpson's paradox laboratory (`Mild` / `Severe`).
    #[default]
    SimpsonParadox,
}

/// Interactive clinical trial randomization and stratified Simpson's paradox explorer.
pub struct RandomizationTool {
    /// Active randomization strategy (`Simple`, `Block`, or `Stratified`).
    pub randomization_type: RandomizationType,
    /// Number of subjects to enroll in `Simple` or `Block` modes (`1..=1000`).
    pub n_subjects: usize,
    /// Permuted block size in `Block` and `Stratified` modes (`2..=100`, even).
    pub block_size: usize,
    /// Base deterministic RNG seed (`OxidizeRng`).
    pub seed: u64,
    /// Monotonic deterministic RNG seed counter incremented on each explicit allocation click.
    pub seed_counter: u64,
    /// Multi-stratum `'Name:Count'` enrollment specification for `Stratified` schedules.
    pub strata_spec: String,
    /// Selected view when `randomization_type == RandomizationType::Stratified`.
    pub stratified_view: StratifiedView,
    /// Latest `Group` vector (`Simple` / `Block` / `strata_spec`).
    pub assignments: Vec<Group>,
    /// Retained per-subject allocation records (`Simple` / `Block` / `strata_spec`).
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
            seed: DEFAULT_RNG_SEED,
            seed_counter: 0,
            strata_spec: default_strata_spec(),
            stratified_view: StratifiedView::SimpsonParadox,
            assignments: Vec::new(),
            schedule: Vec::new(),
            error_message: None,
            simpson_lab: SimpsonParadoxLab::default(),
            io_state: ClinicalIoState::default(),
        };
        tool.recompute_schedule();
        tool
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RandomizationState {
    randomization_type: RandomizationType,
    n_subjects: usize,
    block_size: usize,
    #[serde(default = "default_seed", skip_serializing_if = "is_default_seed")]
    seed: u64,
    #[serde(
        default = "default_strata_spec",
        skip_serializing_if = "is_default_strata_spec"
    )]
    strata_spec: String,
    #[serde(default, skip_serializing_if = "SimpsonParadoxConfig::is_default")]
    simpson_config: SimpsonParadoxConfig,
}

impl RandomizationTool {
    /// Effective deterministic seed combining `seed` and `seed_counter`.
    #[must_use]
    pub const fn effective_seed(&self) -> u64 {
        if self.seed_counter == 0 {
            self.seed
        } else {
            BASE_SEED ^ self.seed ^ self.seed_counter
        }
    }

    /// Increments `seed_counter` and draws a fresh deterministic subject allocation.
    pub fn allocate_subjects(&mut self) {
        self.seed_counter = self.seed_counter.wrapping_add(1);
        self.recompute_schedule();
        if self.randomization_type == RandomizationType::Stratified {
            let mut simpson_seed = self.seed_counter.wrapping_sub(1);
            self.simpson_lab
                .balance_via_stratified_randomizer(&mut simpson_seed);
        }
    }

    /// Returns the active subject allocation schedule across `Simple`, `Block`, or `Stratified`.
    #[must_use]
    pub fn active_schedule(&self) -> &[SubjectAllocationRecord] {
        if self.randomization_type == RandomizationType::Stratified
            && self.stratified_view == StratifiedView::SimpsonParadox
        {
            &self.simpson_lab.allocated_schedule
        } else {
            &self.schedule
        }
    }

    pub(crate) fn recompute_schedule(&mut self) {
        self.error_message = None;
        self.schedule.clear();
        self.assignments.clear();
        let mut rng = OxidizeRng::from_seed(self.effective_seed());

        match self.randomization_type {
            RandomizationType::Simple => match SimpleRandomizer.assign(&mut rng, self.n_subjects) {
                Ok(groups) => {
                    self.schedule = build_unstratified_schedule(&groups, None);
                    self.assignments = groups;
                }
                Err(e) => self.error_message = Some(e.to_string()),
            },
            RandomizationType::Block => {
                match BlockRandomizer::new(self.block_size)
                    .and_then(|r| r.assign(&mut rng, self.n_subjects))
                {
                    Ok(groups) => {
                        self.schedule = build_unstratified_schedule(&groups, Some(self.block_size));
                        self.assignments = groups;
                    }
                    Err(e) => self.error_message = Some(e.to_string()),
                }
            }
            RandomizationType::Stratified => self.recompute_stratified_schedule(&mut rng),
        }
    }

    fn recompute_stratified_schedule(&mut self, rng: &mut OxidizeRng) {
        let res = parse_strata_patients(&self.strata_spec).and_then(|patients| {
            let block = BlockRandomizer::new(self.block_size).map_err(|e| e.to_string())?;
            let map = StratifiedRandomizer::new(block)
                .assign_stratified(rng, &patients)
                .map_err(|e| e.to_string())?;
            Ok((patients, map))
        });
        match res {
            Ok((patients, map)) => {
                self.assignments = patients
                    .iter()
                    .filter_map(|p| map.get(&p.id).copied())
                    .collect();
                self.schedule = build_stratified_schedule_from_map(&patients, &map);
            }
            Err(err) => self.error_message = Some(err),
        }
    }

    fn render_controls_panel(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        ui.heading("Subject Allocation");
        ui.horizontal_wrapped(|ui| {
            for (kind, label) in [
                (RandomizationType::Simple, "Simple"),
                (RandomizationType::Block, "Block"),
                (RandomizationType::Stratified, "Stratified"),
            ] {
                changed |= ui
                    .radio_value(&mut self.randomization_type, kind, label)
                    .changed();
            }
        });
        changed |= self.render_parameter_inputs(ui);
        if changed {
            self.seed_counter = 0;
            self.recompute_schedule();
        }
        self.render_action_buttons_and_roster(ui);
    }

    fn render_parameter_inputs(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("RNG Seed:");
            changed |= ui
                .add(egui::DragValue::new(&mut self.seed).speed(1.0))
                .changed();
            ui.label(format!("(#{})", self.seed_counter));
        });
        if self.randomization_type != RandomizationType::Stratified {
            ui.horizontal(|ui| {
                ui.label("Subjects (N):");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.n_subjects).range(1..=1000))
                    .changed();
            });
        }
        if self.randomization_type != RandomizationType::Simple {
            ui.horizontal(|ui| {
                ui.label("Block Size (even):");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.block_size).range(2..=100))
                    .changed();
            });
        }
        if self.randomization_type == RandomizationType::Stratified {
            ui.label("Strata ('Name:Count, ...'):");
            changed |= ui
                .add(
                    egui::TextEdit::multiline(&mut self.strata_spec)
                        .desired_width(f32::INFINITY)
                        .desired_rows(3),
                )
                .changed();
        }
        changed
    }

    fn render_action_buttons_and_roster(&mut self, ui: &mut egui::Ui) {
        let enter_pressed = crate::framework::is_pane_activation_shortcut_allowed(ui)
            && ui.input(|i| i.key_pressed(egui::Key::Enter));
        if ui
            .button("▶ Allocate Subjects")
            .accessible_hover_text("Run seeded OxidizeRng subject allocation")
            .clicked()
            || enter_pressed
        {
            self.allocate_subjects();
        }
        ui.add_space(4.0);
        let export_sched = self.active_schedule().to_vec();
        let mode_label = self.randomization_type.label();
        let seed = self.effective_seed();
        render_schedule_export_bar(&mut self.io_state, ui, mode_label, seed, &export_sched);
        self.io_state.render_status(ui);
        ui.separator();
        render_roster_summary(ui, self.error_message.as_deref(), self.active_schedule());
    }

    fn render_central_panel(&mut self, ui: &mut egui::Ui) {
        if self.randomization_type == RandomizationType::Stratified {
            ui.horizontal_wrapped(|ui| {
                ui.label("Stratified View:");
                ui.selectable_value(
                    &mut self.stratified_view,
                    StratifiedView::ScheduleBalance,
                    "📈 Cumulative Arm Balance",
                );
                ui.selectable_value(
                    &mut self.stratified_view,
                    StratifiedView::SimpsonParadox,
                    "⚖ Simpson's Paradox Lab",
                );
            });
            ui.separator();
            if self.stratified_view == StratifiedView::SimpsonParadox {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.simpson_lab.show_ui(ui, &mut self.seed_counter);
                });
                return;
            }
        }
        ui.heading("Cumulative Arm Balance Trajectory");
        render_cumulative_balance_plot(ui, "randomization_balance_plot", &self.schedule);
    }
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
            seed: self.seed,
            strata_spec: self.strata_spec.clone(),
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
        if s.randomization_type != RandomizationType::Simple {
            BlockRandomizer::new(s.block_size).map_err(|e| e.to_string())?;
        }
        if s.randomization_type == RandomizationType::Stratified {
            let _ = parse_strata_patients(&s.strata_spec)?;
        }
        s.simpson_config.validate()?;
        self.randomization_type = s.randomization_type;
        self.n_subjects = s.n_subjects;
        self.block_size = s.block_size;
        self.seed = s.seed;
        self.seed_counter = 0;
        self.stratified_view =
            if !is_default_strata_spec(&s.strata_spec) && s.simpson_config.is_default() {
                StratifiedView::ScheduleBalance
            } else {
                StratifiedView::SimpsonParadox
            };
        self.strata_spec = s.strata_spec;
        self.simpson_lab.config = s.simpson_config;
        self.simpson_lab.allocated_schedule = build_deterministic_stratum_schedule(
            self.simpson_lab.config.mild_counts,
            self.simpson_lab.config.severe_counts,
        );
        self.simpson_lab.recalculate();
        self.recompute_schedule();
        if let Some(err) = &self.error_message {
            return Err(err.clone());
        }
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.io_state.poll_save("allocation schedule");
        let panel_id = crate::framework::scoped_panel_id(ui, "randomization_side_panel");
        crate::framework::PaneSidePanel::left(panel_id)
            .min_width(300.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.render_controls_panel(ui));
            });
        egui::CentralPanel::default().show_inside(ui, |ui| self.render_central_panel(ui));
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        self.simpson_lab.observables()
    }

    fn sync_callouts(&mut self, specs: &[crate::framework::CalloutSpec]) {
        if !specs.is_empty() && self.randomization_type == RandomizationType::Stratified {
            self.stratified_view = StratifiedView::SimpsonParadox;
        }
        self.simpson_lab.set_hud_callouts(specs);
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
                "permuted block randomization guaranteeing periodic balance"
            }
            RandomizationType::Stratified => {
                "stratified permuted-block randomization balancing prognostic strata to prevent Simpson's paradox confounding"
            }
        };
        format!(
            "Clinical trial subject allocation using {} (seed {}) across {} subjects to eliminate selection bias.",
            scheme,
            self.effective_seed(),
            self.active_schedule().len()
        )
    }

    fn phonetic_description(&self) -> String {
        "Clinical trial randomization allocation theory".into()
    }

    fn theory_citation(&self) -> String {
        "Randomization in Clinical Trials: Theory and Practice (Rosenberger & Lachin, 2015)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        randomization_available_descriptions()
    }

    fn theory_parameters(
        &self,
    ) -> std::collections::HashMap<String, scientific_metadata::theory::ParameterConstraint> {
        super::simpson_atlas::simpson_theory_parameters()
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "randomization_starter",
        title: "Permuted Block Randomization",
        description: "Simple, permuted-block, and stratified patient allocation maintaining balance across trial arms.",
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

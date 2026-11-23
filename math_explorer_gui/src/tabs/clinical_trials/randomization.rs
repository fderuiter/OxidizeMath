//! Interactive Clinical Trial Subject Allocation and Simpson's Paradox Tool.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::randomization_io::{
    build_cumulative_series, default_seed, default_strata_spec, draw_balance_plot,
    format_schedule_csv, format_schedule_json, is_default_seed, is_default_strata_spec,
    parse_strata_patients, ScheduleRow, ScheduleSaveIo, DEFAULT_RNG_SEED, STARTER_PRESET_JSON,
};
use super::simpson_paradox::{SimpsonParadoxConfig, SimpsonParadoxLab};
use crate::accessibility::AccessibleHoverText;
use crate::framework::{ExportCapabilities, InteractiveTool};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use math_explorer::applied::clinical_trials::design::{
    AllocationStrategy, BlockRandomizer, Group, SimpleRandomizer, StratifiedRandomizer,
};
use oxidize_core::rng::{OxidizeRng, BASE_SEED};

#[derive(Debug, PartialEq, Eq, Clone, Copy, serde::Serialize, serde::Deserialize)]
enum RandomizationType {
    Simple,
    Block,
    Stratified,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
enum StratifiedView {
    #[default]
    ScheduleBalance,
    SimpsonParadox,
}

/// Interactive clinical trial randomization and stratified Simpson's paradox explorer.
pub struct RandomizationTool {
    randomization_type: RandomizationType,
    n_subjects: usize,
    block_size: usize,
    seed: u64,
    seed_counter: u64,
    strata_spec: String,
    stratified_view: StratifiedView,
    assignments: Vec<Group>,
    schedule: Vec<ScheduleRow>,
    error_message: Option<String>,
    save_io: ScheduleSaveIo,
    /// Hosted stratified confounding and Simpson's paradox laboratory.
    pub simpson_lab: SimpsonParadoxLab,
}

impl Default for RandomizationTool {
    fn default() -> Self {
        let mut tool = Self {
            randomization_type: RandomizationType::Simple,
            n_subjects: 24,
            block_size: 4,
            seed: DEFAULT_RNG_SEED,
            seed_counter: 0,
            strata_spec: default_strata_spec(),
            stratified_view: StratifiedView::ScheduleBalance,
            assignments: Vec::new(),
            schedule: Vec::new(),
            error_message: None,
            save_io: ScheduleSaveIo::default(),
            simpson_lab: SimpsonParadoxLab::default(),
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
    const fn effective_seed(&self) -> u64 {
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
    }

    fn recompute_schedule(&mut self) {
        self.error_message = None;
        self.schedule.clear();
        self.assignments.clear();
        let mut rng = OxidizeRng::from_seed(self.effective_seed());

        match self.randomization_type {
            RandomizationType::Simple => match SimpleRandomizer.assign(&mut rng, self.n_subjects) {
                Ok(groups) => self.set_unstratified_schedule(groups),
                Err(e) => self.error_message = Some(e.to_string()),
            },
            RandomizationType::Block => {
                match BlockRandomizer::new(self.block_size)
                    .and_then(|r| r.assign(&mut rng, self.n_subjects))
                {
                    Ok(groups) => self.set_unstratified_schedule(groups),
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
                self.schedule = patients
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, p)| {
                        let group = *map.get(&p.id)?;
                        Some(ScheduleRow {
                            subject_index: i + 1,
                            patient_id: p.id,
                            stratum: p.stratum,
                            group,
                        })
                    })
                    .collect();
                self.assignments = self.schedule.iter().map(|r| r.group).collect();
            }
            Err(err) => self.error_message = Some(err),
        }
    }

    fn set_unstratified_schedule(&mut self, groups: Vec<Group>) {
        self.assignments.clone_from(&groups);
        self.schedule = groups
            .into_iter()
            .enumerate()
            .map(|(i, group)| ScheduleRow {
                subject_index: i + 1,
                patient_id: format!("S{:04}", i + 1),
                stratum: "Unstratified".to_string(),
                group,
            })
            .collect();
    }

    fn render_controls_panel(&mut self, ui: &mut egui::Ui) {
        self.save_io.poll();
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
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("▶ Allocate Subjects")
                .accessible_hover_text("Run seeded OxidizeRng subject allocation")
                .clicked()
                || enter_pressed
            {
                self.allocate_subjects();
            }
            let ready = !self.schedule.is_empty();
            if ui
                .add_enabled(ready, egui::Button::new("💾 Save CSV"))
                .accessible_hover_text("Download seeded allocation schedule as CSV")
                .clicked()
            {
                self.save_io
                    .save_csv(format_schedule_csv(&self.schedule, self.effective_seed()));
            }
            if ui
                .add_enabled(ready, egui::Button::new("💾 Save JSON"))
                .accessible_hover_text("Download seeded allocation schedule as JSON")
                .clicked()
            {
                self.save_io.save_json(format_schedule_json(
                    &self.schedule,
                    self.effective_seed(),
                    self.block_size,
                ));
            }
        });
        self.render_roster_summary(ui);
    }

    fn render_roster_summary(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        if let Some((is_err, msg)) = &self.save_io.status {
            ui.colored_label(if *is_err { palette.error } else { palette.info }, msg);
        }
        if let Some(err) = &self.error_message {
            ui.colored_label(palette.error, format!("Error: {err}"));
            return;
        }
        let trt_n = self
            .schedule
            .iter()
            .filter(|r| r.group == Group::Treatment)
            .count();
        let ctrl_n = self.schedule.len().saturating_sub(trt_n);
        render_copyable_metric(ui, "Treatment / Control", &format!("{trt_n} / {ctrl_n}"));
        egui::ScrollArea::vertical()
            .max_height(200.0)
            .show(ui, |ui| {
                for row in &self.schedule {
                    ui.label(format!(
                        "{} [{}]: {}",
                        row.patient_id,
                        row.stratum,
                        row.group_label()
                    ));
                }
            });
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
        draw_balance_plot(&self.schedule, ui);
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
        if self.schedule.is_empty() {
            return Vec::new();
        }
        let (trt, ctrl, diff) = build_cumulative_series(&self.schedule);
        vec![
            ("Treatment Enrolled".into(), trt),
            ("Control Enrolled".into(), ctrl),
            ("Imbalance (T - C)".into(), diff),
        ]
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
        self.strata_spec = s.strata_spec;
        self.simpson_lab.config = s.simpson_config;
        self.simpson_lab.recalculate();
        self.recompute_schedule();
        if let Some(err) = &self.error_message {
            return Err(err.clone());
        }
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "randomization_side_panel");
        crate::framework::PaneSidePanel::left(panel_id)
            .min_width(300.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.render_controls_panel(ui));
            });
        egui::CentralPanel::default().show_inside(ui, |ui| self.render_central_panel(ui));
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
            self.schedule.len()
        )
    }
    fn phonetic_description(&self) -> String {
        "Clinical trial randomization allocation theory".into()
    }
    fn theory_citation(&self) -> String {
        "Randomization in Clinical Trials: Theory and Practice (Rosenberger & Lachin, 2015)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        [
            (
                "Simple",
                "Independent coin-toss allocation with potential small-sample imbalance",
            ),
            (
                "Block",
                "Permuted block allocation ensuring exact balance at block boundaries",
            ),
            (
                "Stratified",
                "Within-stratum permuted block randomization preventing Simpson's paradox reversals",
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
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

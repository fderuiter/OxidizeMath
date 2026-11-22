#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::AccessibleHoverText;
use crate::framework::{ExportCapabilities, InteractiveTool};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use math_explorer::applied::clinical_trials::design::{
    AllocationStrategy, BlockRandomizer, Group, SimpleRandomizer, StratifiedRandomizer,
};

use super::randomization_io::{
    build_cumulative_series, format_schedule_csv, format_schedule_json, parse_strata_patients,
    ScheduleRow, ScheduleSaveIo,
};

const DEFAULT_RNG_SEED: u64 = 42;
const DEFAULT_STRATA_SPEC: &str = "Age<65:12, Age>=65:12";

const fn default_seed() -> u64 {
    DEFAULT_RNG_SEED
}

const fn is_default_seed(seed: &u64) -> bool {
    *seed == DEFAULT_RNG_SEED
}

fn default_strata_spec() -> String {
    DEFAULT_STRATA_SPEC.to_string()
}

fn is_default_strata_spec(spec: &str) -> bool {
    spec == DEFAULT_STRATA_SPEC
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, serde::Serialize, serde::Deserialize)]
enum RandomizationType {
    Simple,
    Block,
    Stratified,
}

pub struct RandomizationTool {
    randomization_type: RandomizationType,
    n_subjects: usize,
    block_size: usize,
    seed: u64,
    strata_spec: String,
    schedule: Vec<ScheduleRow>,
    error_message: Option<String>,
    save_io: ScheduleSaveIo,
}

impl Default for RandomizationTool {
    fn default() -> Self {
        let mut tool = Self {
            randomization_type: RandomizationType::Simple,
            n_subjects: 24,
            block_size: 4,
            seed: DEFAULT_RNG_SEED,
            strata_spec: default_strata_spec(),
            schedule: Vec::new(),
            error_message: None,
            save_io: ScheduleSaveIo::default(),
        };
        tool.allocate_subjects();
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
}

impl RandomizationTool {
    fn allocate_subjects(&mut self) {
        self.error_message = None;
        self.schedule.clear();
        let mut rng = oxidize_core::rng::OxidizeRng::new(self.seed);

        match self.randomization_type {
            RandomizationType::Simple => match SimpleRandomizer.assign(&mut rng, self.n_subjects) {
                Ok(groups) => self.set_unstratified_schedule(groups),
                Err(e) => self.error_message = Some(e.to_string()),
            },
            RandomizationType::Block => match BlockRandomizer::new(self.block_size)
                .and_then(|r| r.assign(&mut rng, self.n_subjects))
            {
                Ok(groups) => self.set_unstratified_schedule(groups),
                Err(e) => self.error_message = Some(e.to_string()),
            },
            RandomizationType::Stratified => {
                let res = parse_strata_patients(&self.strata_spec).and_then(|patients| {
                    let block = BlockRandomizer::new(self.block_size).map_err(|e| e.to_string())?;
                    let map = StratifiedRandomizer::new(block)
                        .assign_stratified(&mut rng, &patients)
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
                    }
                    Err(err) => self.error_message = Some(err),
                }
            }
        }
    }

    fn set_unstratified_schedule(&mut self, groups: Vec<Group>) {
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
            changed |= ui
                .radio_value(&mut self.randomization_type, RandomizationType::Simple, "Simple")
                .changed();
            changed |= ui
                .radio_value(&mut self.randomization_type, RandomizationType::Block, "Block")
                .changed();
            changed |= ui
                .radio_value(
                    &mut self.randomization_type,
                    RandomizationType::Stratified,
                    "Stratified",
                )
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("RNG Seed:");
            changed |= ui.add(egui::DragValue::new(&mut self.seed).speed(1.0)).changed();
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
        self.render_action_buttons_and_roster(ui, changed);
    }

    fn render_action_buttons_and_roster(&mut self, ui: &mut egui::Ui, changed: bool) {
        let enter_pressed = crate::framework::is_pane_activation_shortcut_allowed(ui)
            && ui.input(|i| i.key_pressed(egui::Key::Enter));
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("▶ Allocate Subjects")
                .accessible_hover_text("Run seeded OxidizeRng subject allocation")
                .clicked()
                || enter_pressed
                || changed
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
                    .save_csv(format_schedule_csv(&self.schedule, self.seed));
            }
            if ui
                .add_enabled(ready, egui::Button::new("💾 Save JSON"))
                .accessible_hover_text("Download seeded allocation schedule as JSON")
                .clicked()
            {
                self.save_io.save_json(format_schedule_json(
                    &self.schedule,
                    self.seed,
                    self.block_size,
                ));
            }
        });
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

    fn render_balance_plot(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.heading("Cumulative Arm Balance Trajectory");
        let (trt_pts, ctrl_pts, diff_pts) = build_cumulative_series(&self.schedule);
        Plot::new("randomization_balance_plot")
            .legend(Legend::default())
            .view_aspect(2.0)
            .x_axis_label("Enrolled Subject Index")
            .y_axis_label("Cumulative Subjects / Imbalance")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Treatment Enrolled", PlotPoints::new(trt_pts))
                        .color(palette.success),
                );
                plot_ui.line(
                    Line::new("Control Enrolled", PlotPoints::new(ctrl_pts))
                        .color(palette.series_color(0)),
                );
                plot_ui.line(
                    Line::new("Imbalance (T - C)", PlotPoints::new(diff_pts))
                        .color(palette.warning),
                );
            });
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
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let s: RandomizationState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid randomization state: {e}"))?;
        if !(1..=1000).contains(&s.n_subjects) || !(2..=100).contains(&s.block_size) {
            return Err("Randomization subject count or block_size out of bounds".into());
        }
        if s.randomization_type != RandomizationType::Simple {
            BlockRandomizer::new(s.block_size).map_err(|e| e.to_string())?;
        }
        if s.randomization_type == RandomizationType::Stratified {
            let _ = parse_strata_patients(&s.strata_spec)?;
        }
        self.randomization_type = s.randomization_type;
        self.n_subjects = s.n_subjects;
        self.block_size = s.block_size;
        self.seed = s.seed;
        self.strata_spec = s.strata_spec;
        self.allocate_subjects();
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
        egui::CentralPanel::default().show_inside(ui, |ui| self.render_balance_plot(ui));
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
            RandomizationType::Block => "permuted block randomization guaranteeing periodic balance",
            RandomizationType::Stratified => "stratified permuted-block randomization balancing prognostic strata",
        };
        format!(
            "Clinical trial subject allocation using {} (seed {}) across {} subjects to eliminate selection bias.",
            scheme,
            self.seed,
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
            "Separate permuted block schedules within prognostic baseline strata".into(),
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
        description: "Simple, permuted-block, and stratified patient allocation maintaining balance across trial arms.",
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
    fn test_legacy_block_state_roundtrip_and_seeded_stratified_export() {
        let mut tool = RandomizationTool::default();
        let legacy = r#"{"randomization_type":"Block","n_subjects":24,"block_size":6}"#;
        tool.try_load_state(legacy).expect("legacy block state loads");
        assert_eq!(tool.save_state().as_deref(), Some(legacy));
        assert_eq!(tool.schedule.len(), 24);

        let stratified = r#"{"randomization_type":"Stratified","n_subjects":24,"block_size":4,"seed":77,"strata_spec":"SiteA:8, SiteB:12"}"#;
        tool.try_load_state(stratified)
            .expect("stratified state loads");
        assert_eq!(tool.schedule.len(), 20);
        let csv = format_schedule_csv(&tool.schedule, tool.seed);
        assert!(csv.contains("SiteA"));
        assert!(csv.contains("SiteB"));
        assert_eq!(tool.export_datasets().len(), 3);
    }
}

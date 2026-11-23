#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::{clamp_selected_child_index, ChildInputsState, ChildState, FavoritismToolState};
use super::theory::FavoritismTheory;
use crate::accessibility::AccessibleHoverText;
use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use egui_plot::{Bar, BarChart, Plot};
use math_explorer::applied::favoritism::{calculate_favoritism_score, FavoritismInputs};
use scientific_metadata::theory::TheoryDescribable;
use std::hash::{Hash, Hasher};

/// Runtime representation of a family member and their evaluated favoritism score.
#[derive(Clone, Debug)]
pub struct ChildEntry {
    /// Display name of the family member.
    pub name: String,
    /// Domain scoring inputs.
    pub inputs: FavoritismInputs,
    /// Cached score computed by `calculate_favoritism_score`.
    pub score: f64,
}

impl ChildEntry {
    /// Creates a new family member entry and evaluates its initial score.
    pub fn new(name: String, inputs: FavoritismInputs) -> Self {
        let score = calculate_favoritism_score(&inputs);
        Self {
            name,
            inputs,
            score,
        }
    }

    /// Recomputes `self.score` from `self.inputs`.
    pub fn update_score(&mut self) {
        self.score = calculate_favoritism_score(&self.inputs);
    }
}

/// Interactive tool for multi-child Unified Favoritism Theory strategy comparison.
pub struct FavoritismTool {
    /// Configured family members.
    pub children: Vec<ChildEntry>,
    /// Index of the active family member being edited.
    pub selected_child_index: Option<usize>,
    /// Draft name for adding a new family member.
    pub new_child_name: String,
    theory: FavoritismTheory,
}

impl Default for FavoritismTool {
    fn default() -> Self {
        let mut child1_inputs = FavoritismInputs::default();
        child1_inputs.personality.wealth = 9.0;
        child1_inputs.gifts.g_practical = 8.0;

        let mut child2_inputs = FavoritismInputs::default();
        child2_inputs.personality.wealth = 2.0;
        child2_inputs.contact.time_since_last_contact = 30.0;

        Self {
            children: vec![
                ChildEntry::new("Golden Child".to_string(), child1_inputs),
                ChildEntry::new("The Disappointment".to_string(), child2_inputs),
            ],
            selected_child_index: Some(0),
            new_child_name: "New Child".to_string(),
            theory: FavoritismTheory,
        }
    }
}

impl FavoritismTool {
    fn sorted_children(&self) -> Vec<&ChildEntry> {
        let mut sorted: Vec<&ChildEntry> = self.children.iter().collect();
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    fn to_tool_state(&self) -> FavoritismToolState {
        FavoritismToolState {
            children: self
                .children
                .iter()
                .map(|c| ChildState {
                    name: c.name.clone(),
                    inputs: ChildInputsState::from_inputs(&c.inputs),
                })
                .collect(),
            selected_child_index: clamp_selected_child_index(
                self.selected_child_index,
                self.children.len(),
            ),
            new_child_name: self.new_child_name.clone(),
        }
    }

    fn render_family_list_panel(&mut self, ui: &mut egui::Ui) {
        let left_id = crate::framework::scoped_panel_id(ui, "favoritism_left_panel");
        crate::framework::PaneSidePanel::left(left_id).show_inside(ui, |ui| {
            ui.heading("Family Members");
            ui.separator();

            let mut to_remove = None;
            for (i, child) in self.children.iter().enumerate() {
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(self.selected_child_index == Some(i), &child.name)
                        .clicked()
                    {
                        self.selected_child_index = Some(i);
                    }
                    if ui
                        .small_button("❌")
                        .accessible_hover_text("Remove family member")
                        .clicked()
                    {
                        to_remove = Some(i);
                    }
                });
            }

            if let Some(i) = to_remove {
                self.remove_child_at(i);
            }

            ui.separator();
            ui.horizontal(|ui| {
                let response = ui.add(
                    egui::TextEdit::singleline(&mut self.new_child_name).hint_text("Enter name"),
                );
                let enter_pressed =
                    response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                if ui.button("➕ Add").clicked() || enter_pressed {
                    self.children.push(ChildEntry::new(
                        self.new_child_name.clone(),
                        FavoritismInputs::default(),
                    ));
                    if self.selected_child_index.is_none() {
                        self.selected_child_index = Some(self.children.len() - 1);
                    }
                    self.new_child_name = "New Child".to_string();
                }
            });
        });
    }

    fn remove_child_at(&mut self, index: usize) {
        if index >= self.children.len() {
            return;
        }
        self.children.remove(index);
        if self.selected_child_index == Some(index) {
            self.selected_child_index =
                clamp_selected_child_index(Some(index), self.children.len());
        } else if let Some(selected) = self.selected_child_index {
            if index < selected {
                self.selected_child_index = Some(selected - 1);
            }
        }
    }

    fn render_leaderboard_panel(&self, ui: &mut egui::Ui) {
        let right_id = crate::framework::scoped_panel_id(ui, "favoritism_right_panel");
        crate::framework::PaneSidePanel::right(right_id)
            .min_width(300.0)
            .show_inside(ui, |ui| {
                ui.heading("Leaderboard");
                ui.separator();

                let sorted_children = self.sorted_children();
                let bars: Vec<Bar> = sorted_children
                    .iter()
                    .enumerate()
                    .map(|(i, child)| Bar::new(i as f64, child.score).name(&child.name).width(0.5))
                    .collect();

                let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
                let chart = BarChart::new("Favoritism Scores", bars).color(palette.series_color(0));

                Plot::new("favoritism_plot")
                    .legend(egui_plot::Legend::default())
                    .show(ui, |plot_ui| plot_ui.bar_chart(chart));

                ui.separator();
                ui.heading("Rankings");
                for (i, child) in sorted_children.iter().enumerate() {
                    ui.label(format!("{}. {} - {:.2}", i + 1, child.name, child.score));
                }
            });
    }

    fn render_strategy_panel(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            if let Some(idx) = self.selected_child_index {
                if let Some(child) = self.children.get_mut(idx) {
                    ui.heading(format!("Strategy: {}", child.name));
                    ui.separator();
                    let mut changed = false;
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        changed |= Self::render_time_and_gifts_groups(ui, &mut child.inputs);
                        changed |= Self::render_personality_group(ui, &mut child.inputs);
                        changed |= Self::render_social_and_contact_groups(ui, &mut child.inputs);
                    });
                    if changed {
                        child.update_score();
                    }
                    return;
                }
            }
            ui.centered_and_justified(|ui| {
                ui.label("Select a family member to adjust their strategy.");
            });
        });
    }

    fn render_time_and_gifts_groups(ui: &mut egui::Ui, inputs: &mut FavoritismInputs) -> bool {
        let mut changed = false;
        ui.group(|ui| {
            ui.label("Time & Proximity");
            changed |= ui
                .add(egui::Slider::new(&mut inputs.time.t, 1.0..=3650.0).text("Days Integrated"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut inputs.time.x_0, 0.0..=1000.0).text("Distance (km)"))
                .changed();
        });
        ui.group(|ui| {
            ui.label("Gifts (Bribery)");
            changed |= ui
                .add(
                    egui::Slider::new(&mut inputs.gifts.g_practical, 0.0..=10.0)
                        .text("Practical Value"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut inputs.gifts.g_emotional, 0.0..=10.0)
                        .text("Emotional Value"),
                )
                .changed();
        });
        changed
    }

    fn render_personality_group(ui: &mut egui::Ui, inputs: &mut FavoritismInputs) -> bool {
        let mut changed = false;
        ui.group(|ui| {
            ui.label("Personality & Success");
            changed |= ui
                .add(egui::Slider::new(&mut inputs.personality.wealth, 0.0..=10.0).text("Wealth"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut inputs.personality.talent, 0.0..=10.0).text("Talent"))
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut inputs.personality.intelligence, 0.0..=10.0)
                        .text("Intelligence"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut inputs.personality.emotional_sensitivity, 0.0..=10.0)
                        .text("Sensitivity"),
                )
                .changed();
        });
        changed
    }

    fn render_social_and_contact_groups(ui: &mut egui::Ui, inputs: &mut FavoritismInputs) -> bool {
        let mut changed = false;
        ui.group(|ui| {
            ui.label("Social & Crisis");
            changed |= ui
                .checkbox(
                    &mut inputs.social.helped_during_crisis,
                    "Helped during Crisis",
                )
                .changed();
            changed |= ui
                .checkbox(
                    &mut inputs.social.active_on_social_media,
                    "Active on Social Media",
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut inputs.social.birth_order_weight, 0.5..=1.5)
                        .text("Birth Order Weight"),
                )
                .changed();
        });
        ui.group(|ui| {
            ui.label("Contact & Decay");
            changed |= ui
                .add(
                    egui::Slider::new(&mut inputs.contact.time_since_last_contact, 0.0..=365.0)
                        .text("Days Since Last Call"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut inputs.contact.f_initial, 0.0..=30.0)
                        .text("Calls per Month"),
                )
                .changed();
        });
        changed
    }
}

impl InteractiveTool for FavoritismTool {
    fn name(&self) -> &'static str {
        "Favoritism"
    }

    fn domain(&self) -> &'static str {
        "favoritism"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let sorted = self.sorted_children();
        if sorted.is_empty() {
            return Vec::new();
        }
        let points: Vec<[f64; 2]> = sorted
            .iter()
            .enumerate()
            .filter(|(_, c)| c.score.is_finite())
            .map(|(i, c)| [i as f64, c.score])
            .collect();
        if points.is_empty() {
            Vec::new()
        } else {
            vec![("Favoritism Scores".to_string(), points)]
        }
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_family_list_panel(ui);
        self.render_leaderboard_panel(ui);
        self.render_strategy_panel(ui);
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.children.len().hash(&mut h);
        for child in &self.children {
            child.name.hash(&mut h);
            ChildInputsState::from_inputs(&child.inputs).hash_into(&mut h);
        }
        self.selected_child_index.hash(&mut h);
        self.new_child_name.hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.to_tool_state()).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        if state.len() > 262_144 {
            return Err("Favoritism state payload exceeds 256 KB limit".to_string());
        }
        let parsed: FavoritismToolState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Favoritism state: {e}"))?;
        if parsed.children.len() > 128 {
            return Err("Favoritism children count exceeds maximum of 128".to_string());
        }
        for child in &parsed.children {
            child.inputs.validate()?;
        }
        self.children = parsed
            .children
            .into_iter()
            .map(|c| ChildEntry::new(c.name, c.inputs.to_inputs()))
            .collect();
        self.selected_child_index =
            clamp_selected_child_index(parsed.selected_child_index, self.children.len());
        self.new_child_name = parsed.new_child_name;
        Ok(())
    }
}

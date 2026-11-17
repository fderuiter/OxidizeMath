// @explorer_feature = "applied"
use crate::accessibility::AccessibleHoverText;
use crate::tabs::ExplorerTab;
use eframe::egui;
use egui_plot::{Bar, BarChart, Plot};
use math_explorer::applied::favoritism::{calculate_favoritism_score, FavoritismInputs};

#[derive(Clone)]
struct Child {
    name: String,
    inputs: FavoritismInputs,
    score: f64,
}

impl Child {
    fn new(name: String, inputs: FavoritismInputs) -> Self {
        let score = calculate_favoritism_score(&inputs);
        Self {
            name,
            inputs,
            score,
        }
    }

    fn update_score(&mut self) {
        self.score = calculate_favoritism_score(&self.inputs);
    }
}

pub struct FavoritismTab {
    children: Vec<Child>,
    selected_child_index: Option<usize>,
    new_child_name: String,
}

impl Default for FavoritismTab {
    fn default() -> Self {
        let mut child1_inputs = FavoritismInputs::default();
        child1_inputs.personality.wealth = 9.0;
        child1_inputs.gifts.g_practical = 8.0;

        let mut child2_inputs = FavoritismInputs::default();
        child2_inputs.personality.wealth = 2.0;
        child2_inputs.contact.time_since_last_contact = 30.0;

        Self {
            children: vec![
                Child::new("Golden Child".to_string(), child1_inputs),
                Child::new("The Disappointment".to_string(), child2_inputs),
            ],
            selected_child_index: Some(0),
            new_child_name: "New Child".to_string(),
        }
    }
}

impl ExplorerTab for FavoritismTab {
    fn name(&self) -> &'static str {
        "Favoritism"
    }

    fn save_state(&self) -> Option<String> {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct FavoritismState {
            selected_child_index: Option<usize>,
            new_child_name: String,
        }

        serde_json::to_string(&FavoritismState {
            selected_child_index: self.selected_child_index,
            new_child_name: self.new_child_name.clone(),
        })
        .ok()
    }

    fn load_state(&mut self, state: &str) {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct FavoritismState {
            selected_child_index: Option<usize>,
            new_child_name: String,
        }

        if let Ok(s) = serde_json::from_str::<FavoritismState>(state) {
            self.selected_child_index = s.selected_child_index;
            self.new_child_name = s.new_child_name;
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("favoritism_left_panel").show(ctx, |ui| {
            ui.heading("Family Members");
            ui.separator();

            // List of children
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
                self.children.remove(i);
                if self.selected_child_index == Some(i) {
                    self.selected_child_index = None;
                } else if let Some(selected) = self.selected_child_index {
                    if i < selected {
                        self.selected_child_index = Some(selected - 1);
                    }
                }
            }

            ui.separator();
            ui.horizontal(|ui| {
                let response = ui.add(
                    egui::TextEdit::singleline(&mut self.new_child_name).hint_text("Enter name"),
                );
                let enter_pressed =
                    response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                if ui.button("➕ Add").clicked() || enter_pressed {
                    self.children.push(Child::new(
                        self.new_child_name.clone(),
                        FavoritismInputs::default(),
                    ));
                    self.new_child_name = "New Child".to_string();
                }
            });
        });

        egui::SidePanel::right("favoritism_right_panel")
            .min_width(300.0)
            .show(ctx, |ui| {
                ui.heading("Leaderboard");
                ui.separator();

                // Sort children by score for display
                let mut sorted_children: Vec<&Child> = self.children.iter().collect();
                sorted_children.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let bars: Vec<Bar> = sorted_children
                    .iter()
                    .enumerate()
                    .map(|(i, child)| Bar::new(i as f64, child.score).name(&child.name).width(0.5))
                    .collect();

                let chart =
                    BarChart::new("Favoritism Scores", bars).color(egui::Color32::LIGHT_BLUE);

                Plot::new("favoritism_plot")
                    .legend(egui_plot::Legend::default())
                    .show(ui, |plot_ui| plot_ui.bar_chart(chart));

                ui.separator();
                ui.heading("Rankings");
                for (i, child) in sorted_children.iter().enumerate() {
                    ui.label(format!("{}. {} - {:.2}", i + 1, child.name, child.score));
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(idx) = self.selected_child_index {
                if idx < self.children.len() {
                    let child = &mut self.children[idx];
                    let inputs = &mut child.inputs;
                    let mut changed = false;

                    ui.heading(format!("Strategy: {}", child.name));
                    ui.separator();

                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.group(|ui| {
                            ui.label("Time & Proximity");
                            if ui
                                .add(
                                    egui::Slider::new(&mut inputs.time.t, 1.0..=3650.0)
                                        .text("Days Integrated"),
                                )
                                .changed()
                            {
                                changed = true;
                            }
                            if ui
                                .add(
                                    egui::Slider::new(&mut inputs.time.x_0, 0.0..=1000.0)
                                        .text("Distance (km)"),
                                )
                                .changed()
                            {
                                changed = true;
                            }
                        });

                        ui.group(|ui| {
                            ui.label("Gifts (Bribery)");
                            if ui
                                .add(
                                    egui::Slider::new(&mut inputs.gifts.g_practical, 0.0..=10.0)
                                        .text("Practical Value"),
                                )
                                .changed()
                            {
                                changed = true;
                            }
                            if ui
                                .add(
                                    egui::Slider::new(&mut inputs.gifts.g_emotional, 0.0..=10.0)
                                        .text("Emotional Value"),
                                )
                                .changed()
                            {
                                changed = true;
                            }
                        });

                        ui.group(|ui| {
                            ui.label("Personality & Success");
                            if ui
                                .add(
                                    egui::Slider::new(&mut inputs.personality.wealth, 0.0..=10.0)
                                        .text("Wealth"),
                                )
                                .changed()
                            {
                                changed = true;
                            }
                            if ui
                                .add(
                                    egui::Slider::new(&mut inputs.personality.talent, 0.0..=10.0)
                                        .text("Talent"),
                                )
                                .changed()
                            {
                                changed = true;
                            }
                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut inputs.personality.intelligence,
                                        0.0..=10.0,
                                    )
                                    .text("Intelligence"),
                                )
                                .changed()
                            {
                                changed = true;
                            }
                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut inputs.personality.emotional_sensitivity,
                                        0.0..=10.0,
                                    )
                                    .text("Sensitivity"),
                                )
                                .changed()
                            {
                                changed = true;
                            }
                        });

                        ui.group(|ui| {
                            ui.label("Social & Crisis");
                            if ui
                                .checkbox(
                                    &mut inputs.social.helped_during_crisis,
                                    "Helped during Crisis",
                                )
                                .changed()
                            {
                                changed = true;
                            }
                            if ui
                                .checkbox(
                                    &mut inputs.social.active_on_social_media,
                                    "Active on Social Media",
                                )
                                .changed()
                            {
                                changed = true;
                            }
                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut inputs.social.birth_order_weight,
                                        0.5..=1.5,
                                    )
                                    .text("Birth Order Weight"),
                                )
                                .changed()
                            {
                                changed = true;
                            }
                        });

                        ui.group(|ui| {
                            ui.label("Contact & Decay");
                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut inputs.contact.time_since_last_contact,
                                        0.0..=365.0,
                                    )
                                    .text("Days Since Last Call"),
                                )
                                .changed()
                            {
                                changed = true;
                            }
                            if ui
                                .add(
                                    egui::Slider::new(&mut inputs.contact.f_initial, 0.0..=30.0)
                                        .text("Calls per Month"),
                                )
                                .changed()
                            {
                                changed = true;
                            }
                        });
                    });

                    if changed {
                        child.update_score();
                    }
                }
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("Select a family member to adjust their strategy.");
                });
            }
        });
    }
}

// [cite:graph_parameters_rust]

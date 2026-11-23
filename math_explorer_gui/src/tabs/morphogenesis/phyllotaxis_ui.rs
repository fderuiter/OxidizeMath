#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::phyllotaxis_tool::{
    LSystemPresetChoice, PhyllotaxisKineticsChoice, PhyllotaxisLSystemTool,
};
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints, Points};
use math_explorer::biology::phyllotaxis::{
    is_consecutive_fibonacci_pair, GOLDEN_ANGLE_DEG, LUCAS_ANGLE_DEG,
};

impl PhyllotaxisLSystemTool {
    fn draw_angle_and_packing_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.label("Divergence Angle Presets:");
        ui.horizontal_wrapped(|ui| {
            if ui.button("Golden (137.508°)").clicked() {
                self.state.divergence_angle_deg = GOLDEN_ANGLE_DEG;
                changed = true;
            }
            if ui.button("Lucas (99.502°)").clicked() {
                self.state.divergence_angle_deg = LUCAS_ANGLE_DEG;
                changed = true;
            }
            if ui.button("Rational (120.0°)").clicked() {
                self.state.divergence_angle_deg = 120.0;
                changed = true;
            }
        });
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.divergence_angle_deg, 45.0..=180.0)
                    .text("Angle α (°)"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.primordia_count, 5..=512).text("Primordia N"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.scale_c, 0.05..=2.0).text("Packing Scale c"))
            .changed();
        changed
    }

    fn draw_nutrient_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.separator();
        ui.strong("Nutrient Uptake Kinetics");
        egui::ComboBox::from_label("Kinetics")
            .selected_text(match self.state.kinetics {
                PhyllotaxisKineticsChoice::MichaelisMenten => "Michaelis-Menten",
                PhyllotaxisKineticsChoice::Hill => "Hill Cooperative",
            })
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_value(
                        &mut self.state.kinetics,
                        PhyllotaxisKineticsChoice::MichaelisMenten,
                        "Michaelis-Menten",
                    )
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut self.state.kinetics,
                        PhyllotaxisKineticsChoice::Hill,
                        "Hill Cooperative",
                    )
                    .changed();
            });
        changed |= ui
            .add(egui::Slider::new(&mut self.state.substrate, 0.0..=25.0).text("Substrate [S]"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.v_max, 0.05..=25.0).text("V_max"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.k_m, 0.05..=25.0).text("K_m"))
            .changed();
        if self.state.kinetics == PhyllotaxisKineticsChoice::Hill {
            changed |= ui
                .add(egui::Slider::new(&mut self.state.hill_n, 0.2..=8.0).text("Hill n"))
                .changed();
        }
        changed
    }

    fn draw_lsystem_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.separator();
        ui.strong("L-System Shoot Grammar");
        egui::ComboBox::from_label("Shoot Rule")
            .selected_text(match self.state.lsystem_preset {
                LSystemPresetChoice::BotanicalShoot => "Botanical Shoot",
                LSystemPresetChoice::SympodialTree => "Sympodial Tree",
                LSystemPresetChoice::MonopodialFern => "Monopodial Fern",
            })
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_value(
                        &mut self.state.lsystem_preset,
                        LSystemPresetChoice::BotanicalShoot,
                        "Botanical Shoot",
                    )
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut self.state.lsystem_preset,
                        LSystemPresetChoice::SympodialTree,
                        "Sympodial Tree",
                    )
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut self.state.lsystem_preset,
                        LSystemPresetChoice::MonopodialFern,
                        "Monopodial Fern",
                    )
                    .changed();
            });
        changed |= ui
            .add(egui::Slider::new(&mut self.state.lsystem_iterations, 1..=6).text("Iterations"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.branch_angle_deg, 5.0..=60.0)
                    .text("Branch Angle δ (°)"),
            )
            .changed();
        changed
    }

    pub(crate) fn draw_controls(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "phyllotaxis_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            ui.heading("Phyllotaxis & L-Systems");
            ui.separator();

            let mut changed = self.draw_angle_and_packing_controls(ui);
            changed |= self.draw_nutrient_controls(ui);
            changed |= self.draw_lsystem_controls(ui);
            if changed {
                self.recompute();
            }

            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("⏭ Step Plastochron").clicked() {
                    self.advance_plastochrons(0.25);
                }
                if ui.button("↺ Reset").clicked() {
                    crate::framework::recovery::FiniteStateGuard::reset_parameters(self);
                }
            });

            ui.separator();
            let (p, q) = self.phyllotaxis.detect_parastichy_pair();
            let fib_flag = is_consecutive_fibonacci_pair(p, q);
            let nutrient = self.phyllotaxis.nutrient();
            ui.strong("Botanical Telemetry");
            let tag = if fib_flag {
                "[Consecutive Fibonacci]"
            } else {
                "[Non-Fibonacci]"
            };
            ui.label(format!("Opposing Parastichies: ({p}, {q}) {tag}"));
            ui.label(format!(
                "Nutrient Velocity v([S]): {:.3} ({:.0}% V_max)",
                nutrient.reaction_velocity(),
                nutrient.normalized_rate() * 100.0
            ));
            ui.label(format!(
                "Turtle Segments: {}",
                self.lsystem.segments().len()
            ));
        });
    }

    fn render_phyllotaxis_plot(
        &self,
        ui: &mut egui::Ui,
        palette: &crate::app::SemanticPalette,
        parastichies: (usize, usize),
    ) -> egui::Rect {
        let (m, n) = parastichies;
        let primordia = self.phyllotaxis.primordia();
        let all_pts: Vec<[f64; 2]> = primordia.iter().map(|p| [p.x, p.y]).collect();
        let spiral_m: Vec<[f64; 2]> = primordia
            .iter()
            .filter(|p| m > 0 && p.index % m == 0)
            .map(|p| [p.x, p.y])
            .collect();
        let spiral_n: Vec<[f64; 2]> = primordia
            .iter()
            .filter(|p| n > 0 && p.index % n == 0)
            .map(|p| [p.x, p.y])
            .collect();
        let offset_x = (self.state.scale_c * ((self.state.primordia_count as f64).sqrt() + 2.0))
            .max(8.0)
            * 1.45;

        Plot::new("phyllotaxis_lsystem_plot")
            .data_aspect(1.0)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.points(
                    Points::new("Primordia Florets", PlotPoints::from(all_pts))
                        .radius(3.2_f32)
                        .color(palette.series[0]),
                );
                if spiral_m.len() >= 2 {
                    plot_ui.line(
                        Line::new(format!("{m}-Parastichy Arm"), PlotPoints::from(spiral_m))
                            .color(palette.series[1])
                            .width(1.8_f32),
                    );
                }
                if spiral_n.len() >= 2 {
                    plot_ui.line(
                        Line::new(format!("{n}-Parastichy Arm"), PlotPoints::from(spiral_n))
                            .color(palette.accent)
                            .width(1.8_f32),
                    );
                }
                for (idx, seg) in self.lsystem.segments().iter().enumerate() {
                    let pts = vec![
                        [seg[0] + offset_x, seg[1] - 4.0],
                        [seg[2] + offset_x, seg[3] - 4.0],
                    ];
                    let name = if idx == 0 {
                        "Nutrient L-System Shoot"
                    } else {
                        ""
                    };
                    plot_ui.line(
                        Line::new(name, PlotPoints::from(pts))
                            .color(palette.axis_y)
                            .width(1.5_f32),
                    );
                }
            })
            .response
            .rect
    }

    pub(crate) fn draw_canvas(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let (m, n) = self.phyllotaxis.detect_parastichy_pair();
            let rect = self.render_phyllotaxis_plot(ui, &palette, (m, n));
            let focus_resp = ui.interact(
                rect,
                ui.id().with("phyllotaxis_canvas_a11y"),
                egui::Sense::focusable_noninteractive(),
            );
            let summary = format!(
                "Parastichy pair ({m}, {n}), {} florets, {} shoot segments",
                self.phyllotaxis.primordia().len(),
                self.lsystem.segments().len()
            );
            crate::accessibility::register_canvas_accessibility(
                ui,
                &focus_resp,
                egui::accesskit::Role::GraphicsDocument,
                "Phyllotaxis and L-System Visualization",
                Some(&summary),
            );
            crate::accessibility::paint_canvas_focus_ring(ui, &focus_resp);
        });
    }
}

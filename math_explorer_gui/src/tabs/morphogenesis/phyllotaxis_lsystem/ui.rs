//! Side-panel controls and Fibonacci parastichy HUD for [`super::PhyllotaxisLSystemTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::PhyllotaxisLSystemTool;
use super::presets::{BotanicalStudioMode, GuiDivergencePreset, GuiLSystemPreset};
use crate::app::SemanticPalette;
use crate::framework::recovery::FiniteStateGuard;
use crate::reflective_ui::AnimatedSlider;
use eframe::egui;
use math_explorer::biology::phyllotaxis::MAX_PRIMORDIA;

impl PhyllotaxisLSystemTool {
    pub(super) fn render_studio(&mut self, ui: &mut egui::Ui) {
        self.draw_side_panel(ui);
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.draw_telemetry_strip(ui);
            ui.add_space(4.0);
            self.draw_botanical_canvas(ui);
        });
    }

    fn draw_side_panel(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "phyllotaxis_lsystem_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Phyllotaxis & L-Systems");
                ui.separator();
                self.draw_recovery_card(ui);
                self.draw_mode_selector(ui);
                ui.separator();
                self.draw_phyllotaxis_controls(ui);
                ui.separator();
                self.draw_lsystem_controls(ui);
            });
        });
    }

    fn draw_recovery_card(&mut self, ui: &mut egui::Ui) {
        if !FiniteStateGuard::diverged(self) {
            return;
        }
        let palette = SemanticPalette::from_visuals(ui.visuals());
        egui::Frame::group(ui.style())
            .stroke(egui::Stroke::new(1.5_f32, palette.error))
            .show(ui, |ui| {
                ui.colored_label(palette.error, "⚠ Botanical Parameter Guard Triggered");
                if let Some(diag) = &self.diagnostic {
                    ui.small(&diag.message);
                }
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Clamp Parameters").clicked() {
                        FiniteStateGuard::clamp_parameters(self);
                    }
                    if ui.button("Reset Studio").clicked() {
                        FiniteStateGuard::reset_parameters(self);
                    }
                });
            });
        ui.separator();
    }

    fn draw_mode_selector(&mut self, ui: &mut egui::Ui) {
        ui.label("Studio View:");
        ui.horizontal_wrapped(|ui| {
            for mode in BotanicalStudioMode::ALL {
                if ui
                    .selectable_label(self.state.mode == mode, mode.label())
                    .clicked()
                {
                    self.state.mode = mode;
                    self.bump_revision();
                }
            }
        });
    }

    fn draw_phyllotaxis_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Vogel Phyllotaxis & Parastichies:");
        egui::ComboBox::from_id_salt("phyllotaxis_divergence_preset")
            .selected_text(self.state.divergence_preset.label())
            .show_ui(ui, |ui| {
                for preset in GuiDivergencePreset::ALL {
                    if ui
                        .selectable_label(self.state.divergence_preset == preset, preset.label())
                        .clicked()
                    {
                        self.apply_divergence_preset(preset);
                    }
                }
            });

        let before = (
            self.state.divergence_deg,
            self.state.primordia_count,
            self.state.radial_scale,
            self.state.show_parastichies,
        );
        ui.add(
            AnimatedSlider::new(&mut self.state.divergence_deg, 10.0..=350.0)
                .text("Divergence Angle α (°)")
                .id_salt("phyl_alpha_deg"),
        );
        let mut count_f64 = self.state.primordia_count as f64;
        if ui
            .add(
                AnimatedSlider::new(&mut count_f64, 8.0..=(MAX_PRIMORDIA as f64))
                    .text("Primordia Count N")
                    .id_salt("phyl_primordia_n"),
            )
            .changed()
        {
            self.state.primordia_count = (count_f64.round() as usize).clamp(8, MAX_PRIMORDIA);
        }
        ui.add(
            AnimatedSlider::new(&mut self.state.radial_scale, 0.2..=5.0)
                .text("Radial Scale c")
                .id_salt("phyl_scale_c"),
        );
        ui.checkbox(
            &mut self.state.show_parastichies,
            "Overlay (F_k, F_{k+1}) Parastichy Spirals",
        );

        let after = (
            self.state.divergence_deg,
            self.state.primordia_count,
            self.state.radial_scale,
            self.state.show_parastichies,
        );
        if before != after {
            self.sync_engines();
        }
    }

    fn draw_lsystem_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("L-System Vascular Grammar:");
        egui::ComboBox::from_id_salt("lsystem_grammar_preset")
            .selected_text(self.state.lsystem_preset.label())
            .show_ui(ui, |ui| {
                for preset in GuiLSystemPreset::ALL {
                    if ui
                        .selectable_label(self.state.lsystem_preset == preset, preset.label())
                        .clicked()
                    {
                        self.apply_lsystem_preset(preset);
                    }
                }
            });

        let before = (
            self.state.generations,
            self.state.branch_angle_deg,
            self.state.contraction_r1,
            self.state.contraction_r2,
        );
        let mut gen_f64 = self.state.generations as f64;
        if ui
            .add(
                AnimatedSlider::new(&mut gen_f64, 1.0..=11.0)
                    .text("Derivation Depth n")
                    .id_salt("lsys_gen_n"),
            )
            .changed()
        {
            self.state.generations = (gen_f64.round() as usize).clamp(1, 11);
        }
        ui.add(
            AnimatedSlider::new(&mut self.state.branch_angle_deg, 5.0..=120.0)
                .text("Branch Angle δ (°)")
                .id_salt("lsys_angle_deg"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.state.contraction_r1, 0.35..=0.95)
                .text("Primary Contraction r₁")
                .id_salt("lsys_r1"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.state.contraction_r2, 0.35..=0.95)
                .text("Lateral Contraction r₂")
                .id_salt("lsys_r2"),
        );

        let after = (
            self.state.generations,
            self.state.branch_angle_deg,
            self.state.contraction_r1,
            self.state.contraction_r2,
        );
        if before != after {
            self.sync_engines();
        }
    }

    fn draw_telemetry_strip(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let analysis = self.vogel.analysis();
        ui.horizontal_wrapped(|ui| {
            let badge_color = if analysis.has_radial_spokes {
                palette.warning
            } else if analysis.is_fibonacci_pair {
                palette.success
            } else {
                palette.info
            };
            let regime = if analysis.has_radial_spokes {
                format!("Radial Orthostichy Spokes ({})", analysis.primary_count)
            } else if analysis.is_fibonacci_pair {
                format!(
                    "Fibonacci Parastichies ({}, {})",
                    analysis.primary_count, analysis.secondary_count
                )
            } else {
                format!(
                    "Opposing Parastichies ({}, {})",
                    analysis.primary_count, analysis.secondary_count
                )
            };
            ui.colored_label(badge_color, format!("● {regime}"));
            ui.separator();
            ui.label(format!(
                "Packing Uniformity: {:.1}%",
                analysis.packing_uniformity * 100.0
            ));
            ui.separator();
            ui.label(format!(
                "L-System Gen {} | {} Symbols | {} Internodes",
                self.lsystem.generation(),
                self.lsystem.symbols().len(),
                self.lsystem.segments().len()
            ));
        });
    }
}

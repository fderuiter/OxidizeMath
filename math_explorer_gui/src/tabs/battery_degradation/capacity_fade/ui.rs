//! Side-panel thermal/electrochemical controls, CSV calibration actions, and
//! `egui_plot` SOH + residual projections for [`super::CapacityFadeTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:battery_degradation]

use super::CapacityFadeTool;
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::reflective_ui::{render_copyable_metric, AnimatedSlider};
use crate::tabs::battery_degradation::cell_card::ChemistryChoice;
use eframe::egui;
use egui_plot::{HLine, Line, Plot, PlotPoints, Points};
use math_explorer::applied::battery_degradation::{
    MAX_C_RATE, MAX_EXPONENT_Z, MAX_TEMPERATURE_C, MIN_C_RATE, MIN_EXPONENT_Z, MIN_TEMPERATURE_C,
};

impl CapacityFadeTool {
    pub(crate) fn render_controls_panel(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "battery_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.render_chemistry_and_stress_controls(ui);
                ui.add_space(10.0);
                ui.separator();
                ui.add_space(8.0);
                self.render_calibration_and_export_controls(ui);
            });
        });
    }

    fn render_chemistry_and_stress_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Electrochemical Parameters");
        ui.add_space(4.0);

        let mut selected = self.chemistry;
        egui::ComboBox::from_label("Chemistry Preset")
            .selected_text(selected.label())
            .show_ui(ui, |ui| {
                for choice in ChemistryChoice::ALL {
                    ui.selectable_value(&mut selected, choice, choice.label());
                }
            });
        if selected != self.chemistry {
            self.apply_chemistry_preset(selected);
        }
        ui.small(self.chemistry.to_preset().description());
        ui.add_space(8.0);
        self.render_operating_sliders(ui);
    }

    fn render_operating_sliders(&mut self, ui: &mut egui::Ui) {
        ui.add(
            AnimatedSlider::new(&mut self.dod, 1.0..=100.0)
                .hard_bounds(1.0..=100.0)
                .text("Depth of Discharge (DoD) - %")
                .id_salt("capacity_fade_dod"),
        )
        .accessible_hover_text("Percentage of nominal cell capacity cycled per discharge.");
        ui.add_space(6.0);

        ui.add(
            AnimatedSlider::new(&mut self.temperature, MIN_TEMPERATURE_C..=MAX_TEMPERATURE_C)
                .hard_bounds(MIN_TEMPERATURE_C..=MAX_TEMPERATURE_C)
                .text("Temperature (°C)")
                .id_salt("capacity_fade_temperature"),
        )
        .accessible_hover_text(
            "Operating cell temperature driving Arrhenius thermal acceleration relative to 25 °C.",
        );
        ui.add_space(6.0);

        ui.add(
            AnimatedSlider::new(&mut self.c_rate, MIN_C_RATE..=MAX_C_RATE)
                .hard_bounds(MIN_C_RATE..=MAX_C_RATE)
                .text("C-Rate (1/h)")
                .id_salt("capacity_fade_c_rate"),
        )
        .accessible_hover_text(
            "Charge/discharge current rate scaling mechanical electrode stress.",
        );
        ui.add_space(6.0);

        ui.add(
            AnimatedSlider::new(&mut self.exponent_z, MIN_EXPONENT_Z..=MAX_EXPONENT_Z)
                .hard_bounds(MIN_EXPONENT_Z..=MAX_EXPONENT_Z)
                .text("Stretched Exponent (z)")
                .id_salt("capacity_fade_exponent_z"),
        )
        .accessible_hover_text(
            "Cycling fade exponent z (z ~ 0.5 for diffusion-limited SEI growth; z ~ 1.0 linear).",
        );
        ui.add_space(6.0);

        ui.add(
            AnimatedSlider::new(&mut self.cycles_to_simulate, 100.0..=10_000.0)
                .hard_bounds(100.0..=10_000.0)
                .text("Simulation Range - Cycles")
                .id_salt("capacity_fade_cycles"),
        );
    }

    fn render_calibration_and_export_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Lab CSV Calibration & Card");
        ui.small("Fit prefactor alpha and exponent z via stack-allocated Levenberg-Marquardt.");
        ui.add_space(4.0);

        ui.horizontal_wrapped(|ui| {
            if ui
                .button("📂 Fit Lab CSV...")
                .accessible_hover_text("Import a 2-column (cycles, SOH) CSV up to 2,000 rows.")
                .clicked()
            {
                self.dialogs.trigger_csv_open();
            }
            if ui
                .button("💾 Save JSON Cell Card...")
                .accessible_hover_text("Export calibrated electrochemical cell parameters as JSON.")
                .clicked()
            {
                let filename = format!("{}_cell_card.json", self.chemistry.label().to_lowercase());
                let card = self.build_cell_card();
                if let Err(err) = self.dialogs.trigger_cell_card_save(&filename, &card) {
                    self.io_status = Some((true, err));
                }
            }
        });

        if self.calibrated_alpha.is_some() || !self.lab_samples.is_empty() {
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.show_residuals, "Show Residuals Plot");
                if ui.button("Reset Fit").clicked() {
                    self.calibrated_alpha = None;
                    self.exponent_z = self.chemistry.to_preset().default_exponent_z();
                    self.lab_samples.clear();
                    self.last_fit = None;
                    self.io_status = None;
                }
            });
        }
        self.render_calibration_status(ui);
    }

    fn render_calibration_status(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        if let Some(fit) = &self.last_fit {
            ui.add_space(4.0);
            ui.label(format!(
                "Source: {} ({} pts) | RMSE: {:.4} | R²: {:.4}",
                fit.source_name, fit.sample_count, fit.rmse, fit.r_squared
            ));
        }
        if let Some((is_err, msg)) = &self.io_status {
            let color = if *is_err {
                palette.error
            } else {
                palette.success
            };
            ui.colored_label(color, msg);
        }
    }

    pub(crate) fn render_projection_panel(&self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Coupled Capacity Fade Projection");
            ui.label(format!(
                "{} kinetics: Arrhenius factor {:.2}x @ {:.1} °C ({:.2} K), {:.2}C, z = {:.2}",
                self.chemistry.label(),
                self.active_model().arrhenius_factor(),
                self.temperature,
                self.active_model().temperature_k(),
                self.c_rate,
                self.exponent_z
            ));

            let palette = SemanticPalette::from_visuals(ui.visuals());
            self.render_soh_plot(ui, &palette);
            if self.show_residuals && !self.lab_samples.is_empty() {
                ui.add_space(6.0);
                self.render_residuals_plot(ui, &palette);
            }
            ui.add_space(8.0);
            self.render_summary_metrics(ui);
        });
    }

    fn render_soh_plot(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let points = self.capacity_curve_points();
        let aspect = if self.show_residuals && !self.lab_samples.is_empty() {
            2.6
        } else {
            2.0
        };
        Plot::new("capacity_fade_plot")
            .view_aspect(aspect)
            .x_axis_label("Cycles")
            .y_axis_label("State of Health (Capacity)")
            .include_y(0.0)
            .include_y(1.0)
            .show(ui, |plot_ui| {
                plot_ui.line(Line::new("Model SOH", PlotPoints::new(points)).color(palette.accent));
                if !self.lab_samples.is_empty() {
                    plot_ui.points(
                        Points::new(
                            "Measured Lab SOH",
                            PlotPoints::new(self.lab_samples.clone()),
                        )
                        .radius(3.5_f32)
                        .color(palette.info),
                    );
                }
                plot_ui.hline(
                    HLine::new("End of Life (70%)", 0.7)
                        .color(palette.error)
                        .style(egui_plot::LineStyle::Dashed { length: 10.0 }),
                );
                plot_ui.hline(
                    HLine::new("First Life (80%)", 0.8)
                        .color(palette.warning)
                        .style(egui_plot::LineStyle::Dashed { length: 10.0 }),
                );
            });
    }

    fn render_residuals_plot(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let residuals = self.residual_points();
        Plot::new("capacity_fade_residuals_plot")
            .view_aspect(4.2)
            .x_axis_label("Cycles")
            .y_axis_label("Residual (Measured - Model)")
            .include_y(-0.02)
            .include_y(0.02)
            .show(ui, |plot_ui| {
                plot_ui.hline(
                    HLine::new("Zero Residual", 0.0)
                        .color(palette.graph_edge)
                        .style(egui_plot::LineStyle::Dashed { length: 6.0 }),
                );
                plot_ui.points(
                    Points::new("SOH Residuals", PlotPoints::new(residuals))
                        .radius(3.5_f32)
                        .color(palette.warning),
                );
            });
    }

    fn render_summary_metrics(&self, ui: &mut egui::Ui) {
        let (n80, n70) = self.projected_life_cycles_coupled();
        let model = self.active_model();
        ui.horizontal_wrapped(|ui| {
            render_copyable_metric(ui, "First Life (80%)", &format!("{n80:.0} cycles"));
            render_copyable_metric(
                ui,
                "Projected Cycle Life (to 70%)",
                &format!("{n70:.0} cycles"),
            );
            render_copyable_metric(
                ui,
                "Arrhenius Factor",
                &format!("{:.2}x", model.arrhenius_factor()),
            );
            render_copyable_metric(ui, "Prefactor alpha", &format!("{:.3e}", model.alpha()));
        });
    }
}

//! Side-panel duty-mission builder (`1..=8` segments), export controls, and
//! `egui_plot` coupled warranty trajectory view for [`super::LifetimeEstimatorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:battery_degradation]

use super::LifetimeEstimatorTool;
use super::duty::{DutyPhaseKind, DutySegmentSpec};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::reflective_ui::{AnimatedSlider, render_copyable_metric};
use crate::tabs::battery_degradation::cell_card::ChemistryChoice;
use eframe::egui;
use egui_plot::{HLine, Line, Plot, PlotPoints};
use math_explorer::applied::battery_degradation::{
    MAX_C_RATE, MAX_DUTY_SEGMENTS, MAX_EXPONENT_Z, MAX_TEMPERATURE_C, MIN_C_RATE, MIN_EXPONENT_Z,
    MIN_TEMPERATURE_C,
};

impl LifetimeEstimatorTool {
    pub(crate) fn render_controls_panel(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "lifetime_duty_controls");
        crate::framework::PaneSidePanel::left(panel_id)
            .default_width(340.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.render_base_cell_controls(ui);
                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(6.0);
                    self.render_duty_mission_builder(ui);
                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(6.0);
                    self.render_export_buttons(ui);
                });
            });
    }

    fn render_base_cell_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Cell & Warranty Target");
        let mut selected = self.chemistry;
        egui::ComboBox::from_label("Chemistry")
            .selected_text(selected.label())
            .show_ui(ui, |ui| {
                for choice in ChemistryChoice::ALL {
                    ui.selectable_value(&mut selected, choice, choice.label());
                }
            });
        if selected != self.chemistry {
            self.chemistry = selected;
            self.exponent_z = selected.to_preset().default_exponent_z();
        }

        ui.add(
            AnimatedSlider::new(&mut self.target_capacity, 10.0..=99.0)
                .hard_bounds(10.0..=99.0)
                .text("Target Capacity (EOL) %")
                .id_salt("lifetime_target_capacity"),
        )
        .accessible_hover_text("End-of-life state-of-health percentage threshold.");

        ui.add(
            AnimatedSlider::new(&mut self.exponent_z, MIN_EXPONENT_Z..=MAX_EXPONENT_Z)
                .hard_bounds(MIN_EXPONENT_Z..=MAX_EXPONENT_Z)
                .text("Stretched Exponent (z)")
                .id_salt("lifetime_exponent_z"),
        )
        .accessible_hover_text("Cycling capacity fade exponent z in [0.1, 2.0].");

        if self.duty_segments.is_empty() {
            self.render_single_profile_sliders(ui);
        }
    }

    fn render_single_profile_sliders(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.strong("Single-Profile Usage");
        ui.add(
            AnimatedSlider::new(&mut self.dod, 0.0..=100.0)
                .hard_bounds(0.0..=100.0)
                .text("DoD (%)")
                .id_salt("lifetime_dod"),
        )
        .accessible_hover_text("Depth of discharge per cycle.");
        ui.add(
            AnimatedSlider::new(&mut self.cycles_per_day, 0.0..=10.0)
                .hard_bounds(0.0..=10.0)
                .text("Cycles / Day")
                .id_salt("lifetime_cycles_per_day"),
        )
        .accessible_hover_text("Daily charge/discharge cycles (0.0 = idle storage).");
        ui.add(
            AnimatedSlider::new(&mut self.temperature, MIN_TEMPERATURE_C..=MAX_TEMPERATURE_C)
                .hard_bounds(MIN_TEMPERATURE_C..=MAX_TEMPERATURE_C)
                .text("Temperature (°C)")
                .id_salt("lifetime_temperature"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.c_rate, MIN_C_RATE..=MAX_C_RATE)
                .hard_bounds(MIN_C_RATE..=MAX_C_RATE)
                .text("C-Rate (1/h)")
                .id_salt("lifetime_c_rate"),
        );
    }

    fn render_duty_mission_builder(&mut self, ui: &mut egui::Ui) {
        ui.heading("Duty Mission Builder (1..=8)");
        ui.small("Combine commute, DC fast-charge, and idle calendar storage phases.");
        ui.horizontal_wrapped(|ui| {
            let can_add = self.duty_segments.len() < MAX_DUTY_SEGMENTS;
            if ui
                .add_enabled(can_add, egui::Button::new("+ Commute"))
                .clicked()
            {
                self.add_duty_segment(DutyPhaseKind::Commute);
            }
            if ui
                .add_enabled(can_add, egui::Button::new("+ Fast-Charge"))
                .clicked()
            {
                self.add_duty_segment(DutyPhaseKind::FastCharge);
            }
            if ui
                .add_enabled(can_add, egui::Button::new("+ Storage"))
                .clicked()
            {
                self.add_duty_segment(DutyPhaseKind::Storage);
            }
            if !self.duty_segments.is_empty() && ui.button("Single Mode").clicked() {
                self.duty_segments.clear();
            }
        });

        let mut remove_idx = None;
        let seg_count = self.duty_segments.len();
        for (idx, seg) in self.duty_segments.iter_mut().enumerate() {
            ui.add_space(4.0);
            if render_single_segment_editor(ui, idx, seg_count > 1, seg) {
                remove_idx = Some(idx);
            }
        }
        if let Some(idx) = remove_idx {
            self.duty_segments.remove(idx);
        }
    }

    fn render_export_buttons(&mut self, ui: &mut egui::Ui) {
        ui.heading("Warranty & Cell Exports");
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("📊 Export Warranty CSV...")
                .accessible_hover_text(
                    "Download yearly SOH schedule with 80% and 70% warranty milestones.",
                )
                .clicked()
            {
                let name = format!(
                    "{}_warranty_schedule.csv",
                    self.chemistry.label().to_lowercase()
                );
                let csv = self.build_warranty_schedule_csv();
                self.dialogs.trigger_schedule_csv_save(&name, csv);
            }
            if ui
                .button("💾 Save JSON Cell Card...")
                .accessible_hover_text("Download JSON cell card with coupled aging parameters.")
                .clicked()
            {
                let name = format!("{}_cell_card.json", self.chemistry.label().to_lowercase());
                let card = self.build_cell_card();
                if let Err(err) = self.dialogs.trigger_cell_card_save(&name, &card) {
                    self.io_status = Some((true, err));
                }
            }
        });
        if let Some((is_err, msg)) = &self.io_status {
            let palette = SemanticPalette::from_visuals(ui.visuals());
            let color = if *is_err {
                palette.error
            } else {
                palette.success
            };
            ui.colored_label(color, msg);
        }
    }

    pub(crate) fn render_lifetime_plot_panel(&self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Coupled SOH & Warranty Lifetime Projection");
            let summary = self.lifetime_summary();
            let palette = SemanticPalette::from_visuals(ui.visuals());
            self.render_trajectory_plot(ui, &palette);
            ui.add_space(8.0);
            self.render_milestone_cards(ui, &summary);
        });
    }

    fn render_trajectory_plot(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let (coupled, cycling_only) = self.soh_trajectory_curves();
        Plot::new("lifetime_estimator_soh_plot")
            .view_aspect(2.1)
            .x_axis_label("Calendar Time (Years)")
            .y_axis_label("State of Health (Capacity)")
            .include_y(0.0)
            .include_y(1.0)
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(
                        "Coupled SOH (Cycling + sqrt(t) SEI)",
                        PlotPoints::new(coupled),
                    )
                    .color(palette.accent),
                );
                plot_ui.line(
                    Line::new("Cycling-Only SOH", PlotPoints::new(cycling_only))
                        .color(palette.info)
                        .style(egui_plot::LineStyle::Dashed { length: 6.0 }),
                );
                plot_ui.hline(
                    HLine::new("80% First-Life Warranty", 0.80)
                        .color(palette.warning)
                        .style(egui_plot::LineStyle::Dashed { length: 10.0 }),
                );
                plot_ui.hline(
                    HLine::new("70% End-of-Life Warranty", 0.70)
                        .color(palette.error)
                        .style(egui_plot::LineStyle::Dashed { length: 10.0 }),
                );
            });
    }

    fn render_milestone_cards(
        &self,
        ui: &mut egui::Ui,
        s: &math_explorer::applied::battery_degradation::DutyLifetimeSummary,
    ) {
        ui.horizontal_wrapped(|ui| {
            render_copyable_metric(
                ui,
                "80% Warranty Life",
                &format!("{:.2} yr ({:.0} cyc)", s.years_to_80, s.cycles_to_80),
            );
            render_copyable_metric(
                ui,
                "70% End-of-Life",
                &format!("{:.2} yr ({:.0} cyc)", s.years_to_70, s.cycles_to_70),
            );
            render_copyable_metric(
                ui,
                &format!("Target ({:.0}% SOH)", self.target_capacity),
                &format!(
                    "{:.2} yr ({:.0} d / {:.0} cyc)",
                    s.years_to_target, s.days_to_target, s.cycles_to_target
                ),
            );
            render_copyable_metric(
                ui,
                "Effective Duty",
                &format!("{:.2} cyc/day", s.effective_daily_cycles),
            );
        });
    }
}

fn render_single_segment_editor(
    ui: &mut egui::Ui,
    idx: usize,
    can_remove: bool,
    seg: &mut DutySegmentSpec,
) -> bool {
    let mut remove = false;
    ui.group(|ui| {
        ui.horizontal(|ui| {
            ui.strong(format!("Segment #{}", idx + 1));
            let mut kind = seg.kind;
            egui::ComboBox::from_id_salt(("duty_kind", idx))
                .selected_text(kind.label())
                .show_ui(ui, |ui| {
                    for k in DutyPhaseKind::ALL {
                        ui.selectable_value(&mut kind, k, k.label());
                    }
                });
            if kind != seg.kind {
                *seg = DutySegmentSpec::from_kind(kind);
            }
            if can_remove && ui.small_button("✕").clicked() {
                remove = true;
            }
        });
        ui.add(egui::Slider::new(&mut seg.days, 0.1..=365.25).text("Days"));
        ui.add(
            egui::Slider::new(&mut seg.temp_c, MIN_TEMPERATURE_C..=MAX_TEMPERATURE_C)
                .text("Temp (°C)"),
        );
        if seg.kind != DutyPhaseKind::Storage {
            ui.add(egui::Slider::new(&mut seg.dod, 1.0..=100.0).text("DoD (%)"));
            ui.add(egui::Slider::new(&mut seg.cycles, 0.0..=10.0).text("Cycles/d"));
            ui.add(egui::Slider::new(&mut seg.c_rate, MIN_C_RATE..=MAX_C_RATE).text("C-Rate"));
        } else {
            ui.small("Idle calendar storage (0 cycles/d): integrates sqrt(t) SEI fade only.");
        }
    });
    remove
}

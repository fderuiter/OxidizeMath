//! Side-panel parameter, calibration, schedule, and artifact download controls.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui::{self, RichText, Slider, SliderClamping};
use math_explorer::applied::pharmacokinetics::{
    MAX_REGIMEN_EVENTS, calibrate_ka_from_half_life, half_life, t_max,
};

use crate::accessibility::AccessibleHoverText;

use super::PharmacokineticRegimenTool;
use super::export::{spawn_csv_schedule_download, spawn_json_report_download};
use super::state::{CompactDoseEvent, PkModelKind, RegimenState};

impl PharmacokineticRegimenTool {
    pub(super) fn render_inside(&mut self, parent_ui: &mut egui::Ui) {
        self.poll_io_status();
        let theme_mode = crate::app::ThemeMode::from_visuals(parent_ui.visuals());
        self.theme_mode = theme_mode;

        let mut changed = false;
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "pk_regimen_controls");
        crate::framework::PaneSidePanel::left(panel_id)
            .default_width(340.0_f32)
            .show_inside(parent_ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    changed |= self.render_side_controls(ui);
                });
            });

        if changed {
            self.state.clamp();
            self.recalculate();
        }

        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.render_kpi_bar(ui, theme_mode);
            ui.separator();
            self.render_plot(ui, theme_mode);
        });
    }

    fn render_side_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.heading("ADME Regimen Planner");
        ui.small("Educational 1-compartment multi-dose & adherence simulator (client-side only).");
        ui.separator();

        changed |= self.render_scenario_presets(ui);
        ui.separator();
        changed |= self.render_model_and_pk_sliders(ui);
        ui.separator();
        changed |= self.render_ka_calibration_box(ui);
        ui.separator();
        changed |= self.render_window_sliders(ui);
        ui.separator();
        changed |= self.render_schedule_editor(ui);
        ui.separator();
        self.render_artifact_downloads(ui);
        changed
    }

    fn render_scenario_presets(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.label(RichText::new("Quick Regimen Scenarios").strong());
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("Uniform q12h")
                .accessible_hover_text("100 mg every 12 hours maintenance baseline")
                .clicked()
            {
                self.state = RegimenState::uniform_maintenance_scenario();
                changed = true;
            }
            if ui
                .button("2D Loading Dose")
                .accessible_hover_text("200 mg loading dose at t=0 followed by 100 mg q12h")
                .clicked()
            {
                self.state = RegimenState::loading_dose_scenario();
                changed = true;
            }
            if ui
                .button("Missed -> 1.5D Catch-Up")
                .accessible_hover_text("Missed 12h dose recovered safely with 150 mg at 18h")
                .clicked()
            {
                self.state = RegimenState::missed_dose_catchup_scenario();
                changed = true;
            }
            if ui
                .button("Missed -> 2D Double")
                .accessible_hover_text("Missed 12h dose followed by 200 mg double dose at 18h")
                .clicked()
            {
                self.state = RegimenState::missed_dose_double_scenario();
                changed = true;
            }
            if ui
                .button("Chiral XR + Booster")
                .accessible_hover_text("3:1 (d:l) biphasic XR capsule with afternoon IR booster")
                .clicked()
            {
                self.state = RegimenState::chiral_xr_scenario();
                changed = true;
            }
        });
        changed
    }

    fn render_model_and_pk_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.label(RichText::new("Drug & Compartment Kinetics").strong());
        egui::ComboBox::from_label("Formulation")
            .selected_text(self.state.model.label())
            .show_ui(ui, |ui| {
                for kind in [
                    PkModelKind::ImmediateRelease,
                    PkModelKind::BiphasicXr,
                    PkModelKind::ChiralEnantiomer,
                ] {
                    changed |= ui
                        .selectable_value(&mut self.state.model, kind, kind.label())
                        .changed();
                }
            });
        changed |= self.render_core_rate_sliders(ui);
        ui.small(format!(
            "Derived single-dose T_max = {:.2} h | Half-life t_1/2 = {:.2} h",
            t_max(self.state.ka, self.state.ke),
            half_life(self.state.ke)
        ));
        changed |= self.render_xr_and_chiral_sliders(ui);
        changed
    }

    fn render_core_rate_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        changed |= Self::clamped_slider(ui, &mut self.state.maintenance_dose_mg, 5.0..=1000.0, "Ref Dose D (mg)");
        changed |= Self::clamped_slider(ui, &mut self.state.tau_h, 2.0..=48.0, "Interval tau (h)");
        changed |= Self::clamped_slider(ui, &mut self.state.horizon_h, 12.0..=168.0, "Horizon (h)");
        changed |= Self::clamped_slider(ui, &mut self.state.f, 0.05..=1.0, "Bioavailability F");
        changed |= Self::clamped_slider(ui, &mut self.state.ka, 0.05..=6.0, "Absorption ka (1/h)");
        changed |= Self::clamped_slider(ui, &mut self.state.ke, 0.01..=1.0, "Elimination ke (1/h)");
        changed |= Self::clamped_slider(ui, &mut self.state.v, 1.0..=200.0, "Volume Vd (L)");
        changed
    }

    fn clamped_slider(
        ui: &mut egui::Ui,
        value: &mut f64,
        range: std::ops::RangeInclusive<f64>,
        label: &str,
    ) -> bool {
        ui.add(Slider::new(value, range).clamping(SliderClamping::Always).text(label))
            .changed()
    }

    fn render_xr_and_chiral_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let has_xr =
            self.state.model == PkModelKind::BiphasicXr || self.state.events.iter().any(|e| e.xr);
        if has_xr {
            changed |= ui
                .add(
                    Slider::new(&mut self.state.xr_lag_h, 0.5..=12.0)
                        .clamping(SliderClamping::Always)
                        .text("XR Lag (h)"),
                )
                .changed();
            changed |= ui
                .add(
                    Slider::new(&mut self.state.xr_f1, 0.1..=0.9)
                        .clamping(SliderClamping::Always)
                        .text("XR Pulse 1 f1"),
                )
                .changed();
        }
        if self.state.model == PkModelKind::ChiralEnantiomer {
            changed |= ui
                .add(
                    Slider::new(&mut self.state.chiral_f_d, 0.0..=1.0)
                        .clamping(SliderClamping::Always)
                        .text("d-Isomer Fraction"),
                )
                .changed();
            changed |= ui
                .add(
                    Slider::new(&mut self.state.chiral_l_ke, 0.01..=1.0)
                        .clamping(SliderClamping::Always)
                        .text("l-Isomer ke (1/h)"),
                )
                .changed();
        }
        changed
    }

    fn render_ka_calibration_box(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.label(RichText::new("Calibrate ka & ke from T_max / Half-Life").strong());
        changed |= ui
            .add(
                Slider::new(&mut self.state.cal_t_max_h, 0.25..=12.0)
                    .clamping(SliderClamping::Always)
                    .text("Target T_max (h)"),
            )
            .changed();
        changed |= ui
            .add(
                Slider::new(&mut self.state.cal_half_life_h, 1.0..=48.0)
                    .clamping(SliderClamping::Always)
                    .text("Target t_1/2 (h)"),
            )
            .changed();
        if ui
            .button("Solve ka & ke (Newton-Raphson)")
            .accessible_hover_text("Calibrate absorption rate ka via solve_ka from T_max and t_1/2")
            .clicked()
        {
            changed |= self.apply_ka_calibration();
        }
        if let Some(msg) = &self.calibration_status {
            ui.small(msg);
        }
        changed
    }

    fn render_window_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.label(RichText::new("Therapeutic Window & Baseline").strong());
        changed |= ui
            .add(
                Slider::new(&mut self.state.mec, 0.05..=25.0)
                    .clamping(SliderClamping::Always)
                    .text("MEC (mg/L)"),
            )
            .changed();
        let min_mtc = (self.state.mec + 0.1).min(49.5);
        changed |= ui
            .add(
                Slider::new(&mut self.state.mtc, min_mtc..=50.0)
                    .clamping(SliderClamping::Always)
                    .text("MTC (mg/L)"),
            )
            .changed();
        changed |= ui
            .checkbox(
                &mut self.state.show_baseline,
                "Overlay Uniform Maintenance Baseline",
            )
            .changed();
        changed
    }

    fn render_schedule_editor(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = self.render_schedule_header(ui);
        let mut remove_idx: Option<usize> = None;
        let can_delete = self.state.events.len() > 1;
        for (idx, ev) in self.state.events.iter_mut().enumerate() {
            if Self::render_dose_row(ui, idx, ev, can_delete, &mut changed) {
                remove_idx = Some(idx);
            }
        }
        if let Some(idx) = remove_idx {
            self.state.events.remove(idx);
            changed = true;
        }
        changed
    }

    fn render_schedule_header(&mut self, ui: &mut egui::Ui) -> bool {
        let mut added = false;
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!(
                    "Dose Schedule ({}/{MAX_REGIMEN_EVENTS})",
                    self.state.events.len()
                ))
                .strong(),
            );
            let can_add = self.state.events.len() < MAX_REGIMEN_EVENTS;
            if ui
                .add_enabled(can_add, egui::Button::new("+ Add Dose"))
                .accessible_hover_text("Append a scheduled dose administration")
                .clicked()
            {
                let next_t = self
                    .state
                    .events
                    .last()
                    .map_or(0.0, |ev| (ev.t + self.state.tau_h).min(768.0));
                self.state.events.push(CompactDoseEvent {
                    t: next_t,
                    d: self.state.maintenance_dose_mg,
                    xr: self.state.model == PkModelKind::BiphasicXr,
                });
                added = true;
            }
        });
        added
    }

    fn render_dose_row(
        ui: &mut egui::Ui,
        idx: usize,
        ev: &mut CompactDoseEvent,
        can_delete: bool,
        changed: &mut bool,
    ) -> bool {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.monospace(format!("#{:02}", idx + 1));
            *changed |= ui
                .add(egui::DragValue::new(&mut ev.t).speed(0.5).range(0.0..=768.0).suffix(" h"))
                .changed();
            *changed |= ui
                .add(egui::DragValue::new(&mut ev.d).speed(5.0).range(0.0..=5000.0).suffix(" mg"))
                .changed();
            *changed |= ui.checkbox(&mut ev.xr, "XR").changed();
            if ui
                .add_enabled(can_delete, egui::Button::new("✕"))
                .accessible_hover_text("Remove dose event")
                .clicked()
            {
                remove = true;
            }
        });
        remove
    }

    fn render_artifact_downloads(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Takeaway Artifacts").strong());
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("Save CSV Schedule...")
                .accessible_hover_text("Export scheduled doses and pre-dose levels to CSV")
                .clicked()
            {
                let (tx, rx) = std::sync::mpsc::channel();
                self.io_rx = Some(rx);
                spawn_csv_schedule_download(tx, &self.state, &self.curves);
            }
            if ui
                .button("Save JSON Exposure Report...")
                .accessible_hover_text("Export PK parameters, schedule, and exposure metrics JSON")
                .clicked()
            {
                if let Some(summary) = &self.curves.summary {
                    let (tx, rx) = std::sync::mpsc::channel();
                    self.io_rx = Some(rx);
                    spawn_json_report_download(tx, &self.state, summary);
                }
            }
        });
        if let Some(status) = &self.io_status {
            ui.small(status);
        }
    }

    pub(super) fn apply_ka_calibration(&mut self) -> bool {
        match calibrate_ka_from_half_life(
            self.state.cal_t_max_h,
            self.state.cal_half_life_h,
            100,
            math_commons::registry::TOLERANCE_FAST,
        ) {
            Ok((ka, ke)) => {
                self.state.ka = ka.clamp(0.05, 10.0);
                self.state.ke = ke.clamp(0.01, 2.0);
                self.calibration_status = Some(format!(
                    "Calibrated ka = {:.3} h^-1, ke = {:.4} h^-1",
                    self.state.ka, self.state.ke
                ));
                true
            }
            Err(err) => {
                self.calibration_status = Some(format!("Calibration error: {err}"));
                false
            }
        }
    }
}

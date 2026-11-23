//! Sidebar controls, copyable metrics, and `egui_plot` rendering for `PharmacokineticsTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    export, plot, AbsorptionMode, EntityMode, PharmacokineticsTool, ReleaseMode, MAX_DOSES,
    MAX_DOSE_MG, MAX_HORIZON_H, MAX_INTERVAL_H, MAX_KA, MAX_KE, MAX_LAG_H, MAX_MEC, MAX_MTC,
    MAX_TARGET_TMAX_H, MAX_VOLUME_L, MIN_HORIZON_H, MIN_INTERVAL_H, MIN_RATE, MIN_VOLUME_L,
};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::framework::{scoped_panel_id, PaneSidePanel};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use math_explorer::applied::pharmacokinetics::half_life;

/// Renders the side panel and central plot canvas for `PharmacokineticsTool`.
pub(crate) fn show_pharmacokinetics_inside(tool: &mut PharmacokineticsTool, ui: &mut egui::Ui) {
    tool.card_export.poll();
    let panel_id = scoped_panel_id(ui, "pharmacokinetics_controls_panel");
    PaneSidePanel::left(panel_id)
        .default_width(360.0)
        .show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                render_sidebar_contents(tool, ui);
            });
        });

    egui::CentralPanel::default().show_inside(ui, |ui| {
        plot::render_concentration_plot(tool, ui);
    });
}

fn render_sidebar_contents(tool: &mut PharmacokineticsTool, ui: &mut egui::Ui) {
    ui.heading("ADME Regimen Planner");
    let prev_summary = tool.last_param_summary.clone();
    let prev_series = export::build_active_trace_series(tool);

    let mut changed = false;
    changed |= render_formulation_section(tool, ui);
    ui.add_space(4.0);
    changed |= render_adme_parameters_section(tool, ui);
    ui.add_space(4.0);
    changed |= render_schedule_section(tool, ui);
    ui.add_space(4.0);
    changed |= render_window_section(tool, ui);

    if changed {
        if !prev_series.is_empty() {
            tool.snapshot_manager
                .push_history_run(prev_summary, prev_series);
        }
        tool.recalculate();
    }

    ui.separator();
    render_metrics_section(tool, ui);
    ui.add_space(4.0);
    render_trace_and_card_actions(tool, ui);
}

fn render_formulation_section(tool: &mut PharmacokineticsTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.group(|ui| {
        ui.label(egui::RichText::new("Formulation & Release").strong());
        ui.horizontal_wrapped(|ui| {
            changed |= ui
                .selectable_value(
                    &mut tool.entity_mode,
                    EntityMode::SingleEntity,
                    "Single Entity",
                )
                .accessible_hover_text("Achiral single-entity Bateman 1-compartment model")
                .changed();
            changed |= ui
                .selectable_value(&mut tool.entity_mode, EntityMode::Enantiomer, "Enantiomer")
                .accessible_hover_text(
                    "Chiral d/l-enantiomer mixture with distinct isomer kinetics",
                )
                .changed();
        });
        ui.horizontal_wrapped(|ui| {
            changed |= ui
                .selectable_value(
                    &mut tool.release_mode,
                    ReleaseMode::ImmediateRelease,
                    "Immediate (IR)",
                )
                .accessible_hover_text("Immediate-release single absorption pulse")
                .changed();
            changed |= ui
                .selectable_value(
                    &mut tool.release_mode,
                    ReleaseMode::TwoPulse,
                    "TwoPulse (XR)",
                )
                .accessible_hover_text("Biphasic extended-release immediate + delayed pulse")
                .changed();
        });
        if tool.release_mode == ReleaseMode::TwoPulse {
            changed |= render_two_pulse_sliders(tool, ui);
        }
        if tool.entity_mode == EntityMode::Enantiomer {
            changed |= render_enantiomer_sliders(tool, ui);
        }
    });
    changed
}

fn render_two_pulse_sliders(tool: &mut PharmacokineticsTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    changed |= ui
        .add(egui::Slider::new(&mut tool.lag_time_hours, 0.0..=MAX_LAG_H).text("Lag Time (h)"))
        .accessible_hover_text("Time delay before second extended-release pulse")
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut tool.f1, 0.0..=1.0).text("Pulse 1 Fraction (f1)"))
        .accessible_hover_text("Fraction of dose released immediately in first pulse")
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut tool.f2, 0.0..=1.0).text("Pulse 2 Fraction (f2)"))
        .accessible_hover_text("Fraction of dose released after lag time in second pulse")
        .changed();
    changed
}

fn render_enantiomer_sliders(tool: &mut PharmacokineticsTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    changed |= ui
        .add(egui::Slider::new(&mut tool.f_d, 0.0..=1.0).text("d-Isomer Fraction (f_d)"))
        .accessible_hover_text("Fraction of d-enantiomer in total administered dose")
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut tool.f_l, 0.0..=1.0).text("l-Isomer Fraction (f_l)"))
        .accessible_hover_text("Fraction of l-enantiomer in total administered dose")
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut tool.ka_l, MIN_RATE..=MAX_KA).text("l-Isomer k_a (1/h)"))
        .accessible_hover_text("First-order absorption rate constant for l-enantiomer")
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut tool.ke_l, MIN_RATE..=MAX_KE).text("l-Isomer k_e (1/h)"))
        .accessible_hover_text("First-order elimination rate constant for l-enantiomer")
        .changed();
    changed
}

fn render_adme_parameters_section(tool: &mut PharmacokineticsTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.group(|ui| {
        ui.label(egui::RichText::new("PK / ADME Parameters").strong());
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.bioavailability, 0.0..=1.0).text("Bioavailability (F)"),
            )
            .accessible_hover_text("Systemic oral bioavailability fraction F (0.0 to 1.0)")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.dose_mg, 0.0..=MAX_DOSE_MG)
                    .text("Maintenance Dose D (mg)"),
            )
            .accessible_hover_text("Maintenance oral dose amount in milligrams")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.volume_l, MIN_VOLUME_L..=MAX_VOLUME_L)
                    .text("Volume V_d (L)"),
            )
            .accessible_hover_text("Apparent volume of distribution in liters")
            .changed();
        changed |= render_elimination_controls(tool, ui);
        changed |= render_absorption_controls(tool, ui);
    });
    changed
}

fn render_elimination_controls(tool: &mut PharmacokineticsTool, ui: &mut egui::Ui) -> bool {
    let mut changed = ui
        .add(egui::Slider::new(&mut tool.ke, MIN_RATE..=MAX_KE).text("Elimination k_e (1/h)"))
        .accessible_hover_text("First-order elimination rate constant k_e (1/h)")
        .changed();
    let mut t_half = half_life(tool.ke).clamp(
        std::f64::consts::LN_2 / MAX_KE,
        std::f64::consts::LN_2 / MIN_RATE,
    );
    if ui
        .add(
            egui::Slider::new(
                &mut t_half,
                (std::f64::consts::LN_2 / MAX_KE)..=(std::f64::consts::LN_2 / MIN_RATE),
            )
            .text("Half-Life T_1/2 (h)"),
        )
        .accessible_hover_text("Elimination half-life T_1/2 = ln(2) / k_e in hours")
        .changed()
    {
        tool.ke = (std::f64::consts::LN_2 / t_half).clamp(MIN_RATE, MAX_KE);
        changed = true;
    }
    changed
}

fn render_absorption_controls(tool: &mut PharmacokineticsTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.horizontal_wrapped(|ui| {
        changed |= ui
            .radio_value(
                &mut tool.absorption_mode,
                AbsorptionMode::DirectKa,
                "Direct k_a",
            )
            .accessible_hover_text("Adjust absorption rate constant k_a directly")
            .changed();
        changed |= ui
            .radio_value(
                &mut tool.absorption_mode,
                AbsorptionMode::SolveFromTMax,
                "Solve ka from T_max",
            )
            .accessible_hover_text("Calibrate k_a from observed peak time T_max via Newton-Raphson")
            .changed();
    });
    match tool.absorption_mode {
        AbsorptionMode::DirectKa => {
            changed |= ui
                .add(
                    egui::Slider::new(&mut tool.ka, MIN_RATE..=MAX_KA).text("Absorption k_a (1/h)"),
                )
                .accessible_hover_text("First-order absorption rate constant k_a (1/h)")
                .changed();
        }
        AbsorptionMode::SolveFromTMax => {
            changed |= ui
                .add(
                    egui::Slider::new(&mut tool.target_t_max, 0.0..=MAX_TARGET_TMAX_H)
                        .text("Target T_max (h)"),
                )
                .accessible_hover_text("Target single-dose time to peak concentration T_max (h)")
                .changed();
            if let Some(ref err) = tool.solver_error {
                let palette = SemanticPalette::from_visuals(ui.visuals());
                ui.colored_label(palette.error, format!("⚠ {err}"));
            } else {
                ui.small(format!("Solved k_a = {:.4} 1/h", tool.effective_ka));
            }
        }
    }
    changed
}

fn render_schedule_section(tool: &mut PharmacokineticsTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.group(|ui| {
        ui.label(egui::RichText::new("Multi-Dose Regimen Schedule").strong());
        changed |= ui
            .add(egui::Slider::new(&mut tool.n_doses, 1..=MAX_DOSES).text("Dose Count (N)"))
            .accessible_hover_text("Number of scheduled dose slots (1 to 64)")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.interval_hours, MIN_INTERVAL_H..=MAX_INTERVAL_H)
                    .text("Interval τ (h)"),
            )
            .accessible_hover_text("Time interval tau between scheduled maintenance doses")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.horizon_hours, MIN_HORIZON_H..=MAX_HORIZON_H)
                    .text("Horizon (h)"),
            )
            .accessible_hover_text("Total plasma simulation duration in hours")
            .changed();
        changed |= ui
            .checkbox(
                &mut tool.use_loading_dose,
                "Enable Initial Loading Dose (t = 0h)",
            )
            .accessible_hover_text("Administer a custom loading dose D_load at slot 1 (t = 0h)")
            .changed();
        if tool.use_loading_dose {
            changed |= ui
                .add(
                    egui::Slider::new(&mut tool.loading_dose_mg, 0.0..=MAX_DOSE_MG)
                        .text("Loading Dose D_load (mg)"),
                )
                .accessible_hover_text("Initial loading dose administered at hour 0")
                .changed();
        }
        if changed {
            tool.missed_slots.retain(|&idx| idx < tool.n_doses);
        }
        changed |= render_missed_slot_chips(tool, ui);
    });
    changed
}

fn render_missed_slot_chips(tool: &mut PharmacokineticsTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.collapsing(
        format!("Missed Dose Slots ({} skipped)", tool.missed_slots.len()),
        |ui| {
            if !tool.missed_slots.is_empty()
                && ui
                    .button("Restore All Doses")
                    .accessible_hover_text("Clear all skipped dose slots")
                    .clicked()
            {
                tool.missed_slots.clear();
                changed = true;
            }
            ui.horizontal_wrapped(|ui| {
                for slot in 0..tool.n_doses {
                    let is_missed = tool.missed_slots.binary_search(&slot).is_ok();
                    let hr = (slot as f64) * tool.interval_hours;
                    let label = format!("#{} ({:.0}h)", slot + 1, hr);
                    if ui
                        .selectable_label(is_missed, label)
                        .accessible_hover_text("Toggle missed dose at this scheduled slot")
                        .clicked()
                    {
                        match tool.missed_slots.binary_search(&slot) {
                            Ok(pos) => {
                                tool.missed_slots.remove(pos);
                            }
                            Err(pos) => {
                                tool.missed_slots.insert(pos, slot);
                            }
                        }
                        changed = true;
                    }
                }
            });
        },
    );
    changed
}

fn render_window_section(tool: &mut PharmacokineticsTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.group(|ui| {
        ui.label(egui::RichText::new("Therapeutic Window & Scale").strong());
        if ui
            .add(egui::Slider::new(&mut tool.mec, 0.0..=MAX_MEC).text("MEC (mg/L)"))
            .accessible_hover_text("Minimum Effective Concentration threshold in mg/L")
            .changed()
        {
            if tool.mtc <= tool.mec {
                tool.mtc = (tool.mec + 0.5).min(MAX_MTC);
            }
            changed = true;
        }
        let min_mtc = (tool.mec + 0.05).min(MAX_MTC);
        changed |= ui
            .add(egui::Slider::new(&mut tool.mtc, min_mtc..=MAX_MTC).text("MTC (mg/L)"))
            .accessible_hover_text("Minimum Toxic Concentration ceiling in mg/L")
            .changed();
        changed |= ui
            .checkbox(&mut tool.log10_y, "Base-10 Logarithmic Y-Axis (log10 C)")
            .accessible_hover_text("Toggle semi-logarithmic base-10 concentration axis")
            .changed();
    });
    changed
}

fn render_metrics_section(tool: &PharmacokineticsTool, ui: &mut egui::Ui) {
    let m = &tool.metrics;
    ui.label(egui::RichText::new("Derived PK Metrics").strong());
    render_copyable_metric(ui, "Peak C_max", &format!("{:.3} mg/L", m.c_max));
    render_copyable_metric(ui, "Trough C_min", &format!("{:.3} mg/L", m.c_min));
    render_copyable_metric(ui, "Half-Life T_1/2", &format!("{:.2} h", m.half_life_h));
    render_copyable_metric(ui, "Peak Time T_max", &format!("{:.2} h", m.t_max_h));
    render_copyable_metric(ui, "Clearance CL", &format!("{:.2} L/h", m.clearance_l_h));
    render_copyable_metric(
        ui,
        "Trapezoidal AUC",
        &format!("{:.2} mg·h/L", m.auc_trapezoidal),
    );
    render_copyable_metric(
        ui,
        "Accumulation R_ac",
        &format!("{:.2}x", m.accumulation_ratio),
    );
    render_copyable_metric(
        ui,
        "Therapeutic Window Time",
        &format!("{:.1}%", m.therapeutic_window_pct),
    );
}

fn render_trace_and_card_actions(tool: &mut PharmacokineticsTool, ui: &mut egui::Ui) {
    let summary = tool.param_summary();
    let next_idx = tool.snapshot_manager.len() + 1;
    let active_series = export::build_active_trace_series(tool);
    if tool.snapshot_manager.render_ui(ui, move || {
        (!active_series.is_empty()).then(|| (format!("Regimen {next_idx}"), summary, active_series))
    }) {
        tool.revision = tool.revision.wrapping_add(1).max(1);
    }

    ui.add_space(4.0);
    render_card_export_buttons(tool, ui);
}

fn render_card_export_buttons(tool: &mut PharmacokineticsTool, ui: &mut egui::Ui) {
    ui.group(|ui| {
        ui.label(egui::RichText::new("Dosing Summary Cards").strong());
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("📋 Copy Markdown Card")
                .accessible_hover_text("Copy Markdown dosing schedule and PK metrics card")
                .clicked()
            {
                ui.ctx().copy_text(export::format_markdown_card(tool));
                tool.card_export
                    .record_info("Copied Markdown dosing card to clipboard".into());
            }
            if ui
                .button("💾 Save .md Card")
                .accessible_hover_text("Download or save Markdown dosing regimen card (.md)")
                .clicked()
            {
                let md = export::format_markdown_card(tool);
                tool.card_export.trigger_save_markdown(md);
            }
        });
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("📋 Copy CSV Card")
                .accessible_hover_text("Copy CSV dosing schedule and PK summary")
                .clicked()
            {
                ui.ctx().copy_text(export::format_csv_card(tool));
                tool.card_export
                    .record_info("Copied CSV dosing summary to clipboard".into());
            }
            if ui
                .button("💾 Save .csv Card")
                .accessible_hover_text("Download or save CSV dosing regimen summary (.csv)")
                .clicked()
            {
                let csv = export::format_csv_card(tool);
                tool.card_export.trigger_save_csv(csv);
            }
        });
        render_card_status(tool, ui);
    });
}

fn render_card_status(tool: &mut PharmacokineticsTool, ui: &mut egui::Ui) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    if let Some(ref info) = tool.card_export.status_info {
        ui.colored_label(palette.success, format!("✓ {info}"));
    }
    if let Some(ref err) = tool.card_export.status_error {
        ui.colored_label(palette.error, format!("⚠ {err}"));
    }
}

// [cite:clinical_trials_statistics]

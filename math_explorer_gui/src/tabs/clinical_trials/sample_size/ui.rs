//! Split-pane UI for `SampleSizeCalculatorTool`: parameter sliders, power curve plot, and protocol table.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::cards::{
    draw_analytical_power_plot, draw_required_n_curve_plot, render_contingency_and_risk_card,
    render_t_test_card,
};
use super::export::{format_protocol_csv, format_protocol_markdown};
use super::{CalculationMode, PowerPlotView, SampleSizeCalculatorTool};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;

/// Renders the `SampleSizeCalculatorTool` sidebar, interactive power curve, and protocol sizing table.
pub fn show_sample_size_inside(tool: &mut SampleSizeCalculatorTool, ui: &mut egui::Ui) {
    tool.io_state.poll_save("protocol sizing table");
    let panel_id = crate::framework::scoped_panel_id(ui, "sample_size_sidebar");
    crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            render_sidebar_controls(tool, ui);
        });
    });

    egui::CentralPanel::default().show_inside(ui, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            render_power_curve_plot(tool, ui);
            ui.add_space(8.0);
            ui.separator();
            render_protocol_sizing_table(tool, ui);
        });
    });
}

fn render_sidebar_controls(tool: &mut SampleSizeCalculatorTool, ui: &mut egui::Ui) {
    ui.heading("Sample Size Design");
    ui.label("Two-arm 1:1 Neyman-Pearson trial sizing.");
    ui.add_space(6.0);

    let mut changed = render_mode_and_error_controls(tool, ui);
    ui.add_space(4.0);
    changed |= render_effect_controls(tool, ui);
    if changed {
        tool.recalculate();
    }

    ui.separator();
    render_required_size_banner(tool, ui);
    ui.separator();
    render_protocol_export_buttons(tool, ui);
    tool.io_state.render_status(ui);
    ui.separator();
    let mut cards_changed = render_t_test_card(ui, &mut tool.cards, &tool.cards_eval);
    ui.add_space(4.0);
    cards_changed |= render_contingency_and_risk_card(ui, &mut tool.cards, &tool.cards_eval);
    if cards_changed {
        tool.recalculate();
    }
}

fn render_mode_and_error_controls(tool: &mut SampleSizeCalculatorTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.group(|ui| {
        ui.label(egui::RichText::new("Endpoint Family").strong());
        changed |= ui
            .radio_value(
                &mut tool.mode,
                CalculationMode::Means,
                "Compare Means (t-test)",
            )
            .changed();
        changed |= ui
            .radio_value(
                &mut tool.mode,
                CalculationMode::Proportions,
                "Compare Proportions (Chi-square)",
            )
            .changed();
    });

    ui.add_space(4.0);
    ui.group(|ui| {
        ui.label(egui::RichText::new("Error Rates").strong());
        changed |= ui
            .add(egui::Slider::new(&mut tool.alpha, 0.001..=0.20).text("α (Two-Sided)"))
            .accessible_hover_text("Significance level α (e.g., 0.05 for 5% Type I error)")
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut tool.power, 0.50..=0.999).text("Power (1 - β)"))
            .accessible_hover_text("Target statistical power 1 - β (e.g., 0.80 or 0.90)")
            .changed();
    });
    changed
}

fn render_effect_controls(tool: &mut SampleSizeCalculatorTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.group(|ui| match tool.mode {
        CalculationMode::Means => {
            ui.label(egui::RichText::new("Continuous Effect Assumptions").strong());
            changed |= ui
                .add(egui::Slider::new(&mut tool.delta, 0.01..=5.0).text("Effect Size (δ)"))
                .accessible_hover_text("Minimum clinically meaningful difference between arms")
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut tool.sigma, 0.1..=10.0).text("Common SD (σ)"))
                .accessible_hover_text("Assumed population standard deviation σ")
                .changed();
        }
        CalculationMode::Proportions => {
            ui.label(egui::RichText::new("Binary Event Proportions").strong());
            changed |= ui
                .add(egui::Slider::new(&mut tool.p1, 0.01..=0.99).text("Control p₁"))
                .accessible_hover_text("Expected event proportion in Control group")
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut tool.p2, 0.01..=0.99).text("Treatment p₂"))
                .accessible_hover_text("Expected event proportion in Treatment group")
                .changed();
        }
    });
    changed
}

fn render_required_size_banner(tool: &SampleSizeCalculatorTool, ui: &mut egui::Ui) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    match &tool.result {
        Ok(n) => {
            render_copyable_metric(ui, "Required per Arm (N)", &n.to_string());
            render_copyable_metric(ui, "Total Enrollment (2N)", &(n * 2).to_string());
            if let Some(achieved) = tool.evaluate_power_at(*n) {
                render_copyable_metric(ui, "Achieved Power", &format!("{:.1}%", achieved * 100.0));
            }
        }
        Err(err) => {
            ui.colored_label(palette.error, format!("⚠ Error: {err}"));
        }
    }
}

fn render_protocol_export_buttons(tool: &mut SampleSizeCalculatorTool, ui: &mut egui::Ui) {
    ui.label(egui::RichText::new("Protocol Sizing Exports").strong());
    ui.horizontal_wrapped(|ui| {
        if ui
            .button("⬇ Protocol (.csv)")
            .accessible_hover_text("Download protocol power-milestone sizing schedule as CSV")
            .clicked()
        {
            tool.io_state
                .trigger_save_csv("protocol_sample_size.csv", format_protocol_csv(tool));
        }
        if ui
            .button("⬇ Protocol (.md)")
            .accessible_hover_text("Download Markdown clinical protocol sample-size card")
            .clicked()
        {
            tool.io_state.trigger_save_markdown(
                "protocol_sample_size_card.md",
                format_protocol_markdown(tool),
            );
        }
        if ui
            .button("📋 Copy Card")
            .accessible_hover_text("Copy Markdown protocol sizing card to clipboard")
            .clicked()
        {
            ui.ctx().copy_text(format_protocol_markdown(tool));
            tool.io_state
                .record_info("Copied protocol sizing card to clipboard".into());
        }
    });
}

fn render_power_curve_plot(tool: &mut SampleSizeCalculatorTool, ui: &mut egui::Ui) {
    ui.horizontal_wrapped(|ui| {
        ui.heading("Statistical Power & Sample Size Curves");
        ui.separator();
        ui.selectable_value(
            &mut tool.plot_view,
            PowerPlotView::PowerBySampleSize,
            "Power (1 - β) vs N",
        );
        ui.selectable_value(
            &mut tool.plot_view,
            PowerPlotView::SampleSizeByPower,
            "Required N vs Power",
        );
    });
    if tool.plot_view == PowerPlotView::PowerBySampleSize {
        let req_marker = tool
            .result
            .as_ref()
            .ok()
            .and_then(|&n| tool.evaluate_power_at(n).map(|p| vec![[n as f64, p]]))
            .unwrap_or_default();
        draw_analytical_power_plot(ui, tool.power_curve.clone(), req_marker, tool.power);
    } else {
        let operating_pt = tool.result.as_ref().ok().map(|&n| [tool.power, n as f64]);
        draw_required_n_curve_plot(ui, &tool.per_arm_curve, &tool.total_curve, operating_pt);
    }
}

fn render_protocol_sizing_table(tool: &SampleSizeCalculatorTool, ui: &mut egui::Ui) {
    ui.heading("Protocol Power-Milestone Sizing Schedule");
    if tool.protocol_rows.is_empty() {
        ui.label("Adjust effect parameters (δ > 0 or p₁ ≠ p₂) to generate protocol milestones.");
        return;
    }
    egui::Grid::new("protocol_sizing_grid")
        .striped(true)
        .spacing([24.0, 4.0])
        .show(ui, |ui| {
            ui.strong("Target Power (1 - β)");
            ui.strong("Per-Arm Sample Size (N)");
            ui.strong("Total 2-Arm Enrollment (2N)");
            ui.strong("Achieved Power");
            ui.end_row();
            for row in &tool.protocol_rows {
                ui.label(format!("{:.0}% ({:.2})", row.power * 100.0, row.power));
                ui.label(row.n_per_arm.to_string());
                ui.label(row.total_n.to_string());
                ui.label(format!("{:.2}%", row.achieved_power * 100.0));
                ui.end_row();
            }
        });
}

// [cite:clinical_trials_statistics]

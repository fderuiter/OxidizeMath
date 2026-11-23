//! Concentration-time `egui_plot` canvas, therapeutic window overlays, and semi-log rendering.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{EntityMode, PharmacokineticsTool, LOG10_CONC_FLOOR};
use crate::app::SemanticPalette;
use eframe::egui;
use egui_plot::{HLine, Legend, Line, LineStyle, Plot, PlotPoints, PlotUi, Polygon};

/// Renders the plasma concentration profile plot with `MEC`/`MTC` therapeutic window and overlays.
pub(crate) fn render_concentration_plot(tool: &PharmacokineticsTool, ui: &mut egui::Ui) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    ui.heading("Plasma Concentration Profile");
    if let Some(ref err) = tool.solver_error {
        ui.colored_label(palette.error, format!("Solver diagnostic: {err}"));
    }

    let y_label = if tool.log10_y {
        "log10 Plasma Concentration (mg/L)"
    } else {
        "Plasma Concentration C(t) (mg/L)"
    };
    let mut plot = Plot::new("pharmacokinetics_adme_plot")
        .view_aspect(2.0)
        .x_axis_label("Time (h)")
        .y_axis_label(y_label)
        .legend(Legend::default());

    if tool.log10_y {
        plot = plot.y_axis_formatter(|mark, _range| {
            let conc = 10.0_f64.powf(mark.value);
            format!("{conc:.3}")
        });
    } else {
        plot = plot.include_y(0.0).include_y(tool.mtc * 1.1);
    }

    plot.show(ui, |plot_ui| {
        draw_window_and_curves(tool, plot_ui, palette);
    });
}

fn draw_window_and_curves<'a>(
    tool: &'a PharmacokineticsTool,
    plot_ui: &mut PlotUi<'a>,
    palette: SemanticPalette,
) {
    let (y_mec, y_mtc) = if tool.log10_y {
        (
            tool.mec.max(LOG10_CONC_FLOOR).log10(),
            tool.mtc.max(LOG10_CONC_FLOOR).log10(),
        )
    } else {
        (tool.mec, tool.mtc)
    };

    let win_fill = egui::Color32::from_rgba_unmultiplied(
        palette.success.r(),
        palette.success.g(),
        palette.success.b(),
        32,
    );
    let rect_pts = vec![
        [0.0, y_mec],
        [tool.horizon_hours, y_mec],
        [tool.horizon_hours, y_mtc],
        [0.0, y_mtc],
    ];
    plot_ui.polygon(
        Polygon::new("Therapeutic Window", PlotPoints::new(rect_pts)).fill_color(win_fill),
    );
    plot_ui.hline(
        HLine::new("MEC", y_mec)
            .color(palette.success)
            .style(LineStyle::Dashed { length: 8.0 }),
    );
    plot_ui.hline(
        HLine::new("MTC", y_mtc)
            .color(palette.error)
            .style(LineStyle::Dashed { length: 8.0 }),
    );

    if !tool.plot_total_points.is_empty() {
        plot_ui.line(
            Line::new("Total C(t)", &tool.plot_total_points[..])
                .color(palette.series_color(0))
                .width(2.2_f32),
        );
    }
    if tool.entity_mode == EntityMode::Enantiomer {
        plot_ui.line(
            Line::new("d-Isomer", &tool.plot_d_points[..])
                .color(palette.series_color(1))
                .style(LineStyle::Dashed { length: 6.0 }),
        );
        plot_ui.line(
            Line::new("l-Isomer", &tool.plot_l_points[..])
                .color(palette.series_color(2))
                .style(LineStyle::Dotted { spacing: 5.0 }),
        );
    }
    draw_trace_overlays(tool, plot_ui, palette);
}

fn draw_trace_overlays<'a>(
    tool: &'a PharmacokineticsTool,
    plot_ui: &mut PlotUi<'a>,
    palette: SemanticPalette,
) {
    if !tool.log10_y {
        tool.snapshot_manager.render_on_plot(plot_ui);
        return;
    }
    for (idx, snap) in tool
        .snapshot_manager
        .history_runs()
        .iter()
        .chain(tool.snapshot_manager.snapshots().iter())
        .enumerate()
    {
        if !snap.visible {
            continue;
        }
        let color = palette.series_color(idx + 3);
        for series in &snap.series {
            let pts: Vec<[f64; 2]> = series
                .points
                .iter()
                .map(|&[x, y]| [x, y.max(LOG10_CONC_FLOOR).log10()])
                .collect();
            let label = format!("{} ({})", series.name, snap.display_label());
            plot_ui.line(
                Line::new(label, PlotPoints::new(pts))
                    .color(color)
                    .style(LineStyle::Dashed { length: 6.0 }),
            );
        }
    }
}

// [cite:clinical_trials_statistics]

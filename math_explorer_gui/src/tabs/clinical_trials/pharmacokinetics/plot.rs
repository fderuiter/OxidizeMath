//! Concentration-time `egui_plot` canvas, therapeutic window overlays, and semi-log rendering.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{EntityMode, PharmacokineticsTool, LOG10_CONC_FLOOR};
use crate::app::SemanticPalette;
use eframe::egui;
use egui_plot::{HLine, Legend, Line, LineStyle, Plot, PlotPoints, PlotUi, Polygon};

/// Renders the plasma concentration profile plot with `MEC`/`MTC` therapeutic window and overlays.
pub(crate) fn render_concentration_plot(tool: &mut PharmacokineticsTool, ui: &mut egui::Ui) {
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

    let (y_mec, y_mtc) = if tool.log10_y {
        (
            tool.mec.max(LOG10_CONC_FLOOR).log10(),
            tool.mtc.max(LOG10_CONC_FLOOR).log10(),
        )
    } else {
        (tool.mec, tool.mtc)
    };
    let horizon_hours = tool.horizon_hours;
    let entity_mode = tool.entity_mode;
    let log10_y = tool.log10_y;
    let PharmacokineticsTool {
        plot_total_points,
        plot_d_points,
        plot_l_points,
        snapshot_manager,
        callout_manager,
        ..
    } = tool;

    plot.show(ui, |plot_ui| {
        draw_therapeutic_window(plot_ui, palette, horizon_hours, y_mec, y_mtc);
        draw_concentration_series(
            plot_ui,
            palette,
            entity_mode,
            plot_total_points,
            plot_d_points,
            plot_l_points,
        );
        draw_trace_overlays(snapshot_manager, log10_y, plot_ui, palette);
        callout_manager.render_on_plot(plot_ui);
    });
}

fn draw_concentration_series<'a>(
    plot_ui: &mut PlotUi<'a>,
    palette: SemanticPalette,
    entity_mode: EntityMode,
    total_pts: &'a [egui_plot::PlotPoint],
    d_pts: &'a [egui_plot::PlotPoint],
    l_pts: &'a [egui_plot::PlotPoint],
) {
    if !total_pts.is_empty() {
        plot_ui.line(
            Line::new("Total C(t)", total_pts)
                .color(palette.series_color(0))
                .width(2.2_f32),
        );
    }
    if entity_mode == EntityMode::Enantiomer {
        plot_ui.line(
            Line::new("d-Isomer", d_pts)
                .color(palette.series_color(1))
                .style(LineStyle::Dashed { length: 6.0 }),
        );
        plot_ui.line(
            Line::new("l-Isomer", l_pts)
                .color(palette.series_color(2))
                .style(LineStyle::Dotted { spacing: 5.0 }),
        );
    }
}

fn draw_therapeutic_window(
    plot_ui: &mut PlotUi<'_>,
    palette: SemanticPalette,
    horizon_hours: f64,
    y_mec: f64,
    y_mtc: f64,
) {
    let win_fill = egui::Color32::from_rgba_unmultiplied(
        palette.success.r(),
        palette.success.g(),
        palette.success.b(),
        32,
    );
    let rect_pts = vec![
        [0.0, y_mec],
        [horizon_hours, y_mec],
        [horizon_hours, y_mtc],
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
}

fn draw_trace_overlays<'a>(
    snapshot_manager: &'a crate::framework::TraceSnapshotManager,
    log10_y: bool,
    plot_ui: &mut PlotUi<'a>,
    palette: SemanticPalette,
) {
    if !log10_y {
        snapshot_manager.render_on_plot(plot_ui);
        return;
    }
    for (idx, snap) in snapshot_manager
        .history_runs()
        .iter()
        .chain(snapshot_manager.snapshots().iter())
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

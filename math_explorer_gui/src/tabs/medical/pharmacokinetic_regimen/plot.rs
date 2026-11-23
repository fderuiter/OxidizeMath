//! Central KPI summary strip and `egui_plot` therapeutic-window visualization.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui::{self, Stroke};
use egui_plot::{HLine, Legend, Line, LineStyle, Plot, PlotPoints, Points, Polygon};

use crate::app::{SemanticPalette, ThemeMode};
use crate::reflective_ui::render_copyable_metric;

use super::PharmacokineticRegimenTool;
use super::state::PkModelKind;

impl PharmacokineticRegimenTool {
    pub(super) fn render_kpi_bar(&self, ui: &mut egui::Ui, theme_mode: ThemeMode) {
        let palette = theme_mode.palette();
        let Some(s) = &self.curves.summary else {
            if let Some(err) = &self.curves.error {
                ui.colored_label(palette.error, err);
            }
            return;
        };
        ui.horizontal_wrapped(|ui| {
            render_copyable_metric(
                ui,
                "C_max",
                &format!("{:.2} mg/L (@ {:.1}h)", s.c_max, s.t_peak_h),
            );
            ui.separator();
            render_copyable_metric(ui, "C_min (tau)", &format!("{:.2} mg/L", s.c_min));
            ui.separator();
            render_copyable_metric(ui, "AUC_0-tau", &format!("{:.2} mg*h/L", s.auc_0_tau));
            ui.separator();
            render_copyable_metric(
                ui,
                "R_ac",
                &format!(
                    "{:.2} (obs {:.2})",
                    s.accumulation_ratio, s.observed_accumulation_ratio
                ),
            );
            ui.separator();
            render_copyable_metric(
                ui,
                "Time in Range",
                &format!("{:.2}% ({:.1} h)", s.time_in_range_pct, s.time_in_range_h),
            );
            ui.separator();
            render_copyable_metric(
                ui,
                "Toxic (>MTC)",
                &format!("{:.2}% ({:.1} h)", s.toxic_pct, s.toxic_hours),
            );
        });
    }

    pub(super) fn render_plot(&self, ui: &mut egui::Ui, theme_mode: ThemeMode) {
        let palette = theme_mode.palette();
        let h = self.state.horizon_h;
        let mec = self.state.mec;
        let mtc = self.state.mtc;
        let y_top = self
            .curves
            .summary
            .map_or(mtc * 1.2, |s| s.c_max.max(mtc * 1.15));

        Plot::new("Pharmacokinetic Plasma Concentration Profile")
            .legend(Legend::default())
            .x_axis_label("Time (hours)")
            .y_axis_label("Plasma Concentration (mg/L)")
            .include_x(0.0)
            .include_x(h)
            .include_y(0.0)
            .include_y(y_top)
            .show(ui, |plot_ui| {
                self.draw_window_bands(plot_ui, h, mec, mtc, y_top, &palette);
                self.draw_concentration_lines(plot_ui, &palette);
            });
    }

    fn draw_window_bands(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        h: f64,
        mec: f64,
        mtc: f64,
        y_top: f64,
        palette: &SemanticPalette,
    ) {
        let mec_col = palette.success;
        let mtc_col = palette.error;
        let win_fill = mec_col.gamma_multiply(0.18_f32);
        let tox_fill = mtc_col.gamma_multiply(0.14_f32);

        let win_poly = vec![[0.0, mec], [h, mec], [h, mtc], [0.0, mtc]];
        plot_ui.polygon(
            Polygon::new("Therapeutic Window [MEC, MTC]", PlotPoints::new(win_poly))
                .fill_color(win_fill)
                .stroke(Stroke::new(0.8_f32, mec_col)),
        );
        if y_top > mtc {
            let tox_poly = vec![[0.0, mtc], [h, mtc], [h, y_top], [0.0, y_top]];
            plot_ui.polygon(
                Polygon::new("Toxic Zone (> MTC)", PlotPoints::new(tox_poly))
                    .fill_color(tox_fill)
                    .stroke(Stroke::new(0.6_f32, mtc_col)),
            );
        }
        plot_ui.hline(
            HLine::new("MEC", mec)
                .color(mec_col)
                .style(LineStyle::Dashed { length: 6.0_f32 }),
        );
        plot_ui.hline(
            HLine::new("MTC", mtc)
                .color(mtc_col)
                .style(LineStyle::Dashed { length: 6.0_f32 }),
        );
    }

    fn zip_points(times: &[f64], values: &[f64]) -> PlotPoints<'static> {
        let pts: Vec<[f64; 2]> = times
            .iter()
            .copied()
            .zip(values.iter().copied())
            .map(|(t, c)| [t, c])
            .collect();
        PlotPoints::new(pts)
    }

    fn draw_concentration_lines(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: &SemanticPalette,
    ) {
        if self.state.show_baseline && self.curves.baseline.len() == self.curves.times.len() {
            plot_ui.line(
                Line::new(
                    "Uniform Maintenance Baseline",
                    Self::zip_points(&self.curves.times, &self.curves.baseline),
                )
                .color(palette.muted_text)
                .width(1.6_f32)
                .style(LineStyle::Dotted { spacing: 5.0_f32 }),
            );
        }
        if self.state.model == PkModelKind::ChiralEnantiomer
            && self.curves.d_isomer.len() == self.curves.times.len()
        {
            plot_ui.line(
                Line::new(
                    "d-Enantiomer",
                    Self::zip_points(&self.curves.times, &self.curves.d_isomer),
                )
                .color(palette.series_color(2))
                .width(1.8_f32),
            );
            plot_ui.line(
                Line::new(
                    "l-Enantiomer",
                    Self::zip_points(&self.curves.times, &self.curves.l_isomer),
                )
                .color(palette.series_color(3))
                .width(1.8_f32),
            );
        }
        plot_ui.line(
            Line::new(
                "Total Plasma C(t)",
                Self::zip_points(&self.curves.times, &self.curves.total),
            )
            .color(palette.series_color(0))
            .width(2.6_f32),
        );
        let dose_markers: Vec<[f64; 2]> = self.state.events.iter().map(|ev| [ev.t, 0.0]).collect();
        plot_ui.points(
            Points::new("Dose Events", PlotPoints::new(dose_markers))
                .color(palette.warning)
                .radius(4.5_f32),
        );
    }
}

//! Shared `egui` controls for empirical CSV column mapping, parameter calibration triggers,
//! copyable confidence-interval diagnostics, and residual CSV downloads.

use super::{CalibrationReport, EmpiricalCsvState};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;

/// Configuration options for rendering [`EmpiricalCsvState::show_panel`].
pub struct EmpiricalPanelOptions<'a> {
    /// Unique `egui` ID salt for `ComboBox` widgets in this tool.
    pub id_salt: &'a str,
    /// Label for the independent variable selector (e.g., `"Time Column (X)"`).
    pub x_label: &'a str,
    /// Label for the observed dependent variable selector (e.g., `"Observed Column (Y)"`).
    pub y_label: &'a str,
    /// Optional label for a third auxiliary column selector (e.g., `"DoD Column (optional)"`).
    pub aux_label: Option<&'a str>,
    /// Default download filename for `Download Residuals CSV` (e.g., `"sir_residuals.csv"`).
    pub residual_filename: &'a str,
    /// Action label on the primary calibration button.
    pub fit_button_label: &'a str,
}

impl EmpiricalCsvState {
    /// Renders CSV import/clear buttons, column-mapping selectors, `Fit Parameters` trigger,
    /// copyable goodness-of-fit & 95% CI rows, and `Download Residuals CSV`.
    ///
    /// Returns `true` when the user clicks `Fit Parameters` with a loaded dataset.
    pub fn show_panel(&mut self, ui: &mut egui::Ui, opts: &EmpiricalPanelOptions<'_>) -> bool {
        let ctx = ui.ctx().clone();
        self.poll_dialogs(&ctx);

        let mut fit_clicked = false;
        ui.collapsing("Empirical CSV Calibration", |ui| {
            self.render_import_row(ui);
            self.render_status_banner(ui);
            if self.table.is_some() {
                ui.add_space(4.0);
                self.render_column_selectors(ui, opts);
                ui.add_space(4.0);
                let btn = ui
                    .button(opts.fit_button_label)
                    .accessible_hover_text("Run damped Levenberg-Marquardt nonlinear calibration");
                if btn.clicked() {
                    fit_clicked = true;
                }
            }
            if self.report.is_some() {
                ui.separator();
                self.render_diagnostics_section(ui, &ctx, opts.residual_filename);
            }
        });
        fit_clicked
    }

    fn render_import_row(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("Import CSV...")
                .accessible_hover_text(
                    "Open a local CSV or TSV file (or drag and drop onto the workspace)",
                )
                .clicked()
            {
                self.trigger_open_dialog();
            }
            if self.table.is_some()
                && ui
                    .button("Clear CSV")
                    .accessible_hover_text("Remove loaded empirical CSV overlay")
                    .clicked()
            {
                self.clear();
            }
        });
        if let Some(ref table) = self.table {
            ui.small(format!(
                "Dataset: {} ({} rows × {} cols)",
                table.name,
                table.row_count(),
                table.column_count()
            ));
        } else {
            ui.small("Drop a .csv/.tsv file onto the plot or click Import CSV...");
        }
    }

    fn render_status_banner(&self, ui: &mut egui::Ui) {
        let Some(ref msg) = self.status_message else {
            return;
        };
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let color = if self.is_error {
            palette.error
        } else {
            palette.success
        };
        ui.colored_label(color, msg);
    }

    fn render_column_selectors(&mut self, ui: &mut egui::Ui, opts: &EmpiricalPanelOptions<'_>) {
        let Some(headers) = self.table.as_ref().map(|t| t.headers.clone()) else {
            return;
        };
        let col_count = headers.len();
        self.x_col = self.x_col.min(col_count.saturating_sub(1));
        self.y_col = self.y_col.min(col_count.saturating_sub(1));

        egui::ComboBox::from_id_salt((opts.id_salt, "x_col"))
            .selected_text(column_title(&headers, self.x_col))
            .show_ui(ui, |ui| {
                for (idx, title) in headers.iter().enumerate() {
                    ui.selectable_value(&mut self.x_col, idx, title);
                }
            });
        ui.small(opts.x_label);

        egui::ComboBox::from_id_salt((opts.id_salt, "y_col"))
            .selected_text(column_title(&headers, self.y_col))
            .show_ui(ui, |ui| {
                for (idx, title) in headers.iter().enumerate() {
                    ui.selectable_value(&mut self.y_col, idx, title);
                }
            });
        ui.small(opts.y_label);

        if let Some(aux_label) = opts.aux_label {
            self.render_aux_selector(ui, opts.id_salt, aux_label, &headers);
        }
    }

    fn render_aux_selector(
        &mut self,
        ui: &mut egui::Ui,
        id_salt: &str,
        aux_label: &str,
        headers: &[String],
    ) {
        if let Some(idx) = self.aux_col {
            if idx >= headers.len() {
                self.aux_col = None;
            }
        }
        let selected_text = match self.aux_col {
            Some(idx) => column_title(headers, idx),
            None => "Fixed Slider Value",
        };
        egui::ComboBox::from_id_salt((id_salt, "aux_col"))
            .selected_text(selected_text)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.aux_col, None, "Fixed Slider Value");
                for (idx, title) in headers.iter().enumerate() {
                    ui.selectable_value(&mut self.aux_col, Some(idx), title);
                }
            });
        ui.small(aux_label);
    }

    fn render_diagnostics_section(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        residual_filename: &str,
    ) {
        if let Some(ref report) = self.report {
            render_report_metrics(ui, report);
        }
        ui.add_space(4.0);
        if ui
            .button("Download Residuals CSV")
            .accessible_hover_text(
                "Export x,y_observed,y_fitted,residual calibration report as CSV",
            )
            .clicked()
        {
            if let Err(err) = self.trigger_save_residuals_dialog(Some(ctx), residual_filename) {
                self.record_error(Some(ctx), err);
            }
        }
    }
}

fn render_report_metrics(ui: &mut egui::Ui, report: &CalibrationReport) {
    ui.strong("Fit Diagnostics & 95% Confidence Intervals");
    for est in &report.parameters {
        let label = format!("{} (95% CI)", est.name);
        let value = format!(
            "{:.4} ± {:.4} (SE {:.4})",
            est.value, est.ci_95_half_width, est.std_error
        );
        render_copyable_metric(ui, &label, &value);
    }
    for (label, val) in &report.derived_metrics {
        render_copyable_metric(ui, label, val);
    }
    render_copyable_metric(ui, "RMSE", &format!("{:.6}", report.rmse));
    render_copyable_metric(ui, "R²", &format!("{:.6}", report.r_squared));
    render_copyable_metric(ui, "AIC", &format!("{:.3}", report.aic));
    render_copyable_metric(
        ui,
        "Solver Convergence",
        &format!("{} iters ({:.2} ms)", report.iterations, report.elapsed_ms),
    );
}

fn column_title(headers: &[String], idx: usize) -> &str {
    headers.get(idx).map_or("Column", String::as_str)
}

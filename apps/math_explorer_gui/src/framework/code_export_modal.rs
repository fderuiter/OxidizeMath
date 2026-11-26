//! Modal drawer for previewing, copying, and downloading LaTeX, Rust, and Python model artifacts.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use eframe::egui;

/// Active preview tab inside the `Code & LaTeX` export modal drawer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CodeExportTab {
    /// Symbolic and parameter-calibrated LaTeX equations (`.tex`).
    #[default]
    Latex,
    /// Standalone Rust 2024 solver snippet (`.rs`).
    Rust,
    /// Standalone Python 3 (`numpy` / `scipy`) solver script (`.py`).
    Python,
}

pub(super) fn wrapped_colored_label(ui: &mut egui::Ui, color: egui::Color32, text: String) {
    ui.add(egui::Label::new(egui::RichText::new(text).color(color)).wrap());
}

pub(super) fn export_action_button(
    ui: &mut egui::Ui,
    enabled: bool,
    label: &str,
    tip: &str,
    close_on_click: bool,
) -> bool {
    let clicked = ui
        .add_enabled(enabled, egui::Button::new(label))
        .accessible_hover_text(tip)
        .clicked();
    if clicked && close_on_click {
        ui.close();
    }
    clicked
}

impl SimulationFramework {
    pub(super) fn code_export_modal_scope_id(&self, id_source: &str) -> egui::Id {
        egui::Id::new(("framework_code_export", id_source, self.instance_id))
    }

    pub(super) fn open_code_export_modal(&mut self) {
        self.show_code_export_modal = true;
    }

    pub(super) fn show_code_export_modal_dialog(&mut self, ctx: &egui::Context, id_source: &str) {
        let modal_id = self.code_export_modal_scope_id(id_source);
        let mut open = self.show_code_export_modal;
        if !open {
            ModalLifecycle::new(modal_id).show(ctx, &mut open, |_, _| {});
            return;
        }

        let tool_name = self
            .active_tool
            .as_ref()
            .map_or("Simulation", |tool| tool.name());
        let artifact = self
            .active_tool
            .as_ref()
            .filter(|tool| tool.has_finite_parameters())
            .and_then(|tool| tool.export_code_artifact());

        ModalLifecycle::new(modal_id)
            .title(format!("Code & LaTeX Export - {id_source}"))
            .default_width(680.0)
            .max_width(760.0)
            .show(ctx, &mut open, |ui, frame_ctx| {
                if self.render_code_modal_contents(ui, ctx, tool_name, artifact.as_ref()) {
                    frame_ctx.close();
                }
            });
        self.show_code_export_modal = open;
    }

    fn render_code_modal_contents(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        tool_name: &str,
        artifact: Option<&CodeArtifact>,
    ) -> bool {
        ui.heading(format!("Reproducible Model & Solver Export ({tool_name})"));
        ui.label(
            egui::RichText::new(
                "Calibrated to active slider parameters on cold preview/export paths.",
            )
            .weak(),
        );
        ui.add_space(6.0);

        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.code_export_tab, CodeExportTab::Latex, "📐 LaTeX");
            ui.selectable_value(
                &mut self.code_export_tab,
                CodeExportTab::Rust,
                "🦀 Rust (.rs)",
            );
            ui.selectable_value(
                &mut self.code_export_tab,
                CodeExportTab::Python,
                "🐍 Python (.py)",
            );
        });
        ui.separator();

        if let Some(art) = artifact {
            match self.code_export_tab {
                CodeExportTab::Latex => self.render_latex_tab(ui, ctx, tool_name, art),
                CodeExportTab::Rust => self.render_source_tab(
                    ui,
                    ctx,
                    tool_name,
                    &art.rust_source,
                    ExportFormat::Rust,
                    "Rust source",
                ),
                CodeExportTab::Python => self.render_source_tab(
                    ui,
                    ctx,
                    tool_name,
                    &art.python_source,
                    ExportFormat::Python,
                    "Python script",
                ),
            }
        } else {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            wrapped_colored_label(
                ui,
                palette.error,
                format!(
                    "⚠️ Code export unavailable for {tool_name}: ensure equation syntax and parameters are finite."
                ),
            );
        }

        ui.separator();
        self.render_code_modal_footer(ui)
    }

    fn render_latex_tab(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        tool_name: &str,
        art: &CodeArtifact,
    ) {
        ui.label(egui::RichText::new("Symbolic Governing Equation:").strong());
        egui_plot::render_math_label(
            ui,
            &art.symbolic_latex,
            egui::TextStyle::Body,
            ui.visuals().text_color(),
        );
        ui.add_space(6.0);
        ui.label(egui::RichText::new("Parameter-Calibrated Equation:").strong());
        egui_plot::render_math_label(
            ui,
            &art.calibrated_latex,
            egui::TextStyle::Body,
            ui.visuals().text_color(),
        );
        ui.add_space(6.0);

        let tex_doc = art.to_tex_document(tool_name);
        render_readonly_code_box(ui, "code_modal_tex_scroll", &tex_doc);
        ui.add_space(6.0);

        let tex_file = Self::build_export_filename(tool_name, ExportFormat::Tex);
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("📋 Copy Calibrated LaTeX")
                .accessible_hover_text("Copy parameter-calibrated LaTeX equation to clipboard")
                .clicked()
            {
                self.copy_code_payload_to_clipboard(
                    "Calibrated LaTeX",
                    art.calibrated_latex.clone(),
                    Some(ctx),
                );
            }
            if ui
                .button("📋 Copy Symbolic LaTeX")
                .accessible_hover_text("Copy symbolic LaTeX equation to clipboard")
                .clicked()
            {
                self.copy_code_payload_to_clipboard(
                    "Symbolic LaTeX",
                    art.symbolic_latex.clone(),
                    Some(ctx),
                );
            }
            if ui
                .button(format!("💾 Save {tex_file}"))
                .accessible_hover_text("Save symbolic and calibrated LaTeX equations to .tex file")
                .clicked()
            {
                self.queue_export(ExportFormat::Tex);
                ctx.request_repaint();
            }
        });
    }

    fn render_source_tab(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        tool_name: &str,
        source: &str,
        format: ExportFormat,
        clip_label: &'static str,
    ) {
        render_readonly_code_box(ui, format.extension(), source);
        ui.add_space(6.0);
        let filename = Self::build_export_filename(tool_name, format);
        ui.horizontal_wrapped(|ui| {
            if ui
                .button(format!("📋 Copy {}", format.label()))
                .accessible_hover_text(format!("Copy {clip_label} to clipboard"))
                .clicked()
            {
                self.copy_code_payload_to_clipboard(clip_label, source.to_string(), Some(ctx));
            }
            if ui
                .button(format!("💾 Save {filename}"))
                .accessible_hover_text(format!("Save standalone {} file to disk", format.label()))
                .clicked()
            {
                self.queue_export(format);
                ctx.request_repaint();
            }
        });
    }

    fn render_code_modal_footer(&mut self, ui: &mut egui::Ui) -> bool {
        let mut close_clicked = false;
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("Close")
                .accessible_hover_text("Close Code & LaTeX export dialog")
                .clicked()
            {
                close_clicked = true;
            }
            if let Some(ref msg) = self.export_status_message {
                let color = self.export_status_severity.palette_color(ui.visuals());
                wrapped_colored_label(ui, color, msg.clone());
            }
        });
        close_clicked
    }
}

fn render_readonly_code_box(ui: &mut egui::Ui, salt: &str, code: &str) {
    egui::ScrollArea::vertical()
        .id_salt(salt)
        .max_height(240.0)
        .show(ui, |ui| {
            let mut preview = code.to_string();
            ui.add(
                egui::TextEdit::multiline(&mut preview)
                    .font(egui::TextStyle::Monospace)
                    .desired_width(f32::INFINITY)
                    .interactive(false),
            );
        });
}

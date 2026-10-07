use eframe::egui;
use federated_registry::Severity;

use super::MathExplorerApp;

impl MathExplorerApp {
    #[allow(clippy::too_many_lines)]
    pub(super) fn render_issues_panel(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("issues_panel")
            .resizable(true)
            .min_height(100.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Issues & Diagnostics");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Clear").clicked() {
                            self.diagnostic_events.clear();
                        }
                        ui.checkbox(&mut self.show_errors, "Errors/Fatal");
                        ui.checkbox(&mut self.show_warnings, "Warnings");
                        ui.checkbox(&mut self.show_info, "Info");
                    });
                });
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for event in &self.diagnostic_events {
                        let show = match event.severity {
                            Severity::Info => self.show_info,
                            Severity::Warning => self.show_warnings,
                            Severity::Error | Severity::Fatal => self.show_errors,
                        };
                        if !show {
                            continue;
                        }

                        let color = match event.severity {
                            Severity::Info => egui::Color32::LIGHT_BLUE,
                            Severity::Warning => egui::Color32::YELLOW,
                            Severity::Error => egui::Color32::RED,
                            Severity::Fatal => egui::Color32::DARK_RED,
                        };

                        ui.group(|ui| {
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "[{} - {}]",
                                        event.source, event.severity
                                    ))
                                    .color(color)
                                    .strong(),
                                );
                                if let Some(thread) = &event.thread_name {
                                    ui.label(
                                        egui::RichText::new(format!("(Thread: {})", thread))
                                            .italics(),
                                    );
                                }
                                ui.label(&event.message);
                            });
                            if !event.metadata.is_empty() {
                                ui.horizontal_wrapped(|ui| {
                                    for (k, v) in &event.metadata {
                                        ui.label(
                                            egui::RichText::new(format!("{}: {}", k, v))
                                                .monospace()
                                                .size(10.0),
                                        );
                                    }
                                });
                            }
                        });
                    }
                });
            });
    }
}

//! Shared fault recovery banners and empty-plot guidance widgets for async worker tools.
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::SimWorkerState;
use crate::accessibility::AccessibleHoverText;
use crate::framework::recovery::{render_recovery_banner, DiagnosticReport, RecoveryAction};
use eframe::egui;

/// User recovery action selected from a shared worker fault banner.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WorkerFaultAction {
    /// Reset simulation state in place or respawn a disconnected worker.
    Reset,
    /// Restore baseline model parameters and reset simulation state.
    ResetParameters,
    /// Apply a structured numerical solver recovery remedy.
    Recovery(RecoveryAction),
}

fn render_fault_frame(
    ui: &mut egui::Ui,
    headline: &str,
    detail: Option<&str>,
    offer_reset_parameters: bool,
) -> Option<WorkerFaultAction> {
    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    let mut chosen = None;
    egui::Frame::new()
        .fill(palette.error_banner_bg)
        .stroke(egui::Stroke::new(1.5_f32, palette.error_banner_stroke))
        .corner_radius(egui::CornerRadius::same(6))
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.label(
                    egui::RichText::new(headline)
                        .color(palette.error)
                        .strong()
                        .size(13.0),
                );
                if let Some(info) = detail {
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new(info).monospace().color(palette.warning));
                }
                ui.add_space(6.0);
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .button("↻ Reset")
                        .accessible_hover_text(
                            "Reset simulation state or respawn a disconnected background worker",
                        )
                        .clicked()
                    {
                        chosen = Some(WorkerFaultAction::Reset);
                    }
                    if offer_reset_parameters
                        && ui
                            .button("↻ Reset Parameters")
                            .accessible_hover_text(
                                "Restore default baseline parameters and reset simulation",
                            )
                            .clicked()
                    {
                        chosen = Some(WorkerFaultAction::ResetParameters);
                    }
                });
            });
        });
    chosen
}

/// Renders a fault banner when a simulation worker is in `SimWorkerState::Faulted`
/// or carries a solver `DiagnosticReport`.
pub fn render_worker_fault_banner(
    ui: &mut egui::Ui,
    state: SimWorkerState,
    report: Option<&DiagnosticReport>,
    supports_param_recovery: bool,
) -> Option<WorkerFaultAction> {
    if state != SimWorkerState::Faulted && report.is_none() {
        return None;
    }

    if supports_param_recovery && report.is_some() {
        if let Some(action) = render_recovery_banner(ui, report) {
            return Some(WorkerFaultAction::Recovery(action));
        }
        return None;
    }

    if let Some(diag) = report {
        let context = format!(
            "{} (t: {:.4}s | Step: {} | Variable: {})",
            diag.message, diag.t_diverged, diag.step_index, diag.variable
        );
        render_fault_frame(
            ui,
            "⚠ Solver Divergence Warning: Non-finite state detected. Simulation halted.",
            Some(&context),
            false,
        )
    } else {
        render_fault_frame(
            ui,
            "⚠ Simulation Worker Faulted: Background worker disconnected or halted.",
            Some("Click Reset to restore Ready state or respawn the worker."),
            false,
        )
    }
}

/// Renders a fault banner with custom diagnostic detail and optional parameter reset button.
pub fn render_custom_fault_banner(
    ui: &mut egui::Ui,
    headline: &str,
    detail: Option<&str>,
    offer_reset_parameters: bool,
) -> Option<WorkerFaultAction> {
    render_fault_frame(ui, headline, detail, offer_reset_parameters)
}

/// Renders subtle guidance text above or inside an empty or baseline plot view.
pub fn render_empty_plot_guidance(ui: &mut egui::Ui, message: &str) {
    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    egui::Frame::new()
        .fill(palette.surface_fill)
        .stroke(egui::Stroke::new(1.0_f32, palette.graph_edge))
        .corner_radius(egui::CornerRadius::same(4))
        .inner_margin(egui::Margin::symmetric(8, 6))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(message)
                    .italics()
                    .color(palette.muted_text),
            );
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{contrast_ratio, ThemeMode};

    fn find_text_color(output: &egui::FullOutput, needle: &str) -> Option<egui::Color32> {
        let mut stack: Vec<&egui::epaint::Shape> = output.shapes.iter().map(|c| &c.shape).collect();
        while let Some(shape) = stack.pop() {
            match shape {
                egui::epaint::Shape::Text(t) if t.galley.job.text.contains(needle) => {
                    return t
                        .galley
                        .job
                        .sections
                        .first()
                        .map(|s| s.format.color)
                        .or(Some(t.fallback_color));
                }
                egui::epaint::Shape::Vec(v) => stack.extend(v.iter()),
                _ => {}
            }
        }
        None
    }

    #[test]
    fn test_worker_fault_and_guidance_frames_meet_wcag_aa_across_themes() {
        for mode in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
            let ctx = egui::Context::default();
            ctx.set_visuals(mode.visuals());
            let palette = mode.palette();
            let out = ctx.run(egui::RawInput::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let _ = render_worker_fault_banner(ui, SimWorkerState::Faulted, None, false);
                    render_empty_plot_guidance(ui, "Simulation ready — click Run.");
                });
            });
            let headline = find_text_color(&out, "Simulation Worker Faulted").expect("headline");
            let detail = find_text_color(&out, "Click Reset to restore").expect("detail");
            let guidance = find_text_color(&out, "Simulation ready").expect("guidance");
            assert!(contrast_ratio(headline, palette.error_banner_bg) >= 4.5);
            assert!(contrast_ratio(detail, palette.error_banner_bg) >= 4.5);
            assert!(contrast_ratio(guidance, palette.surface_fill) >= 4.5);
            assert!(contrast_ratio(palette.error_banner_stroke, palette.error_banner_bg) >= 3.0);
        }
    }
}

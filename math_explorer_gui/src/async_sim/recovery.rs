#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;
use std::collections::HashMap;

/// Available one-click recovery remedies for continuum physics solver divergence.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RecoveryAction {
    /// Halve step size / speed and re-integrate.
    AutoScaleStepSize,
    /// Clamp instability rate parameters and fields to stable bounds.
    ClampParameters,
    /// Reset parameters to baseline configuration.
    ResetParameters,
}

/// Structured diagnostic report generated when a continuum physics simulation encounters non-finite states.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DiagnosticReport {
    /// Time $t_{\text{diverged}}$ at which divergence occurred.
    pub t_diverged: f64,
    /// Step index at which the solver diverged.
    pub step_index: usize,
    /// Name of the variable that became non-finite (e.g., "velocity (u)" or "psi(x)").
    pub variable: String,
    /// Detailed diagnostic description of the failure.
    pub message: String,
}

impl DiagnosticReport {
    /// Creates a new diagnostic report.
    pub fn new(
        t_diverged: f64,
        step_index: usize,
        variable: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            t_diverged,
            step_index,
            variable: variable.into(),
            message: message.into(),
        }
    }
}

/// Publishes a structured divergence telemetry event to the global registry.
pub fn emit_divergence_telemetry(
    source: &str,
    t_div: f64,
    step_idx: usize,
    var_name: &str,
    preset: &str,
    dt: f64,
) {
    let mut metadata = HashMap::new();
    metadata.insert("t_diverged".to_string(), format!("{:.4}", t_div));
    metadata.insert("step_count".to_string(), step_idx.to_string());
    metadata.insert("diverged_variable".to_string(), var_name.to_string());
    metadata.insert("preset".to_string(), preset.to_string());
    metadata.insert("dt".to_string(), format!("{:.6}", dt));

    federated_registry::global_registry().emit(federated_registry::TelemetryEvent {
        source: source.to_string(),
        severity: federated_registry::Severity::Error,
        message: format!(
            "Solver divergence: non-finite state encountered at t={:.4} (step {}) in variable {}",
            t_div, step_idx, var_name
        ),
        metadata,
        thread_name: None,
    });
}

/// Publishes a structured recovery action telemetry event to the global registry.
pub fn emit_recovery_telemetry(source: &str, action_name: &str, details: &str) {
    let mut metadata = HashMap::new();
    metadata.insert("action".to_string(), action_name.to_string());
    metadata.insert("details".to_string(), details.to_string());

    federated_registry::global_registry().emit(federated_registry::TelemetryEvent {
        source: source.to_string(),
        severity: federated_registry::Severity::Info,
        message: format!("Applied recovery action: {}", action_name),
        metadata,
        thread_name: None,
    });
}

fn render_recovery_action_buttons(ui: &mut egui::Ui) -> Option<RecoveryAction> {
    let mut chosen_action = None;
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Recovery Actions:").italics());

        if ui
            .button("Auto-scale Step Size (dt)")
            .on_hover_text("Halve step size or speed and resume simulation")
            .clicked()
        {
            chosen_action = Some(RecoveryAction::AutoScaleStepSize);
        }

        if ui
            .button("Clamp Instability Parameters")
            .on_hover_text("Restrict parameters and fields to stable bounds")
            .clicked()
        {
            chosen_action = Some(RecoveryAction::ClampParameters);
        }

        if ui
            .button("Reset All Parameters")
            .on_hover_text("Restore default baseline configuration")
            .clicked()
        {
            chosen_action = Some(RecoveryAction::ResetParameters);
        }
    });
    chosen_action
}

/// Renders the interactive diagnostic recovery banner offering one-click remedies.
pub fn render_recovery_banner(
    ui: &mut egui::Ui,
    report: Option<&DiagnosticReport>,
) -> Option<RecoveryAction> {
    let mut chosen_action = None;

    egui::Frame::new()
        .fill(egui::Color32::from_rgb(45, 20, 20))
        .stroke(egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(220, 50, 50)))
        .corner_radius(egui::CornerRadius::same(6))
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("⚠️ Solver Divergence Warning: Non-finite state detected (NaN or Inf). Integration terminated.")
                            .color(egui::Color32::from_rgb(255, 100, 100))
                            .strong()
                            .size(13.0),
                    );
                });

                if let Some(diag) = report {
                    ui.add_space(4.0);
                    ui.horizontal_wrapped(|ui| {
                        ui.label(
                            egui::RichText::new("Diagnostic Context: ")
                                .strong()
                                .color(egui::Color32::LIGHT_GRAY),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "t_diverged: {:.4}s  |  Step: {}  |  Variable: {}",
                                diag.t_diverged, diag.step_index, diag.variable
                            ))
                            .monospace()
                            .color(egui::Color32::from_rgb(240, 200, 150)),
                        );
                    });
                }

                ui.add_space(6.0);
                chosen_action = render_recovery_action_buttons(ui);
            });
        });

    chosen_action
}

/// Helper that applies one-click recovery remedies to a UnifiedSimTool.
pub fn apply_recovery_action_helper<M: crate::async_sim::unified::UnifiedModel>(
    action: RecoveryAction,
    params: &std::sync::Arc<std::sync::RwLock<HashMap<String, f64>>>,
    controller: &mut crate::async_sim::BackgroundWorker<
        crate::async_sim::unified::UnifiedSimRunner<M>,
    >,
    steps_per_frame: &mut usize,
) {
    match action {
        RecoveryAction::ResetParameters => {
            let default_params = M::default_parameters();
            if let Ok(mut lock) = params.write() {
                *lock = default_params;
            }
            emit_recovery_telemetry(
                M::tool_source(),
                "Reset All Parameters",
                "Restored default baseline configuration",
            );
            if controller.state() == crate::async_sim::SimWorkerState::Faulted
                || controller.state() == crate::async_sim::SimWorkerState::Terminated
            {
                let runner = crate::async_sim::unified::UnifiedSimRunner::<M>::new(
                    std::sync::Arc::clone(params),
                );
                *controller = crate::async_sim::BackgroundWorker::new(runner);
            } else {
                let _ = controller.send_command(crate::async_sim::SimCommand::Reset);
            }
        }
        RecoveryAction::ClampParameters => {
            if let Ok(mut lock) = params.write() {
                M::clamp_parameters(&mut lock);
            }
            emit_recovery_telemetry(
                M::tool_source(),
                "Clamp Instability Parameters",
                "Clamped parameters to stable bounds",
            );
            let _ = controller.send_command(crate::async_sim::SimCommand::Custom(
                "ClampParameters".to_string(),
            ));
        }
        RecoveryAction::AutoScaleStepSize => {
            *steps_per_frame = (*steps_per_frame / 2).max(1);
            emit_recovery_telemetry(
                M::tool_source(),
                "Auto-scale Step Size (dt)",
                "Halved step size",
            );
            let _ =
                controller.send_command(crate::async_sim::SimCommand::SetSpeed(*steps_per_frame));
        }
    }
}

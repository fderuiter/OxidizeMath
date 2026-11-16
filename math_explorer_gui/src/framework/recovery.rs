#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;

/// Available one-click recovery remedies for solver divergence.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RecoveryAction {
    /// Halve step size (dt) to restore numerical stability.
    AutoScaleStepSize,
    /// Clamp instability rate parameters and initial conditions to stable bounds.
    ClampParameters,
    /// Reset parameters to baseline configuration.
    ResetParameters,
}

/// Structured diagnostic report generated when a simulation encounters non-finite states.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DiagnosticReport {
    /// Time $t_{\text{diverged}}$ or elapsed time at which divergence occurred.
    pub t_diverged: f64,
    /// Step index at which the solver diverged.
    pub step_index: usize,
    /// Name of the variable that became non-finite.
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

/// Trait defining finite-state validation and diagnostic reporting for simulation tools and solvers.
pub trait FiniteStateGuard {
    /// Validates whether the current state consists solely of finite numerical values.
    fn is_finite_state(&self) -> bool {
        true
    }

    /// Validates the current state and returns an error diagnostic report if non-finite values are detected.
    fn validate_state(&self) -> Result<(), DiagnosticReport> {
        if self.is_finite_state() {
            Ok(())
        } else {
            Err(DiagnosticReport::new(
                0.0,
                0,
                "state",
                "Non-finite floating-point state encountered.",
            ))
        }
    }

    /// Returns the diagnostic report if the solver has diverged.
    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        None
    }

    /// Returns whether the simulation state has diverged.
    fn diverged(&self) -> bool {
        !self.is_finite_state() || self.diagnostic_report().is_some()
    }

    /// Halve step size (dt) to restore stability.
    fn auto_scale_step_size(&mut self) {}

    /// Clamp parameters to stable bounds.
    fn clamp_parameters(&mut self) {}

    /// Restore baseline parameters and resume simulation.
    fn reset_parameters(&mut self) {}
}

/// Publishes a structured divergence telemetry event to the global registry.
pub fn emit_divergence_telemetry(
    source: &str,
    t_div: f64,
    step_idx: usize,
    var_name: &str,
    message: &str,
) {
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("t_diverged".to_string(), format!("{:.4}", t_div));
    metadata.insert("step_count".to_string(), step_idx.to_string());
    metadata.insert("diverged_variable".to_string(), var_name.to_string());

    federated_registry::global_registry().emit(federated_registry::TelemetryEvent {
        source: source.to_string(),
        severity: federated_registry::Severity::Error,
        message: message.to_string(),
        metadata,
        thread_name: None,
    });
}

/// Publishes a structured recovery action telemetry event to the global registry.
pub fn emit_recovery_telemetry(source: &str, action_name: &str, details: &str) {
    let mut metadata = std::collections::HashMap::new();
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

/// Renders the recovery action buttons inside a horizontal layout.
pub fn render_recovery_action_buttons(ui: &mut egui::Ui) -> Option<RecoveryAction> {
    let mut chosen_action = None;
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Recovery Actions:").italics());

        if ui
            .button("Auto-scale Step Size (dt)")
            .on_hover_text("Halve step size (dt) and recalculate simulation trajectory")
            .clicked()
        {
            emit_recovery_telemetry(
                "FiniteStateGuard",
                "Auto-scale Step Size (dt)",
                "Halved step size",
            );
            chosen_action = Some(RecoveryAction::AutoScaleStepSize);
        }

        if ui
            .button("Clamp Instability Parameters")
            .on_hover_text(
                "Restrict rate parameters and initial conditions to pre-configured stable bounds",
            )
            .clicked()
        {
            emit_recovery_telemetry(
                "FiniteStateGuard",
                "Clamp Instability Parameters",
                "Clamped rate parameters and ICs to stable bounds",
            );
            chosen_action = Some(RecoveryAction::ClampParameters);
        }

        if ui
            .button("Reset All Parameters")
            .on_hover_text("Restore default baseline configuration")
            .clicked()
        {
            emit_recovery_telemetry(
                "FiniteStateGuard",
                "Reset All Parameters",
                "Restored default baseline configuration",
            );
            chosen_action = Some(RecoveryAction::ResetParameters);
        }
    });
    chosen_action
}

/// Renders the interactive diagnostic recovery banner offering one-click remedies.
/// Matches the ADR 0008 visual design with diagnostic context and recovery actions.
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

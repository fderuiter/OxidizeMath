#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::AccessibleHoverText;
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

/// Available recovery remedies for the centralized framework state guard,
/// supporting step-back state snapshot rollback.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum StateGuardAction {
    /// Restore the most recent valid pre-divergence state snapshot and plot data.
    StepBack,
    /// Halve step size (dt) and recalculate simulation trajectory from snapshot.
    AutoScaleStepSize,
    /// Restrict rate parameters and initial conditions to pre-configured stable bounds.
    ClampParameters,
    /// Restore default baseline configuration.
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
    match render_state_guard_action_buttons(ui, false) {
        Some(StateGuardAction::AutoScaleStepSize) => Some(RecoveryAction::AutoScaleStepSize),
        Some(StateGuardAction::ClampParameters) => Some(RecoveryAction::ClampParameters),
        Some(StateGuardAction::ResetParameters) => Some(RecoveryAction::ResetParameters),
        _ => None,
    }
}

/// Renders the recovery action buttons supporting Step-Back snapshot rollback inside a horizontal layout.
pub fn render_state_guard_action_buttons(
    ui: &mut egui::Ui,
    has_stepback: bool,
) -> Option<StateGuardAction> {
    let mut chosen_action = None;
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Recovery Actions:").italics());

        if has_stepback
            && ui
                .button("⏮ Step Back")
                .accessible_hover_text(
                    "Restore the most recent valid pre-divergence state snapshot and plot data",
                )
                .clicked()
        {
            emit_recovery_telemetry(
                "FiniteStateGuard",
                "Step Back",
                "Restored previous valid state snapshot",
            );
            chosen_action = Some(StateGuardAction::StepBack);
        }

        if ui
            .button("Auto-scale Step Size (dt)")
            .accessible_hover_text(
                "Halve step size (dt) and recalculate simulation trajectory from snapshot",
            )
            .clicked()
        {
            emit_recovery_telemetry(
                "FiniteStateGuard",
                "Auto-scale Step Size (dt)",
                "Halved step size",
            );
            chosen_action = Some(StateGuardAction::AutoScaleStepSize);
        }

        if ui
            .button("Clamp Instability Parameters")
            .accessible_hover_text(
                "Restrict rate parameters and initial conditions to pre-configured stable bounds",
            )
            .clicked()
        {
            emit_recovery_telemetry(
                "FiniteStateGuard",
                "Clamp Instability Parameters",
                "Clamped rate parameters and ICs to stable bounds",
            );
            chosen_action = Some(StateGuardAction::ClampParameters);
        }

        if ui
            .button("Reset All Parameters")
            .accessible_hover_text("Restore default baseline configuration")
            .clicked()
        {
            emit_recovery_telemetry(
                "FiniteStateGuard",
                "Reset All Parameters",
                "Restored default baseline configuration",
            );
            chosen_action = Some(StateGuardAction::ResetParameters);
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
    match render_state_guard_banner(ui, report, false) {
        Some(StateGuardAction::AutoScaleStepSize) => Some(RecoveryAction::AutoScaleStepSize),
        Some(StateGuardAction::ClampParameters) => Some(RecoveryAction::ClampParameters),
        Some(StateGuardAction::ResetParameters) => Some(RecoveryAction::ResetParameters),
        _ => None,
    }
}

/// Renders the interactive diagnostic recovery banner supporting Step-Back rollback.
pub fn render_state_guard_banner(
    ui: &mut egui::Ui,
    report: Option<&DiagnosticReport>,
    has_stepback: bool,
) -> Option<StateGuardAction> {
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
                chosen_action = render_state_guard_action_buttons(ui, has_stepback);
            });
        });

    chosen_action
}

/// Inspects plot datasets for any non-finite floating-point numbers (`NaN`, `+Inf`, `-Inf`).
/// Returns `Err(DiagnosticReport)` if any non-finite point coordinate is detected.
pub fn inspect_datasets_for_divergence(
    datasets: &[(String, Vec<[f64; 2]>)],
) -> Result<(), DiagnosticReport> {
    for (name, points) in datasets {
        for (idx, &[x, y]) in points.iter().enumerate() {
            if !x.is_finite() || !y.is_finite() {
                let (var_name, bad_val) = if !y.is_finite() {
                    (format!("{}.y", name), y)
                } else {
                    (format!("{}.x", name), x)
                };
                return Err(DiagnosticReport::new(
                    0.0,
                    idx,
                    var_name,
                    format!(
                        "Non-finite floating-point value ({}) encountered in plot dataset '{}' at point index {}.",
                        bad_val, name, idx
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Helper to halve step size / time step numerical parameters in a serialized tool state JSON.
/// Inspects JSON fields for standard time step keys ("dt", "time_step", "step_size", etc.)
/// and halves their values.
pub fn auto_scale_state_json(state_json: &str) -> Option<String> {
    let mut val: serde_json::Value = serde_json::from_str(state_json).ok()?;
    let mut modified = false;

    fn halve_step_in_object(obj: &mut serde_json::Map<String, serde_json::Value>) -> bool {
        let step_keys = [
            "dt",
            "time_step",
            "timestep",
            "step_size",
            "stepsize",
            "h",
            "delta_t",
            "viscosity",
            "num_levels",
        ];
        let mut changed = false;
        for key in step_keys {
            if let Some(v) = obj.get_mut(key) {
                if let Some(f) = v.as_f64() {
                    let halved = f * 0.5;
                    if let Some(num) = serde_json::Number::from_f64(halved) {
                        *v = serde_json::Value::Number(num);
                        changed = true;
                    }
                } else if let Some(i) = v.as_i64() {
                    let halved = (i / 2).max(1);
                    *v = serde_json::Value::Number(serde_json::Number::from(halved));
                    changed = true;
                }
            }
        }
        changed
    }

    if let serde_json::Value::Object(ref mut map) = val {
        modified = halve_step_in_object(map);
    }

    if modified {
        serde_json::to_string(&val).ok()
    } else {
        None
    }
}

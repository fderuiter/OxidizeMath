#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::presets::OdePreset;
use eframe::egui;

/// Available one-click recovery remedies for solver divergence.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RecoveryAction {
    /// Halve step size `dt` and re-integrate.
    AutoScaleStepSize,
    /// Clamp instability rate parameters and initial conditions to stable bounds.
    ClampParameters,
    /// Reset parameters to baseline configuration.
    ResetParameters,
}

/// Structured diagnostic report generated when an ODE simulation encounters non-finite states.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OdeDiagnosticReport {
    /// Time $t_{\text{diverged}}$ at which divergence occurred.
    pub t_diverged: f64,
    /// Step index at which the solver diverged.
    pub step_index: usize,
    /// Name of the variable that became non-finite (e.g., "y(t)" or "y'(t)").
    pub variable: String,
    /// Detailed diagnostic description of the failure.
    pub message: String,
}

impl OdeDiagnosticReport {
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
    t_div: f64,
    step_idx: usize,
    var_name: &str,
    preset: &str,
    dt: f64,
) {
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("t_diverged".to_string(), format!("{:.4}", t_div));
    metadata.insert("step_count".to_string(), step_idx.to_string());
    metadata.insert("diverged_variable".to_string(), var_name.to_string());
    metadata.insert("preset".to_string(), preset.to_string());
    metadata.insert("dt".to_string(), format!("{:.6}", dt));

    federated_registry::global_registry().emit(federated_registry::TelemetryEvent {
        source: "OdeSolverTool".to_string(),
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
pub fn emit_recovery_telemetry(action_name: &str, details: &str) {
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("action".to_string(), action_name.to_string());
    metadata.insert("details".to_string(), details.to_string());

    federated_registry::global_registry().emit(federated_registry::TelemetryEvent {
        source: "OdeSolverTool".to_string(),
        severity: federated_registry::Severity::Info,
        message: format!("Applied recovery action: {}", action_name),
        metadata,
        thread_name: None,
    });
}

/// Clamps solver parameters and initial conditions to stable, finite bounds.
pub fn clamp_ode_parameters(
    preset: OdePreset,
    param_k: &mut f64,
    param_r: &mut f64,
    param_cap_k: &mut f64,
    ic_y0: &mut f64,
    ic_v0: &mut f64,
) {
    if !ic_y0.is_finite() {
        *ic_y0 = 1.0;
    } else {
        *ic_y0 = ic_y0.clamp(-10.0, 10.0);
    }

    if !ic_v0.is_finite() {
        *ic_v0 = 0.0;
    } else {
        *ic_v0 = ic_v0.clamp(-10.0, 10.0);
    }

    match preset {
        OdePreset::Exponential => {
            if !param_k.is_finite() {
                *param_k = 1.0;
            } else {
                *param_k = param_k.clamp(-5.0, 5.0);
            }
        }
        OdePreset::HarmonicOscillator => {
            if !param_k.is_finite() {
                *param_k = 1.0;
            } else {
                *param_k = param_k.clamp(0.01, 10.0);
            }
        }
        OdePreset::LogisticGrowth => {
            if !param_r.is_finite() {
                *param_r = 1.0;
            } else {
                *param_r = param_r.clamp(0.01, 5.0);
            }
            if !param_cap_k.is_finite() {
                *param_cap_k = 10.0;
            } else {
                *param_cap_k = param_cap_k.clamp(0.1, 100.0);
            }
        }
        OdePreset::Custom => {
            if !param_k.is_finite() {
                *param_k = 1.0;
            } else {
                *param_k = param_k.clamp(-10.0, 10.0);
            }
        }
    }
}

fn render_recovery_action_buttons(ui: &mut egui::Ui) -> Option<RecoveryAction> {
    let mut chosen_action = None;
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Recovery Actions:").italics());

        if ui
            .button("Auto-scale Step Size (dt)")
            .on_hover_text("Halve step size (dt) and recalculate simulation trajectory")
            .clicked()
        {
            emit_recovery_telemetry("Auto-scale Step Size (dt)", "Halved step size");
            chosen_action = Some(RecoveryAction::AutoScaleStepSize);
        }

        if ui
            .button("Clamp Instability Parameters")
            .on_hover_text("Restrict rate parameters and initial conditions to pre-configured stable bounds")
            .clicked()
        {
            emit_recovery_telemetry("Clamp Instability Parameters", "Clamped rate parameters and ICs to stable bounds");
            chosen_action = Some(RecoveryAction::ClampParameters);
        }

        if ui
            .button("Reset All Parameters")
            .on_hover_text("Restore default baseline configuration")
            .clicked()
        {
            emit_recovery_telemetry("Reset All Parameters", "Restored default baseline configuration");
            chosen_action = Some(RecoveryAction::ResetParameters);
        }
    });
    chosen_action
}

/// Renders the interactive diagnostic banner offering one-click recovery remedies.
pub fn render_recovery_banner(
    ui: &mut egui::Ui,
    report: Option<&OdeDiagnosticReport>,
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

/// Renders the recovery banner and immediately applies recovery remedies to the tool.
pub fn render_recovery_ui(
    ui: &mut egui::Ui,
    tool: &mut super::OdeSolverTool,
    changed: &mut bool,
    plot_points_y: &mut Vec<[f64; 2]>,
    plot_points_v: &mut Vec<[f64; 2]>,
) {
    if let Some(action) = render_recovery_banner(ui, tool.diagnostic_report()) {
        match action {
            RecoveryAction::AutoScaleStepSize => tool.auto_scale_step_size(),
            RecoveryAction::ClampParameters => tool.clamp_instability_parameters(),
            RecoveryAction::ResetParameters => tool.reset_parameters(),
        }
        *changed = true;
        plot_points_y.clear();
        plot_points_v.clear();
        for (i, &t) in tool.time_series.iter().enumerate() {
            if let Some(&y) = tool.y_series.get(i) {
                plot_points_y.push([t, y]);
            }
            if let Some(&v) = tool.v_series.get(i) {
                plot_points_v.push([t, v]);
            }
        }
    }
}

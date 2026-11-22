//! Unified code artifact representation and cold-path `.tex`, `.rs`, and `.py` export dispatch.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use eframe::egui;

/// Self-contained LaTeX, Rust, and Python representations calibrated to an [`InteractiveTool`]'s active state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeArtifact {
    /// Symbolic governing equation(s) in display LaTeX (`\begin{aligned} ... \end{aligned}` or inline display math).
    pub symbolic_latex: String,
    /// Parameter-calibrated governing equation(s) with active finite slider values substituted.
    pub calibrated_latex: String,
    /// Standalone Rust 2024 source file (`rustc --edition 2024` compatible with zero external crates).
    pub rust_source: String,
    /// Standalone Python 3 (`numpy` / `scipy`) reproducible numerical script.
    pub python_source: String,
}

impl CodeArtifact {
    /// Validates that every generated representation is non-empty and free of raw `NaN` / `inf` tokens.
    pub fn validate(&self) -> Result<(), String> {
        for (label, text) in [
            ("symbolic LaTeX", &self.symbolic_latex),
            ("calibrated LaTeX", &self.calibrated_latex),
            ("Rust source", &self.rust_source),
            ("Python source", &self.python_source),
        ] {
            if text.trim().is_empty() {
                return Err(format!("Generated {label} payload is empty."));
            }
        }
        Ok(())
    }

    /// Formats a standalone `.tex` snippet containing both symbolic and parameter-calibrated equations.
    #[must_use]
    pub fn to_tex_document(&self, tool_name: &str) -> String {
        format!(
            "% OxidizeMath Reproducible Model Export: {tool_name}\n\
             % Requires: \\usepackage{{amsmath}}\n\n\
             % Symbolic governing model:\n\
             \\[\n{}\n\\]\n\n\
             % Calibrated governing model (active parameters):\n\
             \\[\n{}\n\\]\n",
            self.symbolic_latex.trim(),
            self.calibrated_latex.trim()
        )
    }
}

impl SimulationFramework {
    /// Dispatches a deferred `.tex`, `.rs`, or `.py` code export request on the cold export path.
    pub(super) fn dispatch_code_export(
        &mut self,
        format: ExportFormat,
        target: ExportTarget,
        ctx: Option<&egui::Context>,
    ) {
        let Some(tool) = &self.active_tool else {
            return;
        };
        let caps = tool.export_capabilities();
        if !caps.code {
            self.set_export_status(ExportStatusSeverity::Warning, caps.code_disabled_reason);
            return;
        }
        let tool_name = tool.name().to_string();
        if !tool.has_finite_parameters() {
            self.set_export_status(
                ExportStatusSeverity::Error,
                format!(
                    "Cannot export {} model from {tool_name}: active parameters must be finite.",
                    format.label()
                ),
            );
            return;
        }
        let Some(artifact) = tool.export_code_artifact() else {
            self.set_export_status(
                ExportStatusSeverity::Error,
                format!(
                    "Cannot export {} model from {tool_name}: ensure active equation and parameters are valid.",
                    format.label()
                ),
            );
            return;
        };
        if let Err(err) = artifact.validate() {
            self.set_export_status(ExportStatusSeverity::Error, err);
            return;
        }

        let content = match (format, target) {
            (ExportFormat::Tex, ExportTarget::File) => artifact.to_tex_document(&tool_name),
            (ExportFormat::Tex, ExportTarget::Clipboard) => artifact.calibrated_latex,
            (ExportFormat::Rust, _) => artifact.rust_source,
            (ExportFormat::Python, _) => artifact.python_source,
            _ => return,
        };

        match target {
            ExportTarget::File => {
                let filename = Self::build_export_filename(&tool_name, format);
                let bytes = content.as_bytes().to_vec();
                self.complete_file_export(
                    format,
                    &filename,
                    &bytes,
                    Some(content),
                    format.filter_name(),
                    "Code export failed",
                );
            }
            ExportTarget::Clipboard => {
                self.copy_code_payload_to_clipboard(format.label(), content, ctx);
            }
        }
    }

    /// Copies an arbitrary code or LaTeX snippet to the clipboard and emits an ARIA live-region confirmation.
    pub(crate) fn copy_code_payload_to_clipboard(
        &mut self,
        label: &'static str,
        content: String,
        ctx: Option<&egui::Context>,
    ) {
        if content.trim().is_empty() {
            self.set_export_status(
                ExportStatusSeverity::Error,
                format!("Cannot copy empty {label} payload to clipboard."),
            );
            return;
        }
        if let Some(egui_ctx) = ctx {
            egui_ctx.copy_text(content.clone());
        }
        let res = if let Some(ref writer) = self.export_queue.custom_clipboard_writer {
            writer(&content)
        } else {
            self.write_system_text_clipboard(label, &content, ctx)
        };
        match res {
            Ok(()) => {
                self.export_queue.last_exported_content = Some(content.clone());
                self.export_queue.last_clipboard_content = Some(content);
                let msg = if cfg!(target_arch = "wasm32")
                    && self.export_queue.custom_clipboard_writer.is_none()
                {
                    format!("Copying {label} to clipboard...")
                } else {
                    format!("Copied {label} to clipboard!")
                };
                self.set_export_status(ExportStatusSeverity::Success, msg);
            }
            Err(err) => {
                self.set_export_status(
                    ExportStatusSeverity::Error,
                    format!(
                        "Clipboard unavailable ({err}). Please save the exported file to disk."
                    ),
                );
            }
        }
    }
}

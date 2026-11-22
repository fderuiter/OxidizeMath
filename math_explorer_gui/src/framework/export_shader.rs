//! Deferred standalone `.wgsl` and `.glsl` fragment shader export dispatch for [`super::SimulationFramework`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;

use super::{
    ExportFormat, ExportStatusSeverity, ExportTarget, ShaderExportFormat, SimulationFramework,
};

impl SimulationFramework {
    /// Queues a direct `.wgsl` or `.glsl` shader source copy to the system clipboard.
    pub(crate) fn queue_shader_clipboard_export(&mut self, format: ExportFormat) -> bool {
        self.export_queue
            .push_target(format, ExportTarget::Clipboard)
    }

    pub(super) fn dispatch_shader_export(
        &mut self,
        format: ExportFormat,
        target: ExportTarget,
        ctx: Option<&egui::Context>,
    ) {
        let shader_format = match format {
            ExportFormat::Wgsl => ShaderExportFormat::Wgsl,
            ExportFormat::Glsl => ShaderExportFormat::Glsl,
            _ => return,
        };
        let Some(tool) = &self.active_tool else {
            return;
        };
        let caps = tool.export_capabilities();
        if !caps.shader {
            self.set_export_status(ExportStatusSeverity::Warning, caps.shader_disabled_reason);
            return;
        }
        let tool_name = tool.name().to_string();
        let source = match tool.export_shader_source(shader_format) {
            Ok(code) if !code.trim().is_empty() => code,
            Ok(_) => {
                self.set_export_status(
                    ExportStatusSeverity::Error,
                    format!("No shader source available to export from {tool_name}."),
                );
                return;
            }
            Err(err) => {
                self.set_export_status(
                    ExportStatusSeverity::Error,
                    format!("Shader export failed: {err}"),
                );
                return;
            }
        };

        match target {
            ExportTarget::File => {
                let filename = Self::build_export_filename(&tool_name, format);
                let bytes = source.as_bytes().to_vec();
                self.complete_file_export(
                    format,
                    &filename,
                    &bytes,
                    Some(source),
                    format.filter_name(),
                    "Shader export failed",
                );
            }
            ExportTarget::Clipboard => {
                self.dispatch_clipboard_text_with_noun(format, "shader", source, ctx);
            }
        }
    }
}

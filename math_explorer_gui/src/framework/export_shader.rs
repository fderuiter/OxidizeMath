//! Deferred WGSL/GLSL/WebGL2/Shadertoy fragment shader export dispatch, clipboard text helpers, and action-bar UI controls.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::export_ui::export_action_button;
use super::*;
use eframe::egui;

const SHADER_EXPORT_ACTIONS: [(&str, &str, ExportFormat, ExportTarget); 8] = [
    (
        "Export WGSL",
        "Export standalone WebGPU WGSL fragment shader (.wgsl)",
        ExportFormat::Wgsl,
        ExportTarget::File,
    ),
    (
        "Copy WGSL",
        "Copy standalone WGSL fragment shader to clipboard",
        ExportFormat::Wgsl,
        ExportTarget::Clipboard,
    ),
    (
        "Export GLSL",
        "Export standalone GLSL ES 3.00 and Shadertoy fragment shader (.glsl)",
        ExportFormat::Glsl,
        ExportTarget::File,
    ),
    (
        "Copy GLSL",
        "Copy standalone GLSL fragment shader to clipboard",
        ExportFormat::Glsl,
        ExportTarget::Clipboard,
    ),
    (
        "Export WebGL2 (.frag)",
        "Download standalone WebGL2 (#version 300 es) fragment shader (.frag)",
        ExportFormat::GlslWebGl2,
        ExportTarget::File,
    ),
    (
        "Copy WebGL2",
        "Copy standalone WebGL2 (#version 300 es) fragment shader to clipboard",
        ExportFormat::GlslWebGl2,
        ExportTarget::Clipboard,
    ),
    (
        "Export Shadertoy (.frag)",
        "Download standalone Shadertoy (mainImage) fragment shader (.frag)",
        ExportFormat::GlslShadertoy,
        ExportTarget::File,
    ),
    (
        "Copy Shadertoy",
        "Copy standalone Shadertoy (mainImage) fragment shader to clipboard",
        ExportFormat::GlslShadertoy,
        ExportTarget::Clipboard,
    ),
];

/// Formats a clipboard failure error with guidance toward disk export.
fn format_clipboard_error(err: &str) -> String {
    if err.contains("Export CSV") {
        format!("Clipboard unavailable: {err}")
    } else {
        format!("Clipboard unavailable ({err}). Please use Export CSV to save to disk.")
    }
}

impl SimulationFramework {
    /// Queues a direct fragment shader clipboard copy for deferred end-of-frame dispatch.
    pub(crate) fn queue_shader_clipboard_export(&mut self, format: ExportFormat) -> bool {
        self.export_queue
            .push_target(format, ExportTarget::Clipboard)
    }

    /// Renders WGSL, GLSL, WebGL2 (`.frag`), and Shadertoy (`.frag`) download and clipboard actions
    /// when the active tool advertises `caps.shader`.
    pub(super) fn render_shader_export_items(
        &mut self,
        ui: &mut egui::Ui,
        caps: ExportCapabilities,
        close_on_click: bool,
    ) {
        if !caps.shader {
            return;
        }
        for (label, tip, format, target) in SHADER_EXPORT_ACTIONS {
            if export_action_button(ui, true, label, tip, close_on_click) {
                match target {
                    ExportTarget::File => self.queue_export(format),
                    ExportTarget::Clipboard => {
                        self.queue_shader_clipboard_export(format);
                    }
                }
            }
        }
    }

    /// Dispatches a queued WGSL, GLSL, WebGL2, or Shadertoy shader export request.
    pub(super) fn dispatch_shader_export(
        &mut self,
        format: ExportFormat,
        target: ExportTarget,
        ctx: Option<&egui::Context>,
    ) {
        let Some(tool) = &self.active_tool else {
            return;
        };
        let caps = tool.export_capabilities();
        if !caps.shader {
            self.set_export_status(ExportStatusSeverity::Warning, caps.shader_disabled_reason);
            return;
        }
        let tool_name = tool.name().to_string();
        let shader_res = match format {
            ExportFormat::Wgsl => tool.export_shader_source(ShaderExportFormat::Wgsl),
            ExportFormat::Glsl => tool.export_shader_source(ShaderExportFormat::Glsl),
            ExportFormat::GlslWebGl2 => tool.export_glsl_shader(ShaderDialect::WebGl2Es300),
            ExportFormat::GlslShadertoy => tool.export_glsl_shader(ShaderDialect::Shadertoy),
            _ => return,
        };
        let shader_src = match shader_res {
            Ok(src) if !src.trim().is_empty() => src,
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
                let bytes = shader_src.as_bytes().to_vec();
                self.complete_file_export(
                    format,
                    &filename,
                    &bytes,
                    Some(shader_src),
                    format.filter_name(),
                    "Shader export failed",
                );
            }
            ExportTarget::Clipboard => {
                self.dispatch_clipboard_text_request(format, shader_src, ctx);
            }
        }
    }

    pub(super) fn dispatch_clipboard_text_request(
        &mut self,
        format: ExportFormat,
        content: String,
        ctx: Option<&egui::Context>,
    ) {
        let label = format.label();
        let noun = format.clipboard_item_noun();
        let res = if let Some(ref writer) = self.export_queue.custom_clipboard_writer {
            writer(&content)
        } else {
            self.write_system_text_clipboard(label, noun, &content, ctx)
        };
        match res {
            Ok(()) => {
                self.export_queue.last_exported_content = Some(content.clone());
                self.export_queue.last_clipboard_content = Some(content);
                let msg = if cfg!(target_arch = "wasm32")
                    && self.export_queue.custom_clipboard_writer.is_none()
                {
                    format!("Copying {label} {noun} to clipboard...")
                } else {
                    format!("Copied {label} {noun} to clipboard!")
                };
                self.set_export_status(ExportStatusSeverity::Success, msg);
            }
            Err(err) => {
                self.set_export_status(ExportStatusSeverity::Error, format_clipboard_error(&err));
            }
        }
    }

    fn write_system_text_clipboard(
        &self,
        label: &'static str,
        noun: &'static str,
        content: &str,
        ctx: Option<&egui::Context>,
    ) -> Result<(), String> {
        #[cfg(test)]
        {
            let _ = (label, noun, content, ctx);
            Ok(())
        }
        #[cfg(not(test))]
        {
            if let Some(egui_ctx) = ctx.cloned() {
                let status_id =
                    egui::Id::new(("framework_action_bar_clipboard_status", self.instance_id));
                egui_plot::export::copy_text_to_clipboard_with_callback(content, move |cb| {
                    egui_ctx.data_mut(|d| {
                        let status = match cb {
                            Ok(()) => ActionBarExportStatus {
                                message: format!("Copied {label} {noun} to clipboard!"),
                                is_error: false,
                                expires_at_time: None,
                            },
                            Err(err) => ActionBarExportStatus {
                                message: format_clipboard_error(&err),
                                is_error: true,
                                expires_at_time: None,
                            },
                        };
                        d.insert_temp(status_id, status);
                    });
                    egui_ctx.request_repaint();
                })
            } else {
                egui_plot::export::copy_text_to_clipboard(content)
            }
        }
    }
}

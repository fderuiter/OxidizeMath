//! Deferred `.frag` download, clipboard copy, and action-bar UI controls for GLSL fragment shaders.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::export_ui::export_action_button;
use super::*;
use eframe::egui;

impl SimulationFramework {
    /// Queues a direct GLSL fragment shader clipboard copy for deferred end-of-frame dispatch.
    pub(crate) fn queue_shader_clipboard_export(&mut self, format: ExportFormat) {
        self.export_queue
            .push_target(format, ExportTarget::Clipboard);
    }

    /// Renders one-click WebGL2 (`#version 300 es`) and Shadertoy (`mainImage`) `.frag` and clipboard
    /// actions when the active tool advertises `caps.shader`.
    pub(super) fn render_shader_export_items(
        &mut self,
        ui: &mut egui::Ui,
        caps: ExportCapabilities,
        close_on_click: bool,
    ) {
        if !caps.shader {
            return;
        }
        if export_action_button(
            ui,
            true,
            "Export WebGL2 (.frag)",
            "Download standalone WebGL2 (#version 300 es) fragment shader (.frag)",
            close_on_click,
        ) {
            self.queue_export(ExportFormat::GlslWebGl2);
        }
        if export_action_button(
            ui,
            true,
            "Copy WebGL2",
            "Copy standalone WebGL2 (#version 300 es) fragment shader to clipboard",
            close_on_click,
        ) {
            self.queue_shader_clipboard_export(ExportFormat::GlslWebGl2);
        }
        if export_action_button(
            ui,
            true,
            "Export Shadertoy (.frag)",
            "Download standalone Shadertoy (mainImage) fragment shader (.frag)",
            close_on_click,
        ) {
            self.queue_export(ExportFormat::GlslShadertoy);
        }
        if export_action_button(
            ui,
            true,
            "Copy Shadertoy",
            "Copy standalone Shadertoy (mainImage) fragment shader to clipboard",
            close_on_click,
        ) {
            self.queue_shader_clipboard_export(ExportFormat::GlslShadertoy);
        }
    }

    /// Dispatches a queued [`ExportFormat::GlslWebGl2`] or [`ExportFormat::GlslShadertoy`] request.
    pub(super) fn dispatch_shader_export(
        &mut self,
        format: ExportFormat,
        target: ExportTarget,
        ctx: Option<&egui::Context>,
    ) {
        let dialect = match format {
            ExportFormat::GlslWebGl2 => ShaderDialect::WebGl2Es300,
            ExportFormat::GlslShadertoy => ShaderDialect::Shadertoy,
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
        let shader_src = match tool.export_glsl_shader(dialect) {
            Ok(src) if !src.trim().is_empty() => src,
            Ok(_) => {
                self.set_export_status(
                    ExportStatusSeverity::Error,
                    format!("No GLSL shader source available to export from {tool_name}."),
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
                            Ok(()) => super::export_queue::ActionBarExportStatus {
                                message: format!("Copied {label} {noun} to clipboard!"),
                                is_error: false,
                                expires_at_time: None,
                            },
                            Err(err) => super::export_queue::ActionBarExportStatus {
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

fn format_clipboard_error(err: &str) -> String {
    if err.contains("Export CSV") {
        format!("Clipboard unavailable: {err}")
    } else {
        format!("Clipboard unavailable ({err}). Please use Export CSV to save to disk.")
    }
}

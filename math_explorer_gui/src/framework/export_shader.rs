//! Deferred `.frag` download, clipboard copy, and action-bar UI controls for GLSL fragment shaders.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::export_ui::export_action_button;
use super::*;
use eframe::egui;
/// Compiles a complex [`MathExpressionInputBar`] into GLSL `const vec2` parameter declarations
/// and a `vec2` expression restricted to `base_vars` plus synthesized free parameters.
#[cfg(any(feature = "physics", feature = "pure_math"))]
pub(crate) fn compile_complex_glsl_bar(
    bar: &MathExpressionInputBar,
    base_vars: &[&str],
    reserved: &[&str],
) -> Result<(String, String), String> {
    use math_commons::ast::format_glsl_float;
    use std::fmt::Write;

    let invalid_err = || {
        bar.error_message()
            .unwrap_or_else(|| "Custom complex expression is invalid".to_string())
    };
    let expr = bar.expr().ok_or_else(invalid_err)?;
    let compiled = bar.compiled().ok_or_else(invalid_err)?;
    let param_names = compiled.param_names();
    let param_vals = bar.param_slice();

    let mut allowed_vars = Vec::with_capacity(base_vars.len() + param_names.len());
    allowed_vars.extend_from_slice(base_vars);
    let mut param_decls = String::new();
    for (idx, name) in param_names.iter().enumerate() {
        if reserved.contains(&name.as_str()) {
            return Err(format!(
                "Parameter '{name}' collides with a reserved GLSL shader symbol"
            ));
        }
        let val = param_vals.get(idx).copied().unwrap_or(1.0);
        let lit = format_glsl_float(val).map_err(|e| e.to_string())?;
        writeln!(param_decls, "    const vec2 {name} = vec2({lit}, 0.0);").unwrap();
        allowed_vars.push(name.as_str());
    }
    let expr_glsl = expr
        .to_glsl_complex_with_vars(&allowed_vars)
        .map_err(|e| e.to_string())?;
    Ok((param_decls, expr_glsl))
}

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

    pub(super) fn complete_file_export(
        &mut self,
        format: ExportFormat,
        filename: &str,
        bytes: &[u8],
        text_content: Option<String>,
        ok_label: &str,
        err_prefix: &str,
    ) {
        match self.persist_export_bytes(format, filename, bytes, text_content) {
            Ok(()) => self.set_export_status(
                ExportStatusSeverity::Success,
                format!("{ok_label} saved successfully."),
            ),
            Err(err) => {
                self.set_export_status(ExportStatusSeverity::Error, format!("{err_prefix}: {err}"))
            }
        }
    }

    pub(super) fn build_export_filename(tool_name: &str, format: ExportFormat) -> String {
        let slug: String = tool_name
            .to_lowercase()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let trimmed = slug.trim_matches('_');
        let base = if trimmed.is_empty() {
            "simulation"
        } else {
            trimmed
        };
        format!("{base}_{}.{}", format.stem_suffix(), format.extension())
    }

    fn persist_export_bytes(
        &mut self,
        format: ExportFormat,
        filename: &str,
        bytes: &[u8],
        text_content: Option<String>,
    ) -> Result<(), String> {
        if bytes.is_empty() {
            return Err("Refusing to write empty export payload".to_string());
        }
        self.export_queue.last_exported_filename = Some(filename.to_string());
        self.export_queue.last_exported_content = text_content;
        self.export_queue.last_exported_bytes = Some(bytes.to_vec());

        if let Some(ref dir) = self.export_queue.destination_dir {
            let path = dir.join(filename);
            std::fs::write(&path, bytes)
                .map_err(|e| format!("Failed to write {}: {e}", path.display()))?;
        } else if let Some(ref writer) = self.export_queue.custom_writer {
            writer(filename, bytes);
        } else {
            #[cfg(not(test))]
            egui_plot::export::save_file_dialog(
                filename,
                bytes,
                format.filter_name(),
                &[format.extension()],
            );
            #[cfg(test)]
            let _ = format;
        }
        Ok(())
    }
}

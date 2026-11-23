use super::*;

impl SimulationFramework {
    pub(in crate::framework) fn dispatch_clipboard_text_request(
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

    pub(in crate::framework) fn complete_file_export(
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

    pub(in crate::framework) fn build_export_filename(
        tool_name: &str,
        format: ExportFormat,
    ) -> String {
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

//! Cold-path file persistence and filename helpers for [`super::SimulationFramework`] exports.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{ExportFormat, ExportStatusSeverity, InteractiveTool, SimulationFramework};

pub(super) fn export_tool_preset_json<T: InteractiveTool + ?Sized>(
    tool: &T,
) -> Result<String, String> {
    if let Some(state_str) = tool.save_state() {
        let config: serde_json::Value =
            serde_json::from_str(&state_str).map_err(|e| format!("Serialization error: {e}"))?;
        crate::presets::serialize_preset_envelope(tool.name(), tool.domain(), config)
    } else {
        Err(format!("Preset export not supported for {}", tool.name()))
    }
}

pub(super) fn import_tool_preset_json<T: InteractiveTool + ?Sized>(
    tool: &mut T,
    json_str: &str,
) -> Result<(), String> {
    let bounds = tool.state_bounds();
    let config: Box<serde_json::value::RawValue> =
        crate::presets::deserialize_preset_envelope_bounded(
            json_str,
            tool.domain(),
            tool.name(),
            bounds,
        )?;
    let config_str = config.get();
    bounds.validate_json(config_str)?;
    tool.try_load_state(config_str)
}

impl SimulationFramework {
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

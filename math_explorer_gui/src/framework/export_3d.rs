//! Deferred end-of-frame 3D mesh and Gaussian splat export dispatch and action-bar buttons.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::export_ui::export_action_button;
use super::*;
use eframe::egui;
use oxidize_core::mesh::{
    export_colored_mesh_to_ply_binary, export_mesh_to_obj_string, export_mesh_to_ply_binary,
    export_mesh_to_stl_binary, ColoredMesh, Mesh,
};

fn resolve_solid_mesh(tool: &dyn InteractiveTool) -> Option<Result<Mesh, String>> {
    if let Some(mesh_res) = tool.export_mesh_3d() {
        return Some(mesh_res);
    }
    tool.export_colored_mesh_3d()
        .map(|res: Result<ColoredMesh, String>| res.map(|cm| cm.mesh))
}

impl SimulationFramework {
    /// Renders `Export STL`, `Export PLY`, and `Export OBJ` toolbar/menu buttons for 3D-capable tools.
    pub(super) fn render_3d_export_action_items(
        &mut self,
        ui: &mut egui::Ui,
        caps: ExportCapabilities,
        close_on_click: bool,
    ) {
        if !caps.has_3d_export() {
            return;
        }
        let stl_tip = if caps.stl {
            "Export watertight binary STL 3D mesh for 3D printing"
        } else {
            caps.stl_disabled_reason
        };
        if export_action_button(ui, caps.stl, "Export STL", stl_tip, close_on_click) {
            self.queue_export(ExportFormat::Stl);
        }

        let ply_tip = if caps.ply {
            "Export binary PLY 3D model (colored mesh or 3DGS splat scene)"
        } else {
            caps.ply_disabled_reason
        };
        if export_action_button(ui, caps.ply, "Export PLY", ply_tip, close_on_click) {
            self.queue_export(ExportFormat::Ply);
        }

        let obj_tip = if caps.obj {
            "Export Wavefront OBJ 3D triangle mesh"
        } else {
            caps.obj_disabled_reason
        };
        if export_action_button(ui, caps.obj, "Export OBJ", obj_tip, close_on_click) {
            self.queue_export(ExportFormat::Obj);
        }
    }

    /// Dispatches a queued `Stl`, `Ply`, or `Obj` 3D export request at end-of-frame.
    pub(super) fn dispatch_3d_export(
        &mut self,
        format: ExportFormat,
        target: ExportTarget,
        _ctx: Option<&egui::Context>,
    ) {
        if target == ExportTarget::Clipboard {
            self.set_export_status(
                ExportStatusSeverity::Warning,
                "Clipboard export is unavailable for 3D models; save STL, PLY, or OBJ to disk.",
            );
            return;
        }
        let Some(tool) = &self.active_tool else {
            return;
        };
        let caps = tool.export_capabilities();
        let tool_name = tool.name().to_string();
        match format {
            ExportFormat::Stl => self.dispatch_stl_file(&tool_name, caps),
            ExportFormat::Ply => self.dispatch_ply_file(&tool_name, caps),
            ExportFormat::Obj => self.dispatch_obj_file(&tool_name, caps),
            _ => {}
        }
    }

    fn dispatch_stl_file(&mut self, tool_name: &str, caps: ExportCapabilities) {
        if !caps.stl {
            self.set_export_status(ExportStatusSeverity::Warning, caps.stl_disabled_reason);
            return;
        }
        let Some(mesh_res) = self.active_tool.as_deref().and_then(resolve_solid_mesh) else {
            self.set_export_status(
                ExportStatusSeverity::Error,
                format!("No 3D mesh available to export as STL from {tool_name}."),
            );
            return;
        };
        match mesh_res.and_then(|mesh| export_mesh_to_stl_binary(&mesh).map_err(|e| e.to_string()))
        {
            Ok(bytes) => {
                let filename = Self::build_export_filename(tool_name, ExportFormat::Stl);
                self.complete_file_export(
                    ExportFormat::Stl,
                    &filename,
                    &bytes,
                    None,
                    ExportFormat::Stl.filter_name(),
                    "3D STL export failed",
                );
            }
            Err(err) => self.set_export_status(
                ExportStatusSeverity::Error,
                format!("3D STL export failed: {err}"),
            ),
        }
    }

    fn dispatch_ply_file(&mut self, tool_name: &str, caps: ExportCapabilities) {
        if !caps.ply {
            self.set_export_status(ExportStatusSeverity::Warning, caps.ply_disabled_reason);
            return;
        }
        let Some(bytes_res) = self.serialize_active_tool_ply() else {
            self.set_export_status(
                ExportStatusSeverity::Error,
                format!("No 3D PLY geometry available to export from {tool_name}."),
            );
            return;
        };
        match bytes_res {
            Ok(bytes) => {
                let filename = Self::build_export_filename(tool_name, ExportFormat::Ply);
                self.complete_file_export(
                    ExportFormat::Ply,
                    &filename,
                    &bytes,
                    None,
                    ExportFormat::Ply.filter_name(),
                    "3D PLY export failed",
                );
            }
            Err(err) => self.set_export_status(
                ExportStatusSeverity::Error,
                format!("3D PLY export failed: {err}"),
            ),
        }
    }

    fn serialize_active_tool_ply(&self) -> Option<Result<Vec<u8>, String>> {
        let tool = self.active_tool.as_deref()?;
        if let Some(splat_res) = tool.export_splat_ply_3d() {
            return Some(splat_res);
        }
        if let Some(colored_res) = tool.export_colored_mesh_3d() {
            return Some(
                colored_res.and_then(|cm| {
                    export_colored_mesh_to_ply_binary(&cm).map_err(|e| e.to_string())
                }),
            );
        }
        tool.export_mesh_3d()
            .map(|res| res.and_then(|m| export_mesh_to_ply_binary(&m).map_err(|e| e.to_string())))
    }

    fn dispatch_obj_file(&mut self, tool_name: &str, caps: ExportCapabilities) {
        if !caps.obj {
            self.set_export_status(ExportStatusSeverity::Warning, caps.obj_disabled_reason);
            return;
        }
        let Some(mesh_res) = self.active_tool.as_deref().and_then(resolve_solid_mesh) else {
            self.set_export_status(
                ExportStatusSeverity::Error,
                format!("No 3D mesh available to export as OBJ from {tool_name}."),
            );
            return;
        };
        match mesh_res.and_then(|mesh| export_mesh_to_obj_string(&mesh).map_err(|e| e.to_string()))
        {
            Ok(obj_text) => {
                let filename = Self::build_export_filename(tool_name, ExportFormat::Obj);
                let bytes = obj_text.as_bytes().to_vec();
                self.complete_file_export(
                    ExportFormat::Obj,
                    &filename,
                    &bytes,
                    Some(obj_text),
                    ExportFormat::Obj.filter_name(),
                    "3D OBJ export failed",
                );
            }
            Err(err) => self.set_export_status(
                ExportStatusSeverity::Error,
                format!("3D OBJ export failed: {err}"),
            ),
        }
    }
}

//! Framework-hosted progressive 2D parameter-space atlas overlay for [`InteractiveTool`]s.
//!
//! Any [`InteractiveTool`] that overrides [`InteractiveTool::atlas_axis_pairs`] and
//! [`InteractiveTool::build_atlas_cartographer`] automatically receives a toolbar toggle and a
//! progressive, non-blocking 2D regime explorer inside [`SimulationFramework`] without needing
//! bespoke UI wiring or extra fields in `#[serde(deny_unknown_fields)]` state structs.

use super::{AccessibleHoverText, InteractiveTool, SimulationFramework, scoped_panel_id};
use crate::async_sim::atlas::ParameterAtlasController;
use eframe::egui;

/// Lazy framework-owned controller state for the active tool's 2D parameter atlas.
#[derive(Default)]
pub struct FrameworkAtlasHost {
    owner_tool_name: Option<&'static str>,
    visible: bool,
    controller: Option<ParameterAtlasController>,
}

impl FrameworkAtlasHost {
    /// Resets the hosted controller when switching to a different tool.
    pub fn reset_for_tool_switch(&mut self) {
        self.owner_tool_name = None;
        self.controller = None;
    }

    /// Returns whether the framework-hosted 2D parameter atlas panel is currently open.
    #[must_use]
    pub fn is_visible(&self) -> bool {
        self.visible
    }

    /// Sets visibility of the framework-hosted 2D parameter atlas panel.
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
        if let Some(ctrl) = &mut self.controller {
            ctrl.set_visible(visible);
        }
    }

    /// Returns a reference to the underlying [`ParameterAtlasController`], if initialized.
    #[must_use]
    pub fn controller(&self) -> Option<&ParameterAtlasController> {
        self.controller.as_ref()
    }

    /// Returns a mutable reference to the underlying [`ParameterAtlasController`], initializing it
    /// for `tool` when `tool.atlas_axis_pairs()` is non-empty.
    pub fn ensure_controller_for(
        &mut self,
        tool: &dyn InteractiveTool,
    ) -> Option<&mut ParameterAtlasController> {
        let pairs = tool.atlas_axis_pairs();
        if pairs.is_empty() {
            return None;
        }
        let name = tool.name();
        if self.owner_tool_name != Some(name) || self.controller.is_none() {
            let mut ctrl = ParameterAtlasController::new(pairs.to_vec());
            ctrl.set_visible(self.visible);
            self.owner_tool_name = Some(name);
            self.controller = Some(ctrl);
        } else if let Some(ctrl) = &mut self.controller {
            ctrl.replace_axis_pair_options(pairs.to_vec());
            ctrl.set_visible(self.visible);
        }
        self.controller.as_mut()
    }
}

impl SimulationFramework {
    /// Returns whether the active tool exposes a framework-hosted 2D parameter atlas.
    #[must_use]
    pub fn active_tool_supports_atlas(&self) -> bool {
        self.active_tool
            .as_ref()
            .is_some_and(|t| !t.atlas_axis_pairs().is_empty())
    }

    /// Opens or closes the framework-hosted 2D parameter atlas for the active tool.
    pub fn set_parameter_atlas_visible(&mut self, visible: bool) {
        self.atlas_host.set_visible(visible);
        if let Some(tool) = self.active_tool.as_deref() {
            let _ = self.atlas_host.ensure_controller_for(tool);
        }
    }

    pub(super) fn render_atlas_bar_toggle(&mut self, ui: &mut egui::Ui) {
        let Some(tool) = self.active_tool.as_deref() else {
            return;
        };
        if tool.atlas_axis_pairs().is_empty() {
            return;
        }
        let is_open = self.atlas_host.is_visible();
        let label = if is_open {
            "🗺 Hide 2D Atlas"
        } else {
            "🗺 2D Atlas"
        };
        let btn = ui.selectable_label(is_open, label).accessible_hover_text(
            "Toggle progressive 2D parameter-space regime explorer and click-to-teleport map",
        );
        if btn.clicked() {
            let next = !is_open;
            self.atlas_host.set_visible(next);
            let _ = self.atlas_host.ensure_controller_for(tool);
        }
    }

    pub(super) fn show_framework_atlas_panel(&mut self, ui: &mut egui::Ui, id_source: &str) {
        if !self.active_tool_supports_atlas() || !self.atlas_host.is_visible() {
            return;
        }
        let panel_id = scoped_panel_id(ui, format!("{id_source}_parameter_atlas"));
        let mut teleported = false;
        egui::TopBottomPanel::bottom(panel_id)
            .resizable(true)
            .default_height(285.0)
            .min_height(190.0)
            .max_height(430.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt(("framework_atlas_scroll", id_source, self.instance_id))
                    .show(ui, |ui| {
                        teleported = self.render_framework_atlas_contents(ui, id_source);
                    });
            });
        if teleported {
            self.bump_framework_revision();
        }
    }

    fn render_framework_atlas_contents(&mut self, ui: &mut egui::Ui, id_source: &str) -> bool {
        let Some(tool) = self.active_tool.as_deref_mut() else {
            return false;
        };
        let Some(ctrl) = self.atlas_host.ensure_controller_for(tool) else {
            return false;
        };
        let axis_keys = ctrl.axis_keys();
        let (ax, ay) = tool.active_atlas_coords(axis_keys);
        let sig = tool.atlas_context_signature(axis_keys);
        let salt = format!("fw_atlas_{id_source}_{}", self.instance_id);
        let ev = ctrl.show_panel(ui, &salt, ax, ay, sig, |keys| {
            tool.build_atlas_cartographer(keys)
                .expect("tool with non-empty atlas_axis_pairs must build a cartographer")
        });
        if !ctrl.is_visible() {
            self.atlas_host.set_visible(false);
        }
        let Some(teleport) = ev else {
            return false;
        };
        tool.apply_atlas_teleport(teleport);
        ui.ctx().request_repaint();
        true
    }
}

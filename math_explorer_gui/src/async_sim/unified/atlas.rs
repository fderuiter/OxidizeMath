#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{UnifiedModel, UnifiedSimTool};
use crate::async_sim::atlas::{AtlasTeleportEvent, ParameterAtlasController};
use crate::async_sim::SimCommand;
use eframe::egui;
use std::hash::{Hash, Hasher};

pub(super) type ParsedUnifiedAtlasCfg = Option<(bool, Option<[String; 2]>)>;

impl<M: UnifiedModel> UnifiedSimTool<M> {
    pub(super) fn create_optional_atlas() -> Option<ParameterAtlasController> {
        let pairs = M::atlas_axis_pairs();
        (!pairs.is_empty()).then(|| ParameterAtlasController::new(pairs.to_vec()))
    }

    pub(super) fn apply_parsed_atlas_state(
        &mut self,
        atlas_cfg: ParsedUnifiedAtlasCfg,
    ) -> Result<(), String> {
        let Some(atlas) = &mut self.atlas else {
            return Ok(());
        };
        match atlas_cfg {
            Some((vis, maybe_axes)) => {
                if let Some([xk, yk]) = maybe_axes {
                    atlas.set_axis_keys(&xk, &yk)?;
                } else {
                    atlas.select_axis_pair_index(0);
                }
                atlas.set_visible(vis);
            }
            None => {
                atlas.select_axis_pair_index(0);
                atlas.set_visible(false);
            }
        }
        Ok(())
    }

    /// Applies an [`AtlasTeleportEvent`], resetting and starting the live model trajectory.
    pub fn apply_atlas_teleport(&mut self, ev: AtlasTeleportEvent, ctx: Option<&egui::Context>) {
        {
            let mut guard = self.params.write().unwrap_or_else(|e| e.into_inner());
            for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
                if let Some((_, c)) = self.param_metadata.iter().find(|(k, _)| k == key) {
                    guard.insert(key.to_string(), val.clamp(c.min, c.max));
                }
            }
        }
        self.reset_simulation(ctx);
        let _ = self.controller.send_command(SimCommand::Start);
    }

    pub(super) fn draw_atlas_toggle_if_supported(&mut self, ui: &mut egui::Ui) {
        if let Some(atlas) = &mut self.atlas {
            ui.separator();
            let _ = atlas.show_toggle_button(ui);
        }
    }

    pub(super) fn is_atlas_visible(&self) -> bool {
        self.atlas
            .as_ref()
            .is_some_and(ParameterAtlasController::is_visible)
    }

    pub(super) fn draw_atlas_panel_ui(&mut self, ui: &mut egui::Ui) -> Option<AtlasTeleportEvent> {
        let atlas = self.atlas.as_mut()?;
        if !atlas.is_visible() {
            return None;
        }
        let (xk, yk) = atlas.axis_keys();
        let snapshot_params = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let active_x = snapshot_params.get(xk).copied().unwrap_or(0.0);
        let active_y = snapshot_params.get(yk).copied().unwrap_or(0.0);

        let mut h = std::collections::hash_map::DefaultHasher::new();
        for (name, _) in &self.param_metadata {
            if name != xk && name != yk {
                name.hash(&mut h);
                if let Some(v) = snapshot_params.get(name) {
                    v.to_bits().hash(&mut h);
                }
            }
        }
        atlas.show_panel(ui, M::name(), active_x, active_y, h.finish(), move |keys| {
            M::build_atlas_cartographer(&snapshot_params, keys)
                .expect("UnifiedModel with atlas_axis_pairs must build cartographer")
        })
    }
}

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{UnifiedModel, UnifiedSimTool};
use crate::async_sim::atlas::{AtlasTeleportEvent, ParameterAtlasController};
use crate::async_sim::SimCommand;
use crate::framework::InteractiveTool;
use eframe::egui;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};

#[derive(Serialize)]
struct UnifiedAtlasStateRef<'a> {
    params: BTreeMap<&'a String, &'a f64>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    atlas_visible: bool,
    atlas_axes: [&'a str; 2],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UnifiedAtlasStateOwned {
    params: HashMap<String, f64>,
    #[serde(default)]
    atlas_visible: bool,
    #[serde(default)]
    atlas_axes: Option<[String; 2]>,
}

type ParsedUnifiedAtlasCfg = Option<(bool, Option<[String; 2]>)>;

impl<M: UnifiedModel> UnifiedSimTool<M> {
    pub(super) fn create_optional_atlas() -> Option<ParameterAtlasController> {
        let pairs = M::atlas_axis_pairs();
        (!pairs.is_empty()).then(|| ParameterAtlasController::new(pairs.to_vec()))
    }

    pub(super) fn serialize_unified_state(&self) -> Option<String> {
        let params = self.params.read().unwrap_or_else(|e| e.into_inner());
        let sorted: BTreeMap<&String, &f64> = params.iter().collect();
        if let Some(atlas) = &self.atlas {
            if !atlas.has_default_persisted_state() {
                let (xk, yk) = atlas.axis_keys();
                return serde_json::to_string(&UnifiedAtlasStateRef {
                    params: sorted,
                    atlas_visible: atlas.is_visible(),
                    atlas_axes: [xk, yk],
                })
                .ok();
            }
        }
        serde_json::to_string(&sorted).ok()
    }

    pub(super) fn parse_unified_state(
        &self,
        state: &str,
    ) -> Result<(HashMap<String, f64>, ParsedUnifiedAtlasCfg), String> {
        self.state_bounds().validate_json(state)?;
        if let Ok(ext) = serde_json::from_str::<UnifiedAtlasStateOwned>(state) {
            if self.atlas.is_none() && (ext.atlas_visible || ext.atlas_axes.is_some()) {
                return Err(format!(
                    "{} does not support 2D parameter atlas state",
                    M::name()
                ));
            }
            return Ok((ext.params, Some((ext.atlas_visible, ext.atlas_axes))));
        }
        let flat: HashMap<String, f64> = serde_json::from_str(state)
            .map_err(|e| format!("Invalid simulation parameter state for {}: {e}", M::name()))?;
        Ok((flat, None))
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

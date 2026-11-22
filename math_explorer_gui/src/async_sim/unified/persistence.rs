#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{UnifiedModel, UnifiedSimTool};
use crate::async_sim::SimCommand;
use crate::framework::{encode_bool_grid, RasterBrushStroke};
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};

const OBSTACLES_RLE_KEY: &str = "obstacles_rle";

type ParsedUnifiedState = (HashMap<String, f64>, Option<(String, Vec<bool>)>);

impl<M: UnifiedModel> UnifiedSimTool<M> {
    pub(crate) fn apply_brush_stroke(&mut self, stroke: RasterBrushStroke, ctx: &egui::Context) {
        let Some((grid_w, grid_h, _)) = M::raster_brush_grid() else {
            return;
        };
        let is_obstacle = !stroke.secondary;
        if let Some(ref mut mask) = self.local_obstacles {
            if mask.len() == grid_w * grid_h {
                for &(x, y) in &stroke.cells {
                    if x < grid_w && y < grid_h {
                        mask[flatten_2d_index(x, y, grid_w)] = is_obstacle;
                    }
                }
                if let Ok(encoded) = encode_bool_grid(grid_w, grid_h, mask) {
                    *self.obstacle_rle.write().unwrap_or_else(|e| e.into_inner()) = Some(encoded);
                }
            }
        }
        if self
            .controller
            .send_command(SimCommand::ApplyBrushCells {
                cells: stroke.cells,
                mode: u8::from(is_obstacle),
            })
            .is_ok()
        {
            self.dirty = true;
            ctx.request_repaint();
        }
    }

    pub(super) fn handle_custom_obstacle_action(&mut self, action: &str, ctx: &egui::Context) {
        let Some((grid_w, grid_h, _)) = M::raster_brush_grid() else {
            return;
        };
        if action == "Clear Obstacles" {
            if let Some(ref mut mask) = self.local_obstacles {
                mask.fill(false);
                if let Ok(encoded) = encode_bool_grid(grid_w, grid_h, mask) {
                    *self.obstacle_rle.write().unwrap_or_else(|e| e.into_inner()) = Some(encoded);
                }
            }
            self.dirty = true;
            ctx.request_repaint();
        } else if action == "Default Obstacles" {
            let current_params = self
                .params
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            self.local_obstacles = M::default_obstacle_mask(&current_params);
            *self.obstacle_rle.write().unwrap_or_else(|e| e.into_inner()) = None;
            self.reset_simulation(Some(ctx));
        }
    }

    pub(super) fn compute_state_revision(&self) -> u64 {
        let params = self.params.read().unwrap_or_else(|e| e.into_inner());
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for (key, _) in &self.param_metadata {
            key.hash(&mut h);
            if let Some(val) = params.get(key) {
                val.to_bits().hash(&mut h);
            }
        }
        if let Some(ref rle) = *self.obstacle_rle.read().unwrap_or_else(|e| e.into_inner()) {
            rle.hash(&mut h);
        }
        h.finish().max(1)
    }

    pub(super) fn serialize_tool_state(&self) -> Option<String> {
        let params = self.params.read().unwrap_or_else(|e| e.into_inner());
        let mut map: BTreeMap<&str, serde_json::Value> = BTreeMap::new();
        for (k, v) in params.iter() {
            let num = serde_json::Number::from_f64(*v)?;
            map.insert(k.as_str(), serde_json::Value::Number(num));
        }
        if let Some(ref rle) = *self.obstacle_rle.read().unwrap_or_else(|e| e.into_inner()) {
            map.insert(OBSTACLES_RLE_KEY, serde_json::Value::String(rle.clone()));
        }
        serde_json::to_string(&map).ok()
    }

    fn parse_state_entries(
        &self,
        raw_map: serde_json::Map<String, serde_json::Value>,
    ) -> Result<ParsedUnifiedState, String> {
        let mut parsed_rle: Option<(String, Vec<bool>)> = None;
        let mut numeric_params: HashMap<String, f64> = HashMap::new();
        for (key, val) in raw_map {
            if key == OBSTACLES_RLE_KEY && M::raster_brush_grid().is_some() {
                if !val.is_null() {
                    let s = val.as_str().ok_or_else(|| {
                        format!(
                            "Parameter '{OBSTACLES_RLE_KEY}' in {} must be a string",
                            M::name()
                        )
                    })?;
                    parsed_rle = Some((s.to_string(), M::validate_obstacle_rle(s)?));
                }
                continue;
            }
            if !self.param_metadata.iter().any(|meta| meta.0 == key) {
                return Err(format!("Unknown parameter '{}' for {}", key, M::name()));
            }
            let num = val.as_f64().filter(|n| n.is_finite()).ok_or_else(|| {
                format!(
                    "Non-finite or non-numeric value for parameter '{}' in {}",
                    key,
                    M::name()
                )
            })?;
            numeric_params.insert(key, num);
        }
        Ok((numeric_params, parsed_rle))
    }

    pub(super) fn deserialize_tool_state(&mut self, state: &str) -> Result<(), String> {
        let raw_map: serde_json::Map<String, serde_json::Value> = serde_json::from_str(state)
            .map_err(|e| format!("Invalid simulation parameter state for {}: {e}", M::name()))?;
        let (numeric_params, parsed_rle) = self.parse_state_entries(raw_map)?;
        if !self.param_metadata.is_empty() && numeric_params.is_empty() {
            return Err(format!(
                "Empty parameter configuration for {}; expected parameters: {}",
                M::name(),
                self.param_metadata
                    .iter()
                    .map(|m| m.0.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        {
            let mut guard = self.params.write().unwrap_or_else(|e| e.into_inner());
            for (key, value) in numeric_params {
                guard.insert(key, value);
            }
        }
        match parsed_rle {
            Some((rle_str, decoded_mask)) => {
                *self.obstacle_rle.write().unwrap_or_else(|e| e.into_inner()) = Some(rle_str);
                self.local_obstacles = Some(decoded_mask);
            }
            None => {
                let p_snap = self
                    .params
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone();
                *self.obstacle_rle.write().unwrap_or_else(|e| e.into_inner()) = None;
                self.local_obstacles = M::default_obstacle_mask(&p_snap);
            }
        }
        self.texture = None;
        self.reset_simulation(None);
        Ok(())
    }
}

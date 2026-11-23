#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{CompiledFormulaPayload, UnifiedFormulaSpec, UnifiedModel, UnifiedSimTool};
use crate::async_sim::SimCommand;
use crate::framework::{
    encode_bool_grid, InteractiveTool, MathExpressionInputBar, RasterBrushStroke,
    MAX_SYNTHESIZED_PARAMS,
};
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

const OBSTACLES_RLE_KEY: &str = "obstacles_rle";

type ParsedUnifiedState = (HashMap<String, f64>, Option<(String, Vec<bool>)>);

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct UnifiedExtendedState {
    params: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    custom_formula_enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    custom_expr: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    custom_params: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    atlas_visible: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    atlas_axes: Option<[String; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    obstacles_rle: Option<String>,
}

pub(super) fn build_formula_bar(spec: &UnifiedFormulaSpec) -> MathExpressionInputBar {
    MathExpressionInputBar::new(spec.lhs_label, spec.default_expr, spec.presets.to_vec())
        .with_axes(spec.axes)
        .with_param_range(spec.param_range.clone(), spec.param_default)
}

pub(super) fn snapshot_formula_payload(
    bar: &MathExpressionInputBar,
    revision: u64,
) -> Option<CompiledFormulaPayload> {
    let compiled = Arc::new(bar.active_compiled()?.clone());
    let slice = bar.param_slice();
    let param_len = slice.len().min(MAX_SYNTHESIZED_PARAMS);
    let mut params = [0.0; MAX_SYNTHESIZED_PARAMS];
    params[..param_len].copy_from_slice(&slice[..param_len]);
    Some(CompiledFormulaPayload {
        revision: revision.max(1),
        compiled,
        params,
        param_len,
    })
}

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
                if let Some(ext) = &mut self.ui_extension {
                    let p = self.params.read().unwrap_or_else(|e| e.into_inner());
                    ext.on_obstacle_mask_loaded(Some(mask.as_slice()), &p);
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

    pub(crate) fn publish_formula_update(&mut self) {
        if !self.custom_formula_enabled {
            *self
                .shared_formula
                .write()
                .unwrap_or_else(|e| e.into_inner()) = None;
            return;
        }
        let Some(ref bar) = self.formula_bar else {
            return;
        };
        self.formula_revision = self.formula_revision.wrapping_add(1).max(1);
        let payload = snapshot_formula_payload(bar, self.formula_revision);
        *self
            .shared_formula
            .write()
            .unwrap_or_else(|e| e.into_inner()) = payload;
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
        if let Some(ext) = &self.ui_extension {
            ext.state_revision_salt().hash(&mut h);
        }
        if let Some(atlas) = &self.atlas {
            atlas.is_visible().hash(&mut h);
            atlas.axis_keys().hash(&mut h);
        }
        self.custom_formula_enabled.hash(&mut h);
        if let Some(ref bar) = self.formula_bar {
            bar.raw_text.hash(&mut h);
            for &pv in bar.param_slice() {
                pv.to_bits().hash(&mut h);
            }
        }
        if let Some(ref rle) = *self.obstacle_rle.read().unwrap_or_else(|e| e.into_inner()) {
            rle.hash(&mut h);
        }
        h.finish().max(1)
    }

    pub(super) fn serialize_tool_state(&self) -> Option<String> {
        let params = self.params.read().unwrap_or_else(|e| e.into_inner());
        let obstacles_rle = self
            .obstacle_rle
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        if self.active_challenge_id().is_some() || obstacles_rle.is_none() {
            if let Some(ext) = &self.ui_extension {
                if let Some(custom) = ext.save_extra_state(&params) {
                    return Some(custom);
                }
            }
        }
        let formula_ext =
            M::formula_spec()
                .zip(self.formula_bar.as_ref())
                .and_then(|(spec, bar)| {
                    let active = self.custom_formula_enabled;
                    let customized = bar.raw_text != spec.default_expr;
                    (active || customized).then(|| {
                        let cp = if active {
                            bar.params_map()
                        } else {
                            BTreeMap::new()
                        };
                        (active, bar.raw_text.clone(), cp)
                    })
                });
        let atlas_ext = self.atlas.as_ref().and_then(|atlas| {
            (!atlas.has_default_persisted_state()).then(|| {
                let (xk, yk) = atlas.axis_keys();
                (atlas.is_visible(), [xk.to_string(), yk.to_string()])
            })
        });
        if formula_ext.is_some() || atlas_ext.is_some() {
            if params.values().any(|v| !v.is_finite()) {
                return None;
            }
            let sorted_params = params.iter().map(|(k, &v)| (k.clone(), v)).collect();
            let (custom_formula_enabled, custom_expr, custom_params) = match formula_ext {
                Some((en, expr, cp)) => (en, Some(expr), cp),
                None => (false, None, BTreeMap::new()),
            };
            let (atlas_visible, atlas_axes) = match atlas_ext {
                Some((vis, axes)) => (vis, Some(axes)),
                None => (false, None),
            };
            return serde_json::to_string(&UnifiedExtendedState {
                params: sorted_params,
                custom_formula_enabled,
                custom_expr,
                custom_params,
                atlas_visible,
                atlas_axes,
                obstacles_rle,
            })
            .ok();
        }
        Self::serialize_flat_state_map(&params, obstacles_rle)
    }

    fn serialize_flat_state_map(
        params: &HashMap<String, f64>,
        obstacles_rle: Option<String>,
    ) -> Option<String> {
        let mut map: BTreeMap<&str, serde_json::Value> = BTreeMap::new();
        for (k, v) in params {
            let num = serde_json::Number::from_f64(*v)?;
            map.insert(k.as_str(), serde_json::Value::Number(num));
        }
        if let Some(rle) = obstacles_rle {
            map.insert(OBSTACLES_RLE_KEY, serde_json::Value::String(rle));
        }
        serde_json::to_string(&map).ok()
    }

    fn apply_parsed_obstacle_rle(&mut self, parsed_rle: Option<(String, Vec<bool>)>) {
        let p_snap = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        match parsed_rle {
            Some((rle_str, decoded_mask)) => {
                if let Some(ext) = &mut self.ui_extension {
                    ext.on_obstacle_mask_loaded(Some(decoded_mask.as_slice()), &p_snap);
                }
                *self.obstacle_rle.write().unwrap_or_else(|e| e.into_inner()) = Some(rle_str);
                self.local_obstacles = Some(decoded_mask);
            }
            None => {
                if let Some(ext) = &mut self.ui_extension {
                    ext.on_obstacle_mask_loaded(None, &p_snap);
                }
                *self.obstacle_rle.write().unwrap_or_else(|e| e.into_inner()) = None;
                self.local_obstacles = M::default_obstacle_mask(&p_snap);
            }
        }
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

    fn try_load_extension_state(&mut self, state: &str) -> Option<Result<(), String>> {
        if self.active_challenge_id().is_none()
            && serde_json::from_str::<serde_json::Value>(state)
                .ok()
                .is_some_and(|v| {
                    v.as_object().is_some_and(|m| {
                        m.contains_key(OBSTACLES_RLE_KEY) || m.contains_key("params")
                    })
                })
        {
            return None;
        }
        let ext = self.ui_extension.as_mut()?;
        let mut next_params = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let res = ext.try_load_extra_state(state, &mut next_params)?;
        Some(res.map(|cmds| {
            *self.obstacle_rle.write().unwrap_or_else(|e| e.into_inner()) = None;
            self.local_obstacles = M::default_obstacle_mask(&next_params);
            *self.params.write().unwrap_or_else(|e| e.into_inner()) = next_params;
            self.texture = None;
            self.reset_simulation(None);
            self.dispatch_extension_commands(cmds, None);
        }))
    }

    pub(super) fn deserialize_tool_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        if let Some(res) = self.try_load_extension_state(state) {
            return res;
        }
        if let Ok(ext) = serde_json::from_str::<UnifiedExtendedState>(state) {
            return self.apply_extended_state(ext);
        }
        let raw_map: serde_json::Map<String, serde_json::Value> = serde_json::from_str(state)
            .map_err(|e| format!("Invalid simulation parameter state for {}: {e}", M::name()))?;
        let (numeric_params, parsed_rle) = self.parse_state_entries(raw_map)?;
        self.validate_param_map(&numeric_params)?;
        self.apply_parsed_atlas_state(None)?;
        {
            let mut guard = self.params.write().unwrap_or_else(|e| e.into_inner());
            for (key, value) in numeric_params {
                guard.insert(key, value);
            }
        }
        self.apply_parsed_obstacle_rle(parsed_rle);
        self.reset_formula_bar_to_default();
        self.texture = None;
        self.reset_simulation(None);
        Ok(())
    }

    fn validate_extended_state(
        &self,
        ext: &UnifiedExtendedState,
    ) -> Result<ParsedUnifiedState, String> {
        let map: HashMap<String, f64> = ext.params.iter().map(|(k, &v)| (k.clone(), v)).collect();
        self.validate_param_map(&map)?;
        let has_formula = ext.custom_formula_enabled
            || ext.custom_expr.is_some()
            || !ext.custom_params.is_empty();
        if M::formula_spec().is_none() && has_formula {
            return Err(format!(
                "{} does not support custom formula state",
                M::name()
            ));
        }
        if ext.custom_params.values().any(|v| !v.is_finite()) {
            return Err(format!(
                "Non-finite custom formula parameter in {}",
                M::name()
            ));
        }
        if self.atlas.is_none() && (ext.atlas_visible || ext.atlas_axes.is_some()) {
            return Err(format!(
                "{} does not support 2D parameter atlas state",
                M::name()
            ));
        }
        let parsed_rle = match ext.obstacles_rle.as_deref() {
            Some(rle) if M::raster_brush_grid().is_some() => {
                let decoded = M::validate_obstacle_rle(rle)?;
                Some((rle.to_string(), decoded))
            }
            Some(_) => {
                return Err(format!(
                    "Unknown parameter '{OBSTACLES_RLE_KEY}' for {}",
                    M::name()
                ));
            }
            None => None,
        };
        Ok((map, parsed_rle))
    }

    fn apply_extended_state(&mut self, ext: UnifiedExtendedState) -> Result<(), String> {
        let (map, parsed_rle) = self.validate_extended_state(&ext)?;
        self.apply_parsed_atlas_state(Some((ext.atlas_visible, ext.atlas_axes)))?;
        {
            let mut guard = self.params.write().unwrap_or_else(|e| e.into_inner());
            for (k, v) in map {
                guard.insert(k, v);
            }
        }
        self.apply_parsed_obstacle_rle(parsed_rle);
        if let Some(spec) = M::formula_spec() {
            if let Some(expr) = ext.custom_expr {
                let bar = self
                    .formula_bar
                    .get_or_insert_with(|| build_formula_bar(&spec));
                bar.set_expression(&expr);
                bar.set_params_map(&ext.custom_params);
                self.custom_formula_enabled = ext.custom_formula_enabled;
                self.publish_formula_update();
            } else {
                self.reset_formula_bar_to_default();
            }
        }
        self.texture = None;
        self.reset_simulation(None);
        Ok(())
    }

    fn reset_formula_bar_to_default(&mut self) {
        self.custom_formula_enabled = false;
        if let (Some(spec), Some(bar)) = (M::formula_spec(), self.formula_bar.as_mut()) {
            bar.set_expression(spec.default_expr);
            bar.set_params_map(&BTreeMap::new());
            self.publish_formula_update();
        }
    }

    fn validate_param_map(&self, params: &HashMap<String, f64>) -> Result<(), String> {
        if !self.param_metadata.is_empty() && params.is_empty() {
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
        for (key, value) in params {
            if !self.param_metadata.iter().any(|meta| &meta.0 == key) {
                return Err(format!("Unknown parameter '{}' for {}", key, M::name()));
            }
            if !value.is_finite() {
                return Err(format!(
                    "Non-finite value for parameter '{}' in {}",
                    key,
                    M::name()
                ));
            }
        }
        Ok(())
    }
}

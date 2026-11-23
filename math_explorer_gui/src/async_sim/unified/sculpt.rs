//! Compact UI-side sculpt state and JSON preset/URL persistence helpers for [`super::UnifiedSimTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{UnifiedModel, UnifiedSimTool};
use crate::async_sim::SculptStatePayload;
use crate::framework::MAX_RLE_MASK_CELLS;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};

impl SculptStatePayload {
    /// Returns `true` when neither RLE 2D masks nor sparse 1D potentials are populated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rle_mask.is_empty() && self.sparse_potential.is_empty()
    }

    /// Clears all persisted sculpt deltas.
    pub fn clear(&mut self) {
        self.rle_mask.clear();
        self.sparse_potential.clear();
    }

    /// Hashes sculpt state into `hasher` so `InteractiveTool::state_revision` tracks brush edits.
    pub fn hash_into(&self, hasher: &mut impl Hasher) {
        self.rle_mask.hash(hasher);
        for &(idx, val) in &self.sparse_potential {
            idx.hash(hasher);
            val.to_bits().hash(hasher);
        }
    }
}

pub(super) fn validate_rle_mask_entries(entries: &[[u32; 2]]) -> Result<(), String> {
    let total_cells: usize = entries.iter().try_fold(0_usize, |acc, &[v, len]| {
        if v > u32::from(u8::MAX) {
            return Err("RLE value out of u8 range".to_string());
        }
        acc.checked_add(len as usize)
            .ok_or_else(|| "RLE length overflow".to_string())
    })?;
    if total_cells > MAX_RLE_MASK_CELLS {
        return Err(format!(
            "rle_mask cell count {total_cells} exceeds {MAX_RLE_MASK_CELLS}"
        ));
    }
    Ok(())
}

pub(super) fn validate_sparse_potential_entries(
    entries: &[(u16, f64)],
    model_name: &str,
) -> Result<(), String> {
    if entries.iter().any(|&(_, v)| !v.is_finite()) {
        return Err(format!("Non-finite sparse_potential value in {model_name}"));
    }
    Ok(())
}

impl<M: UnifiedModel> UnifiedSimTool<M> {
    pub(super) fn serialize_unified_state(&self) -> Option<String> {
        let params = self.params.read().unwrap_or_else(|e| e.into_inner());
        let sculpt = self.sculpt_state.read().unwrap_or_else(|e| e.into_inner());

        let mut map = serde_json::Map::new();
        let sorted: BTreeMap<&String, &f64> = params.iter().collect();
        for (k, v) in sorted {
            if let Some(num) = serde_json::Number::from_f64(*v) {
                map.insert(k.clone(), serde_json::Value::Number(num));
            }
        }

        if !sculpt.rle_mask.is_empty() {
            if let Ok(val) = serde_json::to_value(&sculpt.rle_mask) {
                map.insert("rle_mask".to_string(), val);
            }
        }
        if !sculpt.sparse_potential.is_empty() {
            if let Ok(val) = serde_json::to_value(&sculpt.sparse_potential) {
                map.insert("sparse_potential".to_string(), val);
            }
        }

        serde_json::to_string(&map).ok()
    }

    fn parse_unified_state_entry(
        &self,
        key: String,
        val: serde_json::Value,
        parsed_params: &mut HashMap<String, f64>,
        parsed_rle: &mut Vec<[u32; 2]>,
        parsed_sparse: &mut Vec<(u16, f64)>,
    ) -> Result<(), String> {
        if key == "rle_mask" && M::supports_spatial_brush() && M::is_raster_model() {
            *parsed_rle = serde_json::from_value(val)
                .map_err(|e| format!("Invalid rle_mask in {}: {e}", M::name()))?;
            validate_rle_mask_entries(parsed_rle)?;
            return Ok(());
        }
        if key == "sparse_potential" && M::supports_spatial_brush() && !M::is_raster_model() {
            *parsed_sparse = serde_json::from_value(val)
                .map_err(|e| format!("Invalid sparse_potential in {}: {e}", M::name()))?;
            validate_sparse_potential_entries(parsed_sparse, M::name())?;
            return Ok(());
        }
        if !self.param_metadata.iter().any(|meta| meta.0 == key) {
            return Err(format!("Unknown parameter '{}' for {}", key, M::name()));
        }
        let Some(number) = val.as_f64().filter(|n| n.is_finite()) else {
            return Err(format!(
                "Non-finite or non-numeric value for parameter '{}' in {}",
                key,
                M::name()
            ));
        };
        parsed_params.insert(key, number);
        Ok(())
    }

    pub(super) fn deserialize_unified_state(&mut self, state: &str) -> Result<(), String> {
        let raw_map: serde_json::Map<String, serde_json::Value> = serde_json::from_str(state)
            .map_err(|e| format!("Invalid simulation parameter state for {}: {e}", M::name()))?;

        let mut parsed_params: HashMap<String, f64> = HashMap::new();
        let mut parsed_rle: Vec<[u32; 2]> = Vec::new();
        let mut parsed_sparse: Vec<(u16, f64)> = Vec::new();

        for (key, val) in raw_map {
            self.parse_unified_state_entry(
                key,
                val,
                &mut parsed_params,
                &mut parsed_rle,
                &mut parsed_sparse,
            )?;
        }

        if !self.param_metadata.is_empty()
            && parsed_params.is_empty()
            && parsed_rle.is_empty()
            && parsed_sparse.is_empty()
        {
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
            for (key, value) in parsed_params {
                guard.insert(key, value);
            }
        }
        self.custom_formula_enabled = false;
        if let (Some(spec), Some(bar)) = (M::formula_spec(), self.formula_bar.as_mut()) {
            bar.set_expression(spec.default_expr);
            bar.set_params_map(&BTreeMap::new());
            self.publish_formula_update();
        }
        {
            let mut s_guard = self.sculpt_state.write().unwrap_or_else(|e| e.into_inner());
            s_guard.rle_mask = parsed_rle;
            s_guard.sparse_potential = parsed_sparse;
        }
        self.texture = None;
        self.respawn_preserving_sculpt(None);
        Ok(())
    }
}

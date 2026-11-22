#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{UnifiedModel, UnifiedSimTool};
use crate::url_state::quantize_bounded_url_series;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};

type ParsedToolState = (HashMap<String, f64>, HashMap<String, Vec<f64>>);

fn validate_custom_series_entry<M: UnifiedModel>(
    key: &str,
    val: &serde_json::Value,
) -> Result<Option<Vec<f64>>, String> {
    let Some(&(_, expected_len, min_val, max_val)) = M::custom_series_schema()
        .iter()
        .find(|(schema_key, _, _, _)| *schema_key == key)
    else {
        return Ok(None);
    };
    let arr = val.as_array().ok_or_else(|| {
        format!(
            "Custom series '{key}' in {} must be a JSON array",
            M::name()
        )
    })?;
    if arr.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let mut raw = Vec::with_capacity(arr.len());
    for item in arr {
        let num = item.as_f64().filter(|v| v.is_finite()).ok_or_else(|| {
            format!(
                "Non-finite or non-numeric element in '{key}' for {}",
                M::name()
            )
        })?;
        raw.push(num);
    }
    let quantized = quantize_bounded_url_series(&raw, expected_len, min_val, max_val)?;
    Ok(Some(quantized))
}

impl<M: UnifiedModel> UnifiedSimTool<M> {
    pub(super) fn compute_state_revision(&self) -> u64 {
        let params = self.params.read().unwrap_or_else(|e| e.into_inner());
        let series = self.custom_series.read().unwrap_or_else(|e| e.into_inner());
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for (key, _) in &self.param_metadata {
            key.hash(&mut h);
            if let Some(val) = params.get(key) {
                val.to_bits().hash(&mut h);
            }
        }
        for &(key, _, _, _) in M::custom_series_schema() {
            if let Some(values) = series.get(key).filter(|v| !v.is_empty()) {
                key.hash(&mut h);
                values.len().hash(&mut h);
                for &v in values {
                    v.to_bits().hash(&mut h);
                }
            }
        }
        h.finish().max(1)
    }

    pub(super) fn check_finite_parameters(&self) -> bool {
        let params = self.params.read().unwrap_or_else(|e| e.into_inner());
        let series = self.custom_series.read().unwrap_or_else(|e| e.into_inner());
        params.values().all(|v| v.is_finite())
            && series.values().all(|vec| vec.iter().all(|v| v.is_finite()))
            && !matches!(
                self.controller.state(),
                crate::async_sim::SimWorkerState::Faulted
            )
    }

    pub(super) fn serialize_tool_state(&self) -> Option<String> {
        let params = self.params.read().unwrap_or_else(|e| e.into_inner());
        let series = self.custom_series.read().unwrap_or_else(|e| e.into_inner());
        let has_active_series = M::custom_series_schema()
            .iter()
            .any(|(k, _, _, _)| series.get(*k).is_some_and(|v| !v.is_empty()));

        if !has_active_series {
            let sorted: BTreeMap<&String, &f64> = params.iter().collect();
            return serde_json::to_string(&sorted).ok();
        }

        let mut map = serde_json::Map::new();
        let sorted_params: BTreeMap<&String, &f64> = params.iter().collect();
        for (k, &v) in sorted_params {
            map.insert(k.clone(), serde_json::Value::from(v));
        }
        for &(key, expected_len, min_val, max_val) in M::custom_series_schema() {
            if let Some(values) = series.get(key).filter(|v| !v.is_empty()) {
                let quantized =
                    quantize_bounded_url_series(values, expected_len, min_val, max_val).ok()?;
                let json_arr = quantized.into_iter().map(serde_json::Value::from).collect();
                map.insert(key.to_string(), serde_json::Value::Array(json_arr));
            }
        }
        serde_json::to_string(&map).ok()
    }

    fn parse_state_map_entries(
        &self,
        raw_map: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<ParsedToolState, String> {
        let mut parsed_scalars = HashMap::new();
        let mut parsed_series = HashMap::new();
        for (key, value) in raw_map {
            if self.param_metadata.iter().any(|meta| &meta.0 == key) {
                let num = value.as_f64().ok_or_else(|| {
                    format!(
                        "Invalid numeric value for parameter '{key}' in {}",
                        M::name()
                    )
                })?;
                if !num.is_finite() {
                    return Err(format!(
                        "Non-finite value for parameter '{key}' in {}",
                        M::name()
                    ));
                }
                parsed_scalars.insert(key.clone(), num);
            } else if let Some(series_vec) = validate_custom_series_entry::<M>(key, value)? {
                if !series_vec.is_empty() {
                    parsed_series.insert(key.clone(), series_vec);
                }
            } else {
                return Err(format!("Unknown parameter '{key}' for {}", M::name()));
            }
        }
        Ok((parsed_scalars, parsed_series))
    }

    pub(super) fn deserialize_tool_state(&mut self, state: &str) -> Result<(), String> {
        let raw_map: serde_json::Map<String, serde_json::Value> = serde_json::from_str(state)
            .map_err(|e| format!("Invalid simulation parameter state for {}: {e}", M::name()))?;
        if !self.param_metadata.is_empty() && raw_map.is_empty() {
            let expected: Vec<&str> = self.param_metadata.iter().map(|m| m.0.as_str()).collect();
            return Err(format!(
                "Empty parameter configuration for {}; expected parameters: {}",
                M::name(),
                expected.join(", ")
            ));
        }

        let (parsed_scalars, parsed_series) = self.parse_state_map_entries(&raw_map)?;
        {
            let mut guard = self.params.write().unwrap_or_else(|e| e.into_inner());
            for (name, constraint) in &self.param_metadata {
                let val = parsed_scalars
                    .get(name)
                    .copied()
                    .unwrap_or_else(|| M::default_parameter(name, constraint));
                guard.insert(name.clone(), val);
            }
        }
        *self
            .custom_series
            .write()
            .unwrap_or_else(|e| e.into_inner()) = parsed_series;
        self.texture = None;
        self.reset_simulation(None);
        Ok(())
    }
}

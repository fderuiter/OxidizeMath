#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::persistence::{build_formula_bar, UnifiedExtendedState};
use super::{UnifiedModel, UnifiedSimTool};
use crate::framework::InteractiveTool;
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

fn collect_quantized_series<M: UnifiedModel>(
    series: &HashMap<String, Vec<f64>>,
) -> Option<BTreeMap<String, Vec<f64>>> {
    let mut out = BTreeMap::new();
    for &(key, expected_len, min_val, max_val) in M::custom_series_schema() {
        if let Some(values) = series.get(key).filter(|v| !v.is_empty()) {
            let quantized =
                quantize_bounded_url_series(values, expected_len, min_val, max_val).ok()?;
            out.insert(key.to_string(), quantized);
        }
    }
    Some(out)
}

fn validate_extended_series_map<M: UnifiedModel>(
    raw_series: &BTreeMap<String, Vec<f64>>,
) -> Result<HashMap<String, Vec<f64>>, String> {
    let mut parsed = HashMap::new();
    for (key, values) in raw_series {
        let Some(&(_, expected_len, min_val, max_val)) = M::custom_series_schema()
            .iter()
            .find(|(schema_key, _, _, _)| schema_key == key)
        else {
            return Err(format!("Unknown custom series '{key}' for {}", M::name()));
        };
        if !values.is_empty() {
            let q = quantize_bounded_url_series(values, expected_len, min_val, max_val)?;
            parsed.insert(key.clone(), q);
        }
    }
    Ok(parsed)
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
        let formula_ok = self
            .formula_bar
            .as_ref()
            .is_none_or(|b| b.param_slice().iter().all(|v| v.is_finite()));
        params.values().all(|v| v.is_finite())
            && formula_ok
            && series.values().all(|vec| vec.iter().all(|v| v.is_finite()))
            && !matches!(
                self.controller.state(),
                crate::async_sim::SimWorkerState::Faulted
            )
    }

    pub(super) fn serialize_tool_state(&self) -> Option<String> {
        let params = self.params.read().unwrap_or_else(|e| e.into_inner());
        let series = self.custom_series.read().unwrap_or_else(|e| e.into_inner());
        let quantized_series = collect_quantized_series::<M>(&series)?;
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
                custom_series: quantized_series,
                atlas_visible,
                atlas_axes,
            })
            .ok();
        }

        if quantized_series.is_empty() {
            let sorted: BTreeMap<&String, &f64> = params.iter().collect();
            return serde_json::to_string(&sorted).ok();
        }

        let mut map = serde_json::Map::new();
        let sorted_params: BTreeMap<&String, &f64> = params.iter().collect();
        for (k, &v) in sorted_params {
            map.insert(k.clone(), serde_json::Value::from(v));
        }
        for (key, quantized) in quantized_series {
            let json_arr = quantized.into_iter().map(serde_json::Value::from).collect();
            map.insert(key, serde_json::Value::Array(json_arr));
        }
        serde_json::to_string(&map).ok()
    }

    fn validate_scalar_param_map(&self, params: &BTreeMap<String, f64>) -> Result<(), String> {
        if !self.param_metadata.is_empty() && params.is_empty() {
            let expected: Vec<&str> = self.param_metadata.iter().map(|m| m.0.as_str()).collect();
            return Err(format!(
                "Empty parameter configuration for {}; expected parameters: {}",
                M::name(),
                expected.join(", ")
            ));
        }
        for (key, value) in params {
            if !self.param_metadata.iter().any(|meta| &meta.0 == key) {
                return Err(format!("Unknown parameter '{key}' for {}", M::name()));
            }
            if !value.is_finite() {
                return Err(format!(
                    "Non-finite value for parameter '{key}' in {}",
                    M::name()
                ));
            }
        }
        Ok(())
    }

    fn apply_loaded_scalars_and_series(
        &mut self,
        parsed_scalars: &HashMap<String, f64>,
        parsed_series: HashMap<String, Vec<f64>>,
    ) {
        let mut guard = self.params.write().unwrap_or_else(|e| e.into_inner());
        for (key, value) in parsed_scalars {
            guard.insert(key.clone(), *value);
        }
        drop(guard);
        *self
            .custom_series
            .write()
            .unwrap_or_else(|e| e.into_inner()) = parsed_series;
    }

    fn apply_extended_state(&mut self, ext: UnifiedExtendedState) -> Result<(), String> {
        self.validate_scalar_param_map(&ext.params)?;
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
        let parsed_series = validate_extended_series_map::<M>(&ext.custom_series)?;
        self.apply_parsed_atlas_state(Some((ext.atlas_visible, ext.atlas_axes)))?;
        let scalar_map: HashMap<String, f64> = ext.params.into_iter().collect();
        self.apply_loaded_scalars_and_series(&scalar_map, parsed_series);
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
        self.state_bounds().validate_json(state)?;
        let raw_map: serde_json::Map<String, serde_json::Value> = serde_json::from_str(state)
            .map_err(|e| format!("Invalid simulation parameter state for {}: {e}", M::name()))?;
        let is_extended_envelope = raw_map.contains_key("params")
            || raw_map.contains_key("custom_expr")
            || raw_map.contains_key("custom_formula_enabled")
            || raw_map.contains_key("custom_params")
            || raw_map.contains_key("atlas_visible")
            || raw_map.contains_key("atlas_axes");
        if is_extended_envelope {
            let ext: UnifiedExtendedState = serde_json::from_str(state)
                .map_err(|e| format!("Invalid extended simulation state for {}: {e}", M::name()))?;
            return self.apply_extended_state(ext);
        }
        if !self.param_metadata.is_empty() && raw_map.is_empty() {
            let expected: Vec<&str> = self.param_metadata.iter().map(|m| m.0.as_str()).collect();
            return Err(format!(
                "Empty parameter configuration for {}; expected parameters: {}",
                M::name(),
                expected.join(", ")
            ));
        }

        let (parsed_scalars, parsed_series) = self.parse_state_map_entries(&raw_map)?;
        self.apply_parsed_atlas_state(None)?;
        self.apply_loaded_scalars_and_series(&parsed_scalars, parsed_series);
        self.reset_formula_bar_to_default();
        self.texture = None;
        self.reset_simulation(None);
        Ok(())
    }
}

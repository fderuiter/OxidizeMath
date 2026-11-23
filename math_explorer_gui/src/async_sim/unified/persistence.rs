#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::sculpt::{validate_rle_mask_entries, validate_sparse_potential_entries};
use super::{CompiledFormulaPayload, UnifiedFormulaSpec, UnifiedModel, UnifiedSimTool};
use crate::framework::{MathExpressionInputBar, MAX_SYNTHESIZED_PARAMS};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct UnifiedFormulaState {
    params: BTreeMap<String, f64>,
    #[serde(default)]
    custom_formula_enabled: bool,
    custom_expr: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    custom_params: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    rle_mask: Vec<[u32; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    sparse_potential: Vec<(u16, f64)>,
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

    pub(super) fn serialize_tool_state(&self) -> Option<String> {
        if let (Some(spec), Some(bar)) = (M::formula_spec(), self.formula_bar.as_ref()) {
            let active = self.custom_formula_enabled;
            let customized = bar.raw_text != spec.default_expr;
            if active || customized {
                let params = self.params.read().unwrap_or_else(|e| e.into_inner());
                let sculpt = self.sculpt_state.read().unwrap_or_else(|e| e.into_inner());
                let sorted_params: BTreeMap<String, f64> =
                    params.iter().map(|(k, &v)| (k.clone(), v)).collect();
                let custom_params = if active {
                    bar.params_map()
                } else {
                    BTreeMap::new()
                };
                return serde_json::to_string(&UnifiedFormulaState {
                    params: sorted_params,
                    custom_formula_enabled: active,
                    custom_expr: bar.raw_text.clone(),
                    custom_params,
                    rle_mask: sculpt.rle_mask.clone(),
                    sparse_potential: sculpt.sparse_potential.clone(),
                })
                .ok();
            }
        }
        self.serialize_unified_state()
    }

    pub(super) fn deserialize_tool_state(&mut self, state: &str) -> Result<(), String> {
        if let Some(spec) = M::formula_spec() {
            if let Ok(fs) = serde_json::from_str::<UnifiedFormulaState>(state) {
                return self.apply_deserialized_formula_state(&spec, fs);
            }
        }
        self.deserialize_unified_state(state)
    }

    fn apply_deserialized_formula_state(
        &mut self,
        spec: &UnifiedFormulaSpec,
        fs: UnifiedFormulaState,
    ) -> Result<(), String> {
        let map: HashMap<String, f64> = fs.params.into_iter().collect();
        self.validate_param_map(&map)?;
        if fs.custom_params.values().any(|v| !v.is_finite()) {
            return Err(format!(
                "Non-finite custom formula parameter in {}",
                M::name()
            ));
        }
        validate_rle_mask_entries(&fs.rle_mask)?;
        validate_sparse_potential_entries(&fs.sparse_potential, M::name())?;
        {
            let mut guard = self.params.write().unwrap_or_else(|e| e.into_inner());
            for (k, v) in map {
                guard.insert(k, v);
            }
        }
        let bar = self
            .formula_bar
            .get_or_insert_with(|| build_formula_bar(spec));
        bar.set_expression(&fs.custom_expr);
        bar.set_params_map(&fs.custom_params);
        self.custom_formula_enabled = fs.custom_formula_enabled;
        self.publish_formula_update();
        {
            let mut s_guard = self.sculpt_state.write().unwrap_or_else(|e| e.into_inner());
            s_guard.rle_mask = fs.rle_mask;
            s_guard.sparse_potential = fs.sparse_potential;
        }
        self.texture = None;
        self.respawn_preserving_sculpt(None);
        Ok(())
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

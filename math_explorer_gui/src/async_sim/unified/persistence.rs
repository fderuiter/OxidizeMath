#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::sculpt::{validate_rle_mask_entries, validate_sparse_potential_entries};
use super::{CompiledFormulaPayload, UnifiedFormulaSpec, UnifiedModel, UnifiedSimTool};
use crate::framework::{InteractiveTool, MathExpressionInputBar, MAX_SYNTHESIZED_PARAMS};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

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
    #[serde(
        default,
        skip_serializing_if = "crate::audio::AcousticSettings::is_default"
    )]
    audio: crate::audio::AcousticSettings,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    rle_mask: Vec<[u32; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    sparse_potential: Vec<(u16, f64)>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    atlas_visible: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    atlas_axes: Option<[String; 2]>,
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
        let params = self.params.read().unwrap_or_else(|e| e.into_inner());
        if let Some(ext) = &self.ui_extension {
            if let Some(custom) = ext.save_extra_state(&params) {
                return Some(custom);
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
            return self.serialize_extended_state(&params, formula_ext, atlas_ext);
        }
        drop(params);
        self.serialize_unified_state()
    }

    fn serialize_extended_state(
        &self,
        params: &HashMap<String, f64>,
        formula_ext: Option<(bool, String, BTreeMap<String, f64>)>,
        atlas_ext: Option<(bool, [String; 2])>,
    ) -> Option<String> {
        let sculpt = self.sculpt_state.read().unwrap_or_else(|e| e.into_inner());
        let sorted_params = params.iter().map(|(k, &v)| (k.clone(), v)).collect();
        let (custom_formula_enabled, custom_expr, custom_params) = match formula_ext {
            Some((en, expr, cp)) => (en, Some(expr), cp),
            None => (false, None, BTreeMap::new()),
        };
        let (atlas_visible, atlas_axes) = match atlas_ext {
            Some((vis, axes)) => (vis, Some(axes)),
            None => (false, None),
        };
        let audio = if M::supports_sonification() {
            self.audio_settings
        } else {
            crate::audio::AcousticSettings::default()
        };
        serde_json::to_string(&UnifiedExtendedState {
            params: sorted_params,
            custom_formula_enabled,
            custom_expr,
            custom_params,
            audio,
            rle_mask: sculpt.rle_mask.clone(),
            sparse_potential: sculpt.sparse_potential.clone(),
            atlas_visible,
            atlas_axes,
        })
        .ok()
    }

    fn try_load_extension_state(&mut self, state: &str) -> Option<Result<(), String>> {
        let ext = self.ui_extension.as_mut()?;
        let mut next_params = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let res = ext.try_load_extra_state(state, &mut next_params)?;
        Some(res.map(|cmds| {
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
        self.deserialize_unified_state(state)
    }

    fn validate_sculpt_for_extended_state(
        &self,
        rle_mask: &[[u32; 2]],
        sparse_potential: &[(u16, f64)],
    ) -> Result<(), String> {
        if !rle_mask.is_empty() {
            if !(M::supports_spatial_brush() && M::is_raster_model()) {
                return Err(format!("{} does not support rle_mask state", M::name()));
            }
            validate_rle_mask_entries(rle_mask)?;
        }
        if !sparse_potential.is_empty() {
            if !(M::supports_spatial_brush() && !M::is_raster_model()) {
                return Err(format!(
                    "{} does not support sparse_potential state",
                    M::name()
                ));
            }
            validate_sparse_potential_entries(sparse_potential, M::name())?;
        }
        Ok(())
    }

    fn apply_extended_state(&mut self, ext: UnifiedExtendedState) -> Result<(), String> {
        self.validate_extended_capabilities(&ext)?;
        let mut map: HashMap<String, f64> = ext.params.into_iter().collect();
        let next_audio = self.resolve_extended_audio(&mut map, ext.audio)?;
        self.validate_param_map(&map)?;
        self.apply_parsed_atlas_state(Some((ext.atlas_visible, ext.atlas_axes)))?;
        {
            let mut guard = self.params.write().unwrap_or_else(|e| e.into_inner());
            for (k, v) in map {
                guard.insert(k, v);
            }
        }
        self.audio_settings = next_audio;
        self.apply_extended_formula(
            ext.custom_expr,
            &ext.custom_params,
            ext.custom_formula_enabled,
        );
        {
            let mut s_guard = self.sculpt_state.write().unwrap_or_else(|e| e.into_inner());
            s_guard.rle_mask = ext.rle_mask;
            s_guard.sparse_potential = ext.sparse_potential;
        }
        self.texture = None;
        self.respawn_preserving_sculpt(None);
        Ok(())
    }

    fn validate_extended_capabilities(&self, ext: &UnifiedExtendedState) -> Result<(), String> {
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
        self.validate_sculpt_for_extended_state(&ext.rle_mask, &ext.sparse_potential)
    }

    fn apply_extended_formula(
        &mut self,
        custom_expr: Option<String>,
        custom_params: &BTreeMap<String, f64>,
        enabled: bool,
    ) {
        if let Some(spec) = M::formula_spec() {
            if let Some(expr) = custom_expr {
                let bar = self
                    .formula_bar
                    .get_or_insert_with(|| build_formula_bar(&spec));
                bar.set_expression(&expr);
                bar.set_params_map(custom_params);
                self.custom_formula_enabled = enabled;
                self.publish_formula_update();
            } else {
                self.reset_formula_bar_to_default();
            }
        }
    }

    fn resolve_extended_audio(
        &self,
        map: &mut HashMap<String, f64>,
        ext_audio: crate::audio::AcousticSettings,
    ) -> Result<crate::audio::AcousticSettings, String> {
        let mut next_audio = self.extract_audio_from_map(map)?;
        if M::supports_sonification() {
            if !ext_audio.is_default() {
                next_audio = ext_audio;
            }
            next_audio.validate()?;
        } else if !ext_audio.is_default() {
            return Err(format!("Audio settings not supported for {}", M::name()));
        }
        Ok(next_audio)
    }

    pub(super) fn reset_formula_bar_to_default(&mut self) {
        self.custom_formula_enabled = false;
        if let (Some(spec), Some(bar)) = (M::formula_spec(), self.formula_bar.as_mut()) {
            bar.set_expression(spec.default_expr);
            bar.set_params_map(&BTreeMap::new());
            self.publish_formula_update();
        }
    }

    pub(super) fn extract_audio_from_map(
        &self,
        params: &mut HashMap<String, f64>,
    ) -> Result<crate::audio::AcousticSettings, String> {
        let mut next_audio = crate::audio::AcousticSettings::default();
        if M::supports_sonification() {
            if let Some(g) = params.remove("audio_gain") {
                next_audio.gain = g as f32;
            }
            if let Some(ps) = params.remove("audio_pitch_scale") {
                next_audio.pitch_scale = ps as f32;
            }
            next_audio.validate()?;
        }
        Ok(next_audio)
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

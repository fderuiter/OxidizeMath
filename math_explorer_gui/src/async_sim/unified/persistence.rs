#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{UnifiedModel, UnifiedSimTool};
use crate::async_sim::SimCommand;
use crate::framework::{MathExpressionInputBar, MAX_SYNTHESIZED_PARAMS};
use eframe::egui;
use math_commons::ast::CompiledExpr;
use std::collections::BTreeMap;
use std::ops::RangeInclusive;
use std::sync::Arc;

/// Specification for an opt-in custom formula input bar on a [`UnifiedModel`].
#[derive(Clone, Debug)]
pub struct UnifiedFormulaSpec {
    /// Left-hand side equation label (e.g. `"V(x)"`).
    pub lhs_label: &'static str,
    /// Initial default expression string.
    pub default_expr: &'static str,
    /// Coordinate axis variable names bound to leading bytecode slots (e.g. `&["x"]`).
    pub axes: &'static [&'static str],
    /// Curated `(display_name, expression)` presets.
    pub presets: &'static [(&'static str, &'static str)],
    /// Slider range bounds for auto-synthesized free parameters.
    pub param_range: RangeInclusive<f64>,
    /// Default value assigned to newly discovered free parameters.
    pub param_default: f64,
}

/// `Arc`-shared compiled formula payload streamed to [`super::UnifiedSimRunner`] without worker-side parsing.
#[derive(Clone, Debug)]
pub struct CompiledFormulaPayload {
    /// Monotonic revision incremented on formula or parameter slider edits.
    pub revision: u64,
    /// Compiled postfix AST bytecode.
    pub compiled: Arc<CompiledExpr>,
    /// Stack-allocated parameter values aligned with `compiled`.
    pub params: [f64; MAX_SYNTHESIZED_PARAMS],
    /// Number of active parameter slots in `params`.
    pub param_len: usize,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct UnifiedExtendedState {
    pub(super) params: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(super) custom_formula_enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) custom_expr: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub(super) custom_params: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub(super) custom_series: BTreeMap<String, Vec<f64>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) rle_mask: Vec<[u32; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) sparse_potential: Vec<(u16, f64)>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(super) atlas_visible: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) atlas_axes: Option<[String; 2]>,
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
        self.clear_stepped_state();
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

    pub(super) fn reset_formula_bar_to_default(&mut self) {
        self.custom_formula_enabled = false;
        if let (Some(spec), Some(bar)) = (M::formula_spec(), self.formula_bar.as_mut()) {
            bar.set_expression(spec.default_expr);
            bar.set_params_map(&BTreeMap::new());
            self.publish_formula_update();
        }
    }

    fn clamp_custom_series(&self) {
        let mut series = self
            .custom_series
            .write()
            .unwrap_or_else(|e| e.into_inner());
        series.retain(|key, values| {
            let Some(&(_, expected_len, min_val, max_val)) = M::custom_series_schema()
                .iter()
                .find(|(schema_key, _, _, _)| schema_key == key)
            else {
                return false;
            };
            if values.len() != expected_len {
                return false;
            }
            for v in values.iter_mut() {
                *v = if v.is_finite() {
                    crate::url_state::quantize_url_centi(*v, min_val, max_val)
                } else {
                    0.0
                };
            }
            true
        });
    }

    pub(super) fn validate_constrained_param(&self, key: &str, value: f64) -> Result<(), String> {
        let Some((_, constraint)) = self.param_metadata.iter().find(|meta| meta.0 == key) else {
            return Err(format!("Unknown parameter '{}' for {}", key, M::name()));
        };
        if !value.is_finite() {
            return Err(format!(
                "Non-finite value for parameter '{}' in {}",
                key,
                M::name()
            ));
        }
        if !(constraint.min..=constraint.max).contains(&value) {
            return Err(format!(
                "Parameter '{}' value {} is outside [{}, {}] bounds in {}",
                key,
                value,
                constraint.min,
                constraint.max,
                M::name()
            ));
        }
        Ok(())
    }

    pub(super) fn handle_recovery_action(
        &mut self,
        action: crate::framework::recovery::RecoveryAction,
        ctx: Option<&egui::Context>,
    ) {
        use crate::framework::recovery::RecoveryAction;
        match action {
            RecoveryAction::AutoScaleStepSize => {
                let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
                if let Some(dt) = p.get_mut("dt") {
                    *dt = (*dt * 0.5).max(1e-4);
                } else if let Some(visc) = p.get_mut("viscosity") {
                    *visc = (*visc * 0.5).max(0.00333);
                }
                drop(p);
                self.reset_simulation(ctx);
            }
            RecoveryAction::ClampParameters => {
                let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
                for (name, c) in &self.param_metadata {
                    if let Some(v) = p.get_mut(name) {
                        *v = if !v.is_finite() {
                            M::default_parameter(name, c)
                        } else {
                            v.clamp(c.min, c.max)
                        };
                    }
                }
                drop(p);
                self.clamp_custom_series();
                self.reset_simulation(ctx);
            }
            RecoveryAction::ResetParameters => {
                let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
                for (name, c) in &self.param_metadata {
                    let d = match name.as_str() {
                        "viscosity" => 0.02,
                        "dt" => 0.05,
                        "potential_type" => 0.0,
                        _ => M::default_parameter(name, c),
                    };
                    p.insert(name.clone(), d);
                }
                drop(p);
                self.custom_series
                    .write()
                    .unwrap_or_else(|e| e.into_inner())
                    .clear();
                if let Some(spec) = M::formula_spec() {
                    self.custom_formula_enabled = false;
                    self.formula_bar = Some(build_formula_bar(&spec));
                    self.publish_formula_update();
                }
                self.reset_simulation(ctx);
                let _ = self.controller.send_command(SimCommand::Start);
            }
        }
    }
}

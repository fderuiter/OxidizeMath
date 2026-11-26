#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Compact JSON and URL-shareable preset serialization and validation for [`LossLandscapeTool`].

use super::arena::ArenaConfig;
use super::data::{
    default_huber_delta, format_target_table, is_default_huber_delta, LossMetric, TargetDataMode,
    DEFAULT_TARGET_EXPR, DEFAULT_TARGET_SAMPLES, DEFAULT_TARGET_X_BOUNDS, MAX_TABLE_ROWS,
};
use super::presets::{PredictorMode, DEFAULT_LOSS_EXPR};
use super::surfaces::{
    LandscapeSurface, SculptBump, MAX_SCULPT_BUMPS, SCULPT_AMP_MAX, SCULPT_AMP_MIN,
    SCULPT_SIGMA_MAX, SCULPT_SIGMA_MIN,
};
use super::{
    default_arena_camera, default_sculpt_amplitude, default_sculpt_sigma, LandscapeViewMode,
    LossLandscapeTool, LOSS_PARAM_MAX_BOUND, LOSS_PARAM_MIN_BOUND,
};
use crate::camera::Camera3D;
use std::collections::BTreeMap;

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct LossLandscapeState {
    w_min: f64,
    w_max: f64,
    b_min: f64,
    b_max: f64,
    resolution: usize,
    #[serde(default, skip_serializing_if = "PredictorMode::is_linear")]
    predictor_mode: PredictorMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    custom_expr: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    custom_params: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "LossMetric::is_mse")]
    loss_metric: LossMetric,
    #[serde(
        default = "default_huber_delta",
        skip_serializing_if = "is_default_huber_delta"
    )]
    huber_delta: f64,
    #[serde(default, skip_serializing_if = "TargetDataMode::is_preset_linear")]
    target_mode: TargetDataMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target_expr: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    target_params: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target_x_bounds: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target_sample_count: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target_table: Option<Vec<[f64; 2]>>,
    #[serde(default)]
    surface: LandscapeSurface,
    #[serde(default)]
    view_mode: LandscapeViewMode,
    #[serde(default = "default_arena_camera")]
    camera: Camera3D,
    #[serde(default)]
    arena: ArenaConfig,
    #[serde(default = "default_sculpt_amplitude")]
    sculpt_amplitude: f64,
    #[serde(default = "default_sculpt_sigma")]
    sculpt_sigma: f64,
    #[serde(default)]
    sculpt_bumps: Vec<SculptBump>,
}

pub(super) fn save_tool_state(tool: &LossLandscapeTool) -> Option<String> {
    let is_custom = tool.predictor_mode == PredictorMode::Custom;
    let custom_expr = (is_custom || tool.math_input.raw_text != DEFAULT_LOSS_EXPR)
        .then(|| tool.math_input.raw_text.clone());
    let custom_params = if is_custom {
        tool.math_input.params_map()
    } else {
        BTreeMap::new()
    };
    let is_formula_target = tool.target_mode == TargetDataMode::Formula;
    let is_table_target = tool.target_mode == TargetDataMode::Table;
    serde_json::to_string(&LossLandscapeState {
        w_min: tool.w_min,
        w_max: tool.w_max,
        b_min: tool.b_min,
        b_max: tool.b_max,
        resolution: tool.resolution,
        predictor_mode: tool.predictor_mode,
        custom_expr,
        custom_params,
        loss_metric: tool.loss_metric,
        huber_delta: tool.huber_delta,
        target_mode: tool.target_mode,
        target_expr: is_formula_target.then(|| tool.target_formula_bar.raw_text.clone()),
        target_params: if is_formula_target {
            tool.target_formula_bar.params_map()
        } else {
            BTreeMap::new()
        },
        target_x_bounds: is_formula_target.then_some(tool.target_x_bounds),
        target_sample_count: is_formula_target.then_some(tool.target_sample_count),
        target_table: is_table_target.then(|| tool.cached_targets.clone()),
        surface: tool.surface,
        view_mode: tool.view_mode,
        camera: tool.camera,
        arena: tool.arena_config.clone(),
        sculpt_amplitude: tool.sculpt_amplitude,
        sculpt_sigma: tool.sculpt_sigma,
        sculpt_bumps: tool.sculpt_bumps.clone(),
    })
    .ok()
}

pub(super) fn try_load_tool_state(tool: &mut LossLandscapeTool, state: &str) -> Result<(), String> {
    let p: LossLandscapeState =
        serde_json::from_str(state).map_err(|e| format!("Invalid loss landscape state: {e}"))?;
    validate_bounds_and_expressions(&p)?;
    validate_target_and_arena(&p)?;

    (tool.w_min, tool.w_max, tool.b_min, tool.b_max) = (p.w_min, p.w_max, p.b_min, p.b_max);
    (tool.resolution, tool.surface, tool.view_mode) = (p.resolution, p.surface, p.view_mode);
    tool.predictor_mode = p.predictor_mode;
    tool.loss_metric = p.loss_metric;
    tool.huber_delta = p.huber_delta;
    tool.target_mode = p.target_mode;
    tool.math_input
        .set_expression(p.custom_expr.as_deref().unwrap_or(DEFAULT_LOSS_EXPR));
    tool.math_input.set_params_map(&p.custom_params);
    tool.target_formula_bar
        .set_expression(p.target_expr.as_deref().unwrap_or(DEFAULT_TARGET_EXPR));
    tool.target_formula_bar.set_params_map(&p.target_params);
    tool.target_x_bounds = p.target_x_bounds.unwrap_or(DEFAULT_TARGET_X_BOUNDS);
    tool.target_sample_count = p.target_sample_count.unwrap_or(DEFAULT_TARGET_SAMPLES);
    if let Some(rows) = p.target_table {
        tool.target_table_text = format_target_table(&rows);
    }
    tool.refresh_target_samples();
    tool.camera.apply_pose(p.camera);
    tool.arena_config = p.arena;
    (tool.sculpt_amplitude, tool.sculpt_sigma) = (p.sculpt_amplitude, p.sculpt_sigma);
    tool.sculpt_bumps = p
        .sculpt_bumps
        .into_iter()
        .map(|b| SculptBump::new(b.w, b.b, b.amplitude, b.sigma))
        .collect();
    tool.is_playing = false;
    (
        tool.slingshot_anchor,
        tool.slingshot_current,
        tool.last_brush_wb,
    ) = (None, None, None);
    tool.reset_arena();
    Ok(())
}

fn validate_param_map(params: &BTreeMap<String, f64>) -> bool {
    params.len() <= crate::framework::math_input::MAX_SYNTHESIZED_PARAMS
        && params
            .values()
            .all(|v| v.is_finite() && (-1e4..=1e4).contains(v))
}

fn validate_bounds_and_expressions(p: &LossLandscapeState) -> Result<(), String> {
    let bounds_ok = p.w_min.is_finite()
        && p.w_max.is_finite()
        && (LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&p.w_min)
        && (LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&p.w_max)
        && p.w_min < p.w_max
        && p.b_min.is_finite()
        && p.b_max.is_finite()
        && (LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&p.b_min)
        && (LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&p.b_max)
        && p.b_min < p.b_max
        && (10..=50).contains(&p.resolution)
        && p.huber_delta.is_finite()
        && (0.01..=50.0).contains(&p.huber_delta);
    let expr_ok = p.custom_expr.as_ref().is_none_or(|s| s.len() <= 512)
        && p.target_expr.as_ref().is_none_or(|s| s.len() <= 512)
        && validate_param_map(&p.custom_params)
        && validate_param_map(&p.target_params);
    if !bounds_ok || !expr_ok {
        return Err("Loss landscape parameter bounds, Huber delta, or expressions invalid".into());
    }
    Ok(())
}

fn validate_target_and_arena(p: &LossLandscapeState) -> Result<(), String> {
    if let Some([x0, x1]) = p.target_x_bounds {
        if !x0.is_finite()
            || !x1.is_finite()
            || !(LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&x0)
            || !(LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&x1)
            || x0 >= x1
        {
            return Err("Target x range invalid".into());
        }
    }
    if p.target_sample_count
        .is_some_and(|c| !(2..=MAX_TABLE_ROWS).contains(&c))
    {
        return Err("Target sample count out of valid range".into());
    }
    if let Some(ref rows) = p.target_table {
        if rows.is_empty()
            || rows.len() > MAX_TABLE_ROWS
            || rows
                .iter()
                .any(|[x, y]| !x.is_finite() || !y.is_finite() || x.abs() > 1e6 || y.abs() > 1e6)
        {
            return Err("Target table rows invalid".into());
        }
    }
    let arena_ok = p.camera.is_valid()
        && p.arena.is_valid()
        && p.sculpt_amplitude.is_finite()
        && (SCULPT_AMP_MIN..=SCULPT_AMP_MAX).contains(&p.sculpt_amplitude)
        && p.sculpt_sigma.is_finite()
        && (SCULPT_SIGMA_MIN..=SCULPT_SIGMA_MAX).contains(&p.sculpt_sigma)
        && p.sculpt_bumps.len() <= MAX_SCULPT_BUMPS
        && p.sculpt_bumps.iter().all(SculptBump::is_valid);
    if !arena_ok {
        return Err("Loss landscape arena or sculpt configuration invalid".into());
    }
    Ok(())
}

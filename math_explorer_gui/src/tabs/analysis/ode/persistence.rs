#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::OdeSolverTool;
use crate::framework::{CalloutManager, TraceSnapshotManager};
use crate::presets::OdePreset;
use math_explorer::pure_math::analysis::ode::IntegrationMethod;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

fn default_annotations_enabled() -> bool {
    true
}

fn is_false(v: &bool) -> bool {
    !*v
}

fn default_ode_custom_params(k: f64, r: f64, cap_k: f64) -> BTreeMap<String, f64> {
    BTreeMap::from([
        ("k".to_string(), k),
        ("r".to_string(), r),
        ("cap_k".to_string(), cap_k),
    ])
}

fn sync_one_legacy_param(
    math_input: &mut crate::framework::MathExpressionInputBar,
    name: &str,
    field: &mut f64,
    last_synced: &mut f64,
) {
    if !field.is_finite() {
        *last_synced = *field;
        return;
    }
    if field.to_bits() != last_synced.to_bits() {
        math_input.set_param(name, *field);
        *last_synced = *field;
    } else if let Some(val) = math_input.active_param_value(name) {
        *field = val;
        *last_synced = val;
    }
}

fn format_named_param_summary(names: &[String], vals: &[f64], dt: f64) -> String {
    if names.is_empty() {
        return format!(r"\Delta t={dt:.2}");
    }
    let mut parts: Vec<String> = names
        .iter()
        .zip(vals)
        .map(|(name, val)| format!("{name}={val:.2}"))
        .collect();
    parts.push(format!(r"\Delta t={dt:.2}"));
    parts.join(", ")
}

#[derive(serde::Serialize)]
struct OdeStateRef<'a> {
    preset: Option<OdePreset>,
    dt: f64,
    total_time: f64,
    param_k: f64,
    param_r: f64,
    param_cap_k: f64,
    ic_y0: f64,
    ic_v0: f64,
    custom_expr: &'a str,
    #[serde(skip_serializing_if = "is_false")]
    custom_coupled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    coupled_exprs: Option<[&'a str; 2]>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    custom_params: BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    coupled_params: BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "TraceSnapshotManager::is_empty")]
    snapshot_manager: &'a TraceSnapshotManager,
    callout_manager: &'a CalloutManager,
    method: IntegrationMethod,
    comparison_mode: bool,
    show_annotations: bool,
}

#[derive(serde::Serialize)]
struct OdeRecoveryStateRef<'a> {
    preset: Option<OdePreset>,
    dt: f64,
    total_time: f64,
    param_k: f64,
    param_r: f64,
    param_cap_k: f64,
    ic_y0: f64,
    ic_v0: f64,
    custom_expr: &'a str,
    #[serde(skip_serializing_if = "is_false")]
    custom_coupled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    coupled_exprs: Option<[&'a str; 2]>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    custom_params: BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    coupled_params: BTreeMap<String, f64>,
    callout_manager: &'a CalloutManager,
    method: IntegrationMethod,
    comparison_mode: bool,
    show_annotations: bool,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct OdeState {
    #[serde(default)]
    preset: Option<OdePreset>,
    dt: f64,
    total_time: f64,
    param_k: f64,
    param_r: f64,
    param_cap_k: f64,
    ic_y0: f64,
    ic_v0: f64,
    #[serde(default)]
    custom_expr: String,
    #[serde(default)]
    custom_coupled: bool,
    #[serde(default)]
    coupled_exprs: Option<[String; 2]>,
    #[serde(default)]
    custom_params: BTreeMap<String, f64>,
    #[serde(default)]
    coupled_params: BTreeMap<String, f64>,
    #[serde(default)]
    snapshot_manager: TraceSnapshotManager,
    #[serde(default)]
    callout_manager: CalloutManager,
    #[serde(default)]
    method: IntegrationMethod,
    #[serde(default)]
    comparison_mode: bool,
    #[serde(default = "default_annotations_enabled")]
    show_annotations: bool,
}

impl OdeSolverTool {
    pub(crate) fn seed_default_custom_params(&mut self) {
        self.math_input.set_params_map(&default_ode_custom_params(
            self.param_k,
            self.param_r,
            self.param_cap_k,
        ));
        self.synced_legacy_params = [self.param_k, self.param_r, self.param_cap_k];
    }

    pub(crate) fn reset_custom_parameters(&mut self) {
        self.param_k = 1.0;
        self.param_r = 1.0;
        self.param_cap_k = 10.0;
        self.ic_y0 = 1.0;
        self.seed_default_custom_params();
        self.math_input.set_expression("-k * y * sin(t)");
    }

    pub(crate) fn sync_legacy_params_from_math_input(&mut self) {
        if let Some(k) = self.math_input.active_param_value("k") {
            self.param_k = k;
        }
        if let Some(r) = self.math_input.active_param_value("r") {
            self.param_r = r;
        }
        if let Some(cap_k) = self.math_input.active_param_value("cap_k") {
            self.param_cap_k = cap_k;
        }
        self.synced_legacy_params = [self.param_k, self.param_r, self.param_cap_k];
    }

    pub(crate) fn sync_custom_params(&mut self) {
        sync_one_legacy_param(
            &mut self.math_input,
            "k",
            &mut self.param_k,
            &mut self.synced_legacy_params[0],
        );
        sync_one_legacy_param(
            &mut self.math_input,
            "r",
            &mut self.param_r,
            &mut self.synced_legacy_params[1],
        );
        sync_one_legacy_param(
            &mut self.math_input,
            "cap_k",
            &mut self.param_cap_k,
            &mut self.synced_legacy_params[2],
        );
    }

    pub(crate) fn custom_param_summary(&self) -> String {
        format_named_param_summary(
            self.math_input.active_param_names(),
            self.math_input.param_slice(),
            self.dt,
        )
    }

    pub(crate) fn coupled_param_summary(&self) -> String {
        format_named_param_summary(
            self.coupled_bank.param_names(),
            self.coupled_bank.param_slice(),
            self.dt,
        )
    }

    pub(crate) fn compute_state_revision(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.revision.hash(&mut hasher);
        (self.preset as u8).hash(&mut hasher);
        (self.method as u8).hash(&mut hasher);
        self.dt.to_bits().hash(&mut hasher);
        self.total_time.to_bits().hash(&mut hasher);
        self.param_k.to_bits().hash(&mut hasher);
        self.param_r.to_bits().hash(&mut hasher);
        self.param_cap_k.to_bits().hash(&mut hasher);
        self.ic_y0.to_bits().hash(&mut hasher);
        self.ic_v0.to_bits().hash(&mut hasher);
        self.math_input.raw_text.hash(&mut hasher);
        let param_slice = self.math_input.param_slice();
        param_slice.len().hash(&mut hasher);
        for &val in param_slice {
            val.to_bits().hash(&mut hasher);
        }
        self.custom_coupled.hash(&mut hasher);
        self.coupled_bank.raw_text(0).hash(&mut hasher);
        self.coupled_bank.raw_text(1).hash(&mut hasher);
        let coupled_slice = self.coupled_bank.param_slice();
        coupled_slice.len().hash(&mut hasher);
        for &v in coupled_slice {
            v.to_bits().hash(&mut hasher);
        }
        self.comparison_mode.hash(&mut hasher);
        self.show_annotations.hash(&mut hasher);
        self.snapshot_manager.len().hash(&mut hasher);
        self.snapshot_manager.history_len().hash(&mut hasher);
        self.callout_manager.len().hash(&mut hasher);
        hasher.finish().max(1)
    }
}

struct CustomSerializationMaps<'a> {
    coupled_exprs: Option<[&'a str; 2]>,
    custom_params: BTreeMap<String, f64>,
    coupled_params: BTreeMap<String, f64>,
}

impl OdeSolverTool {
    fn custom_serialization_maps(&self) -> CustomSerializationMaps<'_> {
        if self.preset != OdePreset::Custom {
            return CustomSerializationMaps {
                coupled_exprs: None,
                custom_params: BTreeMap::new(),
                coupled_params: BTreeMap::new(),
            };
        }
        if self.custom_coupled {
            let exprs = Some([self.coupled_bank.raw_text(0), self.coupled_bank.raw_text(1)]);
            CustomSerializationMaps {
                coupled_exprs: exprs,
                custom_params: BTreeMap::new(),
                coupled_params: self.coupled_bank.params_map(),
            }
        } else {
            CustomSerializationMaps {
                coupled_exprs: None,
                custom_params: self.math_input.params_map(),
                coupled_params: BTreeMap::new(),
            }
        }
    }

    pub(crate) fn serialize_full_state(&self) -> Option<String> {
        let CustomSerializationMaps {
            coupled_exprs,
            custom_params,
            coupled_params,
        } = self.custom_serialization_maps();
        serde_json::to_string(&OdeStateRef {
            preset: Some(self.preset),
            dt: self.dt,
            total_time: self.total_time,
            param_k: self.param_k,
            param_r: self.param_r,
            param_cap_k: self.param_cap_k,
            ic_y0: self.ic_y0,
            ic_v0: self.ic_v0,
            custom_expr: &self.math_input.raw_text,
            custom_coupled: self.custom_coupled,
            coupled_exprs,
            custom_params,
            coupled_params,
            snapshot_manager: &self.snapshot_manager,
            callout_manager: &self.callout_manager,
            method: self.method,
            comparison_mode: self.comparison_mode,
            show_annotations: self.show_annotations,
        })
        .ok()
    }

    pub(crate) fn serialize_recovery_state(&self) -> Option<String> {
        let CustomSerializationMaps {
            coupled_exprs,
            custom_params,
            coupled_params,
        } = self.custom_serialization_maps();
        serde_json::to_string(&OdeRecoveryStateRef {
            preset: Some(self.preset),
            dt: self.dt,
            total_time: self.total_time,
            param_k: self.param_k,
            param_r: self.param_r,
            param_cap_k: self.param_cap_k,
            ic_y0: self.ic_y0,
            ic_v0: self.ic_v0,
            custom_expr: &self.math_input.raw_text,
            custom_coupled: self.custom_coupled,
            coupled_exprs,
            custom_params,
            coupled_params,
            callout_manager: &self.callout_manager,
            method: self.method,
            comparison_mode: self.comparison_mode,
            show_annotations: self.show_annotations,
        })
        .ok()
    }

    pub(crate) fn deserialize_state(
        &mut self,
        state: &str,
        preserve_snapshots: bool,
    ) -> Result<(), String> {
        let s: OdeState =
            serde_json::from_str(state).map_err(|e| format!("Invalid OdeSolverTool state: {e}"))?;
        validate_deserialized_ode_state(&s)?;
        self.apply_deserialized_ode_state(s, preserve_snapshots);
        self.recalculate();
        Ok(())
    }

    fn apply_deserialized_ode_state(&mut self, s: OdeState, preserve_snapshots: bool) {
        if let Some(p) = s.preset {
            self.preset = p;
        }
        self.dt = s.dt;
        self.total_time = s.total_time;
        self.param_k = s.custom_params.get("k").copied().unwrap_or(s.param_k);
        self.param_r = s.custom_params.get("r").copied().unwrap_or(s.param_r);
        self.param_cap_k = s
            .custom_params
            .get("cap_k")
            .copied()
            .unwrap_or(s.param_cap_k);
        self.ic_y0 = s.ic_y0;
        self.ic_v0 = s.ic_v0;
        self.custom_coupled = s.custom_coupled;
        let mut merged_params =
            default_ode_custom_params(self.param_k, self.param_r, self.param_cap_k);
        for (k, &v) in &s.custom_params {
            merged_params.insert(k.clone(), v);
        }
        if !s.custom_expr.is_empty() {
            self.math_input.set_expression(&s.custom_expr);
        }
        self.math_input.set_params_map(&merged_params);
        self.synced_legacy_params = [self.param_k, self.param_r, self.param_cap_k];
        if let Some([e0, e1]) = s.coupled_exprs {
            self.coupled_bank.set_expressions(&[e0, e1]);
        }
        if !s.coupled_params.is_empty() {
            self.coupled_bank.set_params_map(&s.coupled_params);
        }
        if !preserve_snapshots {
            self.snapshot_manager = s.snapshot_manager;
        }
        self.callout_manager = s.callout_manager;
        self.method = s.method;
        self.comparison_mode = s.comparison_mode;
        self.show_annotations = s.show_annotations;
        self.pending_sweep_run = None;
    }

    pub(crate) fn collect_full_xy_pairs(&self) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        if self
            .series_cache
            .matches_series(&self.time_series, &self.y_series, &self.v_series)
        {
            let (py, pv) = self.series_cache.shared_points();
            return (py.to_vec(), pv.to_vec());
        }
        let mut plot_points_y = Vec::with_capacity(self.time_series.len());
        let mut plot_points_v = if self.v_series.is_empty() {
            Vec::new()
        } else {
            Vec::with_capacity(self.v_series.len())
        };
        for (i, &t) in self.time_series.iter().enumerate() {
            if let Some(&y) = self.y_series.get(i) {
                plot_points_y.push([t, y]);
            }
            if let Some(&v) = self.v_series.get(i) {
                plot_points_v.push([t, v]);
            }
        }
        (plot_points_y, plot_points_v)
    }

    pub(crate) fn build_current_trace_series(&self) -> Vec<crate::framework::TraceSeries> {
        if self
            .series_cache
            .matches_series(&self.time_series, &self.y_series, &self.v_series)
        {
            let (py, pv) = self.series_cache.shared_points();
            let mut series = vec![crate::framework::TraceSeries::new("y(t)", py)];
            if self.has_second_state() && !pv.is_empty() {
                series.push(crate::framework::TraceSeries::new(
                    self.second_state_label(),
                    pv,
                ));
            }
            return series;
        }
        let (plot_points_y, plot_points_v) = self.collect_full_xy_pairs();
        let mut series = vec![crate::framework::TraceSeries::new("y(t)", plot_points_y)];
        if self.has_second_state() && !plot_points_v.is_empty() {
            series.push(crate::framework::TraceSeries::new(
                self.second_state_label(),
                plot_points_v,
            ));
        }
        series
    }

    pub(crate) fn build_full_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut datasets = Vec::new();
        if self.comparison_mode && !self.comparison_manager.is_empty() {
            datasets.extend(self.comparison_manager.export_datasets());
        } else {
            let (plot_points_y, plot_points_v) = self.collect_full_xy_pairs();
            datasets.push(("y(t)".to_string(), plot_points_y));
            if self.has_second_state() && !plot_points_v.is_empty() {
                datasets.push((self.second_state_label().to_string(), plot_points_v));
            }
        }
        datasets.extend(self.snapshot_manager.export_datasets());
        for (name, pts) in egui_plot::get_last_plot_datasets() {
            if !datasets.iter().any(|(existing, _)| existing == &name) {
                datasets.push((name, pts));
            }
        }
        datasets
    }
}

fn validate_deserialized_ode_state(s: &OdeState) -> Result<(), String> {
    if !s.dt.is_finite()
        || s.dt <= 0.0
        || !s.total_time.is_finite()
        || s.total_time <= 0.0
        || !s.param_k.is_finite()
        || !s.param_r.is_finite()
        || !s.param_cap_k.is_finite()
        || !s.ic_y0.is_finite()
        || !s.ic_v0.is_finite()
        || !s.custom_params.values().all(|v| v.is_finite())
        || !s.coupled_params.values().all(|v| v.is_finite())
    {
        return Err("ODE parameters must be finite with dt > 0 and total_time > 0".to_string());
    }
    if crate::presets::checked_ode_step_count(s.dt, s.total_time).is_none() {
        return Err(format!(
            "ODE step count (total_time / dt) exceeds maximum allowed limit ({})",
            crate::presets::MAX_ODE_STEPS
        ));
    }
    Ok(())
}

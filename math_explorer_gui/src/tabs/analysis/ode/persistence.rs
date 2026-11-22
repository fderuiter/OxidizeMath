#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::OdeSolverTool;
use crate::framework::{CalloutManager, TraceSnapshotManager};
use crate::presets::OdePreset;
use math_explorer::pure_math::analysis::ode::IntegrationMethod;
use std::hash::{Hash, Hasher};

fn default_annotations_enabled() -> bool {
    true
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
    pub fn param_summary(&self) -> String {
        match self.preset {
            OdePreset::Exponential | OdePreset::HarmonicOscillator | OdePreset::Custom => {
                format!(r"k={:.2}, \Delta t={:.2}", self.param_k, self.dt)
            }
            OdePreset::LogisticGrowth => format!(
                r"r={:.2}, K={:.2}, \Delta t={:.2}",
                self.param_r, self.param_cap_k, self.dt
            ),
        }
    }

    pub fn reset_parameters(&mut self) {
        match self.preset {
            OdePreset::Exponential => {
                self.param_k = 1.0;
                self.ic_y0 = 1.0;
            }
            OdePreset::HarmonicOscillator => {
                self.param_k = 1.0;
                self.ic_y0 = 1.0;
                self.ic_v0 = 0.0;
            }
            OdePreset::LogisticGrowth => {
                self.param_r = 1.0;
                self.param_cap_k = 10.0;
                self.ic_y0 = 1.0;
            }
            OdePreset::Custom => {
                self.param_k = 1.0;
                self.ic_y0 = 1.0;
                self.math_input.set_expression("-k * y * sin(t)");
            }
        }
        self.dt = 0.05;
        self.total_time = 10.0;
        self.pending_sweep_run = None;
        self.recalculate();
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
        self.comparison_mode.hash(&mut hasher);
        self.show_annotations.hash(&mut hasher);
        self.snapshot_manager.len().hash(&mut hasher);
        self.snapshot_manager.history_len().hash(&mut hasher);
        self.callout_manager.len().hash(&mut hasher);
        hasher.finish().max(1)
    }

    pub(crate) fn serialize_full_state(&self) -> Option<String> {
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
            snapshot_manager: &self.snapshot_manager,
            callout_manager: &self.callout_manager,
            method: self.method,
            comparison_mode: self.comparison_mode,
            show_annotations: self.show_annotations,
        })
        .ok()
    }

    pub(crate) fn serialize_recovery_state(&self) -> Option<String> {
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
        if !s.dt.is_finite()
            || s.dt <= 0.0
            || !s.total_time.is_finite()
            || s.total_time <= 0.0
            || !s.param_k.is_finite()
            || !s.param_r.is_finite()
            || !s.param_cap_k.is_finite()
            || !s.ic_y0.is_finite()
            || !s.ic_v0.is_finite()
        {
            return Err("ODE parameters must be finite with dt > 0 and total_time > 0".to_string());
        }
        if crate::presets::checked_ode_step_count(s.dt, s.total_time).is_none() {
            return Err(format!(
                "ODE step count (total_time / dt) exceeds maximum allowed limit ({})",
                crate::presets::MAX_ODE_STEPS
            ));
        }
        if let Some(p) = s.preset {
            self.preset = p;
        }
        self.dt = s.dt;
        self.total_time = s.total_time;
        self.param_k = s.param_k;
        self.param_r = s.param_r;
        self.param_cap_k = s.param_cap_k;
        self.ic_y0 = s.ic_y0;
        self.ic_v0 = s.ic_v0;
        if !s.custom_expr.is_empty() {
            self.math_input.set_expression(&s.custom_expr);
        }
        if !preserve_snapshots {
            self.snapshot_manager = s.snapshot_manager;
        }
        self.callout_manager = s.callout_manager;
        self.method = s.method;
        self.comparison_mode = s.comparison_mode;
        self.show_annotations = s.show_annotations;
        self.pending_sweep_run = None;
        self.recalculate();
        Ok(())
    }
}

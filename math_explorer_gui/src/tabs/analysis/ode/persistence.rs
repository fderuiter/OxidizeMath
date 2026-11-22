#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::phase_plane::{
    DEFAULT_PLANAR_X_EXPR, DEFAULT_PLANAR_Y_EXPR, DEFAULT_QUIVER_GRID, MAX_QUIVER_GRID,
    MAX_SEEDED_ORBITS,
};
use super::poincare_map::DEFAULT_MATRIX_A;
use super::OdeSolverTool;
use crate::framework::{CalloutManager, TraceSnapshotManager};
use crate::presets::OdePreset;
use math_explorer::pure_math::analysis::ode::IntegrationMethod;
use std::hash::{Hash, Hasher};

fn default_true() -> bool {
    true
}

fn is_false(val: &bool) -> bool {
    !*val
}

fn is_true(val: &bool) -> bool {
    *val
}

fn default_planar_x_expr() -> String {
    DEFAULT_PLANAR_X_EXPR.to_string()
}

fn is_default_planar_x(val: &&str) -> bool {
    *val == DEFAULT_PLANAR_X_EXPR
}

fn default_planar_y_expr() -> String {
    DEFAULT_PLANAR_Y_EXPR.to_string()
}

fn is_default_planar_y(val: &&str) -> bool {
    *val == DEFAULT_PLANAR_Y_EXPR
}

fn default_matrix_a() -> [[f64; 2]; 2] {
    DEFAULT_MATRIX_A
}

fn is_default_matrix_a(val: &[[f64; 2]; 2]) -> bool {
    *val == DEFAULT_MATRIX_A
}

fn default_quiver_density() -> usize {
    DEFAULT_QUIVER_GRID
}

fn is_default_quiver_density(val: &usize) -> bool {
    *val == DEFAULT_QUIVER_GRID
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
    #[serde(skip_serializing_if = "is_default_planar_x")]
    planar_x_expr: &'a str,
    #[serde(skip_serializing_if = "is_default_planar_y")]
    planar_y_expr: &'a str,
    #[serde(skip_serializing_if = "is_default_matrix_a")]
    matrix_a: [[f64; 2]; 2],
    #[serde(skip_serializing_if = "is_false")]
    phase_plane_mode: bool,
    #[serde(skip_serializing_if = "is_true")]
    show_quiver: bool,
    #[serde(skip_serializing_if = "is_default_quiver_density")]
    quiver_density: usize,
    #[serde(skip_serializing_if = "is_false")]
    show_nullclines: bool,
    #[serde(skip_serializing_if = "slice_is_empty")]
    seeded_ic: &'a [[f64; 2]],
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
    #[serde(skip_serializing_if = "is_default_planar_x")]
    planar_x_expr: &'a str,
    #[serde(skip_serializing_if = "is_default_planar_y")]
    planar_y_expr: &'a str,
    #[serde(skip_serializing_if = "is_default_matrix_a")]
    matrix_a: [[f64; 2]; 2],
    #[serde(skip_serializing_if = "is_false")]
    phase_plane_mode: bool,
    #[serde(skip_serializing_if = "is_true")]
    show_quiver: bool,
    #[serde(skip_serializing_if = "is_default_quiver_density")]
    quiver_density: usize,
    #[serde(skip_serializing_if = "is_false")]
    show_nullclines: bool,
    #[serde(skip_serializing_if = "slice_is_empty")]
    seeded_ic: &'a [[f64; 2]],
    callout_manager: &'a CalloutManager,
    method: IntegrationMethod,
    comparison_mode: bool,
    show_annotations: bool,
}

fn slice_is_empty(s: &&[[f64; 2]]) -> bool {
    s.is_empty()
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
    #[serde(default = "default_planar_x_expr")]
    planar_x_expr: String,
    #[serde(default = "default_planar_y_expr")]
    planar_y_expr: String,
    #[serde(default = "default_matrix_a")]
    matrix_a: [[f64; 2]; 2],
    #[serde(default)]
    phase_plane_mode: bool,
    #[serde(default = "default_true")]
    show_quiver: bool,
    #[serde(default = "default_quiver_density")]
    quiver_density: usize,
    #[serde(default)]
    show_nullclines: bool,
    #[serde(default)]
    seeded_ic: Vec<[f64; 2]>,
    #[serde(default)]
    snapshot_manager: TraceSnapshotManager,
    #[serde(default)]
    callout_manager: CalloutManager,
    #[serde(default)]
    method: IntegrationMethod,
    #[serde(default)]
    comparison_mode: bool,
    #[serde(default = "default_true")]
    show_annotations: bool,
}

fn validate_loaded_state(s: &OdeState) -> Result<(), String> {
    if !s.dt.is_finite()
        || s.dt <= 0.0
        || !s.total_time.is_finite()
        || s.total_time <= 0.0
        || !s.param_k.is_finite()
        || !s.param_r.is_finite()
        || !s.param_cap_k.is_finite()
        || !s.ic_y0.is_finite()
        || !s.ic_v0.is_finite()
        || !s.matrix_a.iter().flatten().all(|v| v.is_finite())
    {
        return Err("ODE parameters must be finite with dt > 0 and total_time > 0".to_string());
    }
    if !(2..=MAX_QUIVER_GRID).contains(&s.quiver_density) {
        return Err(format!(
            "Quiver grid density ({}) must be between 2 and {MAX_QUIVER_GRID}",
            s.quiver_density
        ));
    }
    if s.seeded_ic.len() > MAX_SEEDED_ORBITS
        || !s
            .seeded_ic
            .iter()
            .all(|pt| pt[0].is_finite() && pt[1].is_finite())
    {
        return Err(format!(
            "Seeded orbits must be finite and contain at most {MAX_SEEDED_ORBITS} initial states"
        ));
    }
    if crate::presets::checked_ode_step_count(s.dt, s.total_time).is_none() {
        return Err(format!(
            "ODE step count (total_time / dt) exceeds maximum allowed limit ({})",
            crate::presets::MAX_ODE_STEPS
        ));
    }
    Ok(())
}

impl OdeSolverTool {
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
        self.planar_x_input.raw_text.hash(&mut hasher);
        self.planar_y_input.raw_text.hash(&mut hasher);
        for val in self.matrix_a.iter().flatten() {
            val.to_bits().hash(&mut hasher);
        }
        self.phase_plane_mode.hash(&mut hasher);
        self.show_quiver.hash(&mut hasher);
        self.quiver_density.hash(&mut hasher);
        self.show_nullclines.hash(&mut hasher);
        for &[sx, sy] in &self.seeded_ic {
            sx.to_bits().hash(&mut hasher);
            sy.to_bits().hash(&mut hasher);
        }
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
            planar_x_expr: &self.planar_x_input.raw_text,
            planar_y_expr: &self.planar_y_input.raw_text,
            matrix_a: self.matrix_a,
            phase_plane_mode: self.phase_plane_mode,
            show_quiver: self.show_quiver,
            quiver_density: self.quiver_density,
            show_nullclines: self.show_nullclines,
            seeded_ic: &self.seeded_ic,
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
            planar_x_expr: &self.planar_x_input.raw_text,
            planar_y_expr: &self.planar_y_input.raw_text,
            matrix_a: self.matrix_a,
            phase_plane_mode: self.phase_plane_mode,
            show_quiver: self.show_quiver,
            quiver_density: self.quiver_density,
            show_nullclines: self.show_nullclines,
            seeded_ic: &self.seeded_ic,
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
        validate_loaded_state(&s)?;

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
        self.planar_x_input.set_expression(&s.planar_x_expr);
        self.planar_y_input.set_expression(&s.planar_y_expr);
        self.matrix_a = s.matrix_a;
        self.phase_plane_mode = s.phase_plane_mode;
        self.show_quiver = s.show_quiver;
        self.quiver_density = s.quiver_density;
        self.show_nullclines = s.show_nullclines;
        self.seeded_ic = s.seeded_ic;
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

    pub(crate) fn collect_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut datasets = egui_plot::get_last_plot_datasets();
        if !datasets.is_empty() {
            return datasets;
        }
        let (t1_lbl, t2_lbl, px_lbl, py_lbl) = self.preset.state_labels();
        if self.is_phase_plane_active() {
            let phase_pts: Vec<[f64; 2]> = self
                .y_series
                .iter()
                .copied()
                .zip(self.v_series.iter().copied())
                .map(|(x, y)| [x, y])
                .collect();
            datasets.push((format!("Primary ({px_lbl}, {py_lbl})"), phase_pts));
        } else {
            let pts_y: Vec<[f64; 2]> = self
                .time_series
                .iter()
                .copied()
                .zip(self.y_series.iter().copied())
                .map(|(t, y)| [t, y])
                .collect();
            datasets.push((t1_lbl.to_string(), pts_y));
            if self.preset.is_2d() && !self.v_series.is_empty() {
                let pts_v: Vec<[f64; 2]> = self
                    .time_series
                    .iter()
                    .copied()
                    .zip(self.v_series.iter().copied())
                    .map(|(t, v)| [t, v])
                    .collect();
                datasets.push((t2_lbl.to_string(), pts_v));
            }
        }
        datasets.extend(self.snapshot_manager.export_datasets());
        if self.comparison_mode {
            datasets.extend(self.comparison_manager.export_datasets());
        }
        datasets
    }
}

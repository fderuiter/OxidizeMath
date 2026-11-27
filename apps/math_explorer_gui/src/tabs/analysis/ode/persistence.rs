#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::custom_spec::default_ode_custom_params;
use super::phase_plane::{DEFAULT_PLANAR_X_EXPR, DEFAULT_PLANAR_Y_EXPR, DEFAULT_QUIVER_GRID};
use super::poincare_map::DEFAULT_MATRIX_A;
use super::OdeSolverTool;
use crate::framework::{CalloutManager, TraceSnapshotManager};
use crate::presets::OdePreset;
use math_explorer::pure_math::analysis::ode::IntegrationMethod;
use std::collections::BTreeMap;
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

fn skip_empty_snapshots(mgr: &Option<&TraceSnapshotManager>) -> bool {
    mgr.is_none_or(TraceSnapshotManager::is_empty)
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

fn slice_is_empty(s: &&[[f64; 2]]) -> bool {
    s.is_empty()
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
    #[serde(skip_serializing_if = "is_false")]
    custom_coupled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    coupled_exprs: Option<[&'a str; 2]>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    custom_params: BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    coupled_params: BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    planar_params: BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "skip_empty_snapshots")]
    snapshot_manager: Option<&'a TraceSnapshotManager>,
    callout_manager: &'a CalloutManager,
    method: IntegrationMethod,
    comparison_mode: bool,
    show_annotations: bool,
    #[serde(skip_serializing_if = "super::phase::is_default_view_mode")]
    view_mode: super::phase::PlotViewMode,
    #[serde(skip_serializing_if = "super::phase::is_zero_t_slice")]
    t_slice: f64,
    #[serde(skip_serializing_if = "is_false")]
    atlas_visible: bool,
    #[serde(skip_serializing_if = "super::atlas::is_default_ode_atlas_axes")]
    atlas_axes: [String; 2],
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OdeState {
    #[serde(default)]
    pub(super) preset: Option<OdePreset>,
    pub(super) dt: f64,
    pub(super) total_time: f64,
    pub(super) param_k: f64,
    pub(super) param_r: f64,
    pub(super) param_cap_k: f64,
    pub(super) ic_y0: f64,
    pub(super) ic_v0: f64,
    #[serde(default)]
    pub(super) custom_expr: String,
    #[serde(default = "default_planar_x_expr")]
    pub(super) planar_x_expr: String,
    #[serde(default = "default_planar_y_expr")]
    pub(super) planar_y_expr: String,
    #[serde(default = "default_matrix_a")]
    pub(super) matrix_a: [[f64; 2]; 2],
    #[serde(default)]
    phase_plane_mode: bool,
    #[serde(default = "default_true")]
    show_quiver: bool,
    #[serde(default = "default_quiver_density")]
    pub(super) quiver_density: usize,
    #[serde(default)]
    show_nullclines: bool,
    #[serde(default)]
    pub(super) seeded_ic: Vec<[f64; 2]>,
    #[serde(default)]
    pub(super) custom_coupled: bool,
    #[serde(default)]
    pub(super) coupled_exprs: Option<[String; 2]>,
    #[serde(default)]
    pub(super) custom_params: BTreeMap<String, f64>,
    #[serde(default)]
    pub(super) coupled_params: BTreeMap<String, f64>,
    #[serde(default)]
    pub(super) planar_params: BTreeMap<String, f64>,
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
    #[serde(default)]
    view_mode: super::phase::PlotViewMode,
    #[serde(default)]
    pub(super) t_slice: f64,
    #[serde(default)]
    atlas_visible: bool,
    #[serde(default = "super::atlas::default_ode_atlas_axes")]
    pub(super) atlas_axes: [String; 2],
}

struct CustomSerializationMaps<'a> {
    coupled_exprs: Option<[&'a str; 2]>,
    custom_params: BTreeMap<String, f64>,
    coupled_params: BTreeMap<String, f64>,
    planar_params: BTreeMap<String, f64>,
}

fn hash_callout_entry(c: &crate::framework::CalloutData, hasher: &mut impl Hasher) {
    c.id.hash(hasher);
    c.target_point[0].to_bits().hash(hasher);
    c.target_point[1].to_bits().hash(hasher);
    c.label_offset[0].to_bits().hash(hasher);
    c.label_offset[1].to_bits().hash(hasher);
    c.text.hash(hasher);
    c.tag.hash(hasher);
    c.visible.hash(hasher);
    c.color_rgb.hash(hasher);
}

impl OdeSolverTool {
    fn hash_planar_ode_parameters(&self, hasher: &mut impl Hasher) {
        self.planar_x_input.raw_text.hash(hasher);
        self.planar_y_input.raw_text.hash(hasher);
        if self.preset == OdePreset::PlanarCustom {
            self.planar_bank.param_names().hash(hasher);
            let planar_slice = self.planar_bank.param_slice();
            planar_slice.len().hash(hasher);
            for &v in planar_slice {
                v.to_bits().hash(hasher);
            }
        }
        for val in self.matrix_a.iter().flatten() {
            val.to_bits().hash(hasher);
        }
        self.phase_plane_mode.hash(hasher);
        self.show_quiver.hash(hasher);
        self.quiver_density.hash(hasher);
        self.show_nullclines.hash(hasher);
        self.seeded_ic.len().hash(hasher);
        for &[sx, sy] in &self.seeded_ic {
            sx.to_bits().hash(hasher);
            sy.to_bits().hash(hasher);
        }
    }

    fn hash_shared_ode_parameters(&self, hasher: &mut impl Hasher) {
        (self.preset as u8).hash(hasher);
        (self.method as u8).hash(hasher);
        for val in [
            self.dt,
            self.total_time,
            self.param_k,
            self.param_r,
            self.param_cap_k,
            self.ic_y0,
            self.ic_v0,
        ] {
            val.to_bits().hash(hasher);
        }
        self.math_input.raw_text.hash(hasher);
        if self.preset == OdePreset::Custom {
            self.custom_coupled.hash(hasher);
            if self.custom_coupled {
                self.coupled_bank.raw_text(0).hash(hasher);
                self.coupled_bank.raw_text(1).hash(hasher);
                self.coupled_bank.param_names().hash(hasher);
                let coupled_slice = self.coupled_bank.param_slice();
                coupled_slice.len().hash(hasher);
                for &v in coupled_slice {
                    v.to_bits().hash(hasher);
                }
            } else {
                self.math_input.active_param_names().hash(hasher);
                let param_slice = self.math_input.param_slice();
                param_slice.len().hash(hasher);
                for &val in param_slice {
                    val.to_bits().hash(hasher);
                }
            }
        }
        self.hash_planar_ode_parameters(hasher);
        self.comparison_mode.hash(hasher);
        self.show_annotations.hash(hasher);
        (self.view_mode as u8).hash(hasher);
        self.t_slice.to_bits().hash(hasher);
        self.atlas.is_visible().hash(hasher);
        self.atlas.axis_keys().hash(hasher);
        let callouts = self.callout_manager.callouts();
        callouts.len().hash(hasher);
        for c in callouts {
            hash_callout_entry(c, hasher);
        }
    }

    pub(crate) fn compute_state_revision(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.revision.hash(&mut hasher);
        self.hash_shared_ode_parameters(&mut hasher);
        self.snapshot_manager.len().hash(&mut hasher);
        self.snapshot_manager.history_len().hash(&mut hasher);
        hasher.finish().max(1)
    }

    pub(crate) fn compute_url_state_revision(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash_shared_ode_parameters(&mut hasher);
        hasher.finish().max(1)
    }

    fn custom_serialization_maps(&self) -> CustomSerializationMaps<'_> {
        if self.preset == OdePreset::PlanarCustom {
            let (dx, _, vals, len) = self.planar_bank.snapshot_compiled_and_params(
                &self.planar_x_input,
                &self.planar_y_input,
                [self.param_k, self.param_r, self.param_cap_k],
                self.synced_legacy_params,
            );
            let mut planar_params = BTreeMap::new();
            for (name, &val) in dx.param_names().iter().take(len).zip(&vals[..len]) {
                if val.is_finite() {
                    planar_params.insert(name.clone(), val.clamp(-1e4, 1e4));
                }
            }
            return CustomSerializationMaps {
                coupled_exprs: None,
                custom_params: BTreeMap::new(),
                coupled_params: BTreeMap::new(),
                planar_params,
            };
        }
        if self.preset != OdePreset::Custom {
            return CustomSerializationMaps {
                coupled_exprs: None,
                custom_params: BTreeMap::new(),
                coupled_params: BTreeMap::new(),
                planar_params: BTreeMap::new(),
            };
        }
        if self.custom_coupled {
            CustomSerializationMaps {
                coupled_exprs: Some([self.coupled_bank.raw_text(0), self.coupled_bank.raw_text(1)]),
                custom_params: BTreeMap::new(),
                coupled_params: self.coupled_bank.params_map(),
                planar_params: BTreeMap::new(),
            }
        } else {
            CustomSerializationMaps {
                coupled_exprs: None,
                custom_params: self.math_input.params_map(),
                coupled_params: BTreeMap::new(),
                planar_params: BTreeMap::new(),
            }
        }
    }

    fn serialize_ode_state(&self, include_snapshots: bool) -> Option<String> {
        let CustomSerializationMaps {
            coupled_exprs,
            custom_params,
            coupled_params,
            planar_params,
        } = self.custom_serialization_maps();
        let (ax, ay) = self.atlas.axis_keys();
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
            custom_coupled: self.custom_coupled,
            coupled_exprs,
            custom_params,
            coupled_params,
            planar_params,
            snapshot_manager: include_snapshots.then_some(&self.snapshot_manager),
            callout_manager: &self.callout_manager,
            method: self.method,
            comparison_mode: self.comparison_mode,
            show_annotations: self.show_annotations,
            view_mode: self.view_mode,
            t_slice: self.t_slice,
            atlas_visible: self.atlas.is_visible(),
            atlas_axes: [ax.to_string(), ay.to_string()],
        })
        .ok()
    }

    pub(crate) fn serialize_full_state(&self) -> Option<String> {
        self.serialize_ode_state(true)
    }

    pub(crate) fn serialize_recovery_state(&self) -> Option<String> {
        self.serialize_ode_state(false)
    }

    fn apply_loaded_expressions(&mut self, s: &OdeState) {
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
        if let Some(ref exprs) = s.coupled_exprs {
            self.coupled_bank.set_expressions(exprs);
        }
        if !s.coupled_params.is_empty() {
            self.coupled_bank.set_params_map(&s.coupled_params);
        }
        self.planar_x_input.set_expression(&s.planar_x_expr);
        self.planar_y_input.set_expression(&s.planar_y_expr);
        let mut merged_planar =
            default_ode_custom_params(self.param_k, self.param_r, self.param_cap_k);
        for (k, &v) in &s.planar_params {
            merged_planar.insert(k.clone(), v);
        }
        self.planar_bank
            .load_state(&self.planar_x_input, &self.planar_y_input, &merged_planar);
    }

    pub(crate) fn deserialize_state(
        &mut self,
        state: &str,
        preserve_snapshots: bool,
    ) -> Result<(), String> {
        let s: OdeState =
            serde_json::from_str(state).map_err(|e| format!("Invalid OdeSolverTool state: {e}"))?;
        self.validate_deserialized_ode_state(&s)?;
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
        let legacy_map = if self.preset == OdePreset::PlanarCustom {
            &s.planar_params
        } else {
            &s.custom_params
        };
        self.param_k = legacy_map.get("k").copied().unwrap_or(s.param_k);
        self.param_r = legacy_map.get("r").copied().unwrap_or(s.param_r);
        self.param_cap_k = legacy_map
            .get("cap_k")
            .or_else(|| legacy_map.get("K"))
            .copied()
            .unwrap_or(s.param_cap_k);
        self.ic_y0 = s.ic_y0;
        self.ic_v0 = s.ic_v0;
        self.apply_loaded_expressions(&s);
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
        self.view_mode = s.view_mode;
        self.t_slice = s.t_slice.clamp(0.0, self.total_time);
        let _ = self.atlas.set_axis_keys(&s.atlas_axes[0], &s.atlas_axes[1]);
        self.atlas.set_visible(s.atlas_visible);
        self.phase_cursor = None;
        self.pending_sweep_run = None;
    }
}

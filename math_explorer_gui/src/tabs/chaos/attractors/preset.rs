#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::twin::{
    is_default_initial_state, is_default_twin_epsilon, TwinConfig, DEFAULT_INITIAL_STATE,
    DEFAULT_TWIN_EPSILON, MAX_INITIAL_COORD, MAX_TWIN_EPSILON, MIN_INITIAL_COORD, MIN_TWIN_EPSILON,
};
use super::{
    AttractorMode, AttractorPlotter, SolverExecutionState, DEFAULT_FIXED_CALLOUT_OFFSETS,
    DEFAULT_FLOW_EXPRS,
};
use crate::framework::Camera3D;
use eframe::egui;
use nalgebra::Vector3;
use std::collections::BTreeMap;

/// JSON preset envelope for Lorenz Butterfly Effect Twin Trajectory divergence.
pub const PRESET_3D_ATTRACTOR_TWIN: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Lorenz Butterfly Effect - Twin Trajectories",
    "domain": "chaos",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "sigma": 10.0,
    "rho": 28.0,
    "beta": 2.6666666666666665,
    "dt": 0.01,
    "simulation_speed": 8,
    "execution_state": "Running",
    "show_annotations": true,
    "initial_state": [1.0, 1.0, 1.0],
    "twin_enabled": true,
    "epsilon": 1e-9,
    "camera": {
      "pitch": 0.35,
      "yaw": -0.45,
      "zoom": 1.0
    }
  }
}"#;

const fn default_annotations_enabled() -> bool {
    true
}

const fn default_tube_radius() -> f64 {
    super::export::DEFAULT_TUBE_RADIUS
}

const fn default_initial_state() -> [f64; 3] {
    DEFAULT_INITIAL_STATE
}

const fn default_twin_epsilon() -> f64 {
    DEFAULT_TWIN_EPSILON
}

const fn is_false(v: &bool) -> bool {
    !*v
}

fn is_default_fixed_callout_offsets(v: &[[f32; 2]; 2]) -> bool {
    (0..2).all(|i| {
        (0..2).all(|j| (v[i][j] - DEFAULT_FIXED_CALLOUT_OFFSETS[i][j]).abs() <= f32::EPSILON)
    })
}

#[derive(serde::Serialize)]
struct AttractorSaveState {
    sigma: f64,
    rho: f64,
    beta: f64,
    dt: f64,
    tube_radius: f64,
    simulation_speed: usize,
    execution_state: SolverExecutionState,
    show_annotations: bool,
    #[serde(skip_serializing_if = "is_default_fixed_callout_offsets")]
    fixed_callout_offsets: [[f32; 2]; 2],
    camera: Camera3D,
    #[serde(skip_serializing_if = "AttractorMode::is_lorenz")]
    mode: AttractorMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    custom_exprs: Option<[String; 3]>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    custom_params: BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "is_default_initial_state")]
    initial_state: [f64; 3],
    #[serde(skip_serializing_if = "is_false")]
    twin_enabled: bool,
    #[serde(skip_serializing_if = "is_default_twin_epsilon")]
    epsilon: f64,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    atlas_visible: bool,
    #[serde(skip_serializing_if = "super::atlas::is_default_attractor_atlas_axes")]
    atlas_axes: [String; 2],
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct AttractorState {
    sigma: f64,
    rho: f64,
    beta: f64,
    dt: f64,
    #[serde(default = "default_tube_radius")]
    tube_radius: f64,
    simulation_speed: usize,
    #[serde(default)]
    execution_state: Option<SolverExecutionState>,
    #[serde(default = "default_annotations_enabled")]
    show_annotations: bool,
    #[serde(default)]
    fixed_callout_offsets: Option<[[f32; 2]; 2]>,
    #[serde(default)]
    paused: Option<bool>,
    #[serde(default)]
    camera: Option<Camera3D>,
    #[serde(default)]
    pitch: Option<f32>,
    #[serde(default)]
    yaw: Option<f32>,
    #[serde(default)]
    zoom: Option<f32>,
    #[serde(default)]
    mode: AttractorMode,
    #[serde(default)]
    custom_exprs: Option<[String; 3]>,
    #[serde(default)]
    custom_params: BTreeMap<String, f64>,
    #[serde(default = "default_initial_state")]
    initial_state: [f64; 3],
    #[serde(default)]
    twin_enabled: bool,
    #[serde(default = "default_twin_epsilon")]
    epsilon: f64,
    #[serde(default)]
    atlas_visible: bool,
    #[serde(default = "super::atlas::default_attractor_atlas_axes")]
    atlas_axes: [String; 2],
}

impl AttractorPlotter {
    pub(super) fn compute_state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.mode as u8).hash(&mut h);
        for bar in self.flow_bank.bars() {
            bar.raw_text.hash(&mut h);
        }
        for &pv in self.flow_bank.param_slice() {
            pv.to_bits().hash(&mut h);
        }
        self.system.sigma.to_bits().hash(&mut h);
        self.system.rho.to_bits().hash(&mut h);
        self.system.beta.to_bits().hash(&mut h);
        self.twin.initial_state.x.to_bits().hash(&mut h);
        self.twin.initial_state.y.to_bits().hash(&mut h);
        self.twin.initial_state.z.to_bits().hash(&mut h);
        self.twin.enabled.hash(&mut h);
        self.twin.epsilon.to_bits().hash(&mut h);
        self.dt.to_bits().hash(&mut h);
        self.tube_radius.to_bits().hash(&mut h);
        self.simulation_speed.hash(&mut h);
        self.max_points.hash(&mut h);
        self.is_paused().hash(&mut h);
        self.diverged().hash(&mut h);
        self.show_annotations.hash(&mut h);
        self.atlas.is_visible().hash(&mut h);
        self.atlas.axis_keys().hash(&mut h);
        for offset in &self.fixed_callout_offsets {
            let x_bits = if offset.x == 0.0 {
                0
            } else {
                offset.x.to_bits()
            };
            let y_bits = if offset.y == 0.0 {
                0
            } else {
                offset.y.to_bits()
            };
            x_bits.hash(&mut h);
            y_bits.hash(&mut h);
        }
        self.camera.hash_state(&mut h);
        h.finish().max(1)
    }

    pub(super) fn serialize_state(&self) -> Option<String> {
        let expr_vec = self.flow_bank.expressions();
        let nondefault_exprs = expr_vec
            .iter()
            .zip(DEFAULT_FLOW_EXPRS.iter())
            .any(|(a, b)| a != b);
        let custom_exprs = (self.mode == AttractorMode::Custom || nondefault_exprs).then(|| {
            [
                expr_vec[0].clone(),
                expr_vec[1].clone(),
                expr_vec[2].clone(),
            ]
        });
        let custom_params = if self.mode == AttractorMode::Custom {
            self.flow_bank.params_map()
        } else {
            BTreeMap::new()
        };
        let (xk, yk) = self.atlas.axis_keys();
        serde_json::to_string(&AttractorSaveState {
            sigma: self.system.sigma,
            rho: self.system.rho,
            beta: self.system.beta,
            dt: self.dt,
            tube_radius: self.tube_radius,
            simulation_speed: self.simulation_speed,
            execution_state: self.execution_state,
            show_annotations: self.show_annotations,
            fixed_callout_offsets: self.fixed_callout_offsets.map(|v| [v.x, v.y]),
            camera: self.camera,
            mode: self.mode,
            custom_exprs,
            custom_params,
            initial_state: self.twin.initial_array(),
            twin_enabled: self.twin.enabled,
            epsilon: self.twin.epsilon,
            atlas_visible: self.atlas.is_visible(),
            atlas_axes: [xk.to_string(), yk.to_string()],
        })
        .ok()
    }

    pub(super) fn deserialize_state(&mut self, state: &str) -> Result<(), String> {
        <Self as crate::framework::InteractiveTool>::state_bounds(self).validate_json(state)?;
        let s: AttractorState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid AttractorPlotter state: {e}"))?;
        let offsets = validate_attractor_state(&s)?;
        let camera = Camera3D::from_preset_fields(
            s.camera,
            s.pitch,
            s.yaw,
            s.zoom,
            Some(Camera3D::new(0.0, 0.0, 1.0)),
        )?;

        self.mode = s.mode;
        self.apply_custom_flow_state(s.custom_exprs.as_ref(), &s.custom_params);
        self.system.sigma = s.sigma;
        self.system.rho = s.rho;
        self.system.beta = s.beta;
        self.dt = s.dt;
        self.tube_radius = s.tube_radius;
        self.simulation_speed = s.simulation_speed;
        self.show_annotations = s.show_annotations;
        self.fixed_callout_offsets = offsets.map(egui::Vec2::from);
        self.camera.apply_pose(camera);
        self.twin = TwinConfig {
            initial_state: Vector3::new(s.initial_state[0], s.initial_state[1], s.initial_state[2]),
            enabled: s.twin_enabled,
            epsilon: s.epsilon,
        };
        self.dragged_callout = None;
        self.atlas.set_visible(s.atlas_visible);
        let _ = self.atlas.set_axis_keys(&s.atlas_axes[0], &s.atlas_axes[1]);
        self.reset();
        let _ = self.worker.update();
        let execution_state = s.execution_state.unwrap_or(if s.paused == Some(true) {
            SolverExecutionState::Paused
        } else {
            SolverExecutionState::Ready
        });
        self.execution_state = execution_state;
        if let SolverExecutionState::Diverged {
            invalid_vec,
            diverged_at_step,
        } = execution_state
        {
            self.system.state.vec = invalid_vec;
            self.step_count = diverged_at_step;
        }
        Ok(())
    }

    fn apply_custom_flow_state(
        &mut self,
        custom_exprs: Option<&[String; 3]>,
        custom_params: &BTreeMap<String, f64>,
    ) {
        if let Some(exprs) = custom_exprs {
            if !custom_params.is_empty() {
                self.flow_bank.load_state(exprs, custom_params);
            } else {
                self.flow_bank.set_expressions(exprs);
            }
        } else {
            self.flow_bank = super::default_flow_bank();
            if !custom_params.is_empty() {
                self.flow_bank.set_params_map(custom_params);
            }
        }
    }
}

fn validate_attractor_state(s: &AttractorState) -> Result<[[f32; 2]; 2], String> {
    let coords_valid = s
        .initial_state
        .iter()
        .all(|v| v.is_finite() && (MIN_INITIAL_COORD..=MAX_INITIAL_COORD).contains(v));
    let eps_valid =
        s.epsilon.is_finite() && (MIN_TWIN_EPSILON..=MAX_TWIN_EPSILON).contains(&s.epsilon);
    if !s.sigma.is_finite()
        || !s.rho.is_finite()
        || !s.beta.is_finite()
        || !s.dt.is_finite()
        || s.dt <= 0.0
        || !s.tube_radius.is_finite()
        || !(super::export::MIN_TUBE_RADIUS..=super::export::MAX_TUBE_RADIUS)
            .contains(&s.tube_radius)
        || !(1..=super::MAX_SIMULATION_SPEED).contains(&s.simulation_speed)
        || s.custom_params.values().any(|v| !v.is_finite())
        || !coords_valid
        || !eps_valid
    {
        return Err(format!(
            "AttractorPlotter parameters must be finite with dt > 0, simulation_speed in 1..={}, initial_state in [{MIN_INITIAL_COORD}, {MAX_INITIAL_COORD}], and epsilon in [{MIN_TWIN_EPSILON:e}, {MAX_TWIN_EPSILON:e}]",
            super::MAX_SIMULATION_SPEED
        ));
    }
    let offsets = s
        .fixed_callout_offsets
        .unwrap_or(DEFAULT_FIXED_CALLOUT_OFFSETS);
    if offsets.iter().flatten().any(|c| !c.is_finite()) {
        return Err("AttractorPlotter fixed_callout_offsets must be finite".to_string());
    }
    Ok(offsets)
}

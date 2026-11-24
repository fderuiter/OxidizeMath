#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Preset serialization and `StarterPresetMetadata` cards for [`super::AttractorPlotter`].

use super::twin_state::{
    AttractorSystemKind, ExtraChaosParams, DEFAULT_EPSILON, DEFAULT_PERTURBATION_EXPR,
};
use super::{
    AttractorForkDescriptor, AttractorMode, AttractorPlotter, SolverExecutionState,
    DEFAULT_FIXED_CALLOUT_OFFSETS, DEFAULT_FLOW_EXPRS, MAX_SIMULATION_SPEED,
};
use crate::framework::Camera3D;
use crate::starter_gallery::StarterPresetMetadata;
use eframe::egui;
use std::collections::BTreeMap;

const fn default_annotations_enabled() -> bool {
    true
}

const fn default_initial_state() -> [f64; 3] {
    [1.0, 1.0, 1.0]
}

const fn default_shadow_enabled() -> bool {
    true
}

const fn default_epsilon() -> f64 {
    DEFAULT_EPSILON
}

fn default_perturbation_expr() -> String {
    DEFAULT_PERTURBATION_EXPR.to_string()
}

const fn default_tube_radius() -> f64 {
    super::export::DEFAULT_TUBE_RADIUS
}

fn is_default_tube_radius(r: &f64) -> bool {
    (*r - super::export::DEFAULT_TUBE_RADIUS).abs() < 1e-12
}

fn is_default_fixed_callout_offsets(v: &[[f32; 2]; 2]) -> bool {
    (0..2).all(|i| {
        (0..2).all(|j| (v[i][j] - DEFAULT_FIXED_CALLOUT_OFFSETS[i][j]).abs() <= f32::EPSILON)
    })
}

#[derive(serde::Serialize)]
struct AttractorSaveState<'a> {
    sigma: f64,
    rho: f64,
    beta: f64,
    dt: f64,
    #[serde(skip_serializing_if = "is_default_tube_radius")]
    tube_radius: f64,
    simulation_speed: usize,
    execution_state: SolverExecutionState,
    show_annotations: bool,
    #[serde(skip_serializing_if = "is_default_fixed_callout_offsets")]
    fixed_callout_offsets: [[f32; 2]; 2],
    camera: Camera3D,
    #[serde(skip_serializing_if = "crate::audio::AcousticSettings::is_default")]
    audio: crate::audio::AcousticSettings,
    #[serde(skip_serializing_if = "AttractorMode::is_lorenz")]
    mode: AttractorMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    custom_exprs: Option<[String; 3]>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    custom_params: BTreeMap<String, f64>,
    system_kind: AttractorSystemKind,
    initial_state: [f64; 3],
    shadow_enabled: bool,
    epsilon: f64,
    perturbation_expr: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    rossler_params: Option<[f64; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pendulum_params: Option<[f64; 5]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fork: Option<AttractorForkDescriptor>,
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
    audio: crate::audio::AcousticSettings,
    #[serde(default)]
    mode: AttractorMode,
    #[serde(default)]
    custom_exprs: Option<[String; 3]>,
    #[serde(default)]
    custom_params: BTreeMap<String, f64>,
    #[serde(default)]
    system_kind: AttractorSystemKind,
    #[serde(default = "default_initial_state")]
    initial_state: [f64; 3],
    #[serde(default = "default_shadow_enabled")]
    shadow_enabled: bool,
    #[serde(default = "default_epsilon")]
    epsilon: f64,
    #[serde(default = "default_perturbation_expr")]
    perturbation_expr: String,
    #[serde(default)]
    rossler_params: Option<[f64; 3]>,
    #[serde(default)]
    pendulum_params: Option<[f64; 5]>,
    #[serde(default)]
    fork: Option<AttractorForkDescriptor>,
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
        self.system_kind.hash(&mut h);
        self.system.sigma.to_bits().hash(&mut h);
        self.system.rho.to_bits().hash(&mut h);
        self.system.beta.to_bits().hash(&mut h);
        self.dt.to_bits().hash(&mut h);
        self.tube_radius.to_bits().hash(&mut h);
        self.simulation_speed.hash(&mut h);
        self.max_points.hash(&mut h);
        self.shadow_enabled.hash(&mut h);
        self.epsilon.to_bits().hash(&mut h);
        self.perturbation_expr.hash(&mut h);
        for c in self.initial_state {
            c.to_bits().hash(&mut h);
        }
        self.is_paused().hash(&mut h);
        self.diverged().hash(&mut h);
        self.show_annotations.hash(&mut h);
        self.atlas.is_visible().hash(&mut h);
        self.atlas.axis_keys().hash(&mut h);
        for offset in &self.fixed_callout_offsets {
            let xb = if offset.x == 0.0 {
                0
            } else {
                offset.x.to_bits()
            };
            let yb = if offset.y == 0.0 {
                0
            } else {
                offset.y.to_bits()
            };
            (xb, yb).hash(&mut h);
        }
        self.camera.hash_state(&mut h);
        self.hash_twin_revision(&mut h);
        h.finish().max(1)
    }

    pub(super) fn serialize_state(&self) -> Option<String> {
        let fork = self.twin.as_ref().map(|t| t.to_descriptor(self.step_count));
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
        let rossler_params = (self.system_kind == AttractorSystemKind::Rossler).then_some([
            self.extra_params.rossler_a,
            self.extra_params.rossler_b,
            self.extra_params.rossler_c,
        ]);
        let pendulum_params =
            (self.system_kind == AttractorSystemKind::DoublePendulum).then_some([
                self.extra_params.pendulum_m1,
                self.extra_params.pendulum_m2,
                self.extra_params.pendulum_l1,
                self.extra_params.pendulum_l2,
                self.extra_params.pendulum_g,
            ]);
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
            audio: self.audio_settings,
            mode: self.mode,
            custom_exprs,
            custom_params,
            system_kind: self.system_kind,
            initial_state: self.initial_state,
            shadow_enabled: self.shadow_enabled,
            epsilon: self.epsilon,
            perturbation_expr: &self.perturbation_expr,
            rossler_params,
            pendulum_params,
            fork,
            atlas_visible: self.atlas.is_visible(),
            atlas_axes: [xk.to_string(), yk.to_string()],
        })
        .ok()
    }

    fn validate_deserialized_state(
        s: &AttractorState,
    ) -> Result<(ExtraChaosParams, [[f32; 2]; 2]), String> {
        s.audio.validate()?;
        if !s.sigma.is_finite()
            || !s.rho.is_finite()
            || !s.beta.is_finite()
            || !s.dt.is_finite()
            || s.dt <= 0.0
            || !s.tube_radius.is_finite()
            || !(super::export::MIN_TUBE_RADIUS..=super::export::MAX_TUBE_RADIUS)
                .contains(&s.tube_radius)
            || !(1..=MAX_SIMULATION_SPEED).contains(&s.simulation_speed)
            || s.custom_params.values().any(|v| !v.is_finite())
        {
            return Err(format!(
                "AttractorPlotter parameters must be finite with dt > 0 and simulation_speed in 1..={MAX_SIMULATION_SPEED}"
            ));
        }
        let offsets = s
            .fixed_callout_offsets
            .unwrap_or(DEFAULT_FIXED_CALLOUT_OFFSETS);
        if offsets.iter().flatten().any(|c| !c.is_finite()) {
            return Err("AttractorPlotter fixed_callout_offsets must be finite".into());
        }
        if s.initial_state.iter().any(|c| !c.is_finite())
            || !s.epsilon.is_finite()
            || !(0.0..=1.0).contains(&s.epsilon)
            || s.perturbation_expr.len() > 160
        {
            return Err("Invalid twin initial_state, epsilon, or perturbation_expr".into());
        }
        let mut extra = ExtraChaosParams::default();
        if let Some([a, b, c]) = s.rossler_params {
            extra.rossler_a = a;
            extra.rossler_b = b;
            extra.rossler_c = c;
        }
        if let Some([m1, m2, l1, l2, g]) = s.pendulum_params {
            extra.pendulum_m1 = m1;
            extra.pendulum_m2 = m2;
            extra.pendulum_l1 = l1;
            extra.pendulum_l2 = l2;
            extra.pendulum_g = g;
        }
        if !extra.is_valid() {
            return Err(
                "AttractorPlotter auxiliary model parameters must be finite and positive".into(),
            );
        }
        Ok((extra, offsets))
    }

    pub(super) fn deserialize_state(&mut self, state: &str) -> Result<(), String> {
        <Self as crate::framework::InteractiveTool>::state_bounds(self).validate_json(state)?;
        let s: AttractorState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid AttractorPlotter state: {e}"))?;
        let (extra, offsets) = Self::validate_deserialized_state(&s)?;
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
        self.audio_settings = s.audio;
        self.fixed_callout_offsets = offsets.map(egui::Vec2::from);
        self.system_kind = s.system_kind;
        self.initial_state = s.initial_state;
        self.shadow_enabled = s.shadow_enabled;
        self.epsilon = s.epsilon;
        self.perturbation_expr = s.perturbation_expr;
        self.perturbation_bar
            .set_expression(&self.perturbation_expr);
        self.extra_params = extra;
        self.metrics_cache.correlation_radius = self.system_kind.default_correlation_radius();
        self.camera.apply_pose(camera);
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
        if let Some(fork_desc) = s.fork {
            self.apply_fork_descriptor(fork_desc)?;
        }
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

inventory::submit! {
    StarterPresetMetadata {
        id: "3d_attractor",
        title: "3D Attractor",
        description: "3D Lorenz strange attractor phase-space trajectory (σ=10, ρ=28, β=8/3).",
        category: "Continuous Chaos",
        icon: "🦋",
        domain: "chaos",
        tool_name: Some("AttractorPlotter"),
        preset_generator: || crate::starter_gallery::PRESET_3D_ATTRACTOR,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "butterfly_twins",
        title: "Butterfly Effect Twins",
        description: "On-attractor Lorenz twins separated by ε = 10⁻⁹ diverging exponentially at Wolf slope λ₁ / ln(10).",
        category: "Twin Divergence",
        icon: "👯",
        domain: "chaos",
        tool_name: Some("AttractorPlotter"),
        preset_generator: || crate::starter_gallery::PRESET_BUTTERFLY_TWINS,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "rossler_attractor",
        title: "Rössler Spiral Fold",
        description: "Single-band spiral-stretch-and-fold strange attractor (a=0.2, b=0.2, c=5.7) with shadow twin.",
        category: "Continuous Chaos",
        icon: "🌀",
        domain: "chaos",
        tool_name: Some("AttractorPlotter"),
        preset_generator: || crate::starter_gallery::PRESET_ROSSLER_ATTRACTOR,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "double_pendulum",
        title: "Chaotic Double Pendulum",
        description: "High-energy Lagrangian double pendulum (θ₁=2.4, θ₂=2.4) exhibiting rapid angular twin separation.",
        category: "Hamiltonian Chaos",
        icon: "⏱️",
        domain: "chaos",
        tool_name: Some("AttractorPlotter"),
        preset_generator: || crate::starter_gallery::PRESET_DOUBLE_PENDULUM,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "lorenz_butterfly_perturbation",
        title: "Lorenz Butterfly Perturbation",
        description: "Counterfactual shadow twin forked at t_k = 1.50 with δx = 1e-5 diverging onto opposite butterfly lobes.",
        category: "Counterfactual Chaos",
        icon: "⑂",
        domain: "chaos",
        tool_name: Some("AttractorPlotter"),
        preset_generator: || crate::starter_gallery::PRESET_LORENZ_BUTTERFLY_PERTURBATION,
    }
}

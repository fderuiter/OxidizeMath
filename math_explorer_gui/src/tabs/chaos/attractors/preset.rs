#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Preset serialization, `TheoryDescribable` metadata, and `StarterPresetMetadata` cards.

use super::twin_state::{
    AttractorSystemKind, ExtraChaosParams, DEFAULT_EPSILON, DEFAULT_PERTURBATION_EXPR,
};
use super::{
    AttractorMode, AttractorPlotter, SolverExecutionState, DEFAULT_FLOW_EXPRS, MAX_SIMULATION_SPEED,
};
use crate::framework::Camera3D;
use crate::starter_gallery::StarterPresetMetadata;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
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

#[derive(serde::Serialize)]
struct AttractorSaveState<'a> {
    sigma: f64,
    rho: f64,
    beta: f64,
    dt: f64,
    simulation_speed: usize,
    execution_state: SolverExecutionState,
    show_annotations: bool,
    camera: Camera3D,
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
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct AttractorState {
    sigma: f64,
    rho: f64,
    beta: f64,
    dt: f64,
    simulation_speed: usize,
    #[serde(default)]
    execution_state: Option<SolverExecutionState>,
    #[serde(default = "default_annotations_enabled")]
    show_annotations: bool,
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
}

impl AttractorPlotter {
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
        serde_json::to_string(&AttractorSaveState {
            sigma: self.system.sigma,
            rho: self.system.rho,
            beta: self.system.beta,
            dt: self.dt,
            simulation_speed: self.simulation_speed,
            execution_state: self.execution_state,
            show_annotations: self.show_annotations,
            camera: self.camera,
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
        })
        .ok()
    }

    fn validate_deserialized_state(s: &AttractorState) -> Result<ExtraChaosParams, String> {
        if !s.sigma.is_finite()
            || !s.rho.is_finite()
            || !s.beta.is_finite()
            || !s.dt.is_finite()
            || s.dt <= 0.0
            || !(1..=MAX_SIMULATION_SPEED).contains(&s.simulation_speed)
            || s.custom_params.values().any(|v| !v.is_finite())
        {
            return Err(format!(
                "AttractorPlotter parameters must be finite with dt > 0 and simulation_speed in 1..={MAX_SIMULATION_SPEED}"
            ));
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
        Ok(extra)
    }

    pub(super) fn deserialize_state(&mut self, state: &str) -> Result<(), String> {
        <Self as crate::framework::InteractiveTool>::state_bounds(self).validate_json(state)?;
        let s: AttractorState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid AttractorPlotter state: {e}"))?;
        let extra = Self::validate_deserialized_state(&s)?;
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
        self.simulation_speed = s.simulation_speed;
        self.show_annotations = s.show_annotations;
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

impl TheoryDescribable for AttractorPlotter {
    fn theory_description(&self) -> String {
        if self.mode == AttractorMode::Custom {
            let exprs = self.flow_bank.expressions();
            return format!(
                "Custom 3D dynamical flow: dx/dt = {}, dy/dt = {}, dz/dt = {} (ε={:.1e}).",
                exprs[0], exprs[1], exprs[2], self.epsilon
            );
        }
        match self.system_kind {
            AttractorSystemKind::Lorenz => format!(
                "Lorenz Strange Attractor & Twin Divergence (σ={:.2}, ρ={:.2}, β={:.2}, ε={:.1e})",
                self.system.sigma, self.system.rho, self.system.beta, self.epsilon
            ),
            AttractorSystemKind::Rossler => format!(
                "Rössler Spiral-Fold Strange Attractor (a={:.2}, b={:.2}, c={:.2}, ε={:.1e})",
                self.extra_params.rossler_a,
                self.extra_params.rossler_b,
                self.extra_params.rossler_c,
                self.epsilon
            ),
            AttractorSystemKind::DoublePendulum => format!(
                "Lagrangian Double Pendulum Chaos (m₁={:.2}, m₂={:.2}, l₁={:.2}, l₂={:.2}, ε={:.1e})",
                self.extra_params.pendulum_m1,
                self.extra_params.pendulum_m2,
                self.extra_params.pendulum_l1,
                self.extra_params.pendulum_l2,
                self.epsilon
            ),
        }
    }

    fn phonetic_description(&self) -> String {
        if self.mode == AttractorMode::Custom {
            return "Custom 3D chaotic twin trajectory simulation".to_string();
        }
        format!(
            "{} chaotic twin trajectory simulation",
            self.system_kind.label()
        )
    }

    fn theory_citation(&self) -> String {
        match self.system_kind {
            AttractorSystemKind::Lorenz => {
                "Lorenz, E. N. (1963). Deterministic Nonperiodic Flow; Wolf, A. et al. (1985)."
                    .to_string()
            }
            AttractorSystemKind::Rossler => {
                "Rössler, O. E. (1976). An Equation for Continuous Chaos. Physics Letters A."
                    .to_string()
            }
            AttractorSystemKind::DoublePendulum => {
                "Shinbrot, T. et al. (1992). Chaos in a Double Pendulum. Am. J. Phys.".to_string()
            }
        }
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::with_capacity(2);
        map.insert("default".to_string(), self.theory_description());
        map.insert(
            "lyapunov".to_string(),
            "Exponential separation ||δx(t)|| ≈ ||δx(0)|| exp(λ₁ t) bounded by strange-attractor diameter.".to_string(),
        );
        map
    }

    fn theory_latex_equation(&self) -> Option<String> {
        if self.mode == AttractorMode::Custom {
            return Some(r"\dot{\mathbf{x}} = \mathbf{f}(x, y, z)".to_string());
        }
        let eq = match self.system_kind {
            AttractorSystemKind::Lorenz => {
                r"\dot{x}=\sigma(y-x),\;\dot{y}=x(\rho-z)-y,\;\dot{z}=xy-\beta z,\;\log_{10}\|\delta\mathbf{x}(t)\|\approx \log_{10}\|\delta\mathbf{x}(0)\|+\frac{\lambda_1}{\ln 10}t"
            }
            AttractorSystemKind::Rossler => r"\dot{x}=-y-z,\;\dot{y}=x+ay,\;\dot{z}=b+z(x-c)",
            AttractorSystemKind::DoublePendulum => {
                r"\mathcal{L}=T(\theta_1,\theta_2,\omega_1,\omega_2)-V(\theta_1,\theta_2)"
            }
        };
        Some(eq.to_string())
    }

    fn theory_parameters(&self) -> std::collections::HashMap<String, ParameterConstraint> {
        let mut p = std::collections::HashMap::with_capacity(5);
        p.insert(
            "sigma".into(),
            ParameterConstraint {
                min: 0.0,
                max: 50.0,
                step: 0.5,
            },
        );
        p.insert(
            "rho".into(),
            ParameterConstraint {
                min: 0.0,
                max: 100.0,
                step: 0.5,
            },
        );
        p.insert(
            "beta".into(),
            ParameterConstraint {
                min: 0.0,
                max: 10.0,
                step: 0.1,
            },
        );
        p.insert(
            "dt".into(),
            ParameterConstraint {
                min: 0.001,
                max: 0.05,
                step: 0.001,
            },
        );
        p.insert(
            "epsilon".into(),
            ParameterConstraint {
                min: 1e-12,
                max: 1e-1,
                step: 1e-9,
            },
        );
        p
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

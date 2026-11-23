#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    AttractorForkDescriptor, AttractorMode, AttractorPlotter, SolverExecutionState,
    DEFAULT_FIXED_CALLOUT_OFFSETS, DEFAULT_FLOW_EXPRS,
};
use crate::framework::Camera3D;
use eframe::egui;
use std::collections::BTreeMap;

const fn default_annotations_enabled() -> bool {
    true
}

const fn default_tube_radius() -> f64 {
    super::export::DEFAULT_TUBE_RADIUS
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
    #[serde(skip_serializing_if = "Option::is_none")]
    fork: Option<AttractorForkDescriptor>,
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
    #[serde(default)]
    fork: Option<AttractorForkDescriptor>,
}

impl AttractorPlotter {
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
            fork,
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
        self.dragged_callout = None;
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

fn validate_attractor_state(s: &AttractorState) -> Result<[[f32; 2]; 2], String> {
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
    {
        return Err(format!(
            "AttractorPlotter parameters must be finite with dt > 0 and simulation_speed in 1..={}",
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

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "AttractorPlotter",
        display_name: "Attractor Plotter",
        domain: "chaos",
        tags: &["lorenz", "attractor", "chaos"],
        build: || Box::new(AttractorPlotter::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "3d_attractor",
        title: "3D Attractor",
        description: "ẋ=σ(y-x), ẏ=x(ρ-z)-y, ż=xy-βz: Canonical 3D chaotic strange attractor trajectory.",
        category: "Chaos Theory",
        icon: "🦋",
        domain: "chaos",
        tool_name: Some("AttractorPlotter"),
        preset_generator: || crate::starter_gallery::PRESET_3D_ATTRACTOR,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
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

impl scientific_metadata::theory::TheoryDescribable for AttractorPlotter {
    fn theory_description(&self) -> String {
        match self.mode {
            AttractorMode::Lorenz => format!(
                "Lorenz attractor dynamical system: $\\dot{{x}} = \\sigma(y - x)$, $\\dot{{y}} = x(\\rho - z) - y$, $\\dot{{z}} = xy - \\beta z$ with parameters $\\sigma = {:.2}$, $\\rho = {:.2}$, $\\beta = {:.2}$.",
                self.system.sigma, self.system.rho, self.system.beta
            ),
            AttractorMode::Custom => {
                let exprs = self.flow_bank.expressions();
                format!(
                    "Custom 3D dynamical flow: dx/dt = {}, dy/dt = {}, dz/dt = {}.",
                    exprs[0], exprs[1], exprs[2]
                )
            }
        }
    }

    fn phonetic_description(&self) -> String {
        "3D strange attractor dynamical system theory".into()
    }

    fn theory_citation(&self) -> String {
        "Deterministic Nonperiodic Flow (Lorenz, 1963); Nonlinear Dynamics and Chaos (Strogatz, 2014)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        self.system.available_descriptions()
    }

    fn theory_parameters(
        &self,
    ) -> std::collections::HashMap<String, scientific_metadata::theory::ParameterConstraint> {
        use scientific_metadata::theory::ParameterConstraint;
        std::collections::HashMap::from([
            (
                "sigma".to_string(),
                ParameterConstraint {
                    min: 0.0,
                    max: 50.0,
                    step: 0.1,
                },
            ),
            (
                "rho".to_string(),
                ParameterConstraint {
                    min: 0.0,
                    max: 100.0,
                    step: 0.5,
                },
            ),
            (
                "beta".to_string(),
                ParameterConstraint {
                    min: 0.0,
                    max: 10.0,
                    step: 0.1,
                },
            ),
            (
                "dt".to_string(),
                ParameterConstraint {
                    min: 0.001,
                    max: 0.05,
                    step: 0.001,
                },
            ),
        ])
    }
}

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Scientific metadata (`TheoryDescribable`) and `StarterGallery` registrations for [`OdeSolverTool`].

use super::OdeSolverTool;
use crate::presets::OdePreset;
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;

impl scientific_metadata::theory::TheoryDescribable for OdeSolverTool {
    fn maturity_tier(&self) -> scientific_metadata::theory::MaturityTier {
        scientific_metadata::theory::MaturityTier::Certified
    }
    fn theory_description(&self) -> String {
        match self.preset {
            OdePreset::Exponential => "First-order linear ordinary differential equation describing exponential growth or decay proportional to current quantity.".into(),
            OdePreset::HarmonicOscillator => "Second-order ordinary differential equation modeling simple harmonic motion under a Hooke's law restoring force.".into(),
            OdePreset::LogisticGrowth => "Nonlinear ordinary differential equation modeling density-dependent population growth with an environmental carrying capacity.".into(),
            OdePreset::LinearSystem2D => "Planar 2x2 linear dynamical system classified on the Poincaré trace-determinant bifurcation diagram.".into(),
            OdePreset::PlanarCustom => "Coupled 2D planar ordinary differential equations: x' = f(t, x, y), y' = g(t, x, y).".into(),
            OdePreset::Custom if self.custom_coupled => "Coupled two-state ordinary differential equation system: y' = f(t, y, v), v' = g(t, y, v)".into(),
            OdePreset::Custom => "Custom user-defined ordinary differential equation: y' = f(t, y)".into(),
        }
    }
    fn phonetic_description(&self) -> String {
        "Ordinary differential equations theory".into()
    }
    fn theory_citation(&self) -> String {
        match self.preset {
            OdePreset::Exponential
            | OdePreset::LinearSystem2D
            | OdePreset::PlanarCustom
            | OdePreset::Custom => {
                "Differential Equations and Dynamical Systems (Perko, 2001)".into()
            }
            OdePreset::HarmonicOscillator => {
                "Classical Mechanics (Goldstein et al., 2002); Perko (2001)".into()
            }
            OdePreset::LogisticGrowth => {
                "Elements of Mathematical Biology (Lotka, 1925); Verhulst (1838)".into()
            }
        }
    }
    fn theory_latex_equation(&self) -> Option<String> {
        let eq = match self.preset {
            OdePreset::Exponential => r"\frac{dy}{dt} = k y",
            OdePreset::HarmonicOscillator => r"\frac{d^2 y}{dt^2} = -k y",
            OdePreset::LogisticGrowth => r"\frac{dy}{dt} = r y (1 - \frac{y}{K})",
            OdePreset::LinearSystem2D => r"\dot{\mathbf{x}} = A \mathbf{x}",
            OdePreset::PlanarCustom | OdePreset::Custom => return None,
        };
        Some(eq.into())
    }
    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::from([
            (
                "Exponential".into(),
                "First-order linear ODE representing exponential growth or decay".into(),
            ),
            (
                "Harmonic Oscillator".into(),
                "Second-order linear ODE representing simple harmonic motion".into(),
            ),
            (
                "Logistic Growth".into(),
                "Nonlinear first-order ODE modeling population growth with carrying capacity"
                    .into(),
            ),
            (
                "Linear 2D".into(),
                "Planar 2x2 linear system with Poincaré stability regimes".into(),
            ),
            (
                "Planar Custom".into(),
                "Coupled 2D user-defined ODE phase-plane system".into(),
            ),
            (
                "Custom".into(),
                "Custom user-defined scalar or coupled ordinary differential equation".into(),
            ),
        ])
    }
    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        let pc = |min: f64, max: f64, step: f64| ParameterConstraint { min, max, step };
        HashMap::from([
            ("k".into(), pc(0.05, 5.0, 0.05)),
            ("c".into(), pc(-0.5, 5.0, 0.05)),
            ("dt".into(), pc(0.005, 0.5, 0.005)),
            ("y0".into(), pc(-4.0, 4.0, 0.1)),
            ("v0".into(), pc(-4.0, 4.0, 0.1)),
            ("r".into(), pc(-1.0, 5.0, 0.05)),
            ("cap_k".into(), pc(1.0, 50.0, 0.5)),
            ("mu".into(), pc(-0.5, 5.0, 0.05)),
        ])
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "van_der_pol_phase_portrait",
        title: "Van der Pol Phase Portrait",
        description: "y' = v, v' = μ·(1 - y²)·v - y: Coupled relaxation oscillator limit cycle in 2D (y, v) phase space.",
        category: "Differential Equations",
        icon: "🌀",
        domain: "analysis",
        tool_name: Some("OdeSolverTool"),
        preset_generator: || crate::starter_gallery::PRESET_VAN_DER_POL_PHASE_PORTRAIT,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "harmonic_oscillator",
        title: "Harmonic Oscillator",
        description: "y'' = -k·y: Second-order differential equation modeling undamped periodic spring motion.",
        category: "Differential Equations",
        icon: "〰️",
        domain: "analysis",
        tool_name: Some("OdeSolverTool"),
        preset_generator: || crate::starter_gallery::PRESET_HARMONIC_OSCILLATOR,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "logistic_growth",
        title: "Logistic Growth",
        description: "y' = r·y·(1 - y/K): Non-linear growth constrained by carrying capacity K.",
        category: "Population Dynamics",
        icon: "📈",
        domain: "analysis",
        tool_name: Some("OdeSolverTool"),
        preset_generator: || crate::starter_gallery::PRESET_LOGISTIC_GROWTH,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "exponential_decay",
        title: "Exponential Decay",
        description: "y' = k·y: First-order differential decay process with constant half-life.",
        category: "Calculus",
        icon: "📉",
        domain: "analysis",
        tool_name: Some("OdeSolverTool"),
        preset_generator: || crate::starter_gallery::PRESET_EXPONENTIAL_DECAY,
    }
}

struct ResolvedOdeComputeAst {
    vars: [&'static str; 2],
    dy: math_commons::ast::Expr,
    dv: Option<math_commons::ast::Expr>,
    params: Vec<(String, f64)>,
}

impl super::OdeSolverTool {
    pub(super) fn build_ode_compute_wgsl(&self) -> Result<String, String> {
        use crate::framework::shader_foundry::{
            emit_ode_rollout_compute_wgsl, OdeRolloutComputeSpec,
        };
        use crate::framework::InteractiveTool as _;
        if !self.has_finite_parameters() {
            return Err("ODE parameters must be finite for Compute WGSL export".to_string());
        }
        let steps = crate::presets::checked_ode_step_count(self.dt, self.total_time)
            .unwrap_or(200)
            .clamp(1, 4_096) as u32;
        let resolved = self.resolve_compute_ode_ast()?;
        emit_ode_rollout_compute_wgsl(&OdeRolloutComputeSpec {
            dy_expr: &resolved.dy,
            dv_expr: resolved.dv.as_ref(),
            state_vars: resolved.vars,
            dt: self.dt,
            steps,
            initial_state: [self.ic_y0, self.ic_v0],
            params: resolved.params,
        })
    }

    fn resolve_compute_ode_ast(&self) -> Result<ResolvedOdeComputeAst, String> {
        use crate::presets::OdePreset;
        let mut base_params: Vec<(String, f64)> = self.active_param_map().into_iter().collect();
        base_params.sort_by(|a, b| a.0.cmp(&b.0));
        match self.preset {
            OdePreset::PlanarCustom | OdePreset::Custom => self.resolve_custom_ode_ast(base_params),
            _ => self.resolve_preset_ode_ast(base_params),
        }
    }

    fn resolve_preset_ode_ast(
        &self,
        base_params: Vec<(String, f64)>,
    ) -> Result<ResolvedOdeComputeAst, String> {
        use crate::presets::OdePreset;
        let parse_rhs = |s: &str| {
            math_commons::ast::parse_equation(s)
                .map(|(_, e)| e)
                .map_err(|e| format!("ODE AST parse error: {e}"))
        };
        match self.preset {
            OdePreset::Exponential => Ok(ResolvedOdeComputeAst {
                vars: ["y", "v"],
                dy: parse_rhs("k * y")?,
                dv: None,
                params: base_params,
            }),
            OdePreset::LogisticGrowth => Ok(ResolvedOdeComputeAst {
                vars: ["y", "v"],
                dy: parse_rhs("r * y * (1 - y / cap_k)")?,
                dv: None,
                params: base_params,
            }),
            OdePreset::HarmonicOscillator => Ok(ResolvedOdeComputeAst {
                vars: ["y", "v"],
                dy: parse_rhs("v")?,
                dv: Some(parse_rhs("-k * y")?),
                params: vec![("k".to_string(), self.param_k)],
            }),
            _ => {
                let [[a11, a12], [a21, a22]] = self.matrix_a;
                Ok(ResolvedOdeComputeAst {
                    vars: ["x", "y"],
                    dy: parse_rhs("a11 * x + a12 * y")?,
                    dv: Some(parse_rhs("a21 * x + a22 * y")?),
                    params: vec![
                        ("a11".into(), a11),
                        ("a12".into(), a12),
                        ("a21".into(), a21),
                        ("a22".into(), a22),
                    ],
                })
            }
        }
    }

    fn resolve_custom_ode_ast(
        &self,
        base_params: Vec<(String, f64)>,
    ) -> Result<ResolvedOdeComputeAst, String> {
        use crate::presets::OdePreset;
        if self.preset == OdePreset::PlanarCustom {
            let dx = self
                .planar_x_input
                .expr()
                .cloned()
                .ok_or_else(|| "Invalid planar x' expression".to_string())?;
            let dy = self
                .planar_y_input
                .expr()
                .cloned()
                .ok_or_else(|| "Invalid planar y' expression".to_string())?;
            return Ok(ResolvedOdeComputeAst {
                vars: ["x", "y"],
                dy: dx,
                dv: Some(dy),
                params: filter_ode_params(base_params, &["t", "x", "y"]),
            });
        }
        if self.custom_coupled {
            let dy = self
                .coupled_bank
                .bars()
                .first()
                .and_then(|b| b.expr())
                .cloned()
                .ok_or_else(|| "Invalid coupled y' expression".to_string())?;
            let dv = self
                .coupled_bank
                .bars()
                .get(1)
                .and_then(|b| b.expr())
                .cloned()
                .ok_or_else(|| "Invalid coupled v' expression".to_string())?;
            return Ok(ResolvedOdeComputeAst {
                vars: ["y", "v"],
                dy,
                dv: Some(dv),
                params: filter_ode_params(self.coupled_bank.params_map(), &["t", "y", "v"]),
            });
        }
        let dy = self
            .math_input
            .expr()
            .cloned()
            .ok_or_else(|| "Invalid scalar y' expression".to_string())?;
        Ok(ResolvedOdeComputeAst {
            vars: ["y", "v"],
            dy,
            dv: None,
            params: filter_ode_params(base_params, &["t", "y", "v"]),
        })
    }
}

fn filter_ode_params<I>(raw: I, state_vars: &[&str]) -> Vec<(String, f64)>
where
    I: IntoIterator<Item = (String, f64)>,
{
    raw.into_iter()
        .filter(|(k, _)| !state_vars.contains(&k.as_str()))
        .collect()
}

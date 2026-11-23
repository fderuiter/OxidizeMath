#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::twin_state::AttractorSystemKind;
use super::{AttractorMode, AttractorPlotter};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};

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
        let mut map = self.system.available_descriptions();
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
        [
            ("sigma", 0.0, 50.0, 0.1),
            ("rho", 0.0, 100.0, 0.5),
            ("beta", 0.0, 10.0, 0.1),
            ("dt", 0.001, 0.05, 0.001),
            ("epsilon", 1e-12, 1e-1, 1e-9),
            ("param_0", 0.0, 10.0, 0.05),
            ("param_1", 0.0, 10.0, 0.05),
        ]
        .into_iter()
        .map(|(k, min, max, step)| (k.into(), ParameterConstraint { min, max, step }))
        .collect()
    }
}

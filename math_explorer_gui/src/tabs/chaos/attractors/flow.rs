#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{CoupledExpressionBank, MAX_SYNTHESIZED_PARAMS};
use math_commons::ast::CompiledExpr;
use nalgebra::Vector3;

pub const DEFAULT_FLOW_EXPRS: [&str; 3] = ["-y - z", "x + a * y", "b + z * (x - c)"];

/// Dynamical flow mode for [`super::AttractorPlotter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum AttractorMode {
    /// Canonical Lorenz strange attractor $(\sigma, \rho, \beta)$.
    #[default]
    Lorenz,
    /// User-authored 3D autonomous ODE system $(\dot{x}, \dot{y}, \dot{z})$ backed by [`CompiledExpr`].
    Custom,
}

impl AttractorMode {
    #[must_use]
    pub const fn is_lorenz(&self) -> bool {
        matches!(self, Self::Lorenz)
    }
}

/// Immutable, `Arc`-shareable compiled 3D vector field evaluated on background worker threads
/// without string parsing or heap allocations.
#[derive(Clone, Debug)]
pub struct CompiledFlowJob {
    exprs: [CompiledExpr; 3],
    params: [f64; MAX_SYNTHESIZED_PARAMS],
    param_len: usize,
}

impl CompiledFlowJob {
    /// Snapshots the synchronized bytecode triple and parameter array from a [`CoupledExpressionBank`].
    #[must_use]
    pub fn from_bank(bank: &CoupledExpressionBank) -> Self {
        let compiled = bank.active_compiled();
        let (params, param_len) = bank.inline_params();
        Self {
            exprs: [
                compiled[0].clone(),
                compiled[1].clone(),
                compiled[2].clone(),
            ],
            params,
            param_len,
        }
    }

    /// Evaluates $(\dot{x}, \dot{y}, \dot{z})$ at `pos` on stack buffers.
    #[inline]
    #[must_use]
    pub fn eval_rhs(&self, pos: Vector3<f64>) -> Vector3<f64> {
        let p = &self.params[..self.param_len];
        Vector3::new(
            self.exprs[0].eval_3d(pos.x, pos.y, pos.z, p),
            self.exprs[1].eval_3d(pos.x, pos.y, pos.z, p),
            self.exprs[2].eval_3d(pos.x, pos.y, pos.z, p),
        )
    }

    /// Advances `pos` by one classical 4th-order Runge-Kutta step of size `dt` without heap allocation.
    #[inline]
    #[must_use]
    pub fn rk4_step(&self, pos: Vector3<f64>, dt: f64) -> Vector3<f64> {
        let k1 = self.eval_rhs(pos);
        let k2 = self.eval_rhs(pos + k1 * (0.5 * dt));
        let k3 = self.eval_rhs(pos + k2 * (0.5 * dt));
        let k4 = self.eval_rhs(pos + k3 * dt);
        pos + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0)
    }
}

/// Builds the default 3-equation [`CoupledExpressionBank`] for custom 3D attractors.
#[must_use]
pub fn default_flow_bank() -> CoupledExpressionBank {
    CoupledExpressionBank::new(
        &["x", "y", "z"],
        &[
            ("dx/dt", DEFAULT_FLOW_EXPRS[0]),
            ("dy/dt", DEFAULT_FLOW_EXPRS[1]),
            ("dz/dt", DEFAULT_FLOW_EXPRS[2]),
        ],
    )
    .with_param_range(0.0..=10.0, 0.2)
    .with_initial_params(&[("a", 0.2), ("b", 0.2), ("c", 5.7)])
    .with_system_presets(vec![
        (
            "Rössler Attractor",
            vec!["-y - z", "x + a * y", "b + z * (x - c)"],
        ),
        (
            "Thomas Cyclically Symmetric",
            vec!["sin(y) - b * x", "sin(z) - b * y", "sin(x) - b * z"],
        ),
        (
            "Halvorsen Attractor",
            vec![
                "-a * x - 4 * y - 4 * z - y^2",
                "-a * y - 4 * z - 4 * x - z^2",
                "-a * z - 4 * x - 4 * y - x^2",
            ],
        ),
        (
            "Damped Spiral Sink",
            vec!["-a * x - y", "x - a * y", "-b * z"],
        ),
    ])
}

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Custom `MathExpressionInputBar` UI controls delegating metric construction and horizon
//! estimation to `math_explorer::physics::high_energy`.

use std::collections::BTreeMap;

use crate::framework::math_input::MAX_SYNTHESIZED_PARAMS;
use crate::framework::MathExpressionInputBar;
use eframe::egui;
use math_explorer::physics::high_energy::estimate_compiled_lapse_horizon;
pub use math_explorer::physics::high_energy::{
    build_lapse_static_metric, build_wormhole_static_metric, MIN_REGULAR_LAPSE, MIN_REGULAR_THROAT,
};

use super::GeodesicPlaygroundTool;

/// Default custom 4D radial lapse expression $f(r)$ ($g_{00} = -f(r)$, $g_{11} = 1/f(r)$).
pub const DEFAULT_CUSTOM_LAPSE_EXPR: &str = "1 - Rs / r";

/// Default custom 3D wormhole circumferential profile $b(l)$ ($g_{22} = b(l)^2$).
pub const DEFAULT_CUSTOM_WORMHOLE_EXPR: &str = "sqrt(b0^2 + l^2)";

/// Maximum UTF-8 byte length permitted for persisted custom metric expressions.
pub const MAX_CUSTOM_METRIC_EXPR_BYTES: usize = 512;

pub(crate) fn default_custom_lapse_expr() -> String {
    DEFAULT_CUSTOM_LAPSE_EXPR.to_string()
}

pub(crate) fn is_default_custom_lapse_expr(s: &str) -> bool {
    s == DEFAULT_CUSTOM_LAPSE_EXPR
}

pub(crate) fn default_custom_wormhole_expr() -> String {
    DEFAULT_CUSTOM_WORMHOLE_EXPR.to_string()
}

pub(crate) fn is_default_custom_wormhole_expr(s: &str) -> bool {
    s == DEFAULT_CUSTOM_WORMHOLE_EXPR
}

/// Interactive AST sandbox holding compiled radial lapse $f(r)$ and wormhole profile $b(l)$ bars.
#[derive(Clone, Debug)]
pub struct MetricSandbox {
    /// Expression bar for 4D static radial lapse $f(r)$ with axes `["r", "Rs"]`.
    pub lapse_bar: MathExpressionInputBar,
    /// Expression bar for 3D wormhole areal radius $b(l)$ with axes `["l", "b0"]`.
    pub wormhole_bar: MathExpressionInputBar,
}

impl Default for MetricSandbox {
    fn default() -> Self {
        let lapse_bar = MathExpressionInputBar::new(
            "f(r)",
            DEFAULT_CUSTOM_LAPSE_EXPR,
            vec![
                ("Schwarzschild (1 - Rs/r)", "1 - Rs / r"),
                ("Unit Mass (1 - 2/r)", "1 - 2 / r"),
                ("Reissner-Nordstrom (Charged)", "1 - Rs / r + q^2 / r^2"),
                ("Yukawa Screened Gravity", "1 - (Rs / r) * exp(-m * r)"),
                ("Schwarzschild-de Sitter", "1 - Rs / r - k * r^2"),
            ],
        )
        .with_axes(&["r", "Rs"])
        .with_param_range(-2.0..=2.0, 0.5)
        .with_param_spec("q", "Charge q", -1.8..=1.8, 0.65)
        .with_param_spec("m", "Screening m", 0.0..=2.0, 0.25)
        .with_param_spec("k", "Cosmological k", -0.02..=0.02, 0.004);

        let wormhole_bar = MathExpressionInputBar::new(
            "b(l)",
            DEFAULT_CUSTOM_WORMHOLE_EXPR,
            vec![
                ("Morris-Thorne sqrt(b0^2 + l^2)", "sqrt(b0^2 + l^2)"),
                ("Quartic Throat Flare", "sqrt(b0^2 + l^2 + a * l^4)"),
                ("Hyperbolic Cosh Throat", "b0 * cosh(l / b0)"),
            ],
        )
        .with_axes(&["l", "b0"])
        .with_param_range(0.0..=2.0, 0.1)
        .with_param_spec("a", "Quartic Flare a", 0.0..=0.5, 0.05);

        Self {
            lapse_bar,
            wormhole_bar,
        }
    }
}

impl MetricSandbox {
    /// Validates and restores persisted formula strings and free-parameter slider values.
    pub fn apply_persisted(
        &mut self,
        lapse_expr: &str,
        wormhole_expr: &str,
        params: &BTreeMap<String, f64>,
        require_lapse_valid: bool,
        require_wormhole_valid: bool,
    ) -> Result<(), String> {
        if lapse_expr.len() > MAX_CUSTOM_METRIC_EXPR_BYTES
            || wormhole_expr.len() > MAX_CUSTOM_METRIC_EXPR_BYTES
        {
            return Err("Custom metric expression exceeds byte limit".to_string());
        }
        if params.len() > MAX_SYNTHESIZED_PARAMS || params.values().any(|v| !v.is_finite()) {
            return Err("Invalid or non-finite custom metric parameters".to_string());
        }
        self.lapse_bar.set_expression(lapse_expr);
        self.wormhole_bar.set_expression(wormhole_expr);
        self.lapse_bar.set_params_map(params);
        self.wormhole_bar.set_params_map(params);
        if require_lapse_valid && !self.lapse_bar.is_valid() {
            return Err(self
                .lapse_bar
                .error_message()
                .unwrap_or_else(|| "Invalid custom lapse expression".to_string()));
        }
        if require_wormhole_valid && !self.wormhole_bar.is_valid() {
            return Err(self
                .wormhole_bar
                .error_message()
                .unwrap_or_else(|| "Invalid custom wormhole expression".to_string()));
        }
        Ok(())
    }

    /// Collects merged active free parameters from both bars for compact JSON persistence.
    #[must_use]
    pub fn merged_params_map(&self) -> BTreeMap<String, f64> {
        let mut out = self.lapse_bar.params_map();
        for (k, v) in self.wormhole_bar.params_map() {
            out.entry(k).or_insert(v);
        }
        out
    }

    /// Returns `true` when all synthesized parameter values across both bars are finite.
    #[must_use]
    pub fn has_finite_params(&self) -> bool {
        self.lapse_bar.param_slice().iter().all(|v| v.is_finite())
            && self
                .wormhole_bar
                .param_slice()
                .iter()
                .all(|v| v.is_finite())
    }

    /// Evaluates active custom radial lapse $f(r)$ with zero heap allocations.
    #[inline]
    #[must_use]
    pub fn eval_lapse(&self, r: f64, rs: f64) -> f64 {
        self.lapse_bar
            .active_compiled()
            .map_or(1.0 - rs / r.max(0.1), |c| {
                c.eval_2d(r, rs, self.lapse_bar.param_slice())
            })
    }

    /// Evaluates active custom wormhole profile $b(l)$ with zero heap allocations.
    #[inline]
    #[must_use]
    pub fn eval_wormhole_b(&self, l: f64, b0: f64) -> f64 {
        self.wormhole_bar
            .active_compiled()
            .map_or((b0 * b0 + l * l).sqrt(), |c| {
                c.eval_2d(l, b0, self.wormhole_bar.param_slice())
            })
    }

    /// Estimates the outermost radial horizon where $f(r) \le 0.02$ on $[0.15, 12.0]$.
    #[must_use]
    pub fn estimate_lapse_horizon(&self, rs: f64) -> f64 {
        estimate_compiled_lapse_horizon(
            self.lapse_bar.active_compiled(),
            self.lapse_bar.param_slice(),
            rs,
        )
    }
}

impl GeodesicPlaygroundTool {
    pub(super) fn draw_sandbox_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let changed = match self.state.spacetime {
            super::SpacetimePreset::CustomRadialLapse => self.sandbox.lapse_bar.ui(ui),
            super::SpacetimePreset::CustomWormholeProfile => self.sandbox.wormhole_bar.ui(ui),
            _ => false,
        };
        if changed {
            self.sync_sandbox_to_state();
            self.lensing_dirty = true;
        }
        changed
    }

    pub(crate) fn sync_sandbox_to_state(&mut self) {
        self.state.custom_lapse_expr = self.sandbox.lapse_bar.raw_text.clone();
        self.state.custom_wormhole_expr = self.sandbox.wormhole_bar.raw_text.clone();
        self.state.custom_params = self.sandbox.merged_params_map();
    }
}

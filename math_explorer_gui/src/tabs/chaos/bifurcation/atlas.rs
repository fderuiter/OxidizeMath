#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::runner::{burned_logistic_lyapunov, CompiledMap1DJob};
use super::{
    BifurcationDiagram, BifurcationMapMode, CUSTOM_R_BOUND, DEFAULT_CUSTOM_EXPR, DEFAULT_X0,
};
use crate::async_sim::atlas::{
    atlas_regime_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, AtlasTeleportEvent,
    RegimeCartographer,
};
use eframe::egui;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Supported `(x, y)` parameter axis pairs for [`BifurcationCartographer`].
pub const BIFURCATION_ATLAS_AXIS_PAIRS: &[(&str, &str)] = &[("r", "x0"), ("r", "param_0")];

const LYAP_EDGE_TOL: f64 = 0.015;
const ATLAS_LYAP_STEPS: usize = 80;

/// Discrete 1D map stability regimes classified by largest Lyapunov exponent $\lambda$.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BifurcationRegime {
    /// Attracting fixed point or periodic limit cycle ($\lambda < -0.015$).
    FixedPointOrCycle = 0,
    /// Period-doubling or tangent bifurcation edge ($|\lambda| \le 0.015$).
    BifurcationEdge = 1,
    /// Deterministic chaos with positive exponential divergence ($\lambda > 0.015$).
    ChaoticBand = 2,
    /// Orbit escapes basin of attraction ($|x_n| > 10^6$ or non-finite).
    EscapingOrbit = 3,
}

impl BifurcationRegime {
    /// Converts a raw `u8` cell regime code into [`Self`].
    #[inline]
    #[must_use]
    pub const fn from_u8(code: u8) -> Self {
        match code {
            0 => Self::FixedPointOrCycle,
            1 => Self::BifurcationEdge,
            2 => Self::ChaoticBand,
            _ => Self::EscapingOrbit,
        }
    }

    /// Human-readable regime classification label.
    #[inline]
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::FixedPointOrCycle => "Periodic / Fixed Point",
            Self::BifurcationEdge => "Bifurcation Edge",
            Self::ChaoticBand => "Chaotic Band",
            Self::EscapingOrbit => "Escaping Orbit",
        }
    }
}

const MAX_CUSTOM_EXPR_BYTES: usize = 256;

/// Default serialized atlas axis keys for `BifurcationState`.
#[must_use]
pub fn default_bifurcation_atlas_axes() -> [String; 2] {
    ["r".to_string(), "x0".to_string()]
}

/// Predicate skipping default `atlas_axes` during JSON serialization for backward compatibility.
#[must_use]
pub fn is_default_bifurcation_atlas_axes(axes: &[String; 2]) -> bool {
    axes[0] == "r" && axes[1] == "x0"
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct BifurcationState {
    pub r_min: f64,
    pub r_max: f64,
    pub steps: usize,
    #[serde(default, skip_serializing_if = "BifurcationMapMode::is_logistic")]
    pub map_mode: BifurcationMapMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_r: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x0: Option<f64>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub r_pinned: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub use_custom_map: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_expr: Option<String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub custom_params: std::collections::BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub atlas_visible: bool,
    #[serde(
        default = "default_bifurcation_atlas_axes",
        skip_serializing_if = "is_default_bifurcation_atlas_axes"
    )]
    pub atlas_axes: [String; 2],
}

impl BifurcationState {
    pub(super) fn validate(&self) -> Result<(BifurcationMapMode, f64, f64, String), String> {
        let mode = if self.use_custom_map {
            BifurcationMapMode::Custom
        } else {
            self.map_mode
        };
        let r_bounds = match mode {
            BifurcationMapMode::Logistic => 0.0..=4.0,
            BifurcationMapMode::Custom => -CUSTOM_R_BOUND..=CUSTOM_R_BOUND,
        };
        let x0_bounds = match mode {
            BifurcationMapMode::Logistic => -0.25..=1.25,
            BifurcationMapMode::Custom => -10.0..=10.0,
        };
        if !self.r_min.is_finite()
            || !self.r_max.is_finite()
            || !r_bounds.contains(&self.r_min)
            || !r_bounds.contains(&self.r_max)
            || self.r_min >= self.r_max
            || !(100..=2000).contains(&self.steps)
            || !self.custom_params.values().all(|v| v.is_finite())
        {
            return Err("Bifurcation r bounds, parameters, or step count invalid".into());
        }
        let sel_r = self.selected_r.unwrap_or((self.r_min + self.r_max) * 0.5);
        if !sel_r.is_finite() || !r_bounds.contains(&sel_r) {
            return Err("Bifurcation selected_r out of bounds".into());
        }
        let x0 = self.x0.unwrap_or(DEFAULT_X0);
        if !x0.is_finite() || !x0_bounds.contains(&x0) {
            return Err("Bifurcation x0 out of bounds".into());
        }
        let expr_str = self
            .custom_expr
            .as_deref()
            .unwrap_or(DEFAULT_CUSTOM_EXPR)
            .trim()
            .to_string();
        if expr_str.len() > MAX_CUSTOM_EXPR_BYTES {
            return Err("Custom map expression exceeds maximum length".into());
        }
        if mode.is_custom() && math_commons::ast::parse_equation(&expr_str).is_err() {
            return Err("Invalid custom 1D map syntax in state".into());
        }
        Ok((mode, sel_r, x0, expr_str))
    }
}

fn bifurcation_axis_spec(
    theory_params: &HashMap<String, ParameterConstraint>,
    key: &'static str,
    mode: BifurcationMapMode,
) -> AtlasAxisSpec {
    let (resolved, label, min, max, step) = match key {
        "r" => match mode {
            BifurcationMapMode::Logistic => ("r", "Control Parameter (r)", 0.0, 4.0, 0.01),
            BifurcationMapMode::Custom => ("r", "Control Parameter (r)", -6.0, 6.0, 0.02),
        },
        "param_0" => ("param_0", "Custom Param 0", -2.0, 2.0, 0.05),
        _ => match mode {
            BifurcationMapMode::Logistic => ("x0", "Initial Seed (x₀)", -0.25, 1.25, 0.01),
            BifurcationMapMode::Custom => ("x0", "Initial Seed (x₀)", -2.0, 2.0, 0.02),
        },
    };
    AtlasAxisSpec::from_theory(
        theory_params,
        resolved,
        label,
        ParameterConstraint { min, max, step },
    )
}

/// Zero-allocation 2D Lyapunov and basin-escape cartographer for 1D iterated maps.
#[derive(Clone, Debug)]
pub struct BifurcationCartographer {
    axes: AtlasAxisPair,
    mode: BifurcationMapMode,
    custom_job: Option<CompiledMap1DJob>,
    base_r: f64,
    base_x0: f64,
    base_param0: f64,
}

impl BifurcationCartographer {
    /// Builds a [`BifurcationCartographer`] bounded by `theory_params`.
    #[must_use]
    pub fn new(
        theory_params: &HashMap<String, ParameterConstraint>,
        axis_keys: (&'static str, &'static str),
        mode: BifurcationMapMode,
        custom_job: Option<CompiledMap1DJob>,
        base_r: f64,
        base_x0: f64,
        base_param0: f64,
    ) -> Self {
        Self {
            axes: AtlasAxisPair {
                x: bifurcation_axis_spec(theory_params, axis_keys.0, mode),
                y: bifurcation_axis_spec(theory_params, axis_keys.1, mode),
            },
            mode,
            custom_job,
            base_r,
            base_x0,
            base_param0,
        }
    }

    #[inline]
    fn resolve_params(&self, p1: f64, p2: f64) -> (f64, f64, Option<f64>) {
        let mut r = self.base_r;
        let mut x0 = self.base_x0;
        let mut p0 = self.base_param0;
        let mut has_p0 = false;
        for (k, v) in [(self.axes.x.key, p1), (self.axes.y.key, p2)] {
            match k {
                "r" => r = v,
                "x0" => x0 = v,
                "param_0" => {
                    p0 = v;
                    has_p0 = true;
                }
                _ => {}
            }
        }
        (r, x0, has_p0.then_some(p0))
    }
}

impl RegimeCartographer for BifurcationCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let (r, x0, p0_override) = self.resolve_params(p1, p2);
        let lam_opt = match self.mode {
            BifurcationMapMode::Logistic => burned_logistic_lyapunov(r, x0, ATLAS_LYAP_STEPS),
            BifurcationMapMode::Custom => self
                .custom_job
                .as_ref()
                .and_then(|job| job.lyapunov(r, x0, p0_override, 80, ATLAS_LYAP_STEPS)),
        };
        let Some(lam) = lam_opt else {
            return AtlasCell {
                regime: BifurcationRegime::EscapingOrbit as u8,
                primary: 2.0,
                secondary: x0,
            };
        };
        let regime = if lam > LYAP_EDGE_TOL {
            BifurcationRegime::ChaoticBand
        } else if lam >= -LYAP_EDGE_TOL {
            BifurcationRegime::BifurcationEdge
        } else {
            BifurcationRegime::FixedPointOrCycle
        };
        AtlasCell {
            regime: regime as u8,
            primary: lam,
            secondary: x0,
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let intensity = match BifurcationRegime::from_u8(cell.regime) {
            BifurcationRegime::FixedPointOrCycle => {
                (1.0 / (1.0 - cell.primary).max(1.0)).clamp(0.2, 1.0)
            }
            BifurcationRegime::BifurcationEdge => 0.85,
            BifurcationRegime::ChaoticBand => (cell.primary * 1.4).clamp(0.3, 1.0),
            BifurcationRegime::EscapingOrbit => 0.75,
        };
        atlas_regime_color(cell.regime, 4, intensity, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        BifurcationRegime::from_u8(regime).label()
    }

    fn primary_metric_label(&self) -> &'static str {
        "Lyapunov Exponent λ"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        let reg = BifurcationRegime::from_u8(cell.regime);
        if reg == BifurcationRegime::EscapingOrbit {
            format!(
                "{} = {:.4}, {} = {:.4}\nRegime: {}\nOrbit diverges beyond basin",
                self.axes.x.key,
                p1,
                self.axes.y.key,
                p2,
                reg.label()
            )
        } else {
            format!(
                "{} = {:.4}, {} = {:.4}\nRegime: {}\nLyapunov λ: {:+.4}",
                self.axes.x.key,
                p1,
                self.axes.y.key,
                p2,
                reg.label(),
                cell.primary
            )
        }
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        [
            BifurcationRegime::FixedPointOrCycle,
            BifurcationRegime::BifurcationEdge,
            BifurcationRegime::ChaoticBand,
            BifurcationRegime::EscapingOrbit,
        ]
        .into_iter()
        .map(|r| {
            (
                r.label(),
                atlas_regime_color(r as u8, 4, 0.7, high_contrast),
            )
        })
        .collect()
    }
}

impl BifurcationDiagram {
    pub(crate) fn build_theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        let mut map = HashMap::new();
        let (r_lo, r_hi, r_step, x_lo, x_hi) = match self.map_mode {
            BifurcationMapMode::Logistic => (0.0, 4.0, 0.01, -0.25, 1.25),
            BifurcationMapMode::Custom => (-6.0, 6.0, 0.02, -2.0, 2.0),
        };
        map.insert(
            "r".to_string(),
            ParameterConstraint {
                min: r_lo,
                max: r_hi,
                step: r_step,
            },
        );
        map.insert(
            "x0".to_string(),
            ParameterConstraint {
                min: x_lo,
                max: x_hi,
                step: 0.01,
            },
        );
        map.insert(
            "param_0".to_string(),
            ParameterConstraint {
                min: -2.0,
                max: 2.0,
                step: 0.05,
            },
        );
        map
    }

    fn first_custom_param(&self) -> f64 {
        self.math_input
            .param_slice()
            .first()
            .copied()
            .unwrap_or(0.0)
    }

    fn set_first_custom_param(&mut self, val: f64) {
        let mut pmap = self.math_input.params_map();
        if let Some(first) = pmap.values_mut().next() {
            *first = val.clamp(-2.0, 2.0);
            self.math_input.set_params_map(&pmap);
        }
    }

    /// Applies an [`AtlasTeleportEvent`] to `selected_r`/`probe_r`, `x0`, or `param_0` and refreshes coupled orbits.
    pub fn apply_atlas_teleport(&mut self, ctx: Option<&egui::Context>, ev: AtlasTeleportEvent) {
        let mut needs_sweep = false;
        let (r_lo, r_hi) = match self.map_mode {
            BifurcationMapMode::Logistic => (0.0, 4.0),
            BifurcationMapMode::Custom => (-6.0, 6.0),
        };
        for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
            match key {
                "r" => {
                    let target_r = val.clamp(r_lo, r_hi);
                    if target_r < self.r_min {
                        self.r_min = target_r;
                        needs_sweep = true;
                    } else if target_r > self.r_max {
                        self.r_max = target_r;
                        needs_sweep = true;
                    }
                    let clamped_r = target_r.clamp(self.r_min, self.r_max);
                    self.selected_r = clamped_r;
                    self.probe_r = clamped_r;
                    self.r_from_hover = false;
                }
                "x0" => {
                    let x_bounds = match self.map_mode {
                        BifurcationMapMode::Logistic => (-0.25, 1.25),
                        BifurcationMapMode::Custom => (-10.0, 10.0),
                    };
                    self.x0 = val.clamp(x_bounds.0, x_bounds.1);
                    needs_sweep = true;
                }
                "param_0" => {
                    self.set_first_custom_param(val);
                    if self.map_mode == BifurcationMapMode::Custom {
                        needs_sweep = true;
                    }
                }
                _ => {}
            }
        }
        if needs_sweep {
            self.recompute();
            if let Some(c) = ctx {
                self.dispatch_sweep_request(c);
            }
        } else {
            self.recompute_coupled_views();
        }
    }

    pub(super) fn render_bifurcation_atlas_panel(&mut self, ui: &mut egui::Ui) {
        if !self.atlas.is_visible() {
            return;
        }
        let ctx = ui.ctx().clone();
        let (xk, yk) = self.atlas.axis_keys();
        let p0 = self.first_custom_param();
        let get_val = |k: &str| match k {
            "r" => self.selected_r,
            "x0" => self.x0,
            "param_0" => p0,
            _ => 0.0,
        };
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.map_mode as u8).hash(&mut h);
        if self.map_mode == BifurcationMapMode::Custom {
            self.math_input.raw_text.hash(&mut h);
            for (idx, &pv) in self.math_input.param_slice().iter().enumerate() {
                if !(idx == 0 && (xk == "param_0" || yk == "param_0")) {
                    pv.to_bits().hash(&mut h);
                }
            }
        }
        if xk != "r" && yk != "r" {
            self.selected_r.to_bits().hash(&mut h);
        }
        if xk != "x0" && yk != "x0" {
            self.x0.to_bits().hash(&mut h);
        }

        let theory_params = self.theory_parameters();
        let mode = self.map_mode;
        let custom_job = (mode == BifurcationMapMode::Custom)
            .then(|| CompiledMap1DJob::from_bar(&self.math_input));
        let (base_r, base_x0) = (self.selected_r, self.x0);
        if let Some(ev) = self.atlas.show_panel(
            ui,
            "bifurcation",
            get_val(xk),
            get_val(yk),
            h.finish(),
            |keys| {
                Arc::new(BifurcationCartographer::new(
                    &theory_params,
                    keys,
                    mode,
                    custom_job,
                    base_r,
                    base_x0,
                    p0,
                ))
            },
        ) {
            self.apply_atlas_teleport(Some(&ctx), ev);
        }
    }
}

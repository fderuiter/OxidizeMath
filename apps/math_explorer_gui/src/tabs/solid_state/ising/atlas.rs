//! Analytical 2D parameter cartographer for [`super::IsingModelTool`] backed by
//! [`math_explorer::physics::solid_state::ising_atlas`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::atlas::{
    atlas_scalar_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, RegimeCartographer,
};
use eframe::egui;
pub use math_explorer::physics::solid_state::{
    eval_ising_order_parameter, eval_spatial_dilemma_cell, onsager_critical_temp, IsingAtlasRegime,
};
use scientific_metadata::theory::ParameterConstraint;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported `(x, y)` parameter axis slices for [`IsingCartographer`].
pub const ISING_ATLAS_AXIS_PAIRS: &[(&str, &str)] = &[
    ("temperature", "h_field"),
    ("temperature", "j_coupling"),
    ("T", "S"),
];

/// Default serialized atlas axis keys for [`super::IsingModelTool`].
#[must_use]
pub fn default_ising_atlas_axes() -> [String; 2] {
    ["temperature".to_string(), "h_field".to_string()]
}

/// Compact persisted atlas state embedded inside `IsingState`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IsingAtlasPersisted {
    /// Whether the 2D regime atlas panel is open.
    #[serde(default)]
    pub visible: bool,
    /// Active `(x_key, y_key)` axis pair.
    #[serde(default = "default_ising_atlas_axes")]
    pub axes: [String; 2],
}

impl Default for IsingAtlasPersisted {
    fn default() -> Self {
        Self {
            visible: false,
            axes: default_ising_atlas_axes(),
        }
    }
}

impl IsingAtlasPersisted {
    /// Predicate omitting default hidden atlas state from serialized JSON.
    #[must_use]
    pub fn is_default(&self) -> bool {
        !self.visible && self.axes[0] == "temperature" && self.axes[1] == "h_field"
    }
}

fn ising_axis_spec(
    theory_params: &HashMap<String, ParameterConstraint>,
    key: &'static str,
) -> AtlasAxisSpec {
    match key {
        "h_field" => AtlasAxisSpec::from_theory(
            theory_params,
            "h_field",
            "External Field (h)",
            ParameterConstraint {
                min: -2.0,
                max: 2.0,
                step: 0.02,
            },
        ),
        "j_coupling" => AtlasAxisSpec::from_theory(
            theory_params,
            "j_coupling",
            "Coupling Constant (J)",
            ParameterConstraint {
                min: -2.0,
                max: 2.0,
                step: 0.02,
            },
        ),
        "T" => AtlasAxisSpec {
            key: "T",
            label: "Temptation Payoff (T)",
            constraint: ParameterConstraint {
                min: 0.0,
                max: 2.0,
                step: 0.02,
            },
        },
        "S" => AtlasAxisSpec {
            key: "S",
            label: "Sucker Payoff (S)",
            constraint: ParameterConstraint {
                min: -1.0,
                max: 1.0,
                step: 0.02,
            },
        },
        _ => AtlasAxisSpec {
            key: "temperature",
            label: "Temperature (T)",
            constraint: ParameterConstraint {
                min: 0.1,
                max: 6.0,
                step: 0.05,
            },
        },
    }
}

/// Zero-allocation 2D phase and dilemma cartographer for [`super::IsingModelTool`].
#[derive(Clone, Debug)]
pub struct IsingCartographer {
    axes: AtlasAxisPair,
    base_j: f64,
    base_h: f64,
    base_temp: f64,
    reward: f64,
    punishment: f64,
}

impl IsingCartographer {
    /// Builds an [`IsingCartographer`] for the requested `axis_keys`.
    #[must_use]
    pub fn new(
        theory_params: &HashMap<String, ParameterConstraint>,
        axis_keys: (&'static str, &'static str),
        base_temp: f64,
        base_j: f64,
        base_h: f64,
        game_payoffs: [f64; 4],
    ) -> Self {
        Self {
            axes: AtlasAxisPair {
                x: ising_axis_spec(theory_params, axis_keys.0),
                y: ising_axis_spec(theory_params, axis_keys.1),
            },
            base_j,
            base_h,
            base_temp,
            reward: game_payoffs[0],
            punishment: game_payoffs[3],
        }
    }

    #[inline]
    fn is_game_slice(&self) -> bool {
        self.axes.x.key == "T" && self.axes.y.key == "S"
    }
}

impl RegimeCartographer for IsingCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        if self.is_game_slice() {
            let (regime, x_sp, x_ode) =
                eval_spatial_dilemma_cell(p1, p2, self.reward, self.punishment, self.base_temp);
            return AtlasCell {
                regime: regime as u8,
                primary: x_sp,
                secondary: x_ode,
            };
        }
        let temp = p1;
        let (j, h) = if self.axes.y.key == "j_coupling" {
            (p2, self.base_h)
        } else {
            (self.base_j, p2)
        };
        let (regime, order, tc) = eval_ising_order_parameter(temp, j, h);
        AtlasCell {
            regime: regime as u8,
            primary: order,
            secondary: tc,
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let val = cell.primary.clamp(0.0, 1.0);
        let mapped = match IsingAtlasRegime::from_u8(cell.regime) {
            IsingAtlasRegime::DisorderedOrDefectorFreeze => 0.04 + 0.18 * val,
            IsingAtlasRegime::CriticalOrBistable => 0.28 + 0.18 * val,
            IsingAtlasRegime::AntiferroOrCoexistence => 0.50 + 0.22 * val,
            IsingAtlasRegime::FerroOrSpatialReciprocity => 0.76 + 0.24 * val,
        };
        atlas_scalar_color(mapped, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        IsingAtlasRegime::from_u8(regime).label(self.is_game_slice())
    }

    fn primary_metric_label(&self) -> &'static str {
        if self.is_game_slice() {
            "Spatial Cooperator Survival"
        } else {
            "Order Parameter |M|"
        }
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        if self.is_game_slice() {
            format!(
                "T = {:.3}, S = {:.3} (R = {:.2}, P = {:.2})\nRegime: {}\nSpatial Cooperator: {:.1}%  |  0D Well-Mixed: {:.1}%",
                p1,
                p2,
                self.reward,
                self.punishment,
                self.regime_label(cell.regime),
                cell.primary * 100.0,
                cell.secondary * 100.0
            )
        } else {
            format!(
                "{} = {:.3}, {} = {:.3}\nRegime: {}\nOrder |M|: {:.3}  |  Onsager Tc(J) = {:.3}",
                self.axes.x.key,
                p1,
                self.axes.y.key,
                p2,
                self.regime_label(cell.regime),
                cell.primary,
                cell.secondary
            )
        }
    }

    fn overlay_contour(&self) -> Option<(&'static str, Vec<[f64; 2]>)> {
        if self.is_game_slice() {
            let rx = self.axes.x.clamp(self.reward);
            return Some((
                "T = R",
                vec![
                    [rx, self.axes.y.constraint.min],
                    [rx, self.axes.y.constraint.max],
                ],
            ));
        }
        if self.axes.y.key == "j_coupling" {
            let x_min = self.axes.x.constraint.min;
            let tc_neg = self
                .axes
                .x
                .clamp(onsager_critical_temp(self.axes.y.constraint.min));
            let tc_pos = self
                .axes
                .x
                .clamp(onsager_critical_temp(self.axes.y.constraint.max));
            Some((
                "Onsager Tc(J) = 2|J|/ln(1+√2)",
                vec![
                    [tc_neg, self.axes.y.constraint.min],
                    [x_min, 0.0],
                    [tc_pos, self.axes.y.constraint.max],
                ],
            ))
        } else {
            let tc = self.axes.x.clamp(onsager_critical_temp(self.base_j));
            Some((
                "Onsager Tc(J)",
                vec![
                    [tc, self.axes.y.constraint.min],
                    [tc, self.axes.y.constraint.max],
                ],
            ))
        }
    }

    fn extra_overlay_contours(&self) -> Vec<(&'static str, Vec<[f64; 2]>)> {
        if !self.is_game_slice() {
            return Vec::new();
        }
        let (x_min, x_max) = (self.axes.x.constraint.min, self.axes.x.constraint.max);
        let (y_min, y_max) = (self.axes.y.constraint.min, self.axes.y.constraint.max);
        let mut out = Vec::with_capacity(2);
        if (y_min..=y_max).contains(&self.punishment) {
            out.push((
                "S = P",
                vec![[x_min, self.punishment], [x_max, self.punishment]],
            ));
        }
        let offset = 3.0 * self.punishment - 2.0 * self.reward;
        let t_lo = x_min.max(2.0 * y_min - offset);
        let t_hi = x_max.min(2.0 * y_max - offset);
        if t_hi > t_lo + 1e-6 {
            let s_lo = 0.5 * (t_lo + offset);
            let s_hi = 0.5 * (t_hi + offset);
            out.push(("2R + 2S = T", vec![[t_lo, s_lo], [t_hi, s_hi]]));
        }
        out
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        let g = self.is_game_slice();
        vec![
            (
                IsingAtlasRegime::DisorderedOrDefectorFreeze.label(g),
                atlas_scalar_color(0.10, high_contrast),
            ),
            (
                IsingAtlasRegime::CriticalOrBistable.label(g),
                atlas_scalar_color(0.36, high_contrast),
            ),
            (
                IsingAtlasRegime::AntiferroOrCoexistence.label(g),
                atlas_scalar_color(0.62, high_contrast),
            ),
            (
                IsingAtlasRegime::FerroOrSpatialReciprocity.label(g),
                atlas_scalar_color(0.90, high_contrast),
            ),
        ]
    }
}

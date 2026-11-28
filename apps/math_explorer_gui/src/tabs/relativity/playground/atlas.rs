#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Zero-allocation 2D lensing and orbital regime cartographer UI adapter over `domain_physics`.

use std::collections::HashMap;

use crate::async_sim::atlas::{
    atlas_regime_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, RegimeCartographer,
};
use eframe::egui;
use math_commons::ast::CompiledExpr;
use math_explorer::physics::high_energy::{
    eval_custom_lapse_lensing_cell, eval_kerr_lensing_cell, eval_wormhole_lensing_cell,
    RadialAtlasSummary,
};
pub use math_explorer::physics::high_energy::{kerr_prograde_critical_b, LensingRegime};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};

use super::sim::to_geodesic_kind;
use super::{GeodesicMode, GeodesicPlaygroundTool, SpacetimePreset};

/// Supported `(x, y)` parameter axis pairs for [`RelativityCartographer`].
pub const RELATIVITY_ATLAS_AXIS_PAIRS: &[(&str, &str)] = &[
    ("impact_or_momentum", "initial_radius"),
    ("impact_or_momentum", "spin_chi"),
];

pub(crate) fn default_relativity_atlas_axes() -> [String; 2] {
    ["impact_or_momentum".into(), "initial_radius".into()]
}

pub(crate) fn is_default_relativity_atlas_axes(axes: &[String; 2]) -> bool {
    axes[0] == "impact_or_momentum" && axes[1] == "initial_radius"
}

#[derive(Clone, Debug)]
enum CartographerMetric {
    BlackHole4D {
        rs: f64,
        base_chi: f64,
    },
    Wormhole3D {
        b0: f64,
        custom_b: Option<(CompiledExpr, [f64; 8], usize)>,
    },
    CustomLapse4D {
        rs: f64,
        compiled: CompiledExpr,
        params: [f64; 8],
        param_len: usize,
    },
}

/// Stack-evaluated 2D regime cartographer for curved-spacetime geodesics.
#[derive(Clone, Debug)]
pub struct RelativityCartographer {
    axes: AtlasAxisPair,
    metric: CartographerMetric,
    mode: GeodesicMode,
    base_r0: f64,
    base_b: f64,
    radial_bias: f64,
}

fn copy_fixed_params(slice: &[f64]) -> ([f64; 8], usize) {
    let mut buf = [0.0; 8];
    let n = slice.len().min(8);
    buf[..n].copy_from_slice(&slice[..n]);
    (buf, n)
}

#[inline]
const fn summary_to_cell(s: RadialAtlasSummary) -> AtlasCell {
    AtlasCell {
        regime: s.regime as u8,
        primary: s.primary,
        secondary: s.secondary,
    }
}

fn build_axis_spec(
    theory: &HashMap<String, ParameterConstraint>,
    key: &'static str,
    spacetime: SpacetimePreset,
    rs: f64,
) -> AtlasAxisSpec {
    match key {
        "spin_chi" => AtlasAxisSpec::from_theory(
            theory,
            "spin_chi",
            "Kerr Spin (chi)",
            ParameterConstraint {
                min: -0.95,
                max: 0.95,
                step: 0.02,
            },
        ),
        "initial_radius" => {
            let (label, min, max) = if spacetime.is_wormhole() {
                ("Launch Distance (l0)", -12.0, 12.0)
            } else {
                ("Launch Radius (r0)", (1.2 * rs).max(0.6), 18.0)
            };
            AtlasAxisSpec {
                key: "initial_radius",
                label,
                constraint: ParameterConstraint {
                    min,
                    max,
                    step: 0.1,
                },
            }
        }
        _ => AtlasAxisSpec::from_theory(
            theory,
            "impact_or_momentum",
            "Impact Parameter / Momentum (b)",
            ParameterConstraint {
                min: -14.0,
                max: 14.0,
                step: 0.05,
            },
        ),
    }
}

impl RelativityCartographer {
    fn is_wormhole(&self) -> bool {
        matches!(self.metric, CartographerMetric::Wormhole3D { .. })
    }
}

impl RegimeCartographer for RelativityCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let mut b_val = self.base_b;
        let mut r0_val = self.base_r0;
        let mut chi_override = None;
        for (k, v) in [(self.axes.x.key, p1), (self.axes.y.key, p2)] {
            match k {
                "impact_or_momentum" => b_val = v,
                "initial_radius" => r0_val = v,
                "spin_chi" => chi_override = Some(v),
                _ => {}
            }
        }
        let kind = to_geodesic_kind(self.mode);
        let summary = match &self.metric {
            CartographerMetric::BlackHole4D { rs, base_chi } => eval_kerr_lensing_cell(
                kind,
                self.radial_bias,
                *rs,
                chi_override.unwrap_or(*base_chi),
                r0_val,
                b_val,
            ),
            CartographerMetric::CustomLapse4D {
                rs,
                compiled,
                params,
                param_len,
            } => eval_custom_lapse_lensing_cell(
                kind,
                self.radial_bias,
                *rs,
                compiled,
                &params[..*param_len],
                r0_val,
                b_val,
            ),
            CartographerMetric::Wormhole3D { b0, custom_b } => {
                let custom_ref = custom_b.as_ref().map(|(c, p, n)| (c, &p[..*n]));
                eval_wormhole_lensing_cell(kind, self.radial_bias, *b0, custom_ref, r0_val, b_val)
            }
        };
        summary_to_cell(summary)
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        atlas_regime_color(cell.regime, 4, cell.primary, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        LensingRegime::from_u8(regime).label(self.is_wormhole())
    }

    fn primary_metric_label(&self) -> &'static str {
        if self.is_wormhole() {
            "Throat Clearance"
        } else {
            "Normalized Winding"
        }
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        format!(
            "{} = {:.3}, {} = {:.3}\nRegime: {}\nr_min / r_+ = {:.3}  |  {}: {:.3}",
            self.axes.x.label,
            p1,
            self.axes.y.label,
            p2,
            self.regime_label(cell.regime),
            cell.secondary,
            self.primary_metric_label(),
            cell.primary
        )
    }

    fn overlay_contour(&self) -> Option<(&'static str, Vec<[f64; 2]>)> {
        if self.mode != GeodesicMode::NullPhoton {
            return None;
        }
        let y_min = self.axes.y.constraint.min;
        let y_max = self.axes.y.constraint.max;
        match (&self.metric, self.axes.x.key, self.axes.y.key) {
            (CartographerMetric::BlackHole4D { rs, .. }, "impact_or_momentum", "spin_chi") => {
                let pts = (0..=48)
                    .map(|i| {
                        let chi = -0.95 + 1.90 * (i as f64) / 48.0;
                        [kerr_prograde_critical_b(*rs, chi), chi]
                    })
                    .collect();
                Some(("Prograde Photon Ring b_crit(chi)", pts))
            }
            (
                CartographerMetric::BlackHole4D { rs, base_chi },
                "impact_or_momentum",
                "initial_radius",
            ) => {
                let bc = kerr_prograde_critical_b(*rs, *base_chi);
                Some((
                    "Critical Photon Ring +b_crit",
                    vec![[bc, y_min], [bc, y_max]],
                ))
            }
            (CartographerMetric::Wormhole3D { b0, .. }, "impact_or_momentum", "initial_radius") => {
                Some((
                    "Throat Capture Boundary b = b0",
                    vec![[*b0, y_min], [*b0, y_max]],
                ))
            }
            _ => None,
        }
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        let wh = self.is_wormhole();
        (0..4u8)
            .map(|slot| {
                (
                    LensingRegime::from_u8(slot).label(wh),
                    atlas_regime_color(slot, 4, 0.7, high_contrast),
                )
            })
            .collect()
    }
}

impl GeodesicPlaygroundTool {
    /// Builds a zero-allocation [`RelativityCartographer`] snapshot for `axis_keys`.
    #[must_use]
    pub fn build_cartographer(
        &self,
        axis_keys: (&'static str, &'static str),
    ) -> RelativityCartographer {
        let s = &self.state;
        let theory = self.theory_parameters();
        let eff = if axis_keys.1 == "spin_chi" && s.spacetime == SpacetimePreset::Schwarzschild {
            SpacetimePreset::KerrEquatorial
        } else {
            s.spacetime
        };
        let metric = match eff {
            SpacetimePreset::Schwarzschild => CartographerMetric::BlackHole4D {
                rs: s.scale_radius,
                base_chi: 0.0,
            },
            SpacetimePreset::KerrEquatorial => CartographerMetric::BlackHole4D {
                rs: s.scale_radius,
                base_chi: s.spin_chi,
            },
            SpacetimePreset::MorrisThorneWormhole => CartographerMetric::Wormhole3D {
                b0: s.scale_radius,
                custom_b: None,
            },
            SpacetimePreset::CustomWormholeProfile => {
                let custom_b = self.sandbox.wormhole_bar.active_compiled().map(|c| {
                    let (buf, len) = copy_fixed_params(self.sandbox.wormhole_bar.param_slice());
                    (c.clone(), buf, len)
                });
                CartographerMetric::Wormhole3D {
                    b0: s.scale_radius,
                    custom_b,
                }
            }
            SpacetimePreset::CustomRadialLapse => match self.sandbox.lapse_bar.active_compiled() {
                Some(c) => {
                    let (params, param_len) =
                        copy_fixed_params(self.sandbox.lapse_bar.param_slice());
                    CartographerMetric::CustomLapse4D {
                        rs: s.scale_radius,
                        compiled: c.clone(),
                        params,
                        param_len,
                    }
                }
                None => CartographerMetric::BlackHole4D {
                    rs: s.scale_radius,
                    base_chi: 0.0,
                },
            },
        };
        RelativityCartographer {
            axes: AtlasAxisPair {
                x: build_axis_spec(&theory, axis_keys.0, eff, s.scale_radius),
                y: build_axis_spec(&theory, axis_keys.1, eff, s.scale_radius),
            },
            metric,
            mode: s.mode,
            base_r0: s.initial_radius,
            base_b: s.impact_or_momentum,
            radial_bias: s.radial_bias,
        }
    }
}

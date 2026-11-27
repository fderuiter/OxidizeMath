#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::atlas::{
    atlas_regime_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, RegimeCartographer,
};
use eframe::egui;
pub use math_explorer::biology::morphogenesis::{
    classify_schnakenberg_dispersion, grid_u_variance, is_pattern_forming_regime,
    MorphogenesisRegime, WASHOUT_VARIANCE_THRESHOLD,
};
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;

/// Supported `(x, y)` parameter axis pairs for [`MorphogenesisCartographer`].
pub const MORPHO_ATLAS_AXIS_PAIRS: &[(&str, &str)] = &[("a", "b"), ("d_v", "dt"), ("d_u", "d_v")];

/// Default serialized atlas axis keys for [`super::MorphogenesisToolState`].
#[must_use]
pub fn default_morpho_atlas_axes() -> [String; 2] {
    ["a".to_string(), "b".to_string()]
}

/// Predicate skipping default `atlas_axes` during JSON serialization for backward compatibility.
#[must_use]
pub fn is_default_morpho_atlas_axes(axes: &[String; 2]) -> bool {
    axes[0] == "a" && axes[1] == "b"
}

/// Zero-allocation Schnakenberg dispersion and 2D Forward Euler CFL classifier.
#[must_use]
pub fn classify_schnakenberg_regime(a: f64, b: f64, d_u: f64, d_v: f64, dt: f64) -> AtlasCell {
    let cell = classify_schnakenberg_dispersion(a, b, d_u, d_v, dt);
    AtlasCell {
        regime: cell.regime,
        primary: cell.primary,
        secondary: cell.secondary,
    }
}

fn morpho_axis_spec(
    theory_params: &HashMap<String, ParameterConstraint>,
    key: &'static str,
) -> AtlasAxisSpec {
    let (label, min, max, step) = match key {
        "a" => ("Activator Rate (a)", 0.0, 1.0, 0.01),
        "b" => ("Inhibitor Rate (b)", 0.0, 2.0, 0.01),
        "d_u" => ("Activator Diffusion (D_u)", 0.1, 5.0, 0.05),
        "d_v" => ("Inhibitor Diffusion (D_v)", 10.0, 200.0, 1.0),
        _ => ("Time Step (dt)", 1e-4, 0.1, 5e-4),
    };
    let resolved_key = if matches!(key, "a" | "b" | "d_u" | "d_v" | "dt") {
        key
    } else {
        "dt"
    };
    AtlasAxisSpec::from_theory(
        theory_params,
        resolved_key,
        label,
        ParameterConstraint { min, max, step },
    )
}

/// Zero-allocation 2D cartographer for Schnakenberg reaction-diffusion morphogenesis.
#[derive(Clone, Debug)]
pub struct MorphogenesisCartographer {
    axes: AtlasAxisPair,
    base_a: f64,
    base_b: f64,
    base_du: f64,
    base_dv: f64,
    base_dt: f64,
}

impl MorphogenesisCartographer {
    /// Builds a [`MorphogenesisCartographer`] bounded by [`super::presets::MorphogenesisTheory`].
    #[must_use]
    pub fn new(
        theory_params: &HashMap<String, ParameterConstraint>,
        axis_keys: (&'static str, &'static str),
        a: f64,
        b: f64,
        d_u: f64,
        d_v: f64,
        dt: f64,
    ) -> Self {
        Self {
            axes: AtlasAxisPair {
                x: morpho_axis_spec(theory_params, axis_keys.0),
                y: morpho_axis_spec(theory_params, axis_keys.1),
            },
            base_a: a,
            base_b: b,
            base_du: d_u,
            base_dv: d_v,
            base_dt: dt,
        }
    }

    #[inline]
    fn resolve_params(&self, p1: f64, p2: f64) -> (f64, f64, f64, f64, f64) {
        let mut a = self.base_a;
        let mut b = self.base_b;
        let mut du = self.base_du;
        let mut dv = self.base_dv;
        let mut dt = self.base_dt;
        for (key, val) in [(self.axes.x.key, p1), (self.axes.y.key, p2)] {
            match key {
                "a" => a = val,
                "b" => b = val,
                "d_u" => du = val,
                "d_v" => dv = val,
                "dt" => dt = val,
                _ => {}
            }
        }
        (a, b, du, dv, dt)
    }
}

impl RegimeCartographer for MorphogenesisCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let (a, b, du, dv, dt) = self.resolve_params(p1, p2);
        classify_schnakenberg_regime(a, b, du, dv, dt)
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let intensity = match MorphogenesisRegime::from_u8(cell.regime) {
            MorphogenesisRegime::StableWashout => {
                (1.0 / (1.0 - cell.primary).max(1.0)).clamp(0.0, 1.0)
            }
            MorphogenesisRegime::TuringPattern => (cell.primary * 4.0).clamp(0.15, 1.0),
            MorphogenesisRegime::BulkHopfOscillation => (cell.primary * 2.0).clamp(0.2, 1.0),
            MorphogenesisRegime::EulerCflUnstable => ((cell.primary - 1.0) * 0.25).clamp(0.2, 1.0),
        };
        atlas_regime_color(cell.regime, 4, intensity, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        MorphogenesisRegime::from_u8(regime).label()
    }

    fn primary_metric_label(&self) -> &'static str {
        "Dominant Growth Rate Re(λ)"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        let reg = MorphogenesisRegime::from_u8(cell.regime);
        let (xk, yk, lbl) = (self.axes.x.key, self.axes.y.key, reg.label());
        if reg == MorphogenesisRegime::EulerCflUnstable {
            format!(
                "{xk} = {p1:.4}, {yk} = {p2:.4}\nRegime: {lbl}\nEuler Amplification |G|: {:.3} (> 1.0 unstable)",
                cell.primary
            )
        } else {
            format!(
                "{xk} = {p1:.4}, {yk} = {p2:.4}\nRegime: {lbl}\nDominant Growth Re(λ): {:+.4} (k² = {:.3})",
                cell.primary, cell.secondary
            )
        }
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        [
            MorphogenesisRegime::StableWashout,
            MorphogenesisRegime::TuringPattern,
            MorphogenesisRegime::BulkHopfOscillation,
            MorphogenesisRegime::EulerCflUnstable,
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

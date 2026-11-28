//! 2D parameter-space cartographer and Pearson saddle-node fold contour
//! for Gray-Scott reaction-diffusion morphogenesis backed by `domain_biology`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::{grid_u_variance, WASHOUT_VARIANCE_THRESHOLD};
use crate::async_sim::atlas::{
    atlas_regime_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, RegimeCartographer,
};
use eframe::egui;
pub use math_explorer::biology::morphogenesis::{
    classify_gray_scott_dispersion, is_gray_scott_pattern_viable, is_gray_scott_v_washed_out,
    GrayScottRegime,
};
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;

/// Supported `(x, y)` parameter axis pairs for [`GrayScottCartographer`].
pub const GRAY_SCOTT_ATLAS_AXIS_PAIRS: &[(&str, &str)] = &[("f", "k"), ("f", "dt"), ("d_u", "d_v")];

/// Zero-allocation Gray-Scott saddle-node, Hopf, Turing-mitosis, and 2D Euler CFL classifier.
#[must_use]
pub fn classify_gray_scott_regime(f: f64, k: f64, d_u: f64, d_v: f64, dt: f64) -> AtlasCell {
    let cell = classify_gray_scott_dispersion(f, k, d_u, d_v, dt);
    AtlasCell {
        regime: cell.regime,
        primary: cell.primary,
        secondary: cell.secondary,
    }
}

fn gray_scott_axis_spec(
    theory_params: &HashMap<String, ParameterConstraint>,
    key: &'static str,
) -> AtlasAxisSpec {
    let (label, min, max, step) = match key {
        "f" => ("Feed Rate (f)", 0.005, 0.10, 0.001),
        "k" => ("Kill Rate (k)", 0.01, 0.09, 0.001),
        "d_u" => ("Substrate Diffusion (D_u)", 0.02, 0.50, 0.005),
        "d_v" => ("Catalyst Diffusion (D_v)", 0.01, 0.25, 0.005),
        _ => ("Time Step (dt)", 0.05, 1.60, 0.01),
    };
    let resolved_key = if matches!(key, "f" | "k" | "d_u" | "d_v" | "dt") {
        key
    } else {
        "dt"
    };
    let mut spec = AtlasAxisSpec::from_theory(
        theory_params,
        resolved_key,
        label,
        ParameterConstraint { min, max, step },
    );
    if matches!(resolved_key, "f" | "k") && spec.constraint.min <= 0.0 {
        spec.constraint.min = 0.002;
    }
    spec
}

/// Zero-allocation 2D cartographer for Gray-Scott reaction-diffusion morphogenesis.
#[derive(Clone, Debug)]
pub struct GrayScottCartographer {
    axes: AtlasAxisPair,
    base_f: f64,
    base_k: f64,
    base_du: f64,
    base_dv: f64,
    base_dt: f64,
}

impl GrayScottCartographer {
    /// Builds a [`GrayScottCartographer`] bounded by [`super::presets::MorphogenesisTheory`].
    #[must_use]
    pub fn new(
        theory_params: &HashMap<String, ParameterConstraint>,
        axis_keys: (&'static str, &'static str),
        f: f64,
        k: f64,
        d_u: f64,
        d_v: f64,
        dt: f64,
    ) -> Self {
        Self {
            axes: AtlasAxisPair {
                x: gray_scott_axis_spec(theory_params, axis_keys.0),
                y: gray_scott_axis_spec(theory_params, axis_keys.1),
            },
            base_f: f,
            base_k: k,
            base_du: d_u,
            base_dv: d_v,
            base_dt: dt,
        }
    }

    #[inline]
    fn resolve_params(&self, p1: f64, p2: f64) -> (f64, f64, f64, f64, f64) {
        let mut f = self.base_f;
        let mut k = self.base_k;
        let mut du = self.base_du;
        let mut dv = self.base_dv;
        let mut dt = self.base_dt;
        for (key, val) in [(self.axes.x.key, p1), (self.axes.y.key, p2)] {
            match key {
                "f" => f = val,
                "k" => k = val,
                "d_u" => du = val,
                "d_v" => dv = val,
                "dt" => dt = val,
                _ => {}
            }
        }
        (f, k, du, dv, dt)
    }
}

impl RegimeCartographer for GrayScottCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let (f, k, du, dv, dt) = self.resolve_params(p1, p2);
        classify_gray_scott_regime(f, k, du, dv, dt)
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let intensity = match GrayScottRegime::from_u8(cell.regime) {
            GrayScottRegime::SaddleNodeWashout => {
                (1.0 / (1.0 - 15.0 * cell.primary).max(1.0)).clamp(0.15, 0.9)
            }
            GrayScottRegime::BistableHomogeneous => {
                (1.0 / (1.0 - 8.0 * cell.primary).max(1.0)).clamp(0.2, 0.9)
            }
            GrayScottRegime::HopfWave => (cell.primary * 25.0).clamp(0.25, 1.0),
            GrayScottRegime::TuringMitosis => (cell.primary * 60.0).clamp(0.3, 1.0),
            GrayScottRegime::EulerCflUnstable => ((cell.primary - 1.0) * 0.35).clamp(0.25, 1.0),
        };
        atlas_regime_color(cell.regime, 5, intensity, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        GrayScottRegime::from_u8(regime).label()
    }

    fn primary_metric_label(&self) -> &'static str {
        "Dispersion Growth / Fold Margin"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        let reg = GrayScottRegime::from_u8(cell.regime);
        match reg {
            GrayScottRegime::EulerCflUnstable => format!(
                "{} = {:.4}, {} = {:.4}\nRegime: {}\nEuler Amplification |G|: {:.3} (> 1.0 unstable)",
                self.axes.x.key, p1, self.axes.y.key, p2, reg.label(), cell.primary
            ),
            GrayScottRegime::SaddleNodeWashout => format!(
                "{} = {:.4}, {} = {:.4}\nRegime: {}\nFold Margin √f/2 − (f + k): {:+.4}",
                self.axes.x.key, p1, self.axes.y.key, p2, reg.label(), cell.primary
            ),
            _ => format!(
                "{} = {:.4}, {} = {:.4}\nRegime: {}\nDominant Growth Re(λ): {:+.5} (k² = {:.3})",
                self.axes.x.key, p1, self.axes.y.key, p2, reg.label(), cell.primary, cell.secondary
            ),
        }
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        [
            GrayScottRegime::SaddleNodeWashout,
            GrayScottRegime::BistableHomogeneous,
            GrayScottRegime::HopfWave,
            GrayScottRegime::TuringMitosis,
            GrayScottRegime::EulerCflUnstable,
        ]
        .into_iter()
        .map(|r| {
            (
                r.label(),
                atlas_regime_color(r as u8, 5, 0.7, high_contrast),
            )
        })
        .collect()
    }

    fn overlay_contour(&self) -> Option<(&'static str, Vec<[f64; 2]>)> {
        if self.axes.x.key != "f" || self.axes.y.key != "k" {
            return None;
        }
        let samples = 64usize;
        let f_min = self.axes.x.constraint.min.max(0.0);
        let f_max = self.axes.x.constraint.max.min(0.25);
        if f_max <= f_min {
            return None;
        }
        let k_min = self.axes.y.constraint.min;
        let k_max = self.axes.y.constraint.max;
        let mut pts = Vec::with_capacity(samples + 1);
        for idx in 0..=samples {
            let t = (idx as f64) / (samples as f64);
            let f = f_min + t * (f_max - f_min);
            let k = 0.5 * f.sqrt() - f;
            if k >= k_min && k <= k_max {
                pts.push([f, k]);
            }
        }
        (pts.len() >= 2).then_some(("Pearson Fold 4(f + k)² = f", pts))
    }
}

/// Returns `true` when `system` is spatially washed out while parameters enter a viable pattern regime.
pub(crate) fn should_reseed_washed_out_system(
    system: &math_explorer::biology::morphogenesis::TuringSystem<
        2,
        super::runner::MorphogenesisKineticsAdapter,
        math_explorer::biology::diffusion::MaskedFiniteDifference2D,
    >,
    d_u: f64,
    d_v: f64,
    dt: f64,
) -> bool {
    if !dt.is_finite() || dt <= 0.0 {
        return false;
    }
    let k = &system.kinetics;
    match k.mode {
        crate::presets::MorphogenesisKineticsMode::Schnakenberg => {
            super::atlas::is_pattern_forming_regime(k.a, k.b, d_u, d_v, dt) && {
                let var_u = grid_u_variance(system.u());
                var_u.is_finite() && var_u < WASHOUT_VARIANCE_THRESHOLD
            }
        }
        crate::presets::MorphogenesisKineticsMode::GrayScott => {
            is_gray_scott_pattern_viable(k.f, k.k, d_u, d_v, dt)
                && is_gray_scott_v_washed_out(system.v())
        }
    }
}

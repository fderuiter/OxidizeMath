//! Zero-allocation Reynolds regime cartographer for [`super::turbulence::TurbulenceTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::turbulence::{Geometry, TurbulenceTool};
use crate::async_sim::atlas::{
    atlas_regime_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, AtlasTeleportEvent,
    RegimeCartographer,
};
use eframe::egui;
use math_explorer::physics::fluid_dynamics::regimes::{
    FlatPlateClassifier, FlowClassifier, FlowRegime, PipeFlowClassifier,
};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::hash::{Hash, Hasher};

/// Supported 2D parameter atlas axis pairs for [`TurbulenceTool`].
pub const TURBULENCE_ATLAS_AXIS_PAIRS: &[(&str, &str)] =
    &[("velocity", "dynamic_viscosity"), ("velocity", "length")];

/// Zero-allocation Reynolds-number cartographer for pipe and flat-plate boundary-layer flows.
pub struct TurbulenceCartographer {
    axes: AtlasAxisPair,
    density: f64,
    dynamic_viscosity: f64,
    length: f64,
    geometry: Geometry,
}

fn build_turbulence_axes(
    tool: &TurbulenceTool,
    axis_keys: (&'static str, &'static str),
    rho: f64,
    mu: f64,
    len: f64,
) -> Option<AtlasAxisPair> {
    let theory = tool.theory_parameters();
    let (_, re_turb) = critical_reynolds_bounds(tool.geometry);
    let v_crit = ((re_turb * mu) / (rho * len)).clamp(0.1, 200.0);
    let v_max = (v_crit * 1.6).max(tool.velocity * 1.35).clamp(0.5, 250.0);
    let mut x = AtlasAxisSpec::from_theory(
        &theory,
        "velocity",
        "Flow Velocity v (m/s)",
        ParameterConstraint {
            min: 0.01,
            max: v_max,
            step: 0.05,
            default: None,
        },
    );
    x.constraint.max = v_max.min(x.constraint.max);

    match axis_keys {
        ("velocity", "dynamic_viscosity") => {
            let mu_crit = ((rho * tool.velocity.max(0.1) * len) / re_turb).max(1e-5);
            let mu_max = (mu * 2.5).max(mu_crit * 3.0).clamp(1e-4, 2.0);
            let mu_min = (mu * 0.1).min(mu_crit * 0.25).clamp(1e-6, mu_max * 0.2);
            let mut y = AtlasAxisSpec::from_theory(
                &theory,
                "dynamic_viscosity",
                "Dynamic Viscosity μ (Pa·s)",
                ParameterConstraint {
                    min: mu_min,
                    max: mu_max,
                    step: 1e-4,
                    default: None,
                },
            );
            y.constraint.min = mu_min.max(y.constraint.min);
            y.constraint.max = mu_max.min(y.constraint.max).max(y.constraint.min + 1e-5);
            Some(AtlasAxisPair { x, y })
        }
        ("velocity", "length") => {
            let len_max = (len * 2.0).clamp(0.05, 10.0);
            let mut y = AtlasAxisSpec::from_theory(
                &theory,
                "length",
                "Characteristic Length L (m)",
                ParameterConstraint {
                    min: 0.001,
                    max: len_max,
                    step: 0.01,
                    default: None,
                },
            );
            y.constraint.max = len_max.min(y.constraint.max).max(y.constraint.min + 0.01);
            Some(AtlasAxisPair { x, y })
        }
        _ => None,
    }
}

impl TurbulenceCartographer {
    /// Builds a [`TurbulenceCartographer`] scaled around critical transition thresholds and active state.
    #[must_use]
    pub fn new(tool: &TurbulenceTool, axis_keys: (&'static str, &'static str)) -> Option<Self> {
        let rho = tool.density.max(1e-3);
        let mu = tool.dynamic_viscosity.max(1e-6);
        let len = tool.length.max(1e-4);
        let axes = build_turbulence_axes(tool, axis_keys, rho, mu, len)?;
        Some(Self {
            axes,
            density: rho,
            dynamic_viscosity: mu,
            length: len,
            geometry: tool.geometry,
        })
    }
}

const fn critical_reynolds_bounds(geometry: Geometry) -> (f64, f64) {
    match geometry {
        Geometry::Pipe => (2_300.0, 4_000.0),
        Geometry::FlatPlate => (500_000.0, 1_000_000.0),
    }
}

impl RegimeCartographer for TurbulenceCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let v = p1.max(1e-6);
        let (mu, len) = match (self.axes.x.key, self.axes.y.key) {
            ("velocity", "dynamic_viscosity") => (p2.max(1e-8), self.length),
            _ => (self.dynamic_viscosity, p2.max(1e-6)),
        };
        let re = (self.density * v * len) / mu;
        if !re.is_finite() || re <= 0.0 {
            return AtlasCell {
                regime: 0,
                primary: 0.0,
                secondary: 0.0,
            };
        }
        let regime = match self.geometry {
            Geometry::Pipe => PipeFlowClassifier.classify(re),
            Geometry::FlatPlate => FlatPlateClassifier.classify(re),
        };
        let regime_idx = match regime {
            FlowRegime::Laminar => 0,
            FlowRegime::Transitional => 1,
            FlowRegime::Turbulent => 2,
        };
        let (_, re_turb) = critical_reynolds_bounds(self.geometry);
        AtlasCell {
            regime: regime_idx,
            primary: re,
            secondary: re / re_turb,
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let intensity = (cell.secondary / 2.0).clamp(0.0, 1.0);
        atlas_regime_color(cell.regime, 3, intensity, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        match regime {
            0 => "Laminar Streamline Flow",
            1 => "Transitional Intermittency",
            _ => "Fully Turbulent Cascade",
        }
    }

    fn primary_metric_label(&self) -> &'static str {
        "Reynolds Number Re"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        format!(
            "{} = {p1:.3}, {} = {p2:.5}\nRegime: {}\nRe = {:.1} ({:.2}× Re_turb)",
            self.axes.x.label,
            self.axes.y.label,
            self.regime_label(cell.regime),
            cell.primary,
            cell.secondary
        )
    }

    fn overlay_contour(&self) -> Option<(&'static str, Vec<[f64; 2]>)> {
        let (re_lam, _) = critical_reynolds_bounds(self.geometry);
        let mut pts = Vec::with_capacity(64);
        let steps = 63;
        for i in 0..=steps {
            let frac = (i as f64) / (steps as f64);
            let v = self.axes.x.constraint.min + frac * self.axes.x.span();
            let y = match (self.axes.x.key, self.axes.y.key) {
                ("velocity", "dynamic_viscosity") => (self.density * v * self.length) / re_lam,
                _ => (re_lam * self.dynamic_viscosity) / (self.density * v.max(1e-6)),
            };
            if (self.axes.y.constraint.min..=self.axes.y.constraint.max).contains(&y) {
                pts.push([v, y]);
            }
        }
        (pts.len() >= 2).then_some(("Critical Transition Onset (Re_c)", pts))
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        (0..3_u8)
            .map(|r| {
                (
                    self.regime_label(r),
                    atlas_regime_color(r, 3, 0.75, high_contrast),
                )
            })
            .collect()
    }
}

impl TurbulenceTool {
    pub(crate) fn turb_atlas_coords(&self, axis_keys: (&str, &str)) -> (f64, f64) {
        match axis_keys {
            ("velocity", "dynamic_viscosity") => (self.velocity, self.dynamic_viscosity),
            ("velocity", "length") => (self.velocity, self.length),
            _ => (self.velocity, self.dynamic_viscosity),
        }
    }

    pub(crate) fn turb_atlas_signature(&self, axis_keys: (&str, &str)) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.density.to_bits().hash(&mut h);
        (self.geometry as u8).hash(&mut h);
        match axis_keys {
            ("velocity", "dynamic_viscosity") => self.length.to_bits().hash(&mut h),
            _ => self.dynamic_viscosity.to_bits().hash(&mut h),
        }
        h.finish()
    }

    pub(crate) fn apply_turbulence_teleport(&mut self, ev: AtlasTeleportEvent) {
        for (k, v) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
            if !v.is_finite() {
                continue;
            }
            match k {
                "velocity" => self.velocity = v.clamp(0.01, 250.0),
                "dynamic_viscosity" => self.dynamic_viscosity = v.clamp(1e-6, 2.0),
                "length" => self.length = v.clamp(0.001, 10.0),
                _ => {}
            }
        }
    }
}

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Christoffel Inspector tool: compares closed-form `christoffel_at` against stack-allocated
//! central-difference `static_christoffel_symbols` across analytical and custom AST manifolds.

use std::collections::{BTreeMap, HashMap};

use crate::framework::{
    CalloutManager, CalloutSpec, ExportCapabilities, InteractiveTool, ToolMetadata, ToolStateBounds,
};
use eframe::egui;
use math_explorer::nalgebra::SVector;
use math_explorer::physics::high_energy::{
    KerrEquatorialMetric, MorrisThorneWormhole, SchwarzschildBlackHole,
};
use math_explorer::pure_math::tensor::{static_christoffel_symbols, StaticMetric, TwoSphereMetric};
use scientific_metadata::theory::ParameterConstraint;
use serde::{Deserialize, Serialize};

use super::playground::sandbox::{
    default_custom_lapse_expr, default_custom_wormhole_expr, is_default_custom_lapse_expr,
    is_default_custom_wormhole_expr, MetricSandbox,
};

#[path = "inspector/sandbox.rs"]
mod sandbox;
#[path = "inspector/ui.rs"]
mod ui;

/// Manifold selectable in [`ChristoffelInspectorTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum InspectorManifold {
    /// 4D Schwarzschild black hole $(ct, r, \theta, \varphi)$.
    #[default]
    Schwarzschild4D,
    /// 4D equatorial Kerr rotating spacetime $(ct, r, \theta, \varphi)$.
    KerrEquatorial4D,
    /// 3D Morris-Thorne traversable wormhole $(ct, l, \varphi)$.
    MorrisThorne3D,
    /// 2D Riemannian sphere $S^2$ $(\theta, \varphi)$.
    TwoSphere2D,
    /// User-authored 4D radial lapse $f(r)$ via `StaticRiemannianMetric`.
    CustomRadialLapse4D,
    /// User-authored 3D wormhole profile $b(l)$ via `StaticRiemannianMetric`.
    CustomWormhole3D,
}

impl InspectorManifold {
    /// Ordered list of all selectable inspector manifolds.
    pub const ALL: [Self; 6] = [
        Self::Schwarzschild4D,
        Self::KerrEquatorial4D,
        Self::MorrisThorne3D,
        Self::TwoSphere2D,
        Self::CustomRadialLapse4D,
        Self::CustomWormhole3D,
    ];

    /// Display name for UI selectors.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Schwarzschild4D => "Schwarzschild 4D (ct, r, theta, phi)",
            Self::KerrEquatorial4D => "Kerr Equatorial 4D (ct, r, theta, phi)",
            Self::MorrisThorne3D => "Morris-Thorne Wormhole 3D (ct, l, phi)",
            Self::TwoSphere2D => "Riemannian 2-Sphere S^2 (theta, phi)",
            Self::CustomRadialLapse4D => "Custom Radial Lapse 4D f(r)",
            Self::CustomWormhole3D => "Custom Wormhole 3D b(l)",
        }
    }

    /// Coordinate dimension $N$ of the manifold.
    pub const fn dim(self) -> usize {
        match self {
            Self::Schwarzschild4D | Self::KerrEquatorial4D | Self::CustomRadialLapse4D => 4,
            Self::MorrisThorne3D | Self::CustomWormhole3D => 3,
            Self::TwoSphere2D => 2,
        }
    }

    /// Returns `true` when evaluated from a user-authored `CompiledExpr` metric.
    pub const fn is_custom(self) -> bool {
        matches!(self, Self::CustomRadialLapse4D | Self::CustomWormhole3D)
    }

    /// Returns `true` when parameterized by proper wormhole distance $l \in (-\infty, \infty)$.
    pub const fn is_wormhole(self) -> bool {
        matches!(self, Self::MorrisThorne3D | Self::CustomWormhole3D)
    }

    /// Coordinate symbols for matrix headers.
    pub const fn coord_labels(self) -> &'static [&'static str] {
        match self {
            Self::Schwarzschild4D | Self::KerrEquatorial4D | Self::CustomRadialLapse4D => {
                &["ct", "r", "theta", "phi"]
            }
            Self::MorrisThorne3D | Self::CustomWormhole3D => &["ct", "l", "phi"],
            Self::TwoSphere2D => &["theta", "phi"],
        }
    }
}

/// Serializable state for [`ChristoffelInspectorTool`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChristoffelInspectorState {
    /// Selected curved manifold.
    pub manifold: InspectorManifold,
    /// Characteristic length scale ($R_s$, $b_0$, or sphere radius $R$).
    pub scale_radius: f64,
    /// Dimensionless Kerr spin $\chi \in (-0.92, 0.92)$.
    pub spin_chi: f64,
    /// Radial or proper-distance evaluation coordinate ($r$ or $l$).
    pub radial_coord: f64,
    /// Polar evaluation angle $\theta \in [0.15, 2.99]$.
    pub theta: f64,
    /// Upper index $\mu$ currently displayed in the $\Gamma^\mu_{\alpha\beta}(x)$ matrix table.
    pub selected_mu: usize,
    /// Custom 4D radial lapse formula $f(r)$.
    #[serde(
        default = "default_custom_lapse_expr",
        skip_serializing_if = "is_default_custom_lapse_expr"
    )]
    pub custom_lapse_expr: String,
    /// Custom 3D wormhole throat profile $b(l)$.
    #[serde(
        default = "default_custom_wormhole_expr",
        skip_serializing_if = "is_default_custom_wormhole_expr"
    )]
    pub custom_wormhole_expr: String,
    /// Synthesized free-parameter values for custom AST metric formulas.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub custom_params: BTreeMap<String, f64>,
}

impl Default for ChristoffelInspectorState {
    fn default() -> Self {
        Self {
            manifold: InspectorManifold::Schwarzschild4D,
            scale_radius: 2.0,
            spin_chi: 0.6,
            radial_coord: 4.5,
            theta: std::f64::consts::FRAC_PI_3,
            selected_mu: 1,
            custom_lapse_expr: default_custom_lapse_expr(),
            custom_wormhole_expr: default_custom_wormhole_expr(),
            custom_params: BTreeMap::new(),
        }
    }
}

/// Evaluated analytical and central-difference Christoffel slice at the current point.
#[derive(Debug, Clone)]
pub(crate) struct ChristoffelEvaluation {
    pub(crate) dim: usize,
    pub(crate) analytical: [[[f64; 4]; 4]; 4],
    pub(crate) numerical: [[[f64; 4]; 4]; 4],
    pub(crate) max_abs_diff: f64,
}

/// Interactive Christoffel symbol table and radial connection profile cross-validator.
pub struct ChristoffelInspectorTool {
    pub(crate) state: ChristoffelInspectorState,
    pub(crate) sandbox: MetricSandbox,
    pub(crate) callouts: CalloutManager,
    pub(crate) revision: u64,
}

impl Default for ChristoffelInspectorTool {
    fn default() -> Self {
        Self {
            state: ChristoffelInspectorState::default(),
            sandbox: MetricSandbox::default(),
            callouts: CalloutManager::new(),
            revision: 1,
        }
    }
}

impl ChristoffelInspectorTool {
    /// Replaces the custom 4D lapse expression $f(r)$ and synchronizes persisted state.
    pub fn set_custom_lapse_expression(&mut self, expr: &str) {
        self.sandbox.lapse_bar.set_expression(expr);
        self.sync_sandbox_to_state();
        self.clamp_in_place();
        self.revision = self.revision.wrapping_add(1);
    }

    pub(crate) fn clamp_in_place(&mut self) {
        let min_r = match self.state.manifold {
            InspectorManifold::MorrisThorne3D | InspectorManifold::CustomWormhole3D => -10.0,
            InspectorManifold::CustomRadialLapse4D => {
                (1.12 * self.sandbox.estimate_lapse_horizon(self.state.scale_radius)).max(0.5)
            }
            InspectorManifold::KerrEquatorial4D => {
                let rs = self.state.scale_radius.clamp(0.5, 5.0);
                let a = 0.5 * rs * self.state.spin_chi.clamp(-0.92, 0.92);
                let r_plus =
                    KerrEquatorialMetric::new(rs, a).map_or(rs, |k| k.outer_horizon_radius());
                1.12 * r_plus
            }
            _ => 1.15 * self.state.scale_radius.clamp(0.5, 5.0),
        };
        let s = &mut self.state;
        s.scale_radius = if s.scale_radius.is_finite() {
            s.scale_radius.clamp(0.5, 5.0)
        } else {
            2.0
        };
        s.spin_chi = if s.spin_chi.is_finite() {
            s.spin_chi.clamp(-0.92, 0.92)
        } else {
            0.6
        };
        s.radial_coord = if s.radial_coord.is_finite() {
            s.radial_coord.clamp(min_r, 16.0)
        } else {
            4.5
        };
        s.theta = if s.theta.is_finite() {
            s.theta.clamp(0.15, 2.99)
        } else {
            std::f64::consts::FRAC_PI_3
        };
        s.selected_mu = s.selected_mu.min(s.manifold.dim() - 1);
    }

    pub(crate) fn evaluate_current(&self) -> Option<ChristoffelEvaluation> {
        let s = &self.state;
        match s.manifold {
            InspectorManifold::Schwarzschild4D => {
                let bh = SchwarzschildBlackHole::from_schwarzschild_radius(s.scale_radius).ok()?;
                let pt = SVector::<f64, 4>::new(0.0, s.radial_coord, s.theta, 0.4);
                Self::pack_eval::<_, 4>(&bh, &pt)
            }
            InspectorManifold::KerrEquatorial4D => {
                let kerr =
                    KerrEquatorialMetric::new(s.scale_radius, 0.5 * s.scale_radius * s.spin_chi)
                        .ok()?;
                let pt = SVector::<f64, 4>::new(0.0, s.radial_coord, s.theta, 0.4);
                Self::pack_eval::<_, 4>(&kerr, &pt)
            }
            InspectorManifold::MorrisThorne3D => {
                let wh = MorrisThorneWormhole::new(s.scale_radius).ok()?;
                let pt = SVector::<f64, 3>::new(0.0, s.radial_coord, 0.4);
                Self::pack_eval::<_, 3>(&wh, &pt)
            }
            InspectorManifold::TwoSphere2D => {
                let sphere = TwoSphereMetric::new(s.scale_radius).ok()?;
                let pt = SVector::<f64, 2>::new(s.theta, 0.4);
                Self::pack_eval::<_, 2>(&sphere, &pt)
            }
            InspectorManifold::CustomRadialLapse4D => self.evaluate_custom_lapse(),
            InspectorManifold::CustomWormhole3D => self.evaluate_custom_wormhole(),
        }
    }

    fn pack_eval<M: StaticMetric<N>, const N: usize>(
        metric: &M,
        pt: &SVector<f64, N>,
    ) -> Option<ChristoffelEvaluation> {
        let exact = metric.christoffel_at(pt).ok()?;
        let approx = static_christoffel_symbols(metric, pt).ok()?;
        let mut analytical = [[[0.0; 4]; 4]; 4];
        let mut numerical = [[[0.0; 4]; 4]; 4];
        let mut max_abs_diff = 0.0_f64;
        for mu in 0..N {
            for a in 0..N {
                for b in 0..N {
                    let ev = exact[mu][(a, b)];
                    let nv = approx[mu][(a, b)];
                    analytical[mu][a][b] = ev;
                    numerical[mu][a][b] = nv;
                    max_abs_diff = max_abs_diff.max((ev - nv).abs());
                }
            }
        }
        Some(ChristoffelEvaluation {
            dim: N,
            analytical,
            numerical,
            max_abs_diff,
        })
    }
}

impl InteractiveTool for ChristoffelInspectorTool {
    fn name(&self) -> &'static str {
        "Christoffel Inspector"
    }

    fn domain(&self) -> &'static str {
        "relativity"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn state_revision(&self) -> u64 {
        self.revision
    }

    fn has_finite_parameters(&self) -> bool {
        let s = &self.state;
        s.scale_radius.is_finite()
            && s.spin_chi.is_finite()
            && s.radial_coord.is_finite()
            && s.theta.is_finite()
            && s.custom_params.values().all(|v| v.is_finite())
            && self.sandbox.has_finite_params()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.clamp_in_place();
        self.sandbox.lapse_bar.clamp_active_params(-5.0..=5.0);
        self.sandbox.wormhole_bar.clamp_active_params(-5.0..=5.0);
        self.sync_sandbox_to_state();
        self.revision = self.revision.wrapping_add(1);
        true
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        let ev = self.evaluate_current();
        let diff = ev.as_ref().map_or(0.0, |e| {
            if e.max_abs_diff.is_finite() {
                e.max_abs_diff
            } else {
                0.0
            }
        });
        let g_r_tt = ev
            .as_ref()
            .map_or(0.0, |e| e.numerical[1.min(e.dim - 1)][0][0]);
        vec![
            ("max_abs_diff", diff),
            ("gamma_r_tt", if g_r_tt.is_finite() { g_r_tt } else { 0.0 }),
            ("radial_coord", self.state.radial_coord),
            ("scale_radius", self.state.scale_radius),
        ]
    }

    fn sync_callouts(&mut self, callouts: &[CalloutSpec]) {
        self.callouts.sync_specs(callouts);
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::DEFAULT
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, raw: &str) -> Result<(), String> {
        self.state_bounds().validate_json(raw)?;
        let loaded: ChristoffelInspectorState = serde_json::from_str(raw)
            .map_err(|e| format!("Invalid Christoffel inspector JSON: {e}"))?;
        let mut next_sandbox = MetricSandbox::default();
        next_sandbox.apply_persisted(
            &loaded.custom_lapse_expr,
            &loaded.custom_wormhole_expr,
            &loaded.custom_params,
            loaded.manifold == InspectorManifold::CustomRadialLapse4D,
            loaded.manifold == InspectorManifold::CustomWormhole3D,
        )?;
        self.sandbox = next_sandbox;
        self.state = loaded;
        self.clamp_in_place();
        self.sync_sandbox_to_state();
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.profile_datasets()
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_inspector(ui);
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

impl scientific_metadata::theory::TheoryDescribable for ChristoffelInspectorTool {
    fn theory_description(&self) -> String {
        format!(
            "Evaluates Levi-Civita connection coefficients Gamma^mu_(alpha beta)(x) = 1/2 g^(mu nu)(d_alpha g_(nu beta) + d_beta g_(nu alpha) - d_nu g_(alpha beta)) on {}, cross-validating closed-form analytical formulas and live AST metrics against zero-allocation second-order central differences.",
            self.state.manifold.label()
        )
    }

    fn phonetic_description(&self) -> String {
        "Levi-Civita connection and Christoffel symbol cross-validation".into()
    }

    fn theory_citation(&self) -> String {
        "Levi-Civita (1917); Misner, Thorne, & Wheeler, Gravitation (1973)".into()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        let min_r = if self.state.manifold.is_wormhole() {
            -10.0
        } else {
            0.55
        };
        HashMap::from([
            (
                "scale_radius".into(),
                ParameterConstraint {
                    min: 0.5,
                    max: 5.0,
                    step: 0.05,
                    default: Some(2.0),
                },
            ),
            (
                "spin_chi".into(),
                ParameterConstraint {
                    min: -0.92,
                    max: 0.92,
                    step: 0.02,
                    default: Some(0.6),
                },
            ),
            (
                "radial_coord".into(),
                ParameterConstraint {
                    min: min_r,
                    max: 16.0,
                    step: 0.1,
                    default: Some(4.5),
                },
            ),
            (
                "theta".into(),
                ParameterConstraint {
                    min: 0.15,
                    max: 2.99,
                    step: 0.02,
                    default: Some(std::f64::consts::FRAC_PI_3),
                },
            ),
        ])
    }
}

inventory::submit! {
    ToolMetadata {
        id: "ChristoffelInspectorTool",
        display_name: "Christoffel Inspector",
        domain: "relativity",
        tags: &["tensor", "christoffel", "curvature", "connection"],
        build: || Box::new(ChristoffelInspectorTool::default()),
    }
}

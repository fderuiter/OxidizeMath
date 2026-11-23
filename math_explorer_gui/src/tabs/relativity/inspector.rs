#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Christoffel Inspector tool: compares closed-form `christoffel_at` against stack-allocated
//! central-difference `static_christoffel_symbols` across curved manifolds.

use crate::framework::{ExportCapabilities, InteractiveTool, ToolMetadata, ToolStateBounds};
use eframe::egui;
use math_explorer::nalgebra::SVector;
use math_explorer::physics::high_energy::{
    KerrEquatorialMetric, MorrisThorneWormhole, SchwarzschildBlackHole,
};
use math_explorer::pure_math::tensor::{static_christoffel_symbols, StaticMetric, TwoSphereMetric};
use serde::{Deserialize, Serialize};

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
}

impl InspectorManifold {
    /// Display name for UI selectors.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Schwarzschild4D => "Schwarzschild 4D (ct, r, theta, phi)",
            Self::KerrEquatorial4D => "Kerr Equatorial 4D (ct, r, theta, phi)",
            Self::MorrisThorne3D => "Morris-Thorne Wormhole 3D (ct, l, phi)",
            Self::TwoSphere2D => "Riemannian 2-Sphere S^2 (theta, phi)",
        }
    }

    /// Coordinate dimension $N$ of the manifold.
    pub const fn dim(self) -> usize {
        match self {
            Self::Schwarzschild4D | Self::KerrEquatorial4D => 4,
            Self::MorrisThorne3D => 3,
            Self::TwoSphere2D => 2,
        }
    }

    /// Coordinate symbols for matrix headers.
    pub const fn coord_labels(self) -> &'static [&'static str] {
        match self {
            Self::Schwarzschild4D | Self::KerrEquatorial4D => &["ct", "r", "theta", "phi"],
            Self::MorrisThorne3D => &["ct", "l", "phi"],
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
    pub(crate) revision: u64,
}

impl Default for ChristoffelInspectorTool {
    fn default() -> Self {
        Self {
            state: ChristoffelInspectorState::default(),
            revision: 1,
        }
    }
}

impl ChristoffelInspectorTool {
    pub(crate) fn clamp_in_place(&mut self) {
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
        let min_r = match s.manifold {
            InspectorManifold::MorrisThorne3D => -10.0,
            _ => 1.15 * s.scale_radius,
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
    }

    fn clamp_parameters(&mut self) -> bool {
        self.clamp_in_place();
        self.revision = self.revision.wrapping_add(1);
        true
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
        self.state = loaded;
        self.clamp_in_place();
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
            "Evaluates Levi-Civita connection coefficients Gamma^mu_(alpha beta)(x) = 1/2 g^(mu nu)(d_alpha g_(nu beta) + d_beta g_(nu alpha) - d_nu g_(alpha beta)) on {}, cross-validating closed-form analytical formulas against zero-allocation second-order central differences.",
            self.state.manifold.label()
        )
    }

    fn phonetic_description(&self) -> String {
        "Levi-Civita connection and Christoffel symbol cross-validation".into()
    }

    fn theory_citation(&self) -> String {
        "Levi-Civita (1917); Misner, Thorne, & Wheeler, Gravitation (1973)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
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

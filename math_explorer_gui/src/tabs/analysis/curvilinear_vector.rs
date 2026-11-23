//! Interactive 3D Curvilinear Vector Calculus, Gauss/Stokes Integral Theorem, and
//! Helmholtz-Hodge Decomposition Studio (`CurvilinearVectorStudio`).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:vector_calculus]

use crate::camera::Camera3D;
use crate::framework::math_input::ParameterSliderBank;
use crate::framework::{ExportCapabilities, InteractiveTool, MathExpressionInputBar, ToolMetadata};
use eframe::egui;
use math_commons::ast::{parse, CompiledExpr};
use math_explorer::pure_math::vector_calculus::theorems::{HelmholtzSummary, HelmholtzWorkspace};
use nalgebra::Vector3;
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

#[path = "curvilinear_vector/compute.rs"]
mod compute;
#[path = "curvilinear_vector/preset.rs"]
pub mod preset;
#[path = "curvilinear_vector/ui.rs"]
mod ui;
#[path = "curvilinear_vector/viewport.rs"]
mod viewport;

#[cfg(test)]
#[path = "curvilinear_vector/tests.rs"]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
#[path = "curvilinear_vector/wasm_tests.rs"]
mod wasm_tests;

/// Maximum 3D Helmholtz grid cells per axis (`32^3 = 32,768` cells) to guarantee interactive frame rates on desktop and `wasm32`.
pub const MAX_HELMHOLTZ_GRID: usize = 32;

/// Minimum 3D Helmholtz grid cells per axis required by interior finite-difference stencils.
pub const MIN_HELMHOLTZ_GRID: usize = 5;

/// Orthogonal curvilinear coordinate frame selectable in [`CurvilinearVectorStudio`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum CoordinateFrameKind {
    /// Standard Cartesian coordinates `(x, y, z)` with scale factors `(1, 1, 1)`.
    Cartesian,
    /// Cylindrical coordinates `(rho, phi, z)` with scale factors `(1, rho, 1)`.
    Cylindrical,
    /// Spherical coordinates `(r, theta, phi)` with scale factors `(1, r, r sin(theta))`.
    Spherical,
}

impl CoordinateFrameKind {
    /// Human-readable coordinate system label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cartesian => "Cartesian (x, y, z)",
            Self::Cylindrical => "Cylindrical (rho, phi, z)",
            Self::Spherical => "Spherical (r, theta, phi)",
        }
    }

    /// Canonical variable names `(u1, u2, u3)` accepted in field formulas for this frame.
    #[must_use]
    pub const fn axis_vars(self) -> [&'static str; 3] {
        match self {
            Self::Cartesian => ["x", "y", "z"],
            Self::Cylindrical => ["rho", "phi", "z"],
            Self::Spherical => ["r", "theta", "phi"],
        }
    }

    /// Basis vector display names `(e1, e2, e3)`.
    #[must_use]
    pub const fn basis_labels(self) -> [&'static str; 3] {
        match self {
            Self::Cartesian => ["e_x", "e_y", "e_z"],
            Self::Cylindrical => ["e_rho", "e_phi", "e_z"],
            Self::Spherical => ["e_r", "e_theta", "e_phi"],
        }
    }

    /// Singularity-safe default `(domain_min, domain_max)` for each coordinate frame.
    #[must_use]
    pub const fn default_domain(self) -> ([f64; 3], [f64; 3]) {
        match self {
            Self::Cartesian => ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
            Self::Cylindrical => ([0.8, 0.15, -0.9], [2.0, 1.75, 0.9]),
            Self::Spherical => ([1.0, 0.45, 0.15], [2.1, 1.55, 1.65]),
        }
    }

    /// Singularity-safe default `(normal_axis, fixed_val, u_bounds, v_bounds)` for Stokes' patch.
    #[must_use]
    pub const fn default_stokes_patch(self) -> (usize, f64, [f64; 2], [f64; 2]) {
        match self {
            Self::Cartesian => (2, 0.25, [-0.8, 0.8], [-0.8, 0.8]),
            Self::Cylindrical => (2, 0.2, [0.9, 1.9], [0.2, 1.6]),
            Self::Spherical => (0, 1.6, [0.5, 1.45], [0.2, 1.5]),
        }
    }

    /// Default probe coordinate inside `default_domain`.
    #[must_use]
    pub const fn default_probe(self) -> [f64; 3] {
        match self {
            Self::Cartesian => [0.4, 0.3, 0.2],
            Self::Cylindrical => [1.35, 0.8, 0.25],
            Self::Spherical => [1.5, 0.95, 0.75],
        }
    }

    /// Default `(F1, F2, F3, Phi)` expressions tailored to each coordinate frame.
    #[must_use]
    pub const fn default_expressions(
        self,
    ) -> (&'static str, &'static str, &'static str, &'static str) {
        match self {
            Self::Cartesian => (
                "a * x - y",
                "a * y + x",
                "a * z",
                "0.5 * a * (x^2 + y^2 + z^2)",
            ),
            Self::Cylindrical => ("a * rho", "rho^2", "a * z", "0.5 * a * (rho^2 + z^2)"),
            Self::Spherical => ("a * r^2", "sin(theta)", "r * sin(theta)", "a * r^2"),
        }
    }
}

/// Cached differential operator, Gauss/Stokes integral, and Helmholtz decomposition diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct CurvilinearDiagnostics {
    /// Local curvilinear gradient $\nabla \Phi$ at `probe_point`.
    pub probe_gradient: Vector3<f64>,
    /// Local curvilinear divergence $\nabla \cdot \mathbf{F}$ at `probe_point`.
    pub probe_divergence: f64,
    /// Local curvilinear curl $\nabla \times \mathbf{F}$ at `probe_point`.
    pub probe_curl: Vector3<f64>,
    /// Local curvilinear scalar Laplacian $\nabla^2 \Phi$ at `probe_point`.
    pub probe_laplacian: f64,
    /// Local curvilinear scale factors $(h_1, h_2, h_3)$ at `probe_point`.
    pub probe_scale_factors: Vector3<f64>,
    /// Gauss volume integral $\iiint_V (\nabla \cdot \mathbf{F})\,dV$.
    pub gauss_volume: f64,
    /// Gauss boundary flux $\oiint_{\partial V} (\mathbf{F} \cdot \mathbf{n})\,dS$.
    pub gauss_flux: f64,
    /// Absolute Gauss theorem discrepancy $|\text{LHS} - \text{RHS}|$.
    pub gauss_residual: f64,
    /// Stokes surface curl flux $\iint_\Sigma (\nabla \times \mathbf{F}) \cdot d\mathbf{S}$.
    pub stokes_surface_curl: f64,
    /// Stokes closed-loop circulation $\oint_{\partial \Sigma} \mathbf{F} \cdot d\mathbf{l}$.
    pub stokes_loop_circulation: f64,
    /// Absolute Stokes theorem discrepancy $|\text{LHS} - \text{RHS}|$.
    pub stokes_residual: f64,
    /// Summary of zero-allocation 3D Helmholtz-Hodge decomposition.
    pub helmholtz: HelmholtzSummary,
}

impl Default for CurvilinearDiagnostics {
    fn default() -> Self {
        Self {
            probe_gradient: Vector3::zeros(),
            probe_divergence: 0.0,
            probe_curl: Vector3::zeros(),
            probe_laplacian: 0.0,
            probe_scale_factors: Vector3::new(1.0, 1.0, 1.0),
            gauss_volume: 0.0,
            gauss_flux: 0.0,
            gauss_residual: 0.0,
            stokes_surface_curl: 0.0,
            stokes_loop_circulation: 0.0,
            stokes_residual: 0.0,
            helmholtz: HelmholtzSummary {
                max_solenoidal_divergence: 0.0,
                max_irrotational_curl: 0.0,
                max_reconstruction_error: 0.0,
                iterations: 0,
            },
        }
    }
}

/// Pre-projected Cartesian 3D quiver sample `[pos, f_total, f_irr, f_sol]` for viewport rendering.
#[derive(Debug, Clone, Copy)]
pub(crate) struct QuiverSample3D {
    pub pos: [f64; 3],
    pub total: [f64; 3],
    pub irrotational: [f64; 3],
    pub solenoidal: [f64; 3],
}

/// Interactive 3D Curvilinear Vector Calculus and Helmholtz Decomposition Studio.
pub struct CurvilinearVectorStudio {
    /// Active orthogonal coordinate system (`Cartesian`, `Cylindrical`, `Spherical`).
    pub frame: CoordinateFrameKind,
    /// Expression input bar for component $F_1$.
    pub f1_bar: MathExpressionInputBar,
    /// Expression input bar for component $F_2$.
    pub f2_bar: MathExpressionInputBar,
    /// Expression input bar for component $F_3$.
    pub f3_bar: MathExpressionInputBar,
    /// Expression input bar for scalar potential $\Phi$.
    pub scalar_bar: MathExpressionInputBar,
    /// Shared synthesized parameter sliders across `(F_1, F_2, F_3, Phi)`.
    pub param_sliders: ParameterSliderBank,
    /// Optional formula compilation error banner.
    pub compile_error: Option<String>,
    /// Curvilinear `Domain` lower bounds `[u1_min, u2_min, u3_min]`.
    pub domain_min: [f64; 3],
    /// Curvilinear `Domain` upper bounds `[u1_max, u2_max, u3_max]`.
    pub domain_max: [f64; 3],
    /// Normal coordinate axis (`0`, `1`, or `2`) for `StokesPatch`.
    pub stokes_axis: usize,
    /// Constant coordinate value along `stokes_axis`.
    pub stokes_fixed: f64,
    /// Tangent `u` interval `[u_min, u_max]` for `StokesPatch`.
    pub stokes_u: [f64; 2],
    /// Tangent `v` interval `[v_min, v_max]` for `StokesPatch`.
    pub stokes_v: [f64; 2],
    /// Curvilinear probe point `[u1, u2, u3]` for local operator inspection.
    pub probe_point: [f64; 3],
    /// 3D Helmholtz grid resolution per axis (`5..=32`).
    pub grid_res: usize,
    /// Quadrature subdivisions per axis for Gauss and Stokes verification (`6..=48`).
    pub integration_steps: usize,
    /// Gauss-Seidel SOR sweeps for curvilinear Poisson relaxation (`20..=240`).
    pub poisson_iters: usize,
    /// Visual arrow length scale in 3D viewport units.
    pub quiver_scale: f64,
    /// Show total field $\mathbf{F}$ quivers.
    pub show_total_field: bool,
    /// Show curl-free irrotational component $\mathbf{F}_{\text{irr}} = -\nabla \Phi$.
    pub show_irrotational: bool,
    /// Show divergence-free solenoidal component $\mathbf{F}_{\text{sol}}$.
    pub show_solenoidal: bool,
    /// Show curvilinear `Domain` boundary wireframe.
    pub show_domain_wireframe: bool,
    /// Show `StokesPatch` surface grid and oriented circulation loop.
    pub show_stokes_loop: bool,
    /// Shared 3D orbital camera.
    pub camera: Camera3D,
    /// Latest computed differential and integral diagnostics.
    pub diagnostics: CurvilinearDiagnostics,
    pub(crate) workspace: HelmholtzWorkspace,
    pub(crate) field_samples: Vec<Vector3<f64>>,
    pub(crate) quivers: Vec<QuiverSample3D>,
    pub(crate) compiled_quad: (CompiledExpr, CompiledExpr, CompiledExpr, CompiledExpr),
    pub(crate) dirty: bool,
    pub(crate) state_rev: u64,
}

fn compile_zero_3d() -> CompiledExpr {
    CompiledExpr::compile_with_vars(&parse("0").expect("0 parses"), &["x", "y", "z"])
        .expect("0 compiles")
}

impl Default for CurvilinearVectorStudio {
    fn default() -> Self {
        let frame = CoordinateFrameKind::Cartesian;
        let (f1, f2, f3, phi) = frame.default_expressions();
        let vars = frame.axis_vars();
        let f1_bar = MathExpressionInputBar::new("F_1", f1, vec![])
            .with_axes(&vars)
            .with_inline_sliders(false);
        let f2_bar = MathExpressionInputBar::new("F_2", f2, vec![])
            .with_axes(&vars)
            .with_inline_sliders(false);
        let f3_bar = MathExpressionInputBar::new("F_3", f3, vec![])
            .with_axes(&vars)
            .with_inline_sliders(false);
        let scalar_bar = MathExpressionInputBar::new("Phi", phi, vec![])
            .with_axes(&vars)
            .with_inline_sliders(false);

        let (d_min, d_max) = frame.default_domain();
        let (s_axis, s_fixed, s_u, s_v) = frame.default_stokes_patch();
        let grid_res = 10;
        let zero = compile_zero_3d();

        let mut studio = Self {
            frame,
            f1_bar,
            f2_bar,
            f3_bar,
            scalar_bar,
            param_sliders: ParameterSliderBank::new().with_range(-3.0..=3.0, 1.0),
            compile_error: None,
            domain_min: d_min,
            domain_max: d_max,
            stokes_axis: s_axis,
            stokes_fixed: s_fixed,
            stokes_u: s_u,
            stokes_v: s_v,
            probe_point: frame.default_probe(),
            grid_res,
            integration_steps: 16,
            poisson_iters: 100,
            quiver_scale: 0.22,
            show_total_field: true,
            show_irrotational: true,
            show_solenoidal: true,
            show_domain_wireframe: true,
            show_stokes_loop: true,
            camera: Camera3D::new(0.52, 0.62, 1.0),
            diagnostics: CurvilinearDiagnostics::default(),
            workspace: HelmholtzWorkspace::new(grid_res, grid_res, grid_res),
            field_samples: vec![Vector3::zeros(); grid_res * grid_res * grid_res],
            quivers: Vec::new(),
            compiled_quad: (zero.clone(), zero.clone(), zero.clone(), zero),
            dirty: true,
            state_rev: 1,
        };
        studio.recompile_expressions();
        studio.recompute_if_dirty();
        studio
    }
}

impl TheoryDescribable for CurvilinearVectorStudio {
    fn theory_description(&self) -> String {
        format!(
            "Orthogonal curvilinear vector calculus in {} verifying Gauss's divergence theorem ($|\\Delta| = {:.2e}$), Kelvin-Stokes circulation ($|\\Delta| = {:.2e}$), and zero-allocation Helmholtz-Hodge decomposition $\\mathbf{{F}} = -\\nabla\\Phi + \\mathbf{{F}}_{{\\mathrm{{sol}}}}$ on a ${}^3$ grid.",
            self.frame.label(),
            self.diagnostics.gauss_residual,
            self.diagnostics.stokes_residual,
            self.grid_res
        )
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(
            r"\iiint_V (\nabla \cdot \mathbf{F})\,dV = \oiint_{\partial V} \mathbf{F}\cdot d\mathbf{S}, \quad \iint_\Sigma (\nabla\times\mathbf{F})\cdot d\mathbf{S} = \oint_{\partial\Sigma} \mathbf{F}\cdot d\mathbf{l}".to_string(),
        )
    }

    fn phonetic_description(&self) -> String {
        format!(
            "Curvilinear vector studio in {} with Gauss residual {:.4} and Stokes residual {:.4}.",
            self.frame.label(),
            self.diagnostics.gauss_residual,
            self.diagnostics.stokes_residual
        )
    }

    fn theory_citation(&self) -> String {
        "Arfken, G. B., Weber, H. J., & Harris, F. E. (2012). Mathematical Methods for Physicists (7th ed.). Academic Press; Chorin, A. J., & Marsden, J. E. (1993).".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "helmholtz".to_string(),
            "Splits a 3D vector field into curl-free (-grad Phi) and divergence-free solenoidal components on flat 3D buffers.".to_string(),
        );
        map
    }
}

impl InteractiveTool for CurvilinearVectorStudio {
    fn name(&self) -> &'static str {
        "Curvilinear Vector Studio"
    }

    fn domain(&self) -> &'static str {
        "analysis"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash as _, Hasher as _};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.state_rev.hash(&mut h);
        self.camera.hash_state(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        self.parameters_are_valid()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.clamp_all_bounds();
        true
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.collect_export_datasets()
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        self.collect_export_primitives()
    }

    fn save_state(&self) -> Option<String> {
        self.serialize_studio_state()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.deserialize_studio_state(state)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.recompute_if_dirty();
        self.render_side_panel(ui);
        self.render_viewport_panel(ui);
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

inventory::submit! {
    ToolMetadata {
        id: "CurvilinearVectorStudio",
        display_name: "Curvilinear Vector Studio",
        domain: "analysis",
        tags: &["vector-calculus", "curvilinear", "stokes", "gauss", "helmholtz", "3d"],
        build: || Box::new(CurvilinearVectorStudio::default()),
    }
}

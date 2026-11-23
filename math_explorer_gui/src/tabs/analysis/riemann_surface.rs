//! Interactive 3D Riemann Surface, Monodromy Winding, Schwarz-Christoffel Conformal Polygon,
//! Schwarz Reflection, and Cauchy/Residue Studio (`RiemannSurfaceStudio`).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:graph_parameters_rust]

use crate::camera::Camera3D;
use crate::framework::{ExportCapabilities, InteractiveTool, MathExpressionInputBar, ToolMetadata};
use eframe::egui;
use num_complex::Complex64;
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

#[path = "riemann_surface/conformal_and_cauchy.rs"]
mod conformal_and_cauchy;
#[path = "riemann_surface/preset.rs"]
pub mod preset;
#[path = "riemann_surface/surfaces_3d.rs"]
mod surfaces_3d;
#[path = "riemann_surface/ui.rs"]
mod ui;
#[path = "riemann_surface/viewport.rs"]
mod viewport;

#[cfg(test)]
#[path = "riemann_surface/tests.rs"]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
#[path = "riemann_surface/wasm_tests.rs"]
mod wasm_tests;

/// Multi-sheet Riemann surface family backed by `pure_math::analysis::complex::surfaces`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum RiemannBranchFamily {
    /// Algebraic $n$-th root surface $w = z^{1/n}$ (`NthRootSurface`) with $n$ cyclic sheets.
    NthRoot,
    /// Infinite-sheeted logarithmic helicoid $w = \ln(z) + 2\pi i k$ (`LogSurface`).
    Logarithm,
}

impl RiemannBranchFamily {
    /// Human-readable surface family label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::NthRoot => "Nth-Root Surface w = z^(1/n)",
            Self::Logarithm => "Logarithmic Helicoid w = ln(z)",
        }
    }
}

/// Real scalar component mapped to vertical 3D height $Z$ on the Riemann surface mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SurfaceHeightChannel {
    /// Height $Z = \operatorname{Im}(w)$ (classic helicoid for $\ln z$, smooth $n$-sheeted ramp for $z^{1/n}$).
    ImaginaryPart,
    /// Height $Z = \operatorname{Re}(w)$ (classic self-intersecting Riemann surface for $\operatorname{Re}(z^{1/2})$).
    RealPart,
}

impl SurfaceHeightChannel {
    /// Human-readable height channel label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::ImaginaryPart => "Height Z = Im(w), Color = Re(w)",
            Self::RealPart => "Height Z = Re(w), Color = Im(w)",
        }
    }
}

/// Mode displayed in the synchronized 2D complex companion panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum CompanionInspectorMode {
    /// Upper-half-plane Schwarz-Christoffel polygon image + optional `schwarz_reflect` continuation.
    SchwarzChristoffelPolygon,
    /// Base-plane monodromy loop, pole location, and Cauchy / residue contour inspector.
    CauchyResidueContour,
}

impl CompanionInspectorMode {
    /// Human-readable companion mode label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SchwarzChristoffelPolygon => "Schwarz-Christoffel & Schwarz Reflection",
            Self::CauchyResidueContour => "Cauchy Integral & Residue Contour",
        }
    }
}

/// Preconfigured Schwarz-Christoffel upper-half-plane polygon vertex/exponent configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum PolygonPreset {
    /// Finite triangle with prevertices `[-1, 1]` and interior angles $(\pi/3, \pi/3, \pi/3)$.
    EquilateralTriangle,
    /// Channel corner with prevertices `[-1, 0, 1]` and exponents `[-0.5, 0.5, -0.5]`.
    SteppedChannel,
    /// Symmetric wedge/quadrilateral with prevertices `[-1.5, -0.5, 0.5, 1.5]`.
    SymmetricQuad,
}

impl PolygonPreset {
    /// Human-readable polygon preset label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::EquilateralTriangle => "Equilateral Triangle [-1, 1]",
            Self::SteppedChannel => "Stepped Channel [-1, 0, 1]",
            Self::SymmetricQuad => "Symmetric Quad [-1.5, -0.5, 0.5, 1.5]",
        }
    }
}

/// Analytic / meromorphic test function for Cauchy integral, derivative, and residue evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MeromorphicPreset {
    /// Simple pole $f(z) = \frac{z + 1}{z - z_{\text{pole}}}$ with exact residue $z_{\text{pole}} + 1$.
    SimplePoleRational,
    /// Trigonometric simple pole $f(z) = \frac{\exp(0.5 z)}{z - z_{\text{pole}}}$ with residue $e^{0.5 z_{\text{pole}}}$.
    ExponentialOverPole,
    /// Entire cubic $f(z) = z^3 - 0.5 z + 0.25$ (zero residue, exact Cauchy derivatives).
    EntireCubic,
    /// Custom real-on-real polynomial/trig combination continued via `schwarz_reflect` divided by $(z - z_{\text{pole}})$.
    CustomReflectedPole,
}

impl MeromorphicPreset {
    /// Human-readable meromorphic function label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SimplePoleRational => "f(z) = (z + 1) / (z - z_p)",
            Self::ExponentialOverPole => "f(z) = exp(0.5 z) / (z - z_p)",
            Self::EntireCubic => "f(z) = z^3 - 0.5 z + 0.25 (Entire)",
            Self::CustomReflectedPole => "f(z) = g_custom(z) / (z - z_p)",
        }
    }
}

/// Precomputed 3D triangle quad on a Riemann sheet for `Scene3DBuilder` rendering.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SheetQuad3D {
    pub corners: [[f64; 3]; 4],
    pub color_scalar: f64,
}

/// Cached monodromy, conformal mapping, Schwarz reflection, and Cauchy/residue diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct RiemannDiagnostics {
    /// Sheet count returned by `RiemannSurface::num_sheets()` (`None` for infinite `LogSurface`).
    pub sheet_count: Option<usize>,
    /// Final sheet index after traversing `winding_turns` around `loop_center`.
    pub final_sheet: isize,
    /// Branch value $w(\gamma(0), \text{start\_sheet})$.
    pub start_value: Complex64,
    /// Branch value $w(\gamma(1), \text{final\_sheet})$.
    pub end_value: Complex64,
    /// Monodromy displacement $|w_{\text{end}} - w_{\text{start}}|$ (`0` when sheet cycle closes).
    pub monodromy_jump: f64,
    /// Local `conformal_scale_factor` $|f'(z_0)|$ of the Schwarz-Christoffel map.
    pub conformal_scale: f64,
    /// Local `area_magnification` $|f'(z_0)|^2 = \det J_f(z_0)$ of the Schwarz-Christoffel map.
    pub area_mag: f64,
    /// Value of `schwarz_reflect` analytic continuation at conjugated probe $\bar{z}_0$.
    pub reflected_value: Complex64,
    /// Direct contour integral $\oint_\gamma f(z)\,dz$ via `path_integrate`.
    pub contour_integral: Complex64,
    /// Cauchy integral formula value $\frac{1}{2\pi i}\oint_\gamma \frac{f(z)}{z - z_0}\,dz$.
    pub cauchy_value: Complex64,
    /// Exact analytic value $f(z_0)$ when $z_0$ lies inside $\gamma$.
    pub exact_probe_value: Complex64,
    /// Cauchy generalized derivative $f^{(n)}(z_0)$ via `cauchy_derivative`.
    pub cauchy_nth_deriv: Complex64,
    /// Contour-integral residue $\operatorname{Res}(f, z_p)$ via `residue_via_integral`.
    pub residue_integral: Complex64,
    /// Limit-formula simple-pole residue $\lim_{z\to z_p}(z - z_p)f(z)$ via `residue_simple_limit`.
    pub residue_limit: Complex64,
}

impl Default for RiemannDiagnostics {
    fn default() -> Self {
        let zero = Complex64::new(0.0, 0.0);
        Self {
            sheet_count: Some(2),
            final_sheet: 1,
            start_value: zero,
            end_value: zero,
            monodromy_jump: 0.0,
            conformal_scale: 1.0,
            area_mag: 1.0,
            reflected_value: zero,
            contour_integral: zero,
            cauchy_value: zero,
            exact_probe_value: zero,
            cauchy_nth_deriv: zero,
            residue_integral: zero,
            residue_limit: zero,
        }
    }
}

/// Interactive 3D Riemann Surface, Schwarz-Christoffel, and Cauchy/Residue Studio.
pub struct RiemannSurfaceStudio {
    /// Active multi-sheet surface family (`NthRoot` or `Logarithm`).
    pub branch_family: RiemannBranchFamily,
    /// Root degree $n \in [2, 6]$ for `NthRootSurface`.
    pub root_n: usize,
    /// Symmetric sheet span $K \in [1, 3]$ (`-K..=K`) rendered for `LogSurface`.
    pub log_sheet_span: usize,
    /// Scalar channel mapped to 3D height $Z$.
    pub height_channel: SurfaceHeightChannel,
    /// Outer polar radius $R_{\max} \in [0.6, 2.5]$ of the 3D surface disk.
    pub radial_max: f64,
    /// Radial subdivisions (`8..=28`).
    pub radial_steps: usize,
    /// Angular subdivisions per sheet (`16..=56`).
    pub angular_steps: usize,
    /// Initial sheet index $k_0$ at $t = 0$ on the monodromy loop.
    pub start_sheet: isize,
    /// Signed integer winding turns $W \in [-4, 4]$ around `loop_center`.
    pub winding_turns: isize,
    /// Radius $R_\gamma \in [0.2, 2.0]$ of the closed winding/Cauchy contour $\gamma$.
    pub loop_radius: f64,
    /// Base-plane center `[re, im]` of the winding/Cauchy contour $\gamma$.
    pub loop_center: [f64; 2],
    /// Mode of the companion 2D complex plane inspector.
    pub companion_mode: CompanionInspectorMode,
    /// Schwarz-Christoffel polygon preset.
    pub polygon_preset: PolygonPreset,
    /// Complex scale factor $|A| \in [0.3, 2.5]$ for `SchwarzChristoffel`.
    pub sc_scale: f64,
    /// Toggle `schwarz_reflect` analytic continuation across $\operatorname{Im}(z) = 0$.
    pub enable_schwarz_reflection: bool,
    /// Active meromorphic function for Cauchy & residue integration.
    pub meromorphic_preset: MeromorphicPreset,
    /// Real-axis numerator formula bar `g(x)` for `MeromorphicPreset::CustomReflectedPole`.
    pub custom_numerator_bar: MathExpressionInputBar,
    /// Interior probe point $z_0 = [x_0, y_0]$ for conformal scale & Cauchy evaluation.
    pub probe_z: [f64; 2],
    /// Simple pole coordinate $z_p = [x_p, y_p]$ for residue verification.
    pub pole_z: [f64; 2],
    /// Quadrature step count (`32..=400`) for contour integrals.
    pub contour_steps: usize,
    /// Derivative order $n \in [1, 3]$ for `cauchy_derivative`.
    pub derivative_order: usize,
    /// Render wireframe edges on the 3D Riemann sheets.
    pub show_wireframe: bool,
    /// Render the lifted 3D monodromy curve on `Plot3DViewport`.
    pub show_monodromy_path: bool,
    /// Shared 3D orbital camera.
    pub camera: Camera3D,
    /// Latest monodromy, conformal, and Cauchy/residue diagnostics.
    pub diagnostics: RiemannDiagnostics,
    pub(crate) sheet_quads: Vec<SheetQuad3D>,
    pub(crate) color_range: (f64, f64),
    pub(crate) lifted_path_3d: Vec<[f64; 3]>,
    pub(crate) base_loop_2d: Vec<[f64; 2]>,
    pub(crate) sc_upper_curves: Vec<Vec<[f64; 2]>>,
    pub(crate) sc_reflected_curves: Vec<Vec<[f64; 2]>>,
    pub(crate) sc_probe_image: [f64; 2],
    pub(crate) sc_reflected_probe_image: [f64; 2],
    pub(crate) dirty: bool,
    pub(crate) state_rev: u64,
}

impl Default for RiemannSurfaceStudio {
    fn default() -> Self {
        let custom_bar = MathExpressionInputBar::new("g(x)", "cos(x) + 0.5 * x^2", vec![])
            .with_axes(&["x"])
            .with_inline_sliders(false);
        let mut studio = Self {
            branch_family: RiemannBranchFamily::NthRoot,
            root_n: 2,
            log_sheet_span: 1,
            height_channel: SurfaceHeightChannel::ImaginaryPart,
            radial_max: 1.8,
            radial_steps: 14,
            angular_steps: 32,
            start_sheet: 0,
            winding_turns: 1,
            loop_radius: 1.15,
            loop_center: [0.0, 0.0],
            companion_mode: CompanionInspectorMode::CauchyResidueContour,
            polygon_preset: PolygonPreset::EquilateralTriangle,
            sc_scale: 1.0,
            enable_schwarz_reflection: true,
            meromorphic_preset: MeromorphicPreset::SimplePoleRational,
            custom_numerator_bar: custom_bar,
            probe_z: [0.35, 0.45],
            pole_z: [0.25, -0.2],
            contour_steps: 160,
            derivative_order: 1,
            show_wireframe: true,
            show_monodromy_path: true,
            camera: Camera3D::new(0.54, 0.68, 0.95),
            diagnostics: RiemannDiagnostics::default(),
            sheet_quads: Vec::new(),
            color_range: (-1.0, 1.0),
            lifted_path_3d: Vec::new(),
            base_loop_2d: Vec::new(),
            sc_upper_curves: Vec::new(),
            sc_reflected_curves: Vec::new(),
            sc_probe_image: [0.0, 0.0],
            sc_reflected_probe_image: [0.0, 0.0],
            dirty: true,
            state_rev: 1,
        };
        studio.recompute_if_dirty();
        studio
    }
}

impl TheoryDescribable for RiemannSurfaceStudio {
    fn theory_description(&self) -> String {
        format!(
            "{} with monodromy winding $W = {}$ ($k_0 = {} \\to k_1 = {}$, $|\\Delta w| = {:.4}$), Schwarz-Christoffel conformal scale $|f'(z_0)| = {:.4}$ ($\\det J = {:.4}$), Cauchy integral $f(z_0) = {:.4} + {:.4}i$, and simple-pole residue $\\operatorname{{Res}} = {:.4} + {:.4}i$.",
            self.branch_family.label(),
            self.winding_turns,
            self.start_sheet,
            self.diagnostics.final_sheet,
            self.diagnostics.monodromy_jump,
            self.diagnostics.conformal_scale,
            self.diagnostics.area_mag,
            self.diagnostics.cauchy_value.re,
            self.diagnostics.cauchy_value.im,
            self.diagnostics.residue_integral.re,
            self.diagnostics.residue_integral.im
        )
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(
            r"f^{(n)}(z_0) = \frac{n!}{2\pi i}\oint_\gamma \frac{f(z)}{(z - z_0)^{n+1}}\,dz, \qquad f(\bar{z}) = \overline{f(z)}, \qquad f_{\mathrm{SC}}(z) = A\int_{z_0}^z \prod_k (\zeta - x_k)^{\alpha_k - 1}\,d\zeta + B".to_string(),
        )
    }

    fn phonetic_description(&self) -> String {
        format!(
            "Riemann surface studio showing {} on sheet {} transitioning to sheet {}.",
            self.branch_family.label(),
            self.start_sheet,
            self.diagnostics.final_sheet
        )
    }

    fn theory_citation(&self) -> String {
        "Ahlfors, L. V. (1979). Complex Analysis (3rd ed.). McGraw-Hill; Driscoll, T. A., & Trefethen, L. N. (2002). Schwarz-Christoffel Mapping. Cambridge University Press.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "monodromy".to_string(),
            "Analytic continuation around a branch point permutes sheets via next_sheet(k, winding).".to_string(),
        );
        map
    }
}

impl InteractiveTool for RiemannSurfaceStudio {
    fn name(&self) -> &'static str {
        "Riemann Surface Studio"
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
        self.render_central_workspace(ui);
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
        id: "RiemannSurfaceStudio",
        display_name: "Riemann Surface Studio",
        domain: "analysis",
        tags: &["complex-analysis", "riemann-surface", "monodromy", "schwarz-christoffel", "cauchy", "3d"],
        build: || Box::new(RiemannSurfaceStudio::default()),
    }
}

// [cite:modular_polynomials_review]
//! Interactive 3D Laplace-Beltrami reaction-diffusion and mean-curvature buckling studio.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod export;
mod presets;
mod solver;
mod state;
mod ui;

use super::custom_surface::{CustomSurfaceSandbox, SurfaceType};
use crate::framework::recovery::FiniteStateGuard as _;
use crate::framework::{Camera3D, ExportCapabilities, InteractiveTool, Scene3DBuilder};
use eframe::egui;
use math_explorer::pure_math::differential_geometry::surface::{
    KleinBottle, ParametricSurface, Sphere, Torus,
};
use scientific_metadata::theory::TheoryDescribable;
use solver::MorphogenesisSurfaceEngine;
use state::{
    default_buckling_gain, default_d_u, default_d_v, default_dt, default_gs_feed, default_gs_kill,
    default_schnak_a, default_schnak_b, default_steps, default_tension, default_tether,
    default_wall_thickness,
};
use std::collections::HashMap;
use std::hash::{Hash as _, Hasher as _};

pub(super) const MIN_SURFACE_MORPHO_RES: usize = 8;
pub(super) const MAX_SURFACE_MORPHO_RES: usize = crate::presets::MAX_INTERACTIVE_SURFACE_RESOLUTION;

/// Reaction kinetics coupled to the surface Laplace-Beltrami operator.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum MorphogenesisKineticsKind {
    /// Activator-depleted substrate Schnakenberg Turing stripes/spots.
    #[default]
    Schnakenberg,
    /// Autocatalytic Gray-Scott self-replicating mitosis spots and labyrinths.
    GrayScott,
}

impl MorphogenesisKineticsKind {
    /// Human-readable label for UI controls.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Schnakenberg => "Schnakenberg (Turing)",
            Self::GrayScott => "Gray-Scott (Mitosis)",
        }
    }
}

/// Scalar field rendered as per-vertex colors on the 3D viewport and `.ply`/`.obj` exports.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum SurfaceOverlayMode {
    /// Active morphogen species concentration (`u` for Schnakenberg, `v` for Gray-Scott).
    #[default]
    MorphogenConcentration,
    /// Intrinsic Gaussian curvature $K = (LN - M^2)/(EG - F^2)$.
    GaussianCurvature,
    /// Principal metric stretch ratio $\sqrt{\lambda_{\max}/\lambda_{\min}}$.
    MetricAnisotropy,
}

impl SurfaceOverlayMode {
    /// Human-readable label for UI controls.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::MorphogenConcentration => "Morphogen Concentration",
            Self::GaussianCurvature => "Gaussian Curvature (K)",
            Self::MetricAnisotropy => "Metric Anisotropy",
        }
    }
}

/// Primary pointer gesture mode inside the 3D viewport.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum ViewportInteractionMode {
    /// Standard 3D camera orbit/pan/zoom (`Shift+Drag` still stamps morphogen pulses).
    #[default]
    OrbitCamera,
    /// Direct 3D surface pulse painting along front-most projected manifold vertices.
    PulseBrush,
}

/// Chemical perturbation applied by the 3D viewport pulse brush.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum PulseBrushAction {
    /// Injects activator species `u`.
    #[default]
    ActivatorPulse,
    /// Injects inhibitor species `v`.
    InhibitorPulse,
    /// Restores homogeneous steady-state concentrations inside the brush neighborhood.
    ResetPatch,
}

/// Interactive 3D Laplace-Beltrami morphogenesis and mean-curvature buckling tool.
pub struct SurfaceMorphogenesisTool {
    pub(crate) surface_type: SurfaceType,
    pub(crate) kinetics_kind: MorphogenesisKineticsKind,
    pub(crate) overlay_mode: SurfaceOverlayMode,
    pub(crate) interaction_mode: ViewportInteractionMode,
    pub(crate) brush_action: PulseBrushAction,
    pub(crate) brush_radius: f64,
    pub(crate) brush_strength: f64,
    pub(crate) sphere_radius: f64,
    pub(crate) torus_major_radius: f64,
    pub(crate) torus_minor_radius: f64,
    pub(crate) klein_radius: f64,
    pub(crate) wall_thickness: f64,
    pub(crate) u_resolution: usize,
    pub(crate) v_resolution: usize,
    pub(crate) schnak_a: f64,
    pub(crate) schnak_b: f64,
    pub(crate) gs_feed: f64,
    pub(crate) gs_kill: f64,
    pub(crate) d_u: f64,
    pub(crate) d_v: f64,
    pub(crate) buckling_gain: f64,
    pub(crate) curvature_tension: f64,
    pub(crate) restoring_tether: f64,
    pub(crate) dt: f64,
    pub(crate) steps_per_frame: usize,
    pub(crate) is_playing: bool,
    pub(crate) camera: Camera3D,
    pub(crate) scene_builder: Scene3DBuilder,
    custom_sandbox: CustomSurfaceSandbox,
    engine: Option<MorphogenesisSurfaceEngine>,
    reseed_counter: u64,
}

impl Default for SurfaceMorphogenesisTool {
    fn default() -> Self {
        // Zero PDE steps and zero surface evaluations in Default::default()
        let camera = Camera3D::new(0.55, 0.50, 1.0);
        Self {
            surface_type: SurfaceType::Torus,
            kinetics_kind: MorphogenesisKineticsKind::Schnakenberg,
            overlay_mode: SurfaceOverlayMode::MorphogenConcentration,
            interaction_mode: ViewportInteractionMode::OrbitCamera,
            brush_action: PulseBrushAction::ActivatorPulse,
            brush_radius: 0.55,
            brush_strength: 0.45,
            sphere_radius: 1.5,
            torus_major_radius: 2.2,
            torus_minor_radius: 0.75,
            klein_radius: 2.0,
            wall_thickness: default_wall_thickness(),
            u_resolution: 28,
            v_resolution: 28,
            schnak_a: default_schnak_a(),
            schnak_b: default_schnak_b(),
            gs_feed: default_gs_feed(),
            gs_kill: default_gs_kill(),
            d_u: default_d_u(),
            d_v: default_d_v(),
            buckling_gain: default_buckling_gain(),
            curvature_tension: default_tension(),
            restoring_tether: default_tether(),
            dt: default_dt(),
            steps_per_frame: default_steps(),
            is_playing: false,
            camera,
            scene_builder: Scene3DBuilder::new(camera),
            custom_sandbox: CustomSurfaceSandbox::default(),
            engine: None,
            reseed_counter: 0,
        }
    }
}

fn canonical_bits(v: f64) -> u64 {
    if v == 0.0 {
        0
    } else {
        v.to_bits()
    }
}

impl SurfaceMorphogenesisTool {
    pub(super) fn clamped_u_res(&self) -> usize {
        self.u_resolution
            .clamp(MIN_SURFACE_MORPHO_RES, MAX_SURFACE_MORPHO_RES)
    }

    pub(super) fn clamped_v_res(&self) -> usize {
        self.v_resolution
            .clamp(MIN_SURFACE_MORPHO_RES, MAX_SURFACE_MORPHO_RES)
    }

    pub(super) fn create_base_surface(&self) -> Box<dyn ParametricSurface> {
        match self.surface_type {
            SurfaceType::Sphere => Box::new(Sphere {
                radius: self.sphere_radius,
            }),
            SurfaceType::Torus => Box::new(Torus {
                major_radius: self.torus_major_radius,
                minor_radius: self.torus_minor_radius,
            }),
            SurfaceType::KleinBottle => Box::new(KleinBottle {
                radius: self.klein_radius,
            }),
            SurfaceType::CustomHeightfield => Box::new(self.custom_sandbox.build_heightfield()),
            SurfaceType::CustomParametric => Box::new(self.custom_sandbox.build_parametric()),
        }
    }

    pub(super) fn geometry_fingerprint(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.surface_type as u8).hash(&mut h);
        self.clamped_u_res().hash(&mut h);
        self.clamped_v_res().hash(&mut h);
        match self.surface_type {
            SurfaceType::Sphere => canonical_bits(self.sphere_radius).hash(&mut h),
            SurfaceType::Torus => {
                canonical_bits(self.torus_major_radius).hash(&mut h);
                canonical_bits(self.torus_minor_radius).hash(&mut h);
            }
            SurfaceType::KleinBottle => canonical_bits(self.klein_radius).hash(&mut h),
            SurfaceType::CustomHeightfield | SurfaceType::CustomParametric => {
                self.custom_sandbox
                    .state_fingerprint(self.surface_type)
                    .hash(&mut h);
            }
        }
        h.finish().max(1)
    }

    pub(super) fn ensure_engine(&mut self) {
        let key = self.geometry_fingerprint();
        let rebuild = match self.engine.as_ref() {
            None => true,
            Some(eng) => eng.geometry_key != key || eng.kinetics_kind != self.kinetics_kind,
        };
        if rebuild {
            self.engine = Some(MorphogenesisSurfaceEngine::build(self));
        }
    }

    /// Advances the coupled reaction-diffusion and mean-curvature buckling simulation by one step.
    pub fn step_once(&mut self) -> bool {
        self.ensure_engine();
        let mut eng = self.engine.take().expect("engine initialized");
        let ok = eng.step_coupled(self);
        self.engine = Some(eng);
        ok
    }
}

impl TheoryDescribable for SurfaceMorphogenesisTool {
    fn theory_description(&self) -> String {
        "Couples intrinsic Laplace-Beltrami reaction-diffusion ∂u/∂t = D_u Δ_g u + R(u, v) to mean-curvature surface buckling ∂r/∂t = γ(u - u₀)n + κ H n across curved 2-manifolds.".to_string()
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(r"\partial_t u = D_u \Delta_g u + R_u(u, v)".to_string())
    }

    fn phonetic_description(&self) -> String {
        "partial t u equals D u Laplace-Beltrami u plus reaction R of u and v coupled to mean curvature normal buckling".to_string()
    }

    fn theory_citation(&self) -> String {
        "The Chemical Basis of Morphogenesis (Turing, 1952, DOI: 10.1098/rstb.1952.0012)"
            .to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

impl InteractiveTool for SurfaceMorphogenesisTool {
    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Surface Morphogenesis"
    }

    fn domain(&self) -> &'static str {
        "geometry_topology"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot().with_3d_solid()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let Some(eng) = self.engine.as_ref() else {
            return vec![(
                "Morphogen Equator".to_string(),
                vec![[0.0, 1.0], [1.0, 1.0]],
            )];
        };
        let row_j = eng.nv / 2;
        let profile: Vec<[f64; 2]> = (0..eng.nu)
            .map(|i| {
                let (u, _) = eng
                    .topology
                    .uv_at(eng.range_u, eng.range_v, eng.nu, eng.nv, i, row_j);
                let val = eng.overlay_field(self.overlay_mode)
                    [math_commons::math_kernel::types::flatten_2d_index(row_j, i, eng.nv)];
                [u, val]
            })
            .collect();
        vec![(self.overlay_mode.label().to_string(), profile)]
    }

    fn export_mesh_3d(&self) -> Option<Result<oxidize_core::mesh::Mesh, String>> {
        Some(self.build_buckled_mesh_3d())
    }

    fn export_colored_mesh_3d(&self) -> Option<Result<oxidize_core::mesh::ColoredMesh, String>> {
        Some(self.build_buckled_colored_mesh_3d())
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.geometry_fingerprint().hash(&mut h);
        (self.kinetics_kind as u8).hash(&mut h);
        (self.overlay_mode as u8).hash(&mut h);
        canonical_bits(self.wall_thickness).hash(&mut h);
        canonical_bits(self.d_u).hash(&mut h);
        canonical_bits(self.d_v).hash(&mut h);
        canonical_bits(self.buckling_gain).hash(&mut h);
        canonical_bits(self.curvature_tension).hash(&mut h);
        canonical_bits(self.dt).hash(&mut h);
        self.camera.hash_state(&mut h);
        if let Some(eng) = self.engine.as_ref() {
            eng.step_count.hash(&mut h);
        }
        h.finish().max(1)
    }

    fn state_bounds(&self) -> crate::framework::state_bounds::ToolStateBounds {
        crate::url_state::URL_JSON_BOUNDS
    }

    fn has_finite_parameters(&self) -> bool {
        self.is_finite_state()
    }

    fn save_state(&self) -> Option<String> {
        self.encode_state_json()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        // Zero PDE steps inside try_load_state(); lazy engine rebuild deferred to step/UI/export
        self.apply_loaded_state(state)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.draw_side_panel(ui);
        self.draw_viewport(ui);
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
    crate::framework::ToolMetadata {
        id: "SurfaceMorphogenesisTool",
        display_name: "Surface Morphogenesis",
        domain: "geometry_topology",
        tags: &["morphogenesis", "laplace-beltrami", "turing", "buckling", "3d"],
        build: || Box::new(SurfaceMorphogenesisTool::default()),
    }
}

#[cfg(test)]
mod tests;

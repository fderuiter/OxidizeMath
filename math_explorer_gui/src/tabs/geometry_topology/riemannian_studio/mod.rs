#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Flagship 3D Riemannian & Cortical PDE Studio (`RiemannianPdeStudio`) unifying intrinsic
//! CSR cotangent differential geometry, Turing morphogenesis, Hodgkin-Huxley cortical waves,
//! and FreeSurfer cortical morphometry & group GLM inference.

pub mod audio;
pub mod export;
#[cfg(feature = "applied")]
pub mod freesurfer_overlay;
pub mod handoff;
pub mod mesh_gen;
pub mod presets;
pub mod solver;
mod solver_init;
pub mod state;
mod ui;
mod viewport;

#[cfg(test)]
mod tests;

use super::custom_surface::CustomSurfaceSandbox;
#[cfg(test)]
use super::surface_brush::GeodesicBrushStroke;
use super::surface_brush::SurfaceBrushController;
use crate::framework::{ExportCapabilities, InteractiveTool, Scene3DBuilder};
use eframe::egui;
use export::build_studio_colored_mesh;
use mesh_gen::{build_manifold_geometry, ManifoldGeometry};
use solver::RiemannianSimulationEngine;
use state::{PdeRegime, RiemannianStudioConfig};
use std::collections::HashMap;

/// Interactive 3D manifold PDE, cortical electrophysiology, and FreeSurfer morphometry studio.
pub struct RiemannianPdeStudio {
    /// Serializable studio configuration (`< 4 KiB` JSON).
    pub config: RiemannianStudioConfig,
    /// Shared compiled expression sandbox for custom heightfield and parametric manifolds.
    pub custom_sandbox: CustomSurfaceSandbox,
    /// 3D raycast and WCAG 2.2 AA geodesic cursor brush controller.
    pub brush: SurfaceBrushController,
    pub(crate) geometry: Option<ManifoldGeometry>,
    pub(crate) engine: Option<RiemannianSimulationEngine>,
    pub(crate) scene_builder: Scene3DBuilder,
    pub(crate) last_canvas_id: Option<egui::Id>,
    pub(crate) viewport_response_id: Option<egui::Id>,
}

impl Default for RiemannianPdeStudio {
    fn default() -> Self {
        let config = RiemannianStudioConfig::default();
        let cam = config.camera;
        Self {
            config,
            custom_sandbox: CustomSurfaceSandbox::default(),
            brush: SurfaceBrushController::default(),
            geometry: None,
            engine: None,
            scene_builder: Scene3DBuilder::new(cam),
            last_canvas_id: None,
            viewport_response_id: None,
        }
    }
}

impl RiemannianPdeStudio {
    /// Lazily constructs [`ManifoldGeometry`] and [`RiemannianSimulationEngine`] on first active use.
    pub fn ensure_initialized(&mut self) {
        if self.geometry.is_none() || self.engine.is_none() {
            self.rebuild_all();
        }
    }

    /// Rebuilds discretized manifold geometry, CSR cotangent Laplacian, and all coupled solvers.
    pub fn rebuild_all(&mut self) {
        self.config.clamp_in_place();
        let geom = build_manifold_geometry(&self.config, &self.custom_sandbox);
        self.brush.clamp_cursor(geom.positions.len());
        let eng = RiemannianSimulationEngine::new(&geom, &self.config, self.brush.cursor_vertex);
        self.geometry = Some(geom);
        self.engine = Some(eng);
    }

    /// Synchronizes the rendered vertex scalar field when switching regimes or overlays.
    pub fn sync_active_regime(&mut self) {
        self.ensure_initialized();
        if let (Some(geom), Some(eng)) = (&self.geometry, &mut self.engine) {
            eng.sync_active_field(geom, &self.config);
        }
    }

    /// Advances the active manifold PDE or deformable pial surface by one step.
    pub fn step_once(&mut self) {
        self.ensure_initialized();
        if let (Some(geom), Some(eng)) = (&mut self.geometry, &mut self.engine) {
            eng.step(geom, &self.config, self.brush.cursor_vertex);
            if !eng.scalar_field.iter().all(|v| v.is_finite()) {
                crate::audio::notify_solver_diverged();
            }
        }
    }

    /// Programmatically applies a [`GeodesicBrushStroke`] onto the active manifold.
    #[cfg(test)]
    pub fn apply_stroke(&mut self, stroke: GeodesicBrushStroke) {
        self.ensure_initialized();
        if let (Some(geom), Some(eng)) = (&mut self.geometry, &mut self.engine) {
            eng.apply_brush_stroke(geom, &self.config, stroke);
        }
    }

    fn build_export_colored_mesh(&self) -> Result<oxidize_core::mesh::ColoredMesh, String> {
        if let (Some(geom), Some(eng)) = (&self.geometry, &self.engine) {
            return build_studio_colored_mesh(
                geom,
                &self.config,
                &self.custom_sandbox,
                &eng.scalar_field,
            );
        }
        let geom = build_manifold_geometry(&self.config, &self.custom_sandbox);
        let eng = RiemannianSimulationEngine::new(&geom, &self.config, 0);
        build_studio_colored_mesh(&geom, &self.config, &self.custom_sandbox, &eng.scalar_field)
    }
}

impl InteractiveTool for RiemannianPdeStudio {
    fn name(&self) -> &'static str {
        "Riemannian PDE Studio"
    }

    fn domain(&self) -> &'static str {
        "geometry_topology"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot().with_3d_solid()
    }

    fn export_colored_mesh_3d(&self) -> Option<Result<oxidize_core::mesh::ColoredMesh, String>> {
        Some(self.build_export_colored_mesh())
    }

    fn export_mesh_3d(&self) -> Option<Result<oxidize_core::mesh::Mesh, String>> {
        Some(self.build_export_colored_mesh().map(|cm| cm.mesh))
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.last_canvas_id
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash as _, Hasher as _};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.config.manifold.hash(&mut h);
        self.config.regime.hash(&mut h);
        self.config.overlay.hash(&mut h);
        self.config.u_res.hash(&mut h);
        self.config.v_res.hash(&mut h);
        self.config.dt.to_bits().hash(&mut h);
        self.config.camera.hash_state(&mut h);
        if let Some(st) = self.config.manifold.as_custom_surface_type() {
            self.custom_sandbox.state_fingerprint(st).hash(&mut h);
        }
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        self.config.validate().is_ok()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.config.clamp_in_place();
        true
    }

    fn save_state(&self) -> Option<String> {
        let mut cfg = self.config.clone();
        if let Some(st) = cfg.manifold.as_custom_surface_type() {
            let (ce, pe, cp) = self.custom_sandbox.export_fields(st);
            let (ub, vb) = self.custom_sandbox.export_domain_bounds(st);
            cfg.custom_expr = ce;
            cfg.parametric_exprs = pe;
            cfg.custom_params = cp;
            cfg.u_bounds = ub;
            cfg.v_bounds = vb;
        } else {
            cfg.custom_expr = None;
            cfg.parametric_exprs = None;
            cfg.custom_params.clear();
            cfg.u_bounds = None;
            cfg.v_bounds = None;
        }
        serde_json::to_string(&cfg).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let mut loaded: RiemannianStudioConfig = serde_json::from_str(state)
            .map_err(|e| format!("Invalid Riemannian PDE Studio state: {e}"))?;
        loaded.regime = loaded.regime.coerce_to_enabled_features();
        loaded.validate()?;
        if let Some(st) = loaded.manifold.as_custom_surface_type() {
            self.custom_sandbox
                .load_domain_bounds(st, loaded.u_bounds, loaded.v_bounds)?;
        }
        self.custom_sandbox.load_fields(
            loaded.custom_expr.as_deref(),
            loaded.parametric_exprs.as_ref(),
            &loaded.custom_params,
        );
        self.brush.mode = loaded.brush_mode;
        self.brush.polarity = loaded.brush_polarity;
        self.brush.radius = loaded.brush_radius;
        self.brush.strength = loaded.brush_strength;
        self.config = loaded;
        self.geometry = None;
        self.engine = None;
        Ok(())
    }

    fn step_simulation(&mut self) {
        self.step_once();
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
        id: "RiemannianPdeStudio",
        display_name: "Riemannian PDE Studio",
        domain: "geometry_topology",
        tags: &["cotangent", "laplace-beltrami", "cortical", "turing", "freesurfer", "manifold"],
        build: || Box::new(RiemannianPdeStudio::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for RiemannianPdeStudio {
    fn theory_description(&self) -> String {
        let regime_eq = match self.config.regime {
            PdeRegime::HeatBeltrami => "intrinsic heat flow $\\partial_t u = \\alpha \\Delta_{\\mathcal{M}} u$",
            PdeRegime::MeanCurvatureSmoothing => "mean curvature flow $\\partial_t \\mathbf{r} = H\\,\\mathbf{n}$",
            PdeRegime::SchnakenbergTuring => "Schnakenberg Turing morphogenesis $\\partial_t u = D_u \\Delta_{\\mathcal{M}} u + a - u + u^2 v$",
            PdeRegime::GrayScottTuring => "Gray-Scott mitosis reaction-diffusion $\\partial_t v = D_v \\Delta_{\\mathcal{M}} v + u v^2 - (f+k)v$",
            PdeRegime::CorticalHodgkinHuxley => "thickness-weighted Hodgkin-Huxley cortical sheet $C_m \\partial_t V = g_s \\Delta_{\\mathcal{M}}(h V) - I_{\\text{ion}}$",
            PdeRegime::FreeSurferMorphometryGlm => "FreeSurfer white-to-pial cortical thickness and mass-univariate GLM $t = c^\\top \\hat{\\beta} / \\text{SE}(c^\\top \\hat{\\beta})$",
        };
        format!(
            "Discrete differential geometry studio evaluating {regime_eq} over a CSR Meyer-Desbrun-Schröder-Barr cotangent Laplace-Beltrami operator on {}.",
            self.config.manifold.label()
        )
    }

    fn phonetic_description(&self) -> String {
        "Riemannian cotangent Laplace-Beltrami PDEs, Turing morphogenesis, Hodgkin-Huxley cortical waves, and FreeSurfer morphometry".into()
    }

    fn theory_citation(&self) -> String {
        "Meyer, Desbrun, Schröder & Barr (2003), Discrete Differential-Geometry Operators for Triangulated 2-Manifolds; Turing (1952); Hodgkin & Huxley (1952); Fischl & Dale (2000).".into()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "CotangentLaplacian".into(),
            "Δ_S u_i = (1 / 2 A_i) Σ_j (cot α_ij + cot β_ij)(u_j - u_i) with mixed Voronoi areas"
                .into(),
        );
        map.insert(
            "CorticalThickness".into(),
            "Symmetric white-to-pial surface nearest-neighbor distance (Fischl & Dale, 2000)"
                .into(),
        );
        map
    }
}

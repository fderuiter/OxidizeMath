#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "curvature_heatmap/cache.rs"]
mod cache;
#[path = "curvature_export.rs"]
mod curvature_export;
#[path = "curvature_heatmap/presets.rs"]
mod presets;
#[path = "curvature_heatmap_ui.rs"]
mod ui;

use super::custom_surface::{CustomSurfaceSandbox, SurfaceType};
use crate::framework::{Camera3D, ExportCapabilities, InteractiveTool, Scene3DBuilder};
use eframe::egui;
use std::collections::BTreeMap;

#[derive(PartialEq, Eq, Hash, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub(super) enum CurvatureType {
    Gaussian,
    Mean,
}

#[derive(serde::Serialize)]
struct CurvatureHeatmapSaveState {
    surface_type: SurfaceType,
    curvature_type: CurvatureType,
    sphere_radius: f64,
    torus_major_radius: f64,
    torus_minor_radius: f64,
    klein_radius: f64,
    u_resolution: usize,
    v_resolution: usize,
    camera: Camera3D,
    #[serde(skip_serializing_if = "Option::is_none")]
    custom_expr: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parametric_exprs: Option<[String; 3]>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    custom_params: BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    u_bounds: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    v_bounds: Option<[f64; 2]>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CurvatureHeatmapState {
    surface_type: SurfaceType,
    curvature_type: CurvatureType,
    sphere_radius: f64,
    torus_major_radius: f64,
    torus_minor_radius: f64,
    klein_radius: f64,
    u_resolution: usize,
    v_resolution: usize,
    #[serde(default)]
    camera: Option<Camera3D>,
    #[serde(default)]
    pitch: Option<f32>,
    #[serde(default)]
    yaw: Option<f32>,
    #[serde(default)]
    zoom: Option<f32>,
    #[serde(default)]
    custom_expr: Option<String>,
    #[serde(default)]
    parametric_exprs: Option<[String; 3]>,
    #[serde(default)]
    custom_params: BTreeMap<String, f64>,
    #[serde(default)]
    u_bounds: Option<[f64; 2]>,
    #[serde(default)]
    v_bounds: Option<[f64; 2]>,
}

/// Interactive 3D parametric manifold Gaussian and mean curvature heatmap viewer.
pub struct CurvatureHeatmap {
    surface_type: SurfaceType,
    curvature_type: CurvatureType,
    sphere_radius: f64,
    torus_major_radius: f64,
    torus_minor_radius: f64,
    klein_radius: f64,
    u_resolution: usize,
    v_resolution: usize,
    pub(crate) camera: Camera3D,
    pub(crate) scene_builder: Scene3DBuilder,
    pub(super) custom_sandbox: CustomSurfaceSandbox,
    cache: cache::CurvatureGeometryCache,
}

impl Default for CurvatureHeatmap {
    fn default() -> Self {
        let camera = Camera3D::new(0.5, 0.5, 1.0);
        Self {
            surface_type: SurfaceType::Torus,
            curvature_type: CurvatureType::Gaussian,
            sphere_radius: 1.0,
            torus_major_radius: 2.0,
            torus_minor_radius: 0.5,
            klein_radius: 2.0,
            u_resolution: 30,
            v_resolution: 30,
            camera,
            scene_builder: Scene3DBuilder::new(camera),
            custom_sandbox: CustomSurfaceSandbox::default(),
            cache: cache::CurvatureGeometryCache::default(),
        }
    }
}

impl CurvatureHeatmap {
    /// Populates `self.scene_builder` with cached world `Point3<f64>` curvature-colored grid segments.
    pub fn populate_scene(&mut self, series_color: egui::Color32) {
        self.populate_cached_scene(series_color);
    }

    #[cfg(test)]
    pub(crate) fn debug_cache_counts(&self) -> (usize, usize) {
        (self.cache.curvature_eval_count, self.cache.projection_count)
    }

    #[cfg(test)]
    pub(crate) fn set_grid_resolutions(&mut self, u_res: usize, v_res: usize) {
        self.u_resolution = u_res;
        self.v_resolution = v_res;
    }
}

fn validate_curvature_radii_and_res(loaded: &CurvatureHeatmapState) -> Result<(), String> {
    if !loaded.sphere_radius.is_finite() || !(0.1..=5.0).contains(&loaded.sphere_radius) {
        return Err(format!(
            "sphere_radius out of range: {}",
            loaded.sphere_radius
        ));
    }
    if !loaded.torus_major_radius.is_finite() || !(1.0..=5.0).contains(&loaded.torus_major_radius) {
        return Err(format!(
            "torus_major_radius out of range: {}",
            loaded.torus_major_radius
        ));
    }
    if !loaded.torus_minor_radius.is_finite() || !(0.1..=2.0).contains(&loaded.torus_minor_radius) {
        return Err(format!(
            "torus_minor_radius out of range: {}",
            loaded.torus_minor_radius
        ));
    }
    if !loaded.klein_radius.is_finite() || !(0.5..=5.0).contains(&loaded.klein_radius) {
        return Err(format!(
            "klein_radius out of range: {}",
            loaded.klein_radius
        ));
    }
    let max_res = math_commons::registry::MAX_GRID_SIZE;
    if !(10..=max_res).contains(&loaded.u_resolution)
        || !(10..=max_res).contains(&loaded.v_resolution)
    {
        return Err("Resolution out of valid range".to_string());
    }
    Ok(())
}

impl InteractiveTool for CurvatureHeatmap {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Curvature Heatmap"
    }

    fn domain(&self) -> &'static str {
        "geometry_topology"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot().with_3d_solid()
    }

    fn export_colored_mesh_3d(&self) -> Option<Result<oxidize_core::mesh::ColoredMesh, String>> {
        Some(self.build_colored_mesh_3d())
    }

    fn export_mesh_3d(&self) -> Option<Result<oxidize_core::mesh::Mesh, String>> {
        Some(self.build_colored_mesh_3d().map(|cm| cm.mesh))
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash as _, Hasher as _};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.curvature_geometry_key().hash(&mut h);
        self.camera.hash_state(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        let max_res = crate::presets::MAX_INTERACTIVE_SURFACE_RESOLUTION;
        self.sphere_radius.is_finite()
            && (0.1..=5.0).contains(&self.sphere_radius)
            && self.torus_major_radius.is_finite()
            && (1.0..=5.0).contains(&self.torus_major_radius)
            && self.torus_minor_radius.is_finite()
            && (0.1..=2.0).contains(&self.torus_minor_radius)
            && self.klein_radius.is_finite()
            && (0.5..=5.0).contains(&self.klein_radius)
            && (10..=max_res).contains(&self.u_resolution)
            && (10..=max_res).contains(&self.v_resolution)
            && self.camera.is_valid()
    }

    fn save_state(&self) -> Option<String> {
        let (custom_expr, parametric_exprs, custom_params) =
            self.custom_sandbox.export_fields(self.surface_type);
        let (u_bounds, v_bounds) = self.custom_sandbox.export_domain_bounds(self.surface_type);
        serde_json::to_string(&CurvatureHeatmapSaveState {
            surface_type: self.surface_type,
            curvature_type: self.curvature_type,
            sphere_radius: self.sphere_radius,
            torus_major_radius: self.torus_major_radius,
            torus_minor_radius: self.torus_minor_radius,
            klein_radius: self.klein_radius,
            u_resolution: self.u_resolution,
            v_resolution: self.v_resolution,
            camera: self.camera,
            custom_expr,
            parametric_exprs,
            custom_params,
            u_bounds,
            v_bounds,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let loaded: CurvatureHeatmapState =
            serde_json::from_str(state).map_err(|e| format!("Invalid curvature state: {e}"))?;
        validate_curvature_radii_and_res(&loaded)?;
        let camera = Camera3D::from_preset_fields(
            loaded.camera,
            loaded.pitch,
            loaded.yaw,
            loaded.zoom,
            Some(Camera3D::default()),
        )?;
        self.custom_sandbox.load_domain_bounds(
            loaded.surface_type,
            loaded.u_bounds,
            loaded.v_bounds,
        )?;
        let cap = crate::presets::MAX_INTERACTIVE_SURFACE_RESOLUTION;
        self.surface_type = loaded.surface_type;
        self.curvature_type = loaded.curvature_type;
        self.sphere_radius = loaded.sphere_radius;
        self.torus_major_radius = loaded.torus_major_radius;
        self.torus_minor_radius = loaded.torus_minor_radius;
        self.klein_radius = loaded.klein_radius;
        self.u_resolution = loaded.u_resolution.min(cap);
        self.v_resolution = loaded.v_resolution.min(cap);
        self.camera.apply_pose(camera);
        self.custom_sandbox.load_fields(
            loaded.custom_expr.as_deref(),
            loaded.parametric_exprs.as_ref(),
            &loaded.custom_params,
        );
        Ok(())
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

// [cite:graph_parameters_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "CurvatureHeatmap",
        display_name: "Curvature Heatmap",
        domain: "geometry_topology",
        tags: &[],
        build: || Box::new(CurvatureHeatmap::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for CurvatureHeatmap {
    fn theory_description(&self) -> String {
        let surf_name = match self.surface_type {
            SurfaceType::Sphere => "sphere",
            SurfaceType::Torus => "torus",
            SurfaceType::KleinBottle => "Klein bottle",
            SurfaceType::CustomHeightfield => "custom compiled heightfield",
            SurfaceType::CustomParametric => "custom compiled parametric manifold",
        };
        let curv_formula = match self.curvature_type {
            CurvatureType::Gaussian => "Gaussian curvature $K = \\kappa_1 \\kappa_2 = \\frac{LN - M^2}{EG - F^2}$ (an intrinsic invariant by Gauss's Theorema Egregium)",
            CurvatureType::Mean => "mean curvature $H = \\frac{\\kappa_1 + \\kappa_2}{2} = \\frac{EN - 2FM + GL}{2(EG - F^2)}$",
        };
        format!(
            "Differential geometry surface analysis evaluating {curv_formula} on a parametric {surf_name}."
        )
    }

    fn phonetic_description(&self) -> String {
        "Differential geometry surface curvature and Gauss-Bonnet theory".into()
    }

    fn theory_citation(&self) -> String {
        "Differential Geometry of Curves and Surfaces (do Carmo, 2016)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Gaussian".into(),
            "K = kappa_1 * kappa_2 intrinsic curvature invariant under isometries".into(),
        );
        map.insert(
            "Mean".into(),
            "H = (kappa_1 + kappa_2) / 2 extrinsic curvature related to minimal surfaces".into(),
        );
        map
    }
}

#[cfg(test)]
#[path = "curvature_heatmap/tests.rs"]
mod tests;

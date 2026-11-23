#[path = "curvature_heatmap/cache.rs"]
mod cache;

use super::custom_surface::CustomSurfaceSandbox;
use crate::framework::{Camera3D, InteractiveTool, Plot3DViewport};
use eframe::egui;
use egui_plot::Line;
use math_commons::primitives::PositiveFloat;
use math_explorer::pure_math::differential_geometry::surface::{
    CurvatureKind, ParametricSurfaceKind,
};
use std::collections::BTreeMap;

#[derive(serde::Serialize)]
struct CurvatureHeatmapSaveState {
    surface_type: ParametricSurfaceKind,
    curvature_type: CurvatureKind,
    sphere_radius: PositiveFloat,
    torus_major_radius: PositiveFloat,
    torus_minor_radius: PositiveFloat,
    klein_radius: PositiveFloat,
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
    surface_type: ParametricSurfaceKind,
    curvature_type: CurvatureKind,
    sphere_radius: PositiveFloat,
    torus_major_radius: PositiveFloat,
    torus_minor_radius: PositiveFloat,
    klein_radius: PositiveFloat,
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

pub struct CurvatureHeatmap {
    surface_type: ParametricSurfaceKind,
    curvature_type: CurvatureKind,
    sphere_radius: f64,
    torus_major_radius: f64,
    torus_minor_radius: f64,
    klein_radius: f64,
    u_resolution: usize,
    v_resolution: usize,
    camera: Camera3D,
    custom_sandbox: CustomSurfaceSandbox,
    cache: cache::CurvatureGeometryCache,
}

impl Default for CurvatureHeatmap {
    fn default() -> Self {
        Self {
            surface_type: ParametricSurfaceKind::Torus,
            curvature_type: CurvatureKind::Gaussian,
            sphere_radius: 1.0,
            torus_major_radius: 2.0,
            torus_minor_radius: 0.5,
            klein_radius: 2.0,
            u_resolution: 30,
            v_resolution: 30,
            camera: Camera3D::new(0.5, 0.5, 1.0),
            custom_sandbox: CustomSurfaceSandbox::default(),
            cache: cache::CurvatureGeometryCache::default(),
        }
    }
}

impl CurvatureHeatmap {
    fn draw_side_controls(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "curvature_heatmap_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Curvature Heatmap");
            ui.separator();

            ui.label("Select Surface:");
            ui.horizontal_wrapped(|ui| {
                ui.radio_value(
                    &mut self.surface_type,
                    ParametricSurfaceKind::Sphere,
                    "Sphere",
                );
                ui.radio_value(
                    &mut self.surface_type,
                    ParametricSurfaceKind::Torus,
                    "Torus",
                );
                ui.radio_value(
                    &mut self.surface_type,
                    ParametricSurfaceKind::KleinBottle,
                    "Klein Bottle",
                );
                ui.radio_value(
                    &mut self.surface_type,
                    ParametricSurfaceKind::CustomHeightfield,
                    "Custom z=f(x,y)",
                );
                ui.radio_value(
                    &mut self.surface_type,
                    ParametricSurfaceKind::CustomParametric,
                    "Custom r(u,v)",
                );
            });

            ui.separator();
            ui.label("Select Curvature Type:");
            ui.horizontal(|ui| {
                ui.radio_value(
                    &mut self.curvature_type,
                    CurvatureKind::Gaussian,
                    "Gaussian",
                );
                ui.radio_value(&mut self.curvature_type, CurvatureKind::Mean, "Mean");
            });

            ui.separator();
            self.draw_surface_parameters(ui);

            egui::CollapsingHeader::new("View")
                .default_open(true)
                .show(ui, |ui| self.camera.ui(ui));

            ui.collapsing("Resolution", |ui| {
                let max_res = crate::presets::MAX_INTERACTIVE_SURFACE_RESOLUTION;
                self.u_resolution = self.u_resolution.clamp(10, max_res);
                self.v_resolution = self.v_resolution.clamp(10, max_res);
                ui.add(egui::Slider::new(&mut self.u_resolution, 10..=max_res).text("Grid Size U"));
                ui.add(egui::Slider::new(&mut self.v_resolution, 10..=max_res).text("Grid Size V"));
            });
        });
    }

    fn draw_surface_parameters(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Parameters")
            .default_open(matches!(
                self.surface_type,
                ParametricSurfaceKind::CustomHeightfield | ParametricSurfaceKind::CustomParametric
            ))
            .show(ui, |ui| match self.surface_type {
                ParametricSurfaceKind::Sphere => {
                    ui.add(egui::Slider::new(&mut self.sphere_radius, 0.1..=5.0).text("Radius"));
                }
                ParametricSurfaceKind::Torus => {
                    ui.add(
                        egui::Slider::new(&mut self.torus_major_radius, 1.0..=5.0)
                            .text("Major Radius (R)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.torus_minor_radius, 0.1..=2.0)
                            .text("Minor Radius (r)"),
                    );
                }
                ParametricSurfaceKind::KleinBottle => {
                    ui.add(egui::Slider::new(&mut self.klein_radius, 0.5..=5.0).text("Radius"));
                }
                ParametricSurfaceKind::CustomHeightfield
                | ParametricSurfaceKind::CustomParametric => {
                    self.custom_sandbox.ui_for_type(ui, self.surface_type);
                }
            });
    }
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

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash as _, Hasher as _};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.curvature_geometry_key().hash(&mut h);
        self.camera_pose_bits().hash(&mut h);
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
            sphere_radius: PositiveFloat::new(self.sphere_radius).ok()?,
            torus_major_radius: PositiveFloat::new(self.torus_major_radius).ok()?,
            torus_minor_radius: PositiveFloat::new(self.torus_minor_radius).ok()?,
            klein_radius: PositiveFloat::new(self.klein_radius).ok()?,
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
        let loaded: CurvatureHeatmapState =
            serde_json::from_str(state).map_err(|e| format!("Invalid curvature state: {e}"))?;
        let (sphere_radius, torus_major_radius, torus_minor_radius, klein_radius) =
            validate_curvature_radii(&loaded)?;
        let max_schema = math_commons::registry::MAX_GRID_SIZE;
        if !(10..=max_schema).contains(&loaded.u_resolution)
            || !(10..=max_schema).contains(&loaded.v_resolution)
        {
            return Err("Resolution out of valid range".to_string());
        }
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
        self.sphere_radius = sphere_radius;
        self.torus_major_radius = torus_major_radius;
        self.torus_minor_radius = torus_minor_radius;
        self.klein_radius = klein_radius;
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

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.draw_side_controls(ui);
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let batches = self.ensure_projected_batches().to_vec();
            Plot3DViewport::new("curvature_plot", 4.0)
                .helper_text("Drag canvas or use WASD to orbit, scroll or Q/E to zoom, double-click or press R to reset view. Blue = Low Curvature, Red = High Curvature.")
                .show(ui, &mut self.camera, |plot_ui| {
                    for batch in batches {
                        plot_ui.line(
                            Line::from_segments(
                                format!("curvature_bin_{}", batch.bin),
                                batch.segments,
                            )
                            .with_bounds(batch.bounds)
                            .color(batch.color),
                        );
                    }
                });
        });
    }
}

fn validate_curvature_radii(
    loaded: &CurvatureHeatmapState,
) -> Result<(f64, f64, f64, f64), String> {
    let sphere_radius = loaded.sphere_radius.value();
    let torus_major_radius = loaded.torus_major_radius.value();
    let torus_minor_radius = loaded.torus_minor_radius.value();
    let klein_radius = loaded.klein_radius.value();
    if !(0.1..=5.0).contains(&sphere_radius) {
        return Err(format!("sphere_radius out of range: {sphere_radius}"));
    }
    if !(1.0..=5.0).contains(&torus_major_radius) {
        return Err(format!(
            "torus_major_radius out of range: {torus_major_radius}"
        ));
    }
    if !(0.1..=2.0).contains(&torus_minor_radius) {
        return Err(format!(
            "torus_minor_radius out of range: {torus_minor_radius}"
        ));
    }
    if !(0.5..=5.0).contains(&klein_radius) {
        return Err(format!("klein_radius out of range: {klein_radius}"));
    }
    Ok((
        sphere_radius,
        torus_major_radius,
        torus_minor_radius,
        klein_radius,
    ))
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
            ParametricSurfaceKind::Sphere => "sphere",
            ParametricSurfaceKind::Torus => "torus",
            ParametricSurfaceKind::KleinBottle => "Klein bottle",
            ParametricSurfaceKind::CustomHeightfield => "custom compiled heightfield",
            ParametricSurfaceKind::CustomParametric => "custom compiled parametric manifold",
        };
        let curv_formula = match self.curvature_type {
            CurvatureKind::Gaussian => "Gaussian curvature $K = \\kappa_1 \\kappa_2 = \\frac{LN - M^2}{EG - F^2}$ (an intrinsic invariant by Gauss's Theorema Egregium)",
            CurvatureKind::Mean => "mean curvature $H = \\frac{\\kappa_1 + \\kappa_2}{2} = \\frac{EN - 2FM + GL}{2(EG - F^2)}$",
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

#[path = "curvature_heatmap/presets.rs"]
mod presets;

#[cfg(test)]
#[path = "curvature_heatmap/tests.rs"]
mod tests;

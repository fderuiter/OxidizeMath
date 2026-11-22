#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "curvature_heatmap_ui.rs"]
mod ui;

#[path = "curvature_heatmap/presets.rs"]
mod presets;

use super::custom_surface::{CustomSurfaceSandbox, SurfaceType};
use crate::framework::{Camera3D, InteractiveTool, Scene3DBuilder};
use eframe::egui;
use egui_plot::LineStyle;
use math_commons::math_kernel::colormap::mapped_gradient;
use math_explorer::pure_math::differential_geometry::surface::{
    KleinBottle, ParametricSurface, Sphere, SurfaceAnalysis, Torus,
};
use nalgebra::Point3;
use std::collections::BTreeMap;

#[derive(PartialEq, Eq, Hash, Clone, Copy, serde::Serialize, serde::Deserialize)]
enum CurvatureType {
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
    curvature_grid: Vec<f64>,
    surface_vertices: Vec<Point3<f64>>,
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
            curvature_grid: Vec::new(),
            surface_vertices: Vec::new(),
        }
    }
}

impl CurvatureHeatmap {
    fn map_color(value: f64, min_val: f64, max_val: f64) -> egui::Color32 {
        let rgb = mapped_gradient(value, min_val, max_val);
        egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2])
    }

    /// Populates `self.scene_builder` with world `Point3<f64>` curvature-colored grid segments.
    pub fn populate_scene(&mut self, series_color: egui::Color32) {
        self.scene_builder.begin(self.camera);
        match self.surface_type {
            SurfaceType::Sphere => {
                let s = Sphere {
                    radius: self.sphere_radius,
                };
                self.build_surface_segments(&s, series_color);
            }
            SurfaceType::Torus => {
                let s = Torus {
                    major_radius: self.torus_major_radius,
                    minor_radius: self.torus_minor_radius,
                };
                self.build_surface_segments(&s, series_color);
            }
            SurfaceType::KleinBottle => {
                let s = KleinBottle {
                    radius: self.klein_radius,
                };
                self.build_surface_segments(&s, series_color);
            }
            SurfaceType::CustomHeightfield => {
                let s = self.custom_sandbox.build_heightfield();
                self.build_surface_segments(&s, series_color);
            }
            SurfaceType::CustomParametric => {
                let s = self.custom_sandbox.build_parametric();
                self.build_surface_segments(&s, series_color);
            }
        }
    }

    fn sample_surface_grid<T: ParametricSurface>(&mut self, surface: &T) -> (f64, f64) {
        let ((u_min, u_max), (v_min, v_max)) = self.surface_type.uv_domain();
        let cols = self.v_resolution + 1;
        let total = (self.u_resolution + 1) * cols;
        self.curvature_grid.clear();
        self.curvature_grid.reserve(total);
        self.surface_vertices.clear();
        self.surface_vertices.reserve(total);

        let mut min_c = f64::INFINITY;
        let mut max_c = f64::NEG_INFINITY;
        let u_span = u_max - u_min;
        let v_span = v_max - v_min;

        for i in 0..=self.u_resolution {
            let u = u_min + u_span * (i as f64 / self.u_resolution as f64);
            for j in 0..=self.v_resolution {
                let v = v_min + v_span * (j as f64 / self.v_resolution as f64);
                let raw = match self.curvature_type {
                    CurvatureType::Gaussian => surface.gaussian_curvature(u, v),
                    CurvatureType::Mean => surface.mean_curvature(u, v),
                };
                let c = if raw.is_finite() {
                    raw.clamp(-1e4, 1e4)
                } else {
                    0.0
                };
                min_c = min_c.min(c);
                max_c = max_c.max(c);
                self.curvature_grid.push(c);
                self.surface_vertices.push(surface.position(u, v));
            }
        }
        if !min_c.is_finite() || !max_c.is_finite() {
            (0.0, 1.0)
        } else {
            (min_c, max_c)
        }
    }

    fn build_surface_segments<T: ParametricSurface>(
        &mut self,
        surface: &T,
        series_color: egui::Color32,
    ) {
        let (min_c, max_c) = self.sample_surface_grid(surface);
        let cols = self.v_resolution + 1;
        for i in 0..self.u_resolution {
            for j in 0..self.v_resolution {
                let idx = i * cols + j;
                let p1 = self.surface_vertices[idx];
                let p2 = self.surface_vertices[idx + 1];
                let p3 = self.surface_vertices[idx + cols];
                let c1 = self.curvature_grid[idx];
                let col_v = Self::map_color(0.5 * (c1 + self.curvature_grid[idx + 1]), min_c, max_c);
                let col_u =
                    Self::map_color(0.5 * (c1 + self.curvature_grid[idx + cols]), min_c, max_c);
                self.scene_builder.add_styled_segment(
                    "Curvature Surface",
                    [p1, p2],
                    egui::Stroke::new(1.25_f32, col_v),
                    LineStyle::Solid,
                    true,
                    false,
                );
                self.scene_builder.add_styled_segment(
                    "Curvature Surface",
                    [p1, p3],
                    egui::Stroke::new(1.25_f32, col_u),
                    LineStyle::Solid,
                    true,
                    false,
                );
            }
        }
        self.scene_builder
            .record_series("Curvature Surface", &self.surface_vertices, series_color);
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
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.surface_type.hash(&mut h);
        self.curvature_type.hash(&mut h);
        self.sphere_radius.to_bits().hash(&mut h);
        self.torus_major_radius.to_bits().hash(&mut h);
        self.torus_minor_radius.to_bits().hash(&mut h);
        self.klein_radius.to_bits().hash(&mut h);
        self.u_resolution.hash(&mut h);
        self.v_resolution.hash(&mut h);
        self.camera.hash_state(&mut h);
        self.custom_sandbox.hash_state(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        let (custom_expr, parametric_exprs, custom_params) =
            self.custom_sandbox.export_fields(self.surface_type);
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
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let loaded: CurvatureHeatmapState =
            serde_json::from_str(state).map_err(|e| format!("Invalid curvature state: {e}"))?;
        validate_curvature_radii(&loaded)?;
        let max_res = math_commons::registry::MAX_GRID_SIZE;
        if !(10..=max_res).contains(&loaded.u_resolution)
            || !(10..=max_res).contains(&loaded.v_resolution)
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
        self.surface_type = loaded.surface_type;
        self.curvature_type = loaded.curvature_type;
        self.sphere_radius = loaded.sphere_radius;
        self.torus_major_radius = loaded.torus_major_radius;
        self.torus_minor_radius = loaded.torus_minor_radius;
        self.klein_radius = loaded.klein_radius;
        self.u_resolution = loaded.u_resolution;
        self.v_resolution = loaded.v_resolution;
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

fn validate_curvature_radii(loaded: &CurvatureHeatmapState) -> Result<(), String> {
    if !loaded.sphere_radius.is_finite() || !(0.1..=5.0).contains(&loaded.sphere_radius) {
        return Err(format!("sphere_radius out of range: {}", loaded.sphere_radius));
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
        return Err(format!("klein_radius out of range: {}", loaded.klein_radius));
    }
    Ok(())
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

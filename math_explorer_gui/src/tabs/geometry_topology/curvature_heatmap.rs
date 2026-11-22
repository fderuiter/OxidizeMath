#[path = "curvature_export.rs"]
mod curvature_export;

use super::custom_surface::{CustomSurfaceSandbox, SurfaceType};
use crate::framework::{Camera3D, ExportCapabilities, InteractiveTool, Plot3DViewport};
use eframe::egui;
use egui_plot::{Line, PlotPoints};
use math_commons::math_kernel::colormap::mapped_gradient;
use math_explorer::pure_math::differential_geometry::surface::{
    KleinBottle, ParametricSurface, Sphere, SurfaceAnalysis, Torus,
};
use nalgebra::Point3;
use std::collections::BTreeMap;

#[derive(PartialEq, Eq, Clone, Copy, serde::Serialize, serde::Deserialize)]
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

pub struct CurvatureHeatmap {
    surface_type: SurfaceType,
    curvature_type: CurvatureType,
    sphere_radius: f64,
    torus_major_radius: f64,
    torus_minor_radius: f64,
    klein_radius: f64,
    u_resolution: usize,
    v_resolution: usize,
    camera: Camera3D,
    custom_sandbox: CustomSurfaceSandbox,
}

impl Default for CurvatureHeatmap {
    fn default() -> Self {
        Self {
            surface_type: SurfaceType::Torus,
            curvature_type: CurvatureType::Gaussian,
            sphere_radius: 1.0,
            torus_major_radius: 2.0,
            torus_minor_radius: 0.5,
            klein_radius: 2.0,
            u_resolution: 30,
            v_resolution: 30,
            camera: Camera3D::new(0.5, 0.5, 1.0),
            custom_sandbox: CustomSurfaceSandbox::default(),
        }
    }
}

impl CurvatureHeatmap {
    fn project(&self, p: Point3<f64>) -> [f64; 2] {
        self.camera.project(&[p.x, p.y, p.z])
    }

    fn map_color(&self, value: f64, min_val: f64, max_val: f64) -> egui::Color32 {
        let rgb = mapped_gradient(value, min_val, max_val);
        egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2])
    }

    fn sample_curvature_grid<T: ParametricSurface>(
        &self,
        surface: &T,
        u_range: (f64, f64),
        v_range: (f64, f64),
    ) -> (Vec<Vec<f64>>, f64, f64) {
        let (u_min, u_max) = u_range;
        let (v_min, v_max) = v_range;
        let mut min_k = f64::INFINITY;
        let mut max_k = f64::NEG_INFINITY;
        let mut grid = vec![vec![0.0; self.v_resolution + 1]; self.u_resolution + 1];

        for (i, row) in grid.iter_mut().enumerate().take(self.u_resolution + 1) {
            let u = u_min + (u_max - u_min) * (i as f64 / self.u_resolution as f64);
            for (j, item) in row.iter_mut().enumerate().take(self.v_resolution + 1) {
                let v = v_min + (v_max - v_min) * (j as f64 / self.v_resolution as f64);
                let raw = match self.curvature_type {
                    CurvatureType::Gaussian => surface.gaussian_curvature(u, v),
                    CurvatureType::Mean => surface.mean_curvature(u, v),
                };
                let c = if raw.is_finite() {
                    raw.clamp(-1e4, 1e4)
                } else {
                    0.0
                };
                *item = c;
                min_k = min_k.min(c);
                max_k = max_k.max(c);
            }
        }
        if !min_k.is_finite() || !max_k.is_finite() {
            (grid, 0.0, 1.0)
        } else {
            (grid, min_k, max_k)
        }
    }

    fn generate_colored_lines<T: ParametricSurface>(
        &self,
        surface: &T,
    ) -> Vec<(Vec<[f64; 2]>, egui::Color32)> {
        let (u_range, v_range) = self.surface_type.uv_domain();
        let (u_min, u_max) = u_range;
        let (v_min, v_max) = v_range;
        let (curvature_map, min_k, max_k) = self.sample_curvature_grid(surface, u_range, v_range);
        let mut lines = Vec::with_capacity(self.u_resolution * self.v_resolution * 2);

        for i in 0..self.u_resolution {
            let u_start = u_min + (u_max - u_min) * (i as f64 / self.u_resolution as f64);
            let u_end = u_min + (u_max - u_min) * ((i + 1) as f64 / self.u_resolution as f64);
            for j in 0..self.v_resolution {
                let v_start = v_min + (v_max - v_min) * (j as f64 / self.v_resolution as f64);
                let v_end = v_min + (v_max - v_min) * ((j + 1) as f64 / self.v_resolution as f64);

                let p1 = self.project(surface.position(u_start, v_start));
                let p2 = self.project(surface.position(u_start, v_end));
                let c1 = curvature_map[i][j];
                if p1[0].is_finite() && p1[1].is_finite() && p2[0].is_finite() && p2[1].is_finite()
                {
                    let avg_v = 0.5 * (c1 + curvature_map[i][j + 1]);
                    lines.push((vec![p1, p2], self.map_color(avg_v, min_k, max_k)));
                }

                let p3 = self.project(surface.position(u_end, v_start));
                if p1[0].is_finite() && p1[1].is_finite() && p3[0].is_finite() && p3[1].is_finite()
                {
                    let avg_u = 0.5 * (c1 + curvature_map[i + 1][j]);
                    lines.push((vec![p1, p3], self.map_color(avg_u, min_k, max_k)));
                }
            }
        }
        lines
    }

    fn build_colored_lines(&self) -> Vec<(Vec<[f64; 2]>, egui::Color32)> {
        match self.surface_type {
            SurfaceType::Sphere => self.generate_colored_lines(&Sphere {
                radius: self.sphere_radius,
            }),
            SurfaceType::Torus => self.generate_colored_lines(&Torus {
                major_radius: self.torus_major_radius,
                minor_radius: self.torus_minor_radius,
            }),
            SurfaceType::KleinBottle => self.generate_colored_lines(&KleinBottle {
                radius: self.klein_radius,
            }),
            SurfaceType::CustomHeightfield => {
                self.generate_colored_lines(&self.custom_sandbox.build_heightfield())
            }
            SurfaceType::CustomParametric => {
                self.generate_colored_lines(&self.custom_sandbox.build_parametric())
            }
        }
    }

    fn draw_side_controls(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "curvature_heatmap_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Curvature Heatmap");
            ui.separator();

            ui.label("Select Surface:");
            ui.horizontal_wrapped(|ui| {
                ui.radio_value(&mut self.surface_type, SurfaceType::Sphere, "Sphere");
                ui.radio_value(&mut self.surface_type, SurfaceType::Torus, "Torus");
                ui.radio_value(
                    &mut self.surface_type,
                    SurfaceType::KleinBottle,
                    "Klein Bottle",
                );
                ui.radio_value(
                    &mut self.surface_type,
                    SurfaceType::CustomHeightfield,
                    "Custom z=f(x,y)",
                );
                ui.radio_value(
                    &mut self.surface_type,
                    SurfaceType::CustomParametric,
                    "Custom r(u,v)",
                );
            });

            ui.separator();
            ui.label("Select Curvature Type:");
            ui.horizontal(|ui| {
                ui.radio_value(
                    &mut self.curvature_type,
                    CurvatureType::Gaussian,
                    "Gaussian",
                );
                ui.radio_value(&mut self.curvature_type, CurvatureType::Mean, "Mean");
            });

            ui.separator();
            self.draw_surface_parameters(ui);

            egui::CollapsingHeader::new("View")
                .default_open(true)
                .show(ui, |ui| self.camera.ui(ui));

            ui.collapsing("Resolution", |ui| {
                let max_res = math_commons::registry::MAX_GRID_SIZE;
                ui.add(egui::Slider::new(&mut self.u_resolution, 10..=max_res).text("Grid Size U"));
                ui.add(egui::Slider::new(&mut self.v_resolution, 10..=max_res).text("Grid Size V"));
            });
        });
    }

    fn draw_surface_parameters(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Parameters")
            .default_open(matches!(
                self.surface_type,
                SurfaceType::CustomHeightfield | SurfaceType::CustomParametric
            ))
            .show(ui, |ui| match self.surface_type {
                SurfaceType::Sphere => {
                    ui.add(egui::Slider::new(&mut self.sphere_radius, 0.1..=5.0).text("Radius"));
                }
                SurfaceType::Torus => {
                    ui.add(
                        egui::Slider::new(&mut self.torus_major_radius, 1.0..=5.0)
                            .text("Major Radius (R)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.torus_minor_radius, 0.1..=2.0)
                            .text("Minor Radius (r)"),
                    );
                }
                SurfaceType::KleinBottle => {
                    ui.add(egui::Slider::new(&mut self.klein_radius, 0.5..=5.0).text("Radius"));
                }
                SurfaceType::CustomHeightfield | SurfaceType::CustomParametric => {
                    self.custom_sandbox.ui_for_type(ui, self.surface_type);
                }
            });
    }
}

fn validate_curvature_radii_and_res(loaded: &CurvatureHeatmapState) -> Result<(), String> {
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
        self.draw_side_controls(ui);
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let lines = self.build_colored_lines();
            Plot3DViewport::new("curvature_plot", 4.0)
                .helper_text("Drag canvas or use WASD to orbit, scroll or Q/E to zoom, double-click or press R to reset view. Blue = Low Curvature, Red = High Curvature.")
                .show(ui, &mut self.camera, |plot_ui| {
                    for (i, (line_points, color)) in lines.into_iter().enumerate() {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(line_points))
                                .name(format!("line_{i}"))
                                .color(color),
                        );
                    }
                });
        });
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

#[path = "curvature_heatmap/presets.rs"]
mod presets;

#[cfg(test)]
#[path = "curvature_heatmap/tests.rs"]
mod tests;

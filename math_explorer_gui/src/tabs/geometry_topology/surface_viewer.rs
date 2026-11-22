use crate::framework::{Camera3D, InteractiveTool, Plot3DViewport};
use crate::web_io::{SaveAsDialogConfig, download_mesh_bytes};
use eframe::egui;
use egui_plot::{Line, PlotPoints};
use math_explorer::pure_math::differential_geometry::surface::{
    KleinBottle, ParametricSurface, Sphere, Torus,
};
use nalgebra::Point3;
use oxidize_core::mesh::{analyze_mesh, export_mesh_to_obj_string, export_mesh_to_stl_bytes};

#[derive(PartialEq, Clone, Copy, serde::Serialize, serde::Deserialize)]
enum SurfaceType {
    Sphere,
    Torus,
    KleinBottle,
}

const fn default_scale_mm() -> f32 {
    20.0
}

#[derive(serde::Serialize)]
struct SurfaceViewerSaveState {
    surface_type: SurfaceType,
    sphere_radius: f64,
    torus_major_radius: f64,
    torus_minor_radius: f64,
    klein_radius: f64,
    u_resolution: usize,
    v_resolution: usize,
    scale_mm: f32,
    camera: Camera3D,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SurfaceViewerState {
    surface_type: SurfaceType,
    sphere_radius: f64,
    torus_major_radius: f64,
    torus_minor_radius: f64,
    klein_radius: f64,
    u_resolution: usize,
    v_resolution: usize,
    #[serde(default = "default_scale_mm")]
    scale_mm: f32,
    #[serde(default)]
    camera: Option<Camera3D>,
    #[serde(default)]
    pitch: Option<f32>,
    #[serde(default)]
    yaw: Option<f32>,
    #[serde(default)]
    zoom: Option<f32>,
}

pub struct SurfaceViewer {
    surface_type: SurfaceType,
    sphere_radius: f64,
    torus_major_radius: f64,
    torus_minor_radius: f64,
    klein_radius: f64,
    u_resolution: usize,
    v_resolution: usize,
    scale_mm: f32,
    camera: crate::framework::Camera3D,
}

impl Default for SurfaceViewer {
    fn default() -> Self {
        Self {
            surface_type: SurfaceType::Torus,
            sphere_radius: 1.0,
            torus_major_radius: 2.0,
            torus_minor_radius: 0.5,
            klein_radius: 2.0,
            u_resolution: 30,
            v_resolution: 30,
            scale_mm: default_scale_mm(),
            camera: crate::framework::Camera3D::new(0.5, 0.5, 1.0),
        }
    }
}

impl SurfaceViewer {
    /// Projects a 3D point to 2D based on rotation and zoom
    fn project(&self, p: Point3<f64>) -> [f64; 2] {
        self.camera.project(&[p.x, p.y, p.z])
    }

    fn generate_grid_lines(&self, surface: &dyn ParametricSurface) -> Vec<Vec<[f64; 2]>> {
        let mut lines = Vec::new();
        let (u_min, u_max) = surface.u_domain();
        let (v_min, v_max) = surface.v_domain();

        for i in 0..=self.u_resolution {
            let u = u_min + (u_max - u_min) * (i as f64 / self.u_resolution as f64);
            let mut line = Vec::new();
            for j in 0..=self.v_resolution {
                let v = v_min + (v_max - v_min) * (j as f64 / self.v_resolution as f64);
                let p = surface.position(u, v);
                line.push(self.project(p));
            }
            lines.push(line);
        }

        for j in 0..=self.v_resolution {
            let v = v_min + (v_max - v_min) * (j as f64 / self.v_resolution as f64);
            let mut line = Vec::new();
            for i in 0..=self.u_resolution {
                let u = u_min + (u_max - u_min) * (i as f64 / self.u_resolution as f64);
                let p = surface.position(u, v);
                line.push(self.project(p));
            }
            lines.push(line);
        }

        lines
    }

    fn create_surface(&self) -> Box<dyn ParametricSurface> {
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
        }
    }

    fn export_stem(&self) -> &'static str {
        match self.surface_type {
            SurfaceType::Sphere => "sphere",
            SurfaceType::Torus => "torus",
            SurfaceType::KleinBottle => "klein_bottle",
        }
    }

    fn build_scaled_mesh(&self) -> oxidize_core::mesh::Mesh {
        let surface = self.create_surface();
        if (self.scale_mm - 1.0).abs() <= f32::EPSILON {
            super::export_utils::surface_to_mesh(
                surface.as_ref(),
                self.u_resolution,
                self.v_resolution,
            )
        } else {
            super::export_utils::surface_to_mesh_scaled(
                surface.as_ref(),
                self.u_resolution,
                self.v_resolution,
                self.scale_mm,
            )
        }
    }
}

impl InteractiveTool for SurfaceViewer {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Surface Viewer"
    }

    fn domain(&self) -> &'static str {
        "geometry_topology"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&SurfaceViewerSaveState {
            surface_type: self.surface_type,
            sphere_radius: self.sphere_radius,
            torus_major_radius: self.torus_major_radius,
            torus_minor_radius: self.torus_minor_radius,
            klein_radius: self.klein_radius,
            u_resolution: self.u_resolution,
            v_resolution: self.v_resolution,
            scale_mm: self.scale_mm,
            camera: self.camera,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let loaded: SurfaceViewerState =
            serde_json::from_str(state).map_err(|e| format!("Invalid SurfaceViewer state: {e}"))?;
        if !loaded.sphere_radius.is_finite()
            || !(0.1..=5.0).contains(&loaded.sphere_radius)
            || !loaded.torus_major_radius.is_finite()
            || !(1.0..=5.0).contains(&loaded.torus_major_radius)
            || !loaded.torus_minor_radius.is_finite()
            || !(0.1..=2.0).contains(&loaded.torus_minor_radius)
            || !loaded.klein_radius.is_finite()
            || !(0.5..=5.0).contains(&loaded.klein_radius)
            || !loaded.scale_mm.is_finite()
            || !(1.0..=200.0).contains(&loaded.scale_mm)
        {
            return Err("Surface radii or millimeter scale out of valid bounds".to_string());
        }
        let max_res = math_commons::registry::MAX_GRID_SIZE;
        if !(10..=max_res).contains(&loaded.u_resolution)
            || !(10..=max_res).contains(&loaded.v_resolution)
        {
            return Err("Surface resolution out of valid bounds".to_string());
        }
        let camera = Camera3D::from_preset_fields(
            loaded.camera,
            loaded.pitch,
            loaded.yaw,
            loaded.zoom,
            Some(Camera3D::default()),
        )?;
        self.surface_type = loaded.surface_type;
        self.sphere_radius = loaded.sphere_radius;
        self.torus_major_radius = loaded.torus_major_radius;
        self.torus_minor_radius = loaded.torus_minor_radius;
        self.klein_radius = loaded.klein_radius;
        self.u_resolution = loaded.u_resolution;
        self.v_resolution = loaded.v_resolution;
        self.scale_mm = loaded.scale_mm;
        self.camera.apply_pose(camera);
        Ok(())
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "surface_viewer_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            ui.heading("Surface Viewer");
            ui.separator();

            ui.label("Select Surface:");
            ui.horizontal(|ui| {
                ui.radio_value(&mut self.surface_type, SurfaceType::Sphere, "Sphere");
                ui.radio_value(&mut self.surface_type, SurfaceType::Torus, "Torus");
                ui.radio_value(
                    &mut self.surface_type,
                    SurfaceType::KleinBottle,
                    "Klein Bottle",
                );
            });

            ui.separator();

            ui.collapsing("Parameters", |ui| {
                match self.surface_type {
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
                }
                ui.add(
                    egui::Slider::new(&mut self.scale_mm, 1.0..=200.0)
                        .suffix(" mm")
                        .text("Physical Scale"),
                );
            });

            egui::CollapsingHeader::new("View")
                .default_open(true)
                .show(ui, |ui| self.camera.ui(ui));

            ui.collapsing("Resolution", |ui| {
                ui.add(
                    egui::Slider::new(
                        &mut self.u_resolution,
                        10..=math_commons::registry::MAX_GRID_SIZE,
                    )
                    .text("Grid Size U"),
                );
                ui.add(
                    egui::Slider::new(
                        &mut self.v_resolution,
                        10..=math_commons::registry::MAX_GRID_SIZE,
                    )
                    .text("Grid Size V"),
                );
            });

            ui.separator();
            let preview_mesh = self.build_scaled_mesh();
            let diag = analyze_mesh(&preview_mesh);
            ui.small(format!(
                "Bounds: {:.1} × {:.1} × {:.1} mm | Open edges: {} | NaN normals: {}",
                diag.dimensions_mm.x,
                diag.dimensions_mm.y,
                diag.dimensions_mm.z,
                diag.open_edges,
                diag.nan_normals
            ));
            ui.small(format!(
                "Area: {:.1} mm² | Signed Vol: {:.1} mm³",
                diag.area_mm2, diag.signed_volume_mm3
            ));
            ui.horizontal(|ui| {
                if ui.button("Download .STL (mm)").clicked() {
                    if let Ok(bytes) = export_mesh_to_stl_bytes(&preview_mesh) {
                        let filename = format!("{}_mm.stl", self.export_stem());
                        download_mesh_bytes(
                            SaveAsDialogConfig {
                                filter_name: "Binary STL Mesh (*.stl)",
                                extensions: &["stl"],
                                fallback_filename: "surface.stl",
                                write_error_prefix: "Failed to write STL file",
                            },
                            &filename,
                            bytes,
                        );
                    }
                }
                if ui.button("Download .OBJ (mm)").clicked() {
                    if let Ok(obj) = export_mesh_to_obj_string(&preview_mesh) {
                        let filename = format!("{}_mm.obj", self.export_stem());
                        download_mesh_bytes(
                            SaveAsDialogConfig {
                                filter_name: "Wavefront OBJ Mesh (*.obj)",
                                extensions: &["obj"],
                                fallback_filename: "surface.obj",
                                write_error_prefix: "Failed to write OBJ file",
                            },
                            &filename,
                            obj.into_bytes(),
                        );
                    }
                }
            });
        });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let surface = self.create_surface();

            let grid_lines = self.generate_grid_lines(surface.as_ref());

            Plot3DViewport::new("surface_plot", 4.0).show(ui, &mut self.camera, |plot_ui| {
                for (i, line_points) in grid_lines.into_iter().enumerate() {
                    plot_ui.line(
                        Line::new("", PlotPoints::new(line_points))
                            .name(format!("line_{}", i))
                            .color(palette.series_color(0)),
                    );
                }
            });
        });
    }
}

// [cite:modular_polynomials_review]


inventory::submit! {
    crate::framework::ToolMetadata {
        id: "SurfaceViewer",
        display_name: "Surface Viewer",
        domain: "geometry_topology",
        tags: &[],
        build: || Box::new(SurfaceViewer::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for SurfaceViewer {
    fn theory_description(&self) -> String {
        let surf_desc = match self.surface_type {
            SurfaceType::Sphere => "sphere $S^2$: $\\vec{r}(u, v) = (R \\sin u \\cos v, R \\sin u \\sin v, R \\cos u)$",
            SurfaceType::Torus => "torus $T^2$: $\\vec{r}(u, v) = ((R + r \\cos v)\\cos u, (R + r \\cos v)\\sin u, r \\sin v)$",
            SurfaceType::KleinBottle => "non-orientable Klein bottle embedded in 3D: $\\chi(M) = 0$",
        };
        format!(
            "Parametric 2-manifold surface representation: {}.",
            surf_desc
        )
    }
    fn phonetic_description(&self) -> String { "Parametric 2-manifold differential geometry theory".into() }
    fn theory_citation(&self) -> String { "Differential Geometry of Curves and Surfaces (do Carmo, 2016)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert("Sphere".into(), "Compact orientable manifold of genus 0 with Euler characteristic 2".into());
        map.insert("Torus".into(), "Compact orientable manifold of genus 1 with Euler characteristic 0".into());
        map.insert("KleinBottle".into(), "Non-orientable closed surface with Euler characteristic 0".into());
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Parametric Surface Viewer",
    "domain": "geometry_topology",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "surface_type": "KleinBottle",
    "sphere_radius": 1.0,
    "torus_major_radius": 2.0,
    "torus_minor_radius": 0.5,
    "klein_radius": 2.2,
    "u_resolution": 40,
    "v_resolution": 40,
    "camera": {
      "pitch": 0.6,
      "yaw": 0.4,
      "zoom": 1.0
    }
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "surface_viewer_starter",
        title: "Parametric Surface Viewer",
        description: "Parametric 3D wireframe manifold visualization for Sphere, Torus, and Klein Bottle surfaces.",
        category: "Differential Geometry",
        icon: "🔮",
        domain: "geometry_topology",
        tool_name: Some("SurfaceViewer"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::*;

    #[test]
    fn surface_viewer_viewport_and_plot_export_work() {
        let mut tool = SurfaceViewer::default();
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));
        assert!(!egui_plot::get_last_plot_datasets().is_empty());
        assert!(!egui_plot::get_last_plot_primitives().is_empty());
    }
}

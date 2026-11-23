#[path = "surface_viewer/cache.rs"]
mod cache;

use super::custom_surface::{CustomSurfaceSandbox, SurfaceType};
use crate::framework::{Camera3D, InteractiveTool, Plot3DViewport};
use eframe::egui;
use egui_plot::Line;
use math_explorer::pure_math::differential_geometry::surface::{
    KleinBottle, ParametricSurface, Sphere, Torus,
};
use std::collections::BTreeMap;

#[derive(serde::Serialize)]
struct SurfaceViewerSaveState {
    surface_type: SurfaceType,
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
struct SurfaceViewerState {
    surface_type: SurfaceType,
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

pub struct SurfaceViewer {
    surface_type: SurfaceType,
    sphere_radius: f64,
    torus_major_radius: f64,
    torus_minor_radius: f64,
    klein_radius: f64,
    u_resolution: usize,
    v_resolution: usize,
    camera: Camera3D,
    custom_sandbox: CustomSurfaceSandbox,
    cache: cache::SurfaceGeometryCache,
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
            camera: Camera3D::new(0.5, 0.5, 1.0),
            custom_sandbox: CustomSurfaceSandbox::default(),
            cache: cache::SurfaceGeometryCache::default(),
        }
    }
}

impl SurfaceViewer {
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
            SurfaceType::CustomHeightfield => Box::new(self.custom_sandbox.build_heightfield()),
            SurfaceType::CustomParametric => Box::new(self.custom_sandbox.build_parametric()),
        }
    }

    fn build_obj_string(&self) -> String {
        let surface = self.create_surface();
        let (u_range, v_range) = self.surface_type.uv_domain();
        let mesh = super::export_utils::surface_to_mesh_with_domain(
            surface.as_ref(),
            u_range,
            v_range,
            self.u_resolution,
            self.v_resolution,
        );
        oxidize_core::mesh::export_mesh_to_obj_string(&mesh).unwrap_or_default()
    }

    fn draw_side_controls(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "surface_viewer_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Surface Viewer");
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

            ui.separator();
            if ui.button("Export to OBJ").clicked() {
                self.trigger_obj_export();
            }
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

    fn trigger_obj_export(&self) {
        let obj = self.build_obj_string();
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(path) = rfd::FileDialog::new()
                .set_file_name("surface.obj")
                .add_filter("OBJ File", &["obj"])
                .save_file()
            {
                let _ = std::fs::write(path, obj);
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let filename = "surface.obj";
            wasm_bindgen_futures::spawn_local(async move {
                use oxidize_core::vfs::VirtualFileSystem;
                let vfs = oxidize_core::vfs::WasmVfs;
                let _ = vfs.write_to_file(filename, obj.as_bytes()).await;
            });
        }
    }
}

impl InteractiveTool for SurfaceViewer {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Surface Viewer"
    }

    fn domain(&self) -> &'static str {
        "geometry_topology"
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash as _, Hasher as _};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.surface_geometry_key().hash(&mut h);
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
        serde_json::to_string(&SurfaceViewerSaveState {
            surface_type: self.surface_type,
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
        {
            return Err("Surface radii out of valid bounds".to_string());
        }
        let max_schema = math_commons::registry::MAX_GRID_SIZE;
        if !(10..=max_schema).contains(&loaded.u_resolution)
            || !(10..=max_schema).contains(&loaded.v_resolution)
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
        let cap = crate::presets::MAX_INTERACTIVE_SURFACE_RESOLUTION;
        self.surface_type = loaded.surface_type;
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
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let (segments, bounds) = self.ensure_projected_wireframe();

            Plot3DViewport::new("surface_plot", 4.0).show(ui, &mut self.camera, |plot_ui| {
                if !segments.is_empty() {
                    plot_ui.line(
                        Line::from_segments("surface_wireframe", segments)
                            .with_bounds(bounds)
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
            SurfaceType::CustomHeightfield => "custom Monge heightfield $z = f(x, y)$ compiled to stack bytecode",
            SurfaceType::CustomParametric => "custom 3D parametric immersion $\\vec{r}(u, v) = (x(u,v), y(u,v), z(u,v))$",
        };
        format!("Parametric 2-manifold surface representation: {surf_desc}.")
    }

    fn phonetic_description(&self) -> String {
        "Parametric 2-manifold differential geometry theory".into()
    }

    fn theory_citation(&self) -> String {
        "Differential Geometry of Curves and Surfaces (do Carmo, 2016)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Sphere".into(),
            "Compact orientable manifold of genus 0 with Euler characteristic 2".into(),
        );
        map.insert(
            "Torus".into(),
            "Compact orientable manifold of genus 1 with Euler characteristic 0".into(),
        );
        map.insert(
            "KleinBottle".into(),
            "Non-orientable closed surface with Euler characteristic 0".into(),
        );
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

const CUSTOM_RIPPLE_MANIFOLD_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Custom Ripple Manifold",
    "domain": "geometry_topology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "surface_type": "CustomHeightfield",
    "sphere_radius": 1.0,
    "torus_major_radius": 2.0,
    "torus_minor_radius": 0.5,
    "klein_radius": 2.0,
    "u_resolution": 36,
    "v_resolution": 36,
    "camera": {
      "pitch": 0.55,
      "yaw": 0.45,
      "zoom": 1.0
    },
    "custom_expr": "a * cos(k * sqrt(x^2 + y^2 + 0.01))",
    "custom_params": {
      "a": 0.7,
      "k": 2.8
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

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "surface_viewer_custom_ripple",
        title: "Custom Ripple Manifold",
        description: "z = a*cos(k*sqrt(x^2 + y^2)): Compiled Monge heightfield wireframe with .obj export and live sliders.",
        category: "Differential Geometry",
        icon: "🌊",
        domain: "geometry_topology",
        tool_name: Some("SurfaceViewer"),
        preset_generator: || CUSTOM_RIPPLE_MANIFOLD_PRESET_JSON,
    }
}

#[cfg(test)]
#[path = "surface_viewer/tests.rs"]
mod tests;

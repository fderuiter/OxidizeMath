#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "surface_viewer/cache.rs"]
mod cache;
#[path = "surface_viewer_ui.rs"]
mod ui;

use super::custom_surface::{CustomSurfaceSandbox, SurfaceType};
use crate::framework::{Camera3D, InteractiveTool, Scene3DBuilder};
use crate::web_io::{spawn_save_as_file_dialog, SaveAsDialogConfig};
use eframe::egui;
use math_explorer::pure_math::differential_geometry::surface::{
    KleinBottle, ParametricSurface, Sphere, Torus,
};
use std::collections::BTreeMap;
use std::sync::mpsc::{self, Receiver, TryRecvError};

const OBJ_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "OBJ File",
    extensions: &["obj"],
    fallback_filename: "surface.obj",
    write_error_prefix: "Failed to save OBJ mesh",
};

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
    #[serde(skip_serializing_if = "Option::is_none")]
    u_bounds: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    v_bounds: Option<[f64; 2]>,
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
    #[serde(default)]
    u_bounds: Option<[f64; 2]>,
    #[serde(default)]
    v_bounds: Option<[f64; 2]>,
}

/// Interactive 3D parametric wireframe manifold viewer with asynchronous `.obj` export.
pub struct SurfaceViewer {
    surface_type: SurfaceType,
    sphere_radius: f64,
    torus_major_radius: f64,
    torus_minor_radius: f64,
    klein_radius: f64,
    u_resolution: usize,
    v_resolution: usize,
    pub(crate) camera: Camera3D,
    pub(crate) scene_builder: Scene3DBuilder,
    pub(crate) custom_sandbox: CustomSurfaceSandbox,
    pub(super) obj_export_rx: Option<Receiver<Result<String, String>>>,
    pub(super) obj_export_status: Option<(String, bool)>,
    cache: cache::SurfaceGeometryCache,
}

impl Default for SurfaceViewer {
    fn default() -> Self {
        let camera = Camera3D::new(0.5, 0.5, 1.0);
        Self {
            surface_type: SurfaceType::Torus,
            sphere_radius: 1.0,
            torus_major_radius: 2.0,
            torus_minor_radius: 0.5,
            klein_radius: 2.0,
            u_resolution: 30,
            v_resolution: 30,
            camera,
            scene_builder: Scene3DBuilder::new(camera),
            custom_sandbox: CustomSurfaceSandbox::default(),
            obj_export_rx: None,
            obj_export_status: None,
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

    /// Generates serialized Wavefront `.obj` text for the active parametric surface mesh.
    pub fn build_obj_mesh_string(&self) -> Result<String, String> {
        let surface = self.create_surface();
        let (u_range, v_range) = self.custom_sandbox.uv_domain_for(self.surface_type);
        let mesh = super::export_utils::surface_to_mesh_with_domain(
            surface.as_ref(),
            u_range,
            v_range,
            self.u_resolution,
            self.v_resolution,
        );
        oxidize_core::mesh::export_mesh_to_obj_string(&mesh)
            .map_err(|e| format!("Failed to serialize OBJ mesh: {e}"))
    }

    #[cfg(test)]
    fn build_obj_string(&self) -> String {
        self.build_obj_mesh_string().unwrap_or_default()
    }

    pub(super) fn trigger_obj_export(&mut self) {
        match self.build_obj_mesh_string() {
            Ok(obj_content) => {
                let (tx, rx) = mpsc::channel();
                self.obj_export_rx = Some(rx);
                self.obj_export_status = None;
                spawn_save_as_file_dialog(
                    tx,
                    OBJ_SAVE_CONFIG,
                    "surface.obj",
                    obj_content,
                    |path, default_name| {
                        path.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(default_name)
                            .to_string()
                    },
                );
            }
            Err(err) => {
                self.obj_export_status = Some((err, true));
            }
        }
    }

    /// Polls any in-flight asynchronous `.obj` export channel and dispatches live status updates.
    pub fn poll_obj_export(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.obj_export_rx else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(saved_name)) => {
                let msg = format!("Saved OBJ mesh to {saved_name}");
                crate::accessibility::announce_live_status(ctx, &msg);
                self.obj_export_status = Some((msg, false));
                self.obj_export_rx = None;
            }
            Ok(Err(err)) => {
                crate::accessibility::announce_live_assertive_status(ctx, &err);
                self.obj_export_status = Some((err, true));
                self.obj_export_rx = None;
            }
            Err(TryRecvError::Empty) => {
                ctx.request_repaint();
            }
            Err(TryRecvError::Disconnected) => {
                self.obj_export_rx = None;
            }
        }
    }

    /// Populates `self.scene_builder` with cached world `Point3<f64>` wireframe grid polylines.
    pub fn populate_scene(&mut self, color: egui::Color32) {
        self.populate_cached_wireframe(color);
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
            u_bounds,
            v_bounds,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
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
        self.custom_sandbox.load_domain_bounds(
            loaded.surface_type,
            loaded.u_bounds,
            loaded.v_bounds,
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

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.poll_obj_export(ui.ctx());
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

#[path = "surface_viewer/presets.rs"]
mod presets;
#[cfg(test)]
use presets::{CUSTOM_RIPPLE_MANIFOLD_PRESET_JSON, STARTER_PRESET_JSON};

#[cfg(test)]
#[path = "surface_viewer_tests.rs"]
mod async_obj_tests;

#[cfg(test)]
#[path = "surface_viewer/tests.rs"]
mod tests;

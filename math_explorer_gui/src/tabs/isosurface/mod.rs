// @explorer_feature = "applied"

use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod presets;
pub mod render;
pub mod tool;
mod ui;

/// Applied mathematics tab hosting the Implicit Isosurface Workbench (`isosurface` domain).
pub struct IsosurfaceTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for IsosurfaceTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("isosurface"),
        }
    }
}

impl ExplorerTab for IsosurfaceTab {
    fn name(&self) -> &'static str {
        "Volumetric Isosurfaces"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "isosurface");
    }

    fn state_revision(&self) -> u64 {
        self.framework.state_revision()
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        self.framework.load_state(state);
    }

    fn select_tool_by_name(&mut self, name: &str) -> bool {
        self.framework.select_tool_by_name(name)
    }

    fn framework(&self) -> Option<&crate::framework::SimulationFramework> {
        Some(&self.framework)
    }

    fn framework_mut(&mut self) -> Option<&mut crate::framework::SimulationFramework> {
        Some(&mut self.framework)
    }
}

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::presets::BUILTIN_ISOSURFACE_PRESETS;
    use super::tool::IsosurfaceTool;
    use super::*;
    use crate::accessibility::{all_announcements, clear_test_announcements};
    use crate::app::{MathExplorerApp, ThemeMode};
    use crate::framework::InteractiveTool;

    #[test]
    fn test_all_builtin_presets_produce_closed_manifolds_and_valid_exports() {
        let mut tool = IsosurfaceTool::default();
        for spec in BUILTIN_ISOSURFACE_PRESETS {
            tool.apply_preset_spec(spec, false);
            assert!(
                tool.field_error.is_none(),
                "Preset '{}' reported field error: {:?}",
                spec.id,
                tool.field_error
            );
            assert!(
                tool.stats.is_watertight_two_manifold(),
                "Preset '{}' must be watertight: {:?}",
                spec.id,
                tool.stats
            );
            let stl = oxidize_core::mesh::export_mesh_to_stl_binary(&tool.mesh).expect("stl");
            assert_eq!(stl.len(), 84 + 50 * tool.stats.triangle_count);
            let obj = oxidize_core::mesh::export_mesh_to_obj_string(&tool.mesh).expect("obj");
            assert!(obj.contains("f "));
        }
    }

    #[test]
    fn test_isosurface_url_roundtrip_and_invalid_field_guidance() {
        let _guard = crate::reflective_ui::tests::lock_test_mutex();
        let mut app = MathExplorerApp::default();
        let idx = app
            .tabs
            .iter()
            .position(|t| t.url_slug() == "isosurface")
            .expect("isosurface tab registered");
        app.selected_tab = idx;

        let custom_envelope = r#"{
          "metadata": {
            "schema_version": "1.0",
            "title": "Custom SDF",
            "domain": "isosurface",
            "timestamp": "2026-11-21T00:00:00Z"
          },
          "config": {
            "half_extent": 2.5,
            "iso_level": 0.0,
            "thickness": 0.4,
            "resolution": 24,
            "sheet_mode": false,
            "show_wireframe": true,
            "expression": "max(sqrt(x^2 + y^2 + z^2) - 2.0, -min(abs(x), abs(y)) + 0.4)",
            "camera": { "pitch": 0.4, "yaw": 0.8, "zoom": 1.1 }
          }
        }"#;
        app.tabs[idx]
            .import_url_preset_json(Some("IsosurfaceTool"), Some(custom_envelope))
            .expect("import custom SDF URL preset");

        let share_url = app.encode_current_share_url().expect("encode share url");
        assert!(share_url.contains("#tab=isosurface"));
        let hash = &share_url[share_url.find('#').unwrap()..];

        let mut restored = MathExplorerApp::default();
        assert!(restored
            .apply_share_location_strings(hash, "", None)
            .unwrap());
        assert_eq!(restored.selected_tab, idx);
        assert_eq!(
            restored.tabs[idx].export_url_preset_json().unwrap(),
            app.tabs[idx].export_url_preset_json().unwrap()
        );

        let mut bad_tool = IsosurfaceTool::default();
        bad_tool.input_bar.set_expression("1 / 0");
        bad_tool.rebuild_mesh(false);
        assert!(bad_tool.mesh.indices.is_empty());
        assert!(bad_tool.field_error.is_some());
    }

    #[test]
    fn test_isosurface_viewport_accesskit_live_announce_and_themes() {
        clear_test_announcements();
        let ctx = egui::Context::default();
        ctx.enable_accesskit();

        let mut tool = IsosurfaceTool::default();
        tool.apply_preset_spec(&BUILTIN_ISOSURFACE_PRESETS[1], true);

        for theme in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
            ctx.set_visuals(theme.visuals());
            let out = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));
            let ak = out
                .platform_output
                .accesskit_update
                .as_ref()
                .expect("accesskit enabled");
            let has_graphics_doc = ak
                .nodes
                .iter()
                .any(|(_, n)| n.role() == egui::accesskit::Role::GraphicsDocument);
            assert!(
                has_graphics_doc,
                "Plot3DViewport must expose GraphicsDocument in {theme:?}"
            );
        }

        let announces = all_announcements();
        assert!(
            announces.iter().any(
                |(msg, prio)| prio == "polite" && msg.contains("Rebuilt watertight isosurface")
            ),
            "Expected polite rebuild announcement, got {announces:?}"
        );
        assert!(!egui_plot::get_last_plot_primitives().is_empty());
    }

    fn capture_isosurface_3d(
        ctx: &egui::Context,
        tool: &mut IsosurfaceTool,
    ) -> Vec<(String, Vec<[f64; 3]>)> {
        egui_plot::reset_detect_critical_points_call_count();
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(960.0, 720.0),
            )),
            ..Default::default()
        };
        let _ = ctx.run(raw, |c| {
            egui_plot::clear_last_plot_datasets();
            tool.show(c);
        });
        assert_eq!(egui_plot::detect_critical_points_call_count(), 0);
        egui_plot::get_last_plot_datasets_3d()
    }

    #[test]
    fn test_isosurface_camera_pan_perspective_sidecar_invariance_and_zero_2d_pois() {
        use super::render::project_mesh_facets;
        use crate::camera::ProjectionMode;

        let ctx = egui::Context::default();
        let mut tool = IsosurfaceTool::default();
        let rev_initial = tool.state_revision();

        let sidecar_a = capture_isosurface_3d(&ctx, &mut tool);
        assert_eq!(sidecar_a.len(), 1);
        assert_eq!(sidecar_a[0].0, "Isosurface Mesh");
        assert!(!sidecar_a[0].1.is_empty());

        tool.camera.pan_x = 0.85;
        tool.camera.pan_y = -0.55;
        tool.camera.projection = ProjectionMode::Perspective;
        assert_ne!(tool.state_revision(), rev_initial);

        let sidecar_b = capture_isosurface_3d(&ctx, &mut tool);
        assert_eq!(sidecar_a, sidecar_b);

        let facets = project_mesh_facets(
            &tool.mesh,
            &tool.camera,
            &ThemeMode::Dark.palette(),
            true,
            false,
        );
        assert!(!facets.is_empty());
        let p0 = facets[0].points[0];
        assert!(tool.mesh.vertices.iter().any(|v| {
            let proj = tool
                .camera
                .project(&[f64::from(v.x), f64::from(v.y), f64::from(v.z)]);
            (p0[0] - proj[0]).hypot(p0[1] - proj[1]) < 1e-12
        }));
    }

    #[test]
    fn test_isosurface_parameterized_3d_formula_synthesizes_sliders_and_roundtrips() {
        let mut tool = IsosurfaceTool::default();
        tool.sheet_mode = false;
        tool.resolution = 20;
        tool.half_extent = 2.5;
        tool.input_bar.set_expression("sqrt(x^2 + y^2 + z^2) - r");
        assert_eq!(tool.input_bar.active_param_names(), &["r".to_string()]);
        tool.input_bar.set_param("r", 1.4);
        tool.rebuild_mesh(false);
        assert!(tool.field_error.is_none());
        assert!(tool.stats.is_watertight_two_manifold());
        let tris_r14 = tool.stats.triangle_count;
        let rev_r14 = tool.state_revision();

        tool.input_bar.set_param("r", 1.9);
        assert_ne!(tool.state_revision(), rev_r14);
        tool.rebuild_mesh(false);
        assert!(tool.stats.is_watertight_two_manifold());
        assert!(tool.stats.triangle_count > tris_r14);

        let saved = tool.save_state().expect("serialize parameterized state");
        assert!(saved.contains("\"params\""));
        let mut restored = IsosurfaceTool::default();
        restored
            .try_load_state(&saved)
            .expect("restore parameterized state");
        assert_eq!(restored.input_bar.active_param_value("r"), Some(1.9));
        assert_eq!(restored.stats, tool.stats);
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::tool::IsosurfaceTool;
    use eframe::egui;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn wasm_isosurface_tool_exports_watertight_stl_and_obj_downloads() {
        let ctx = egui::Context::default();
        let mut tool = IsosurfaceTool::default();
        assert!(tool.stats.is_watertight_two_manifold());

        tool.trigger_stl_export(&ctx);
        let (stl_ok, ref stl_msg) = tool
            .export_status
            .clone()
            .expect("wasm STL export delivers synchronous status");
        assert!(stl_ok, "WASM STL export failed: {stl_msg}");
        assert!(stl_msg.contains(".stl"));

        tool.trigger_obj_export(&ctx);
        let (obj_ok, ref obj_msg) = tool
            .export_status
            .clone()
            .expect("wasm OBJ export delivers synchronous status");
        assert!(obj_ok, "WASM OBJ export failed: {obj_msg}");
        assert!(obj_msg.contains(".obj"));
    }
}

// [cite:isosurface_extraction]

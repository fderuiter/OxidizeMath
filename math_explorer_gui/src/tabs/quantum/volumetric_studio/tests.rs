#![cfg_attr(any(), verified(opt_out = "tests"))]
//! Unit, zero-allocation (`VerifiedAllocator`), cutting-plane, `.obj` export, and contract tests.

use super::presets::{
    PRESET_1S_2PZ_DIPOLE_JSON, PRESET_3DZ2_CONE_JSON, PRESET_3D_TURING_GYROID_JSON,
};
use super::state::{FieldSourceMode, SliceAxis, VolumetricRenderMode, VolumetricStudioConfig};
use super::VolumetricFieldStudioTool;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::InteractiveTool;
use eframe::egui;

#[test]
fn test_zero_alloc_hot_path_quantum_and_turing_under_verified_allocator() {
    let mut tool = VolumetricFieldStudioTool::default();
    tool.config.grid_resolution = 22;
    tool.config.render_mode = VolumetricRenderMode::HybridShellAndCloud;
    tool.config.cut_enabled = true;
    tool.refresh_zero_alloc();
    tool.step_zero_alloc();

    let mut turing_tool = VolumetricFieldStudioTool::default();
    turing_tool.config.source = FieldSourceMode::TuringMorphogenesis;
    turing_tool.config.grid_resolution = 20;
    turing_tool.config.render_mode = VolumetricRenderMode::HybridShellAndCloud;
    turing_tool.config.cut_enabled = true;
    turing_tool.refresh_zero_alloc();
    turing_tool.step_zero_alloc();

    let _alloc_lock = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        tool.step_zero_alloc();
        turing_tool.step_zero_alloc();
    });

    assert!(!tool.engine.pos_mesh.vertices.is_empty());
    assert!(!tool.engine.splat_ws.projected_splats.is_empty());
    assert!(!turing_tool.engine.splat_ws.projected_splats.is_empty());
}

#[test]
fn test_cutting_plane_reveals_2s_and_3s_interior_radial_nodes_and_gyroid_channels() {
    let mut tool = VolumetricFieldStudioTool::default();
    tool.config.source = FieldSourceMode::QuantumSuperposition;
    tool.config.render_mode = VolumetricRenderMode::MarchingCubesMesh;
    tool.config.grid_resolution = 36;
    tool.config.n1 = 2;
    tool.config.l1 = 0;
    tool.config.m1 = 0;
    tool.config.superposition_mix = 0.0;
    tool.config.iso_threshold = 0.012;
    tool.config.cut_enabled = true;
    tool.config.cut_axis = SliceAxis::X;
    tool.config.cut_offset = 0.0;
    tool.refresh_zero_alloc();

    let nodes_2s = tool.engine.count_cutting_plane_radial_nodes(&tool.config);
    assert_eq!(
        nodes_2s, 1,
        "2s orbital must exhibit 1 interior radial node"
    );
    assert!(!tool.engine.slice_contours.is_empty());

    tool.config.n1 = 3;
    tool.config.l1 = 0;
    tool.config.m1 = 0;
    tool.config.iso_threshold = 0.0025;
    tool.config.cut_enabled = false;
    tool.refresh_zero_alloc();
    let uncut_facets = tool.engine.projected_facets.len();

    tool.config.cut_enabled = true;
    tool.config.cut_offset = 0.0;
    tool.refresh_zero_alloc();
    let nodes_3s = tool.engine.count_cutting_plane_radial_nodes(&tool.config);
    assert_eq!(
        nodes_3s, 2,
        "3s orbital must exhibit 2 interior radial nodes"
    );
    assert!(tool.engine.projected_facets.len() < uncut_facets);

    tool.import_preset_json(PRESET_3D_TURING_GYROID_JSON)
        .expect("3D Turing gyroid preset valid");
    assert!(tool.config.cut_enabled);
    assert!(!tool.engine.slice_contours.is_empty());
}

#[test]
fn test_obj_export_produces_valid_wavefront_vertices_unit_normals_and_faces() {
    let mut tool = VolumetricFieldStudioTool::default();
    for preset in [PRESET_3DZ2_CONE_JSON, PRESET_3D_TURING_GYROID_JSON] {
        tool.import_preset_json(preset).expect("valid preset");
        let obj = tool.export_obj_string().expect("non-empty OBJ mesh");
        assert!(obj.starts_with("o oxidize_mesh\n"));
        let mut v_count = 0usize;
        let mut vn_count = 0usize;
        let mut f_count = 0usize;
        for line in obj.lines() {
            if let Some(rest) = line.strip_prefix("v ") {
                v_count += 1;
                let coords: Vec<f32> = rest
                    .split_whitespace()
                    .map(|t| t.parse::<f32>().expect("float vertex"))
                    .collect();
                assert_eq!(coords.len(), 3);
                assert!(coords.iter().all(|c| c.is_finite()));
            } else if let Some(rest) = line.strip_prefix("vn ") {
                vn_count += 1;
                let n: Vec<f32> = rest
                    .split_whitespace()
                    .map(|t| t.parse::<f32>().expect("float normal"))
                    .collect();
                assert_eq!(n.len(), 3);
                let mag = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                assert!(
                    (mag - 1.0).abs() < 1e-3,
                    "normal must be unit length, got {mag}"
                );
            } else if line.starts_with("f ") {
                f_count += 1;
            }
        }
        assert!(v_count >= 12 && v_count == vn_count && f_count >= 4);
    }
}

#[test]
fn test_viewport_render_modes_and_turing_perturbation_pulse() {
    let mut tool = VolumetricFieldStudioTool::default();
    tool.config.grid_resolution = 18;
    let ctx = egui::Context::default();
    for mode in [
        VolumetricRenderMode::MarchingCubesMesh,
        VolumetricRenderMode::GaussianRadianceCloud,
        VolumetricRenderMode::HybridShellAndCloud,
    ] {
        tool.config.render_mode = mode;
        tool.refresh_zero_alloc();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::Vec2::new(960.0, 640.0),
            )),
            ..Default::default()
        };
        let _ = ctx.run(input, |ctx| {
            tool.show(ctx);
        });
        assert!(tool.engine.projected_facets.iter().all(|f| {
            f.pts.iter().all(|p| p.x.is_finite() && p.y.is_finite()) && f.depth.is_finite()
        }));
    }

    tool.config.source = FieldSourceMode::TuringMorphogenesis;
    tool.refresh_zero_alloc();
    let mid = tool.engine.voxel_grid.width() / 2;
    let before = tool.engine.voxel_grid.get(mid, mid, mid);
    let readout = tool
        .engine
        .apply_probe_or_pulse(&mut tool.config, [0.0, 0.0, 0.0]);
    let after = tool.engine.voxel_grid.get(mid, mid, mid);
    assert!(
        after > before,
        "Activator pulse must elevate local concentration"
    );
    assert!(readout.legendre_polar.is_finite());
}

#[test]
fn test_state_bounds_validation_starter_presets_and_finite_state_guard() {
    let mut tool = VolumetricFieldStudioTool::default();
    for preset in [
        PRESET_1S_2PZ_DIPOLE_JSON,
        PRESET_3DZ2_CONE_JSON,
        PRESET_3D_TURING_GYROID_JSON,
    ] {
        tool.import_preset_json(preset)
            .expect("starter preset must load cleanly");
        assert!(FiniteStateGuard::is_finite_state(&tool));
    }

    let mut bad_json: serde_json::Value =
        serde_json::from_str(&tool.save_state().unwrap()).unwrap();
    bad_json["unknown_extra_field"] = serde_json::json!(42);
    assert!(tool.try_load_state(&bad_json.to_string()).is_err());

    let mut invalid_cfg = VolumetricStudioConfig::default();
    invalid_cfg.l1 = invalid_cfg.n1;
    assert!(invalid_cfg.validate().is_err());
    invalid_cfg.grid_resolution = 96;
    assert!(invalid_cfg.validate().is_err());

    tool.engine.diverged = true;
    assert!(FiniteStateGuard::diverged(&tool));
    FiniteStateGuard::reset_parameters(&mut tool);
    assert!(!FiniteStateGuard::diverged(&tool));
    assert!(FiniteStateGuard::is_finite_state(&tool));
}

#[test]
fn test_custom_implicit_ast_zero_alloc_48_cubed_and_20_cubed_drag_preview() {
    use super::ast_field::{DEFAULT_CUSTOM_3D_EXPR, GYROID_3D_EXPR, ORBITAL_4F_XYZ_EXPR};

    let mut tool = VolumetricFieldStudioTool::default();
    tool.config.source = FieldSourceMode::CustomImplicitAst;
    tool.config.render_mode = VolumetricRenderMode::HybridShellAndCloud;
    tool.config.grid_resolution = 48;
    tool.config.iso_threshold = 0.045;
    tool.config.cut_enabled = true;
    tool.config.cut_axis = SliceAxis::Z;
    tool.config.cut_offset = 0.2;

    for formula in [DEFAULT_CUSTOM_3D_EXPR, ORBITAL_4F_XYZ_EXPR, GYROID_3D_EXPR] {
        tool.set_custom_expression(formula)
            .expect("template 3D AST must compile");
        assert!(!tool.engine.pos_mesh.vertices.is_empty());
    }
    tool.config.iso_threshold = 0.25;
    tool.set_custom_param_zero_alloc(0, 2.4, true);
    assert_eq!(tool.engine.voxel_grid.width(), 20);
    tool.set_custom_param_zero_alloc(0, 2.4, false);
    assert_eq!(tool.engine.voxel_grid.width(), 48);

    {
        let _alloc_lock = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            tool.set_custom_param_zero_alloc(0, 2.75, true);
            tool.set_custom_param_zero_alloc(0, 2.75, false);
        });
    }

    assert_eq!(tool.engine.voxel_grid.width(), 48);
    assert!(!tool.engine.pos_mesh.vertices.is_empty());
    assert!(!tool.engine.neg_mesh.vertices.is_empty());
    assert!(!tool.engine.splat_ws.projected_splats.is_empty());
    assert!(!tool.engine.slice_contours.is_empty());
    let obj = tool.export_obj_string().expect("AST isosurface OBJ export");
    assert!(obj.starts_with("o oxidize_mesh\n") && obj.contains("\nf "));

    #[cfg(not(target_arch = "wasm32"))]
    save_gyroid_preview_png(&tool.engine, "/tmp/volumetric_3d_gyroid_ast.png");
}

#[cfg(not(target_arch = "wasm32"))]
fn save_gyroid_preview_png(engine: &super::engine::VolumetricEngine, path: &str) {
    let side = 384usize;
    let mut rgba = vec![16u8; side * side * 4];
    for px in rgba.as_chunks_mut::<4>().0 {
        *px = [14, 18, 28, 255];
    }
    let map_xy = |x: f64, y: f64| -> (isize, isize) {
        let sx = ((x + 3.1) / 6.2 * (side as f64 - 1.0)).round() as isize;
        let sy = ((3.1 - y) / 6.2 * (side as f64 - 1.0)).round() as isize;
        (sx, sy)
    };
    for facet in &engine.projected_facets {
        let p0 = map_xy(facet.pts[0].x, facet.pts[0].y);
        let p1 = map_xy(facet.pts[1].x, facet.pts[1].y);
        let p2 = map_xy(facet.pts[2].x, facet.pts[2].y);
        let min_x = p0.0.min(p1.0).min(p2.0).clamp(0, side as isize - 1);
        let max_x = p0.0.max(p1.0).max(p2.0).clamp(0, side as isize - 1);
        let min_y = p0.1.min(p1.1).min(p2.1).clamp(0, side as isize - 1);
        let max_y = p0.1.max(p1.1).max(p2.1).clamp(0, side as isize - 1);
        let edge = |a: (isize, isize), b: (isize, isize), c: (isize, isize)| {
            (c.0 - a.0) * (b.1 - a.1) - (c.1 - a.1) * (b.0 - a.0)
        };
        let area = edge(p0, p1, p2);
        if area == 0 {
            continue;
        }
        let [r, g, b, _] = facet.fill.to_srgba_unmultiplied();
        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let w0 = edge(p1, p2, (px, py));
                let w1 = edge(p2, p0, (px, py));
                let w2 = edge(p0, p1, (px, py));
                if (area > 0 && w0 >= 0 && w1 >= 0 && w2 >= 0)
                    || (area < 0 && w0 <= 0 && w1 <= 0 && w2 <= 0)
                {
                    let idx = (py as usize * side + px as usize) * 4;
                    rgba[idx..idx + 4].copy_from_slice(&[r, g, b, 255]);
                }
            }
        }
    }
    if let Ok(file) = std::fs::File::create(path) {
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), side as u32, side as u32);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        if let Ok(mut writer) = enc.write_header() {
            let _ = writer.write_image_data(&rgba);
        }
    }
}

#[test]
fn test_singular_3d_ast_sanitization_and_custom_state_url_roundtrip() {
    let mut tool = VolumetricFieldStudioTool::default();
    tool.config.source = FieldSourceMode::CustomImplicitAst;
    tool.config.grid_resolution = 21;
    tool.set_custom_expression("1 / (x^2 + y^2 + z^2)")
        .expect("singular 3D AST compiles cleanly");
    assert!(!FiniteStateGuard::diverged(&tool));
    assert!(FiniteStateGuard::is_finite_state(&tool));
    assert_eq!(tool.engine.voxel_grid.get(10, 10, 10), 0.0);

    assert!(tool
        .set_custom_expression("a + b + c + d + f + g + h + x")
        .is_err());

    tool.set_custom_expression("x * y * z * exp(-a * sqrt(x^2 + y^2 + z^2))")
        .expect("4f_xyz compiles");
    let saved = tool.save_state().expect("serialize custom AST state");
    crate::url_state::URL_JSON_BOUNDS
        .validate_json(&saved)
        .expect("fits URL_JSON_BOUNDS");

    let mut fw = crate::framework::SimulationFramework::new("quantum");
    assert!(fw.select_tool_by_name("VolumetricFieldStudioTool"));
    fw.active_tool
        .as_mut()
        .unwrap()
        .try_load_state(&saved)
        .expect("restore custom AST state");
    let share_url = fw.encode_share_url().expect("encode share URL");
    let fragment = format!("#{}", share_url.split_once('#').unwrap().1);
    let params = crate::url_state::parse_share_location(&fragment, "")
        .unwrap()
        .unwrap();
    let mut restored_fw = crate::framework::SimulationFramework::new("quantum");
    restored_fw
        .import_url_preset_json(params.tool.as_deref(), params.state_json.as_deref())
        .expect("import shared URL preset");
    assert_eq!(
        restored_fw.active_tool.as_ref().unwrap().save_state(),
        Some(saved)
    );

    let oversized = VolumetricStudioConfig {
        custom_expr: "x+".repeat(300),
        ..Default::default()
    };
    assert!(oversized.validate().is_err());
}

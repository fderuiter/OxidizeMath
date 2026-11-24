#![cfg_attr(any(), verified(opt_out = "tests"))]
//! `CustomImplicitAst` zero-allocation `48^3` / `20^3` drag-preview, singularity clamping, and URL tests.

use super::super::ast_field::{DEFAULT_CUSTOM_3D_EXPR, GYROID_3D_EXPR, ORBITAL_4F_XYZ_EXPR};
use super::super::state::{
    FieldSourceMode, SliceAxis, VolumetricRenderMode, VolumetricStudioConfig,
};
use super::super::VolumetricFieldStudioTool;
use super::assert_watertight_outward_solid_stl_ply_and_obj;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::InteractiveTool;

#[test]
fn test_custom_implicit_ast_zero_alloc_48_cubed_and_20_cubed_drag_preview() {
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
    assert_watertight_outward_solid_stl_ply_and_obj(
        &tool,
        tool.config.export_size_mm as f32,
        "48^3 Cut Gyroid AST",
    );

    #[cfg(not(target_arch = "wasm32"))]
    save_gyroid_preview_png(&tool.engine, "/tmp/volumetric_3d_gyroid_ast.png");
}

#[cfg(not(target_arch = "wasm32"))]
fn save_gyroid_preview_png(engine: &super::super::engine::VolumetricEngine, path: &str) {
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

    let baseline_state = tool.save_state().expect("baseline state");
    for bad_json in [
        r#"{"source":"custom_implicit_ast","custom_expr":"sin(x +"}"#,
        r#"{"source":"custom_implicit_ast","custom_expr":"a + b + c + d + f + g + h + x"}"#,
        r#"{"source":"custom_implicit_ast","custom_expr":"cos(k * x)","custom_params":{"k":1e9}}"#,
    ] {
        assert!(
            tool.try_load_state(bad_json).is_err(),
            "Rejected state must return Err: {bad_json}"
        );
        assert_eq!(tool.save_state().as_deref(), Some(baseline_state.as_str()));
        assert!(!FiniteStateGuard::diverged(&tool));
    }
}

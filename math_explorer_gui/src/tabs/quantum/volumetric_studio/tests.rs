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

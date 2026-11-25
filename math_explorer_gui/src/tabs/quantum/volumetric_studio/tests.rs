#![cfg_attr(any(), verified(opt_out = "tests"))]
//! Unit, zero-allocation (`VerifiedAllocator`), cutting-plane, millimeter-scaled manifold 3D
//! solid export, custom 3D implicit CAD/AST, and 3DGS `.ply` tests for `VolumetricFieldStudioTool`.

#[path = "tests_ast.rs"]
mod tests_ast;

use super::presets::{
    PRESET_1S_2PZ_DIPOLE_JSON, PRESET_3DZ2_CONE_JSON, PRESET_3D_IMPLICIT_GYROID_AST_JSON,
    PRESET_3D_TURING_GYROID_JSON, PRESET_CSG_BOOLEAN_SOLID_JSON, PRESET_GYROID_TPMS_JSON,
    PRESET_SCHWARZ_P_SHELL_JSON,
};
use super::state::{
    FieldSourceMode, ImplicitClosureMode, SliceAxis, VolumetricRenderMode, VolumetricStudioConfig,
};
use super::VolumetricFieldStudioTool;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{InteractiveTool, SimulationFramework};
use crate::tabs::geometry_topology::export_utils::{
    count_open_and_nonmanifold_edges, signed_volume,
};
use crate::url_state::URL_JSON_BOUNDS;
use eframe::egui;
use oxidize_core::mesh::{export_colored_mesh_to_ply_binary, export_mesh_to_stl_binary, Mesh};

#[test]
fn test_zero_alloc_hot_path_quantum_turing_and_implicit_under_verified_allocator() {
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

    let mut implicit_tool = VolumetricFieldStudioTool::default();
    implicit_tool
        .import_preset_json(PRESET_GYROID_TPMS_JSON)
        .expect("gyroid preset loads");
    implicit_tool.config.grid_resolution = 24;
    implicit_tool.config.render_mode = VolumetricRenderMode::HybridShellAndCloud;
    implicit_tool.config.cut_enabled = true;
    implicit_tool.config.cut_axis = SliceAxis::Y;
    implicit_tool.config.cut_offset = 0.15;
    implicit_tool.refresh_zero_alloc();

    let _alloc_lock = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        tool.step_zero_alloc();
        turing_tool.step_zero_alloc();
        implicit_tool.config.implicit_level = 0.12;
        implicit_tool.config.wall_thickness = 0.62;
        implicit_tool.config.cut_offset = -0.10;
        implicit_tool.refresh_zero_alloc();
        implicit_tool.step_zero_alloc();
    });

    assert!(!tool.engine.pos_mesh.vertices.is_empty());
    assert!(!tool.engine.splat_ws.projected_splats.is_empty());
    assert!(!turing_tool.engine.splat_ws.projected_splats.is_empty());
    assert!(!implicit_tool.engine.pos_mesh.vertices.is_empty());
    assert!(!implicit_tool.engine.splat_ws.projected_splats.is_empty());
    assert!(!implicit_tool.engine.slice_contours.is_empty());
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

fn compute_mesh_max_span(mesh: &Mesh) -> f32 {
    let mut min_v = [f32::INFINITY; 3];
    let mut max_v = [f32::NEG_INFINITY; 3];
    for v in &mesh.vertices {
        min_v[0] = min_v[0].min(v.x);
        min_v[1] = min_v[1].min(v.y);
        min_v[2] = min_v[2].min(v.z);
        max_v[0] = max_v[0].max(v.x);
        max_v[1] = max_v[1].max(v.y);
        max_v[2] = max_v[2].max(v.z);
    }
    (max_v[0] - min_v[0])
        .max(max_v[1] - min_v[1])
        .max(max_v[2] - min_v[2])
}

pub(super) fn assert_watertight_outward_solid_stl_ply_and_obj(
    tool: &VolumetricFieldStudioTool,
    expected_span_mm: f32,
    label: &str,
) {
    let caps = tool.export_capabilities();
    assert!(caps.stl && caps.ply && caps.obj);
    let mesh = tool
        .export_mesh_3d()
        .expect("solid 3D capability")
        .unwrap_or_else(|e| panic!("{label} export_mesh_3d failed: {e}"));
    assert!(!mesh.indices.is_empty(), "{label} mesh empty");
    assert!(mesh.normals.iter().all(|n| n.is_finite()));
    assert_eq!(
        count_open_and_nonmanifold_edges(&mesh),
        (0, 0),
        "{label} must have (0, 0) open/non-manifold edges"
    );
    assert!(
        signed_volume(&mesh) > 0.0,
        "{label} must have positive outward signed_volume"
    );
    let unscaled_stl = export_mesh_to_stl_binary(&mesh).expect("unitless binary STL");
    assert!(!std::str::from_utf8(&unscaled_stl[..80])
        .unwrap_or("")
        .contains("UNITS=mm"));
    let mm_mesh = tool
        .export_millimeter_mesh_3d()
        .expect("MillimeterMesh capability")
        .unwrap_or_else(|e| panic!("{label} export_millimeter_mesh_3d failed: {e}"));
    let stl = export_mesh_to_stl_binary(&mm_mesh).expect("binary STL serialization");
    assert_eq!(stl.len(), 84 + 50 * (mesh.indices.len() / 3));
    assert!(std::str::from_utf8(&stl[..80])
        .unwrap_or("")
        .contains("UNITS=mm"));
    assert!((compute_mesh_max_span(&mesh) - expected_span_mm).abs() <= 0.01);

    let colored = tool
        .export_colored_mesh_3d()
        .expect("colored mesh capability")
        .unwrap_or_else(|e| panic!("{label} export_colored_mesh_3d failed: {e}"));
    assert_eq!(colored.colors.len(), colored.mesh.vertices.len());
    assert_eq!(count_open_and_nonmanifold_edges(&colored.mesh), (0, 0));
    assert!(signed_volume(&colored.mesh) > 0.0);
    assert!((compute_mesh_max_span(&colored.mesh) - expected_span_mm).abs() <= 0.01);
    let ply = export_colored_mesh_to_ply_binary(&colored).expect("valid colored PLY");
    assert!(ply.starts_with(b"ply\nformat binary_little_endian 1.0\n"));

    let obj = tool.export_obj_string().expect("non-empty OBJ mesh");
    assert!(obj.contains("\nv ") && obj.contains("\nvn ") && obj.contains("\nf "));
}

#[test]
fn test_watertight_manifold_topology_and_mm_stl_across_implicit_orbital_and_turing() {
    let mut tool = VolumetricFieldStudioTool::default();
    let cases = [
        ("Gyroid TPMS", PRESET_GYROID_TPMS_JSON, 60.0_f32),
        ("Schwarz-P Shell", PRESET_SCHWARZ_P_SHELL_JSON, 50.0_f32),
        ("CSG Cube-Sphere", PRESET_CSG_BOOLEAN_SOLID_JSON, 45.0_f32),
        ("3dz2 Orbital", PRESET_3DZ2_CONE_JSON, 85.0_f32),
        ("Cut-Plane Turing", PRESET_3D_TURING_GYROID_JSON, 120.0_f32),
        (
            "Schoen Gyroid AST",
            PRESET_3D_IMPLICIT_GYROID_AST_JSON,
            55.0_f32,
        ),
    ];
    for (label, preset, span_mm) in cases {
        tool.import_preset_json(preset).expect("valid preset");
        tool.config.grid_resolution = 26;
        tool.config.export_size_mm = f64::from(span_mm);
        tool.refresh_zero_alloc();
        assert_watertight_outward_solid_stl_ply_and_obj(&tool, span_mm, label);
    }

    // Verify clamp + exp + cut-plane half-space implicit solid also seals with (0, 0) open edges.
    tool.config.source = FieldSourceMode::CustomImplicit;
    tool.config.implicit_expr =
        "exp(-0.35 * (x^2 + y^2 + z^2)) - clamp(0.5 + 0.15 * z, 0.2, 0.8) = 0".into();
    tool.config.implicit_closure = ImplicitClosureMode::HalfSpaceSolid;
    tool.config.cut_enabled = true;
    tool.config.cut_axis = SliceAxis::X;
    tool.config.cut_offset = 0.05;
    tool.config.export_size_mm = 32.5;
    tool.refresh_zero_alloc();
    assert_watertight_outward_solid_stl_ply_and_obj(&tool, 32.5, "Cut Clamped-Gaussian Implicit");
}

#[test]
fn test_custom_implicit_csg_6_params_url_roundtrip_and_transient_syntax_resilience() {
    let mut tool = VolumetricFieldStudioTool::default();
    tool.config.source = FieldSourceMode::CustomImplicit;
    tool.config.render_mode = VolumetricRenderMode::MarchingCubesMesh;
    tool.config.grid_resolution = 24;
    tool.config.implicit_closure = ImplicitClosureMode::HalfSpaceSolid;
    tool.config.x_bounds = [-3.0, 3.0];
    tool.config.y_bounds = [-3.0, 3.0];
    tool.config.z_bounds = [-2.0, 2.0];
    let expr_6p = "max(min(sqrt(x^2 + y^2 + z^2) - R, max(abs(x), abs(y)) - b), r - sqrt(x^2 + y^2)) + a * sin(k * clamp(z, -1.5, 1.5) + c * atan2(y, x))";
    tool.implicit_ui.bar.set_expression(expr_6p);
    assert!(tool.implicit_ui.bar.is_valid());
    assert_eq!(tool.implicit_ui.bar.active_param_names().len(), 6);
    for (name, val) in [
        ("R", 1.7),
        ("b", 1.3),
        ("r", 0.55),
        ("a", 0.12),
        ("k", 1.8),
        ("c", 1.0),
    ] {
        tool.implicit_ui.bar.set_param(name, val);
    }
    tool.config.implicit_expr = expr_6p.to_string();
    tool.config.implicit_params = tool.implicit_ui.bar.params_map();
    tool.refresh_zero_alloc();
    let valid_mesh_tris = tool.engine.pos_mesh.indices.len() / 3;
    assert!(valid_mesh_tris > 20);

    let saved = tool.save_url_state().expect("url state");
    URL_JSON_BOUNDS
        .validate_json(&saved)
        .expect("within 4 KiB URL bounds");
    let mut restored = VolumetricFieldStudioTool::default();
    restored
        .try_load_state(&saved)
        .expect("6-param state loads");
    assert_eq!(restored.config, tool.config);

    assert_transient_syntax_resilience(tool, valid_mesh_tris);
}

fn assert_transient_syntax_resilience(tool: VolumetricFieldStudioTool, valid_mesh_tris: usize) {
    let mut fw = SimulationFramework::new("quantum");
    fw.active_tool = Some(Box::new(tool));
    fw.cache_valid_snapshot();
    {
        let active = fw
            .active_tool
            .as_mut()
            .and_then(|t| t.as_any_mut())
            .and_then(|a| a.downcast_mut::<VolumetricFieldStudioTool>())
            .unwrap();
        active.implicit_ui.bar.set_expression("sin(k * x + ");
        assert!(!active.implicit_ui.bar.is_valid());
        assert!(FiniteStateGuard::is_finite_state(active));
        assert!(!FiniteStateGuard::diverged(active));
        active.refresh_zero_alloc();
        assert_eq!(active.engine.pos_mesh.indices.len() / 3, valid_mesh_tris);
    }
    let ctx = egui::Context::default();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(960.0, 640.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(input, |ctx| {
        fw.show(ctx, "quantum");
    });
    assert!(!fw.is_diverged);
}

#[test]
fn test_viewport_render_modes_splat_ply_and_turing_perturbation_pulse() {
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

    // Verify GaussianRadianceCloud disables STL/OBJ with hover reasons and exports INRIA 3DGS PLY.
    tool.config.render_mode = VolumetricRenderMode::GaussianRadianceCloud;
    tool.refresh_zero_alloc();
    let caps = tool.export_capabilities();
    assert!(!caps.stl && caps.ply && !caps.obj);
    assert!(!caps.stl_disabled_reason.is_empty());
    assert!(!caps.obj_disabled_reason.is_empty());
    assert!(tool.export_mesh_3d().is_none());
    assert!(tool.export_colored_mesh_3d().is_none());
    let splat_ply = tool
        .export_splat_ply_3d()
        .expect("3DGS PLY hook active")
        .expect("3DGS PLY bytes");
    let header_end = splat_ply
        .windows(b"end_header\n".len())
        .position(|w| w == b"end_header\n")
        .expect("PLY header");
    let header = std::str::from_utf8(&splat_ply[..header_end]).unwrap();
    assert!(header.contains("property float f_dc_0") && header.contains("property float rot_0"));

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
        PRESET_GYROID_TPMS_JSON,
        PRESET_SCHWARZ_P_SHELL_JSON,
        PRESET_CSG_BOOLEAN_SOLID_JSON,
        PRESET_3D_IMPLICIT_GYROID_AST_JSON,
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

#![cfg_attr(any(), verified(opt_out = "tests"))]
//! Unit, zero-allocation (`VerifiedAllocator`), cutting-plane, manifold 3D solid export,
//! custom 3D implicit formula, and 3DGS `.ply` tests for `VolumetricFieldStudioTool`.

use super::presets::{
    PRESET_1S_2PZ_DIPOLE_JSON, PRESET_3DZ2_CONE_JSON, PRESET_3D_GYROID_TPMS_JSON,
    PRESET_3D_TURING_GYROID_JSON,
};
use super::state::{FieldSourceMode, SliceAxis, VolumetricRenderMode, VolumetricStudioConfig};
use super::VolumetricFieldStudioTool;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{InteractiveTool, SimulationFramework};
use crate::url_state::URL_JSON_BOUNDS;
use eframe::egui;
use oxidize_core::mesh::{export_colored_mesh_to_ply_binary, export_mesh_to_stl_binary, Mesh};
use std::collections::HashMap;

fn count_open_and_nonmanifold_edges(mesh: &Mesh) -> (usize, usize) {
    let mut directed: HashMap<(usize, usize), usize> = HashMap::new();
    for &[a, b, c] in mesh.indices.as_chunks::<3>().0 {
        for (u, v) in [(a, b), (b, c), (c, a)] {
            *directed.entry((u, v)).or_insert(0) += 1;
        }
    }
    let mut open = 0usize;
    let mut nonmanifold = 0usize;
    for (&(u, v), &cnt) in &directed {
        let rev = directed.get(&(v, u)).copied().unwrap_or(0);
        if cnt != 1 || rev > 1 {
            nonmanifold += 1;
        } else if rev == 0 {
            open += 1;
        }
    }
    (open, nonmanifold)
}

fn compute_signed_volume(mesh: &Mesh) -> f64 {
    let mut sum = 0.0_f64;
    for &[i0, i1, i2] in mesh.indices.as_chunks::<3>().0 {
        let (a, b, c) = (mesh.vertices[i0], mesh.vertices[i1], mesh.vertices[i2]);
        let (ax, ay, az) = (f64::from(a.x), f64::from(a.y), f64::from(a.z));
        let (bx, by, bz) = (f64::from(b.x), f64::from(b.y), f64::from(b.z));
        let (cx, cy, cz) = (f64::from(c.x), f64::from(c.y), f64::from(c.z));
        sum += ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
    }
    sum / 6.0
}

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
        .import_preset_json(PRESET_3D_GYROID_TPMS_JSON)
        .expect("gyroid preset loads");
    implicit_tool.config.grid_resolution = 24;
    implicit_tool.config.render_mode = VolumetricRenderMode::HybridShellAndCloud;
    implicit_tool.config.cut_enabled = true;
    implicit_tool.refresh_zero_alloc();

    let _alloc_lock = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        tool.step_zero_alloc();
        turing_tool.step_zero_alloc();
        implicit_tool.step_zero_alloc();
    });

    assert!(!tool.engine.pos_mesh.vertices.is_empty());
    assert!(!tool.engine.splat_ws.projected_splats.is_empty());
    assert!(!turing_tool.engine.splat_ws.projected_splats.is_empty());
    assert!(!implicit_tool.engine.pos_mesh.vertices.is_empty());
    assert!(!implicit_tool.engine.splat_ws.projected_splats.is_empty());
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
fn test_watertight_manifold_solid_exports_for_gyroid_tpms_and_sliced_turing() {
    let mut tool = VolumetricFieldStudioTool::default();
    for preset in [
        PRESET_3DZ2_CONE_JSON,
        PRESET_3D_TURING_GYROID_JSON,
        PRESET_3D_GYROID_TPMS_JSON,
    ] {
        tool.import_preset_json(preset).expect("valid preset");
        let colored = tool
            .export_colored_mesh_3d()
            .expect("mesh mode active")
            .expect("solid colored mesh");
        assert_eq!(colored.colors.len(), colored.mesh.vertices.len());
        assert_eq!(
            count_open_and_nonmanifold_edges(&colored.mesh),
            (0, 0),
            "Preset mesh must be a closed 2-manifold"
        );
        assert!(
            compute_signed_volume(&colored.mesh) > 0.0,
            "Preset mesh must have positive outward signed volume"
        );
        let stl = export_mesh_to_stl_binary(&colored.mesh).expect("valid STL");
        assert_eq!(stl.len(), 84 + 50 * (colored.mesh.indices.len() / 3));
        let ply = export_colored_mesh_to_ply_binary(&colored).expect("valid colored PLY");
        assert!(ply.starts_with(b"ply\nformat binary_little_endian 1.0\n"));

        let obj = tool.export_obj_string().expect("non-empty OBJ mesh");
        assert!(obj.starts_with("o oxidize_mesh\n"));
        assert!(obj.contains("\nv ") && obj.contains("\nvn ") && obj.contains("\nf "));
    }

    // Verify 32^3 Gyroid TPMS remains a closed manifold with active cutting plane as well.
    tool.import_preset_json(PRESET_3D_GYROID_TPMS_JSON).unwrap();
    tool.config.cut_enabled = true;
    tool.config.cut_axis = SliceAxis::Y;
    tool.config.cut_offset = 0.15;
    tool.refresh_zero_alloc();
    let sliced_tpms = tool.export_mesh_3d().unwrap().unwrap();
    assert_eq!(count_open_and_nonmanifold_edges(&sliced_tpms), (0, 0));
    assert!(compute_signed_volume(&sliced_tpms) > 0.0);
}

#[test]
fn test_custom_implicit_csg_6_params_url_roundtrip_and_transient_syntax_resilience() {
    let mut tool = VolumetricFieldStudioTool::default();
    tool.config.source = FieldSourceMode::CustomImplicitFormula;
    tool.config.render_mode = VolumetricRenderMode::MarchingCubesMesh;
    tool.config.grid_resolution = 24;
    tool.config.shell_enabled = false;
    tool.config.x_bounds = [-3.0, 3.0];
    tool.config.y_bounds = [-3.0, 3.0];
    tool.config.z_bounds = [-2.0, 2.0];
    let expr_6p = "max(min(sqrt(x^2 + y^2 + z^2) - R, max(abs(x), abs(y)) - b), r - sqrt(x^2 + y^2)) + a * sin(k * clamp(z, -1.5, 1.5) + c * atan2(y, x))";
    tool.formula_bar.set_expression(expr_6p);
    assert!(tool.formula_bar.is_valid());
    assert_eq!(tool.formula_bar.active_param_names().len(), 6);
    for (name, val) in [
        ("R", 1.7),
        ("b", 1.3),
        ("r", 0.55),
        ("a", 0.12),
        ("k", 1.8),
        ("c", 1.0),
    ] {
        tool.formula_bar.set_param(name, val);
    }
    tool.sync_valid_formula_to_config_and_engine();
    tool.refresh_zero_alloc();
    let valid_mesh_tris = tool.engine.pos_mesh.indices.len() / 3;
    assert!(valid_mesh_tris > 20);

    // Verify compact JSON/URL round-trip within URL_JSON_BOUNDS (<= 4 KiB).
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
        active.formula_bar.set_expression("sin(k * x + ");
        assert!(!active.formula_bar.is_valid());
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
        PRESET_3D_GYROID_TPMS_JSON,
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

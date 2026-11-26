// [cite:modular_polynomials_review]
#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::tabs::geometry_topology::export_utils::{
    count_open_and_nonmanifold_edges, signed_volume,
};
use crate::url_state::URL_JSON_BOUNDS;
use oxidize_core::mesh::{
    export_colored_mesh_to_ply_binary, export_mesh_to_obj_string, export_mesh_to_stl_binary,
};

#[test]
fn default_and_try_load_state_run_zero_pde_steps_and_fit_url_bounds() {
    let mut tool = SurfaceMorphogenesisTool::default();
    assert!(
        tool.engine.is_none(),
        "Default::default() must run zero PDE steps and defer engine allocation"
    );

    tool.surface_type = SurfaceType::CustomHeightfield;
    tool.kinetics_kind = MorphogenesisKineticsKind::GrayScott;
    tool.overlay_mode = SurfaceOverlayMode::MetricAnisotropy;
    tool.u_resolution = 24;
    tool.v_resolution = 20;

    let saved = tool.save_state().expect("save_state must succeed");
    assert!(
        URL_JSON_BOUNDS.validate_json(&saved).is_ok(),
        "Saved JSON ({} bytes) must fit within URL_JSON_BOUNDS",
        saved.len()
    );
    let fragment = crate::url_state::encode_share_fragment(
        "geometry_topology",
        Some("SurfaceMorphogenesisTool"),
        Some(&saved),
    )
    .expect("URL share fragment must encode cleanly");
    let parsed = crate::url_state::parse_share_location(&fragment, "")
        .expect("fragment parses")
        .expect("share params present");

    let mut restored = SurfaceMorphogenesisTool::default();
    restored
        .try_load_state(&parsed.state_json.expect("state_json"))
        .expect("try_load_state succeeds");
    assert!(
        restored.engine.is_none(),
        "try_load_state() must run zero PDE steps"
    );
    assert_eq!(restored.surface_type, SurfaceType::CustomHeightfield);
    assert_eq!(restored.kinetics_kind, MorphogenesisKineticsKind::GrayScott);
    assert_eq!(restored.overlay_mode, SurfaceOverlayMode::MetricAnisotropy);

    // Reject unknown fields via #[serde(deny_unknown_fields)]
    let bad = saved.replacen('{', r#"{"unexpected_field":123,"#, 1);
    assert!(restored.try_load_state(&bad).is_err());
}

#[test]
fn coupled_buckling_exports_watertight_stl_ply_obj_across_all_topologies() {
    for (surf_type, kinetics) in [
        (SurfaceType::Sphere, MorphogenesisKineticsKind::Schnakenberg),
        (SurfaceType::Torus, MorphogenesisKineticsKind::Schnakenberg),
        (
            SurfaceType::KleinBottle,
            MorphogenesisKineticsKind::GrayScott,
        ),
        (
            SurfaceType::CustomHeightfield,
            MorphogenesisKineticsKind::Schnakenberg,
        ),
    ] {
        let mut tool = SurfaceMorphogenesisTool {
            surface_type: surf_type,
            kinetics_kind: kinetics,
            u_resolution: 20,
            v_resolution: 18,
            buckling_gain: 0.30,
            ..Default::default()
        };
        if kinetics == MorphogenesisKineticsKind::GrayScott {
            tool.d_u = 0.04;
            tool.d_v = 0.02;
        }
        for _ in 0..8 {
            assert!(tool.step_once());
        }

        let mesh = tool
            .export_mesh_3d()
            .expect("3D mesh hook enabled")
            .expect("buckled mesh builds");
        assert!(mesh.normals.iter().all(|n| n.is_finite()));
        assert_eq!(
            count_open_and_nonmanifold_edges(&mesh),
            (0, 0),
            "Buckled {surf_type:?} solid must have zero open or non-manifold edges"
        );
        assert!(
            signed_volume(&mesh) > 0.0,
            "Buckled {surf_type:?} solid must have positive outward volume"
        );
        let stl = export_mesh_to_stl_binary(&mesh).expect("binary STL encodes");
        assert!(stl.len() > 84);
        verify_colored_overlays(&mut tool);
    }
}

fn verify_colored_overlays(tool: &mut SurfaceMorphogenesisTool) {
    for overlay in [
        SurfaceOverlayMode::MorphogenConcentration,
        SurfaceOverlayMode::GaussianCurvature,
        SurfaceOverlayMode::MetricAnisotropy,
    ] {
        tool.overlay_mode = overlay;
        let colored = tool
            .export_colored_mesh_3d()
            .expect("colored mesh hook enabled")
            .expect("colored mesh builds");
        assert_eq!(colored.colors.len(), colored.mesh.vertices.len());
        assert_eq!(count_open_and_nonmanifold_edges(&colored.mesh), (0, 0));
        let ply = export_colored_mesh_to_ply_binary(&colored).expect("PLY encodes");
        assert!(ply.starts_with(b"ply\n"));
        let obj = export_mesh_to_obj_string(&colored.mesh).expect("OBJ encodes");
        assert!(obj.contains("\nv ") && obj.contains("\nf "));
    }
}

#[test]
fn viewport_pulse_brush_sonification_and_finite_state_guard_rollback() {
    let mut tool = SurfaceMorphogenesisTool {
        u_resolution: 20,
        v_resolution: 20,
        ..Default::default()
    };
    tool.ensure_engine();
    let tube_xy = tool.camera.project(&[2.2, 0.0, 0.0]);
    let hit = tool
        .pick_frontmost_surface_vertex(tube_xy)
        .expect("frontmost 3D vertex picked on torus tube");
    let (_, hit_vz) = tool.camera.project_with_depth(&[hit.x, hit.y, hit.z]);
    assert!(hit_vz > -0.5, "Picked vertex must lie on front-facing side");

    let max_u_before = tool
        .engine
        .as_ref()
        .unwrap()
        .runner
        .u()
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    tool.brush_action = PulseBrushAction::ActivatorPulse;
    tool.brush_strength = 1.25;
    tool.apply_brush_at_world_point(hit);
    let max_u_after = tool
        .engine
        .as_ref()
        .unwrap()
        .runner
        .u()
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_u_after > max_u_before + 0.5,
        "3D pulse brush must elevate local morphogen concentration ({max_u_before} -> {max_u_after})"
    );

    for _ in 0..6 {
        assert!(tool.step_once());
    }
    let frame = tool.current_sonification_frame();
    assert!(
        frame.is_audible(),
        "Active morphogen variance and buckled curvature shifts must synthesize an audible SonificationFrame"
    );

    // Inject non-finite parameter and verify FiniteStateGuard rollback preserves finite state
    let snapshot_pts = tool.engine.as_ref().unwrap().flow.points().to_vec();
    tool.dt = f64::NAN;
    assert!(!tool.has_finite_parameters());
    assert!(!tool.step_once());
    assert_eq!(tool.engine.as_ref().unwrap().rollback_count, 1);
    assert_eq!(
        tool.engine.as_ref().unwrap().flow.points(),
        snapshot_pts.as_slice()
    );
    tool.sanitize_and_recover();
    assert!(tool.has_finite_parameters());
    assert!(tool.step_once());
}

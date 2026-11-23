#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::audio::synthesize_manifold_frame;
use super::presets::{
    PRESET_CORTICAL_SPREADING_DEPRESSION_JSON, PRESET_FREESURFER_GLM_CONTRAST_JSON,
    PRESET_MEAN_CURVATURE_FLOW_JSON, PRESET_SPHERICAL_TURING_MITOSIS_JSON,
};
use super::state::ManifoldKind;
#[cfg(all(feature = "applied", feature = "biology"))]
use super::state::PdeRegime;
use super::RiemannianPdeStudio;
use crate::accessibility::{all_announcements, clear_test_announcements};
use crate::framework::InteractiveTool;
use crate::tabs::geometry_topology::export_utils::{
    count_open_and_nonmanifold_edges, signed_volume,
};
use crate::tabs::geometry_topology::surface_brush::{
    pick_front_facing_vertex, BrushPolarity, GeodesicBrushStroke, ViewportBrushMode,
};
use eframe::egui;

fn run_studio_frame(
    studio: &mut RiemannianPdeStudio,
    ctx: &egui::Context,
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    ctx.run(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 800.0),
            )),
            predicted_dt: 1.0 / 60.0,
            events,
            ..Default::default()
        },
        |sub_ctx| studio.show(sub_ctx),
    )
}

#[test]
fn test_default_studio_is_lightweight_and_exposes_3d_accessibility_controls() {
    let mut studio = RiemannianPdeStudio::default();
    assert!(
        studio.geometry.is_none(),
        "Default must stay lazy and lightweight"
    );

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let out = run_studio_frame(&mut studio, &ctx, Vec::new());
    let ak = out
        .platform_output
        .accesskit_update
        .expect("AccessKit update expected");

    assert!(
        ak.nodes.iter().any(|(_, n)| {
            n.role() == egui::accesskit::Role::Button
                && n.label().unwrap_or("").contains("Reset View")
        }),
        "Must expose Reset View button on initial open"
    );
    assert!(
        ak.nodes.iter().any(|(_, n)| {
            [n.label(), n.value()]
                .into_iter()
                .flatten()
                .any(|s| s.contains("Drag canvas or use WASD to orbit"))
        }),
        "Must expose Orbit helper caption on initial open"
    );
    assert!(
        ak.nodes
            .iter()
            .any(|(_, n)| n.role() == egui::accesskit::Role::Canvas),
        "Must expose AccessKit Role::Canvas for the 3D manifold viewport"
    );
}

#[test]
fn test_front_facing_raycast_picks_front_hemisphere_and_geodesic_brush_perturbs_field() {
    let mut studio = RiemannianPdeStudio::default();
    studio.ensure_initialized();
    let geom = studio.geometry.as_ref().expect("geometry initialized");

    let hit = pick_front_facing_vertex(
        &studio.config.camera,
        &geom.positions,
        &geom.triangles,
        &geom.normals,
        [0.0, 0.0],
        geom.two_sided,
        0.35,
    )
    .expect("Origin ray must intersect front hemisphere of sphere");
    assert!(hit.vertex < geom.positions.len());
    let n_cam = studio
        .config
        .camera
        .rotate_to_camera_space(&geom.normals[hit.vertex]);
    assert!(n_cam[2] > -0.05, "Picked vertex must be front-facing");

    let before = studio.engine.as_ref().unwrap().scalar_field[hit.vertex];
    studio.apply_stroke(GeodesicBrushStroke {
        center_vertex: hit.vertex,
        radius: 0.30,
        strength: 0.80,
        polarity: BrushPolarity::Excite,
        via_keyboard: false,
    });
    let after = studio.engine.as_ref().unwrap().scalar_field[hit.vertex];
    assert!(after > before + 0.5, "Excite brush stroke must raise field");
}

fn key_press(key: egui::Key) -> Vec<egui::Event> {
    vec![egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }]
}

#[test]
fn test_keyboard_geodesic_cursor_moves_and_stamps() {
    let mut studio = RiemannianPdeStudio::default();
    studio.config.brush_mode = ViewportBrushMode::Sculpt;
    let ctx = egui::Context::default();
    ctx.enable_accesskit();

    let _ = run_studio_frame(&mut studio, &ctx, Vec::new());
    let plot_id = studio
        .viewport_response_id
        .expect("Plot3DViewport must record persistent response ID");
    ctx.memory_mut(|m| m.request_focus(plot_id));
    let _ = run_studio_frame(&mut studio, &ctx, Vec::new());

    let start_cursor = studio.brush.cursor_vertex;
    clear_test_announcements();
    let _ = run_studio_frame(&mut studio, &ctx, key_press(egui::Key::ArrowRight));
    assert_ne!(
        studio.brush.cursor_vertex, start_cursor,
        "ArrowRight must advance geodesic cursor along CSR 1-ring"
    );

    let v_idx = studio.brush.cursor_vertex;
    let field_before = studio.engine.as_ref().unwrap().scalar_field[v_idx];
    let _ = run_studio_frame(&mut studio, &ctx, key_press(egui::Key::Space));
    let field_after = studio.engine.as_ref().unwrap().scalar_field[v_idx];
    assert!(
        (field_after - field_before).abs() > 1e-3,
        "Space key must inject geodesic brush perturbation"
    );
    assert!(
        all_announcements()
            .iter()
            .any(|(msg, _)| msg.contains("Applied geodesic brush")),
        "Keyboard brush stroke must emit polite ARIA announcement"
    );
}

#[test]
fn test_sculpt_mode_registers_shortcuts_and_tab_exits_focus() {
    let mut studio = RiemannianPdeStudio::default();
    studio.config.brush_mode = ViewportBrushMode::Sculpt;
    let ctx = egui::Context::default();
    let _ = run_studio_frame(&mut studio, &ctx, Vec::new());
    let plot_id = studio.viewport_response_id.expect("viewport ID");
    ctx.memory_mut(|m| m.request_focus(plot_id));
    let _ = run_studio_frame(&mut studio, &ctx, Vec::new());

    let cmds = ctx.data(|d| {
        d.get_temp::<egui_plot::commands::CommandRegistryData>(egui::Id::new("CMD_REGISTRY"))
            .unwrap_or_default()
            .commands
    });
    assert!(
        cmds.iter()
            .any(|c| c.context == "Geodesic Brush" && c.name == "Arrow Keys"),
        "Sculpt mode must register Geodesic Brush shortcuts in CommandRegistryData"
    );

    let _ = run_studio_frame(&mut studio, &ctx, key_press(egui::Key::Tab));
    assert_ne!(
        ctx.memory(|m| m.focused()),
        Some(plot_id),
        "Tab key must move focus cleanly out of the 3D canvas without trapping"
    );
}

#[test]
fn test_all_six_manifolds_export_watertight_outward_colored_ply_meshes() {
    for kind in ManifoldKind::ALL {
        let mut studio = RiemannianPdeStudio::default();
        studio.config.manifold = kind;
        studio.config.u_res = 16;
        studio.config.v_res = 14;
        studio.rebuild_all();
        studio.step_once();

        let colored = studio
            .export_colored_mesh_3d()
            .expect("supports 3D colored mesh")
            .unwrap_or_else(|e| panic!("{kind:?} export failed: {e}"));
        assert_eq!(colored.colors.len(), colored.mesh.vertices.len());
        assert!(colored.mesh.normals.iter().all(|n| n.is_finite()));
        assert_eq!(
            count_open_and_nonmanifold_edges(&colored.mesh),
            (0, 0),
            "{kind:?} exported mesh is not watertight 2-manifold"
        );
        assert!(
            signed_volume(&colored.mesh) > 0.0,
            "{kind:?} exported mesh winding must have positive signed volume"
        );

        let ply_buf = oxidize_core::mesh::export_colored_mesh_to_ply_binary(&colored)
            .expect("binary PLY serialization succeeds");
        assert!(ply_buf.starts_with(b"ply\nformat binary_little_endian 1.0\n"));
        let obj_txt = oxidize_core::mesh::export_mesh_to_obj_string(&colored.mesh)
            .expect("OBJ serialization succeeds");
        assert!(obj_txt.starts_with("o oxidize_mesh\nv "));
        let stl_buf = oxidize_core::mesh::export_mesh_to_stl_binary(&colored.mesh)
            .expect("binary STL serialization succeeds");
        assert_eq!(stl_buf.len(), 84 + 50 * (colored.mesh.indices.len() / 3));
    }
}

#[test]
fn test_export_utils_welded_sphere_cotangent_laplacian_l2_convergence() {
    use crate::tabs::geometry_topology::export_utils::polar_surface_to_manifold_mesh;
    use math_explorer::pure_math::differential_geometry::surface::Sphere;
    use math_explorer::pure_math::differential_geometry::CotangentLaplacian;

    let eval_err = |nu: usize, nv: usize, l: usize| -> f64 {
        let mesh = polar_surface_to_manifold_mesh(&Sphere { radius: 1.0 }, nu, nv)
            .expect("welded sphere mesh");
        let lap = CotangentLaplacian::from_mesh(&mesh);
        let eig = -((l * (l + 1)) as f64);
        let field: Vec<f64> = mesh
            .vertices
            .iter()
            .map(|v| {
                let z = f64::from(v.z);
                if l == 1 {
                    z
                } else {
                    0.5 * (3.0 * z * z - 1.0)
                }
            })
            .collect();
        let mut out = vec![0.0; field.len()];
        lap.apply(&field, &mut out);
        let mut err_sq = 0.0;
        let mut norm_sq = 0.0;
        for (i, (&ap, &u)) in out.iter().zip(field.iter()).enumerate() {
            assert!(ap.is_finite(), "Polar/interior NaN at vertex {i}");
            let a = lap.vertex_area(i);
            let ex = eig * u;
            err_sq += a * (ap - ex).powi(2);
            norm_sq += a * ex.powi(2);
        }
        (err_sq / norm_sq).sqrt()
    };

    let (c1, f1) = (eval_err(20, 14, 1), eval_err(40, 28, 1));
    let (c2, f2) = (eval_err(20, 14, 2), eval_err(40, 28, 2));
    assert!(f1 < c1 * 0.5 && f1 < 0.025, "l=1 err {c1} -> {f1}");
    assert!(f2 < c2 * 0.5 && f2 < 0.045, "l=2 err {c2} -> {f2}");
}

fn simulate_primary_drag(
    studio: &mut RiemannianPdeStudio,
    ctx: &egui::Context,
    p0: egui::Pos2,
    p1: egui::Pos2,
) {
    let _ = run_studio_frame(
        studio,
        ctx,
        vec![
            egui::Event::PointerMoved(p0),
            egui::Event::PointerButton {
                pos: p0,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    let _ = run_studio_frame(studio, ctx, vec![egui::Event::PointerMoved(p1)]);
    let _ = run_studio_frame(
        studio,
        ctx,
        vec![egui::Event::PointerButton {
            pos: p1,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }],
    );
}

#[test]
fn test_pointer_drag_sculpts_in_sculpt_mode_and_orbits_camera_in_orbit_mode() {
    let mut studio = RiemannianPdeStudio::default();
    let ctx = egui::Context::default();
    let _ = run_studio_frame(&mut studio, &ctx, Vec::new());

    let p0 = egui::pos2(820.0, 400.0);
    let p1 = egui::pos2(860.0, 420.0);
    let yaw_init = studio.config.camera.yaw;
    let field_orbit_before = studio.engine.as_ref().unwrap().scalar_field.clone();

    simulate_primary_drag(&mut studio, &ctx, p0, p1);
    assert_ne!(
        studio.config.camera.yaw, yaw_init,
        "Orbit mode must rotate Camera3D"
    );
    assert_eq!(
        studio.engine.as_ref().unwrap().scalar_field,
        field_orbit_before
    );

    studio.config.brush_mode = ViewportBrushMode::Sculpt;
    let yaw_sculpt = studio.config.camera.yaw;
    let field_sculpt_before = studio.engine.as_ref().unwrap().scalar_field.clone();
    clear_test_announcements();
    simulate_primary_drag(&mut studio, &ctx, p0, p0);
    assert_eq!(
        studio.config.camera.yaw, yaw_sculpt,
        "Sculpt mode must lock primary orbit"
    );
    assert_ne!(
        studio.engine.as_ref().unwrap().scalar_field,
        field_sculpt_before,
        "Sculpt pointer stroke must perturb manifold vertices"
    );
    assert!(
        all_announcements()
            .iter()
            .any(|(msg, _)| msg.contains("Sculpt stroke on manifold vertex")),
        "Sculpt pointer stroke must announce accessible status"
    );
}

#[test]
fn test_cross_tab_handoff_opens_riemannian_studio_with_preset() {
    use super::handoff::{request_open_riemannian_preset, RiemannianPresetTarget};
    use crate::app::{domain_bar::poll_cross_tab_tool_handoffs, MathExplorerApp};

    let mut app = MathExplorerApp::default();
    let ctx = egui::Context::default();
    request_open_riemannian_preset(&ctx, RiemannianPresetTarget::MeanCurvatureFlow);
    poll_cross_tab_tool_handoffs(&mut app, &ctx);
    assert_eq!(app.tabs[app.selected_tab].name(), "Geometry & Topology");
    let saved = app.tabs[app.selected_tab]
        .save_state()
        .expect("tab state serialized");
    assert!(saved.contains("RiemannianPdeStudio") || saved.contains("Riemannian PDE Studio"));
    assert!(saved.contains("MeanCurvatureSmoothing"));
}

#[test]
fn test_presets_under_4kib_roundtrip_and_sonification_frame_is_finite() {
    for json in [
        PRESET_MEAN_CURVATURE_FLOW_JSON,
        PRESET_SPHERICAL_TURING_MITOSIS_JSON,
        PRESET_CORTICAL_SPREADING_DEPRESSION_JSON,
        PRESET_FREESURFER_GLM_CONTRAST_JSON,
    ] {
        assert!(json.len() < 4096, "Preset must stay well under 4 KiB");
        let mut studio = RiemannianPdeStudio::default();
        studio
            .import_preset_json(json)
            .expect("Curated preset must load cleanly");
        let exported = studio.export_preset_json().expect("export succeeds");
        let mut restored = RiemannianPdeStudio::default();
        restored
            .import_preset_json(&exported)
            .expect("roundtrip import succeeds");
        assert_eq!(studio.save_state(), restored.save_state());

        studio.step_once();
        let geom = studio.geometry.as_ref().unwrap();
        let eng = studio.engine.as_ref().unwrap();
        assert!(eng.scalar_field.iter().all(|v| v.is_finite()));
        let frame = synthesize_manifold_frame(geom, eng, studio.config.regime);
        assert!(frame.is_audible());
        assert!(frame.voices[0].freq_hz.is_finite());
    }

    let mut strict_check = RiemannianPdeStudio::default();
    assert!(
        strict_check
            .try_load_state(r#"{"manifold":"Sphere","unknown_injected_key":42}"#)
            .is_err(),
        "RiemannianStudioConfig must reject unknown fields via #[serde(deny_unknown_fields)]"
    );

    strict_check
        .import_preset_json(PRESET_MEAN_CURVATURE_FLOW_JSON)
        .unwrap();
    strict_check.ensure_initialized();
    let area_before: f64 = strict_check
        .geometry
        .as_ref()
        .unwrap()
        .laplacian
        .vertex_areas()
        .iter()
        .sum();
    for _ in 0..8 {
        strict_check.step_once();
    }
    let area_after: f64 = strict_check
        .geometry
        .as_ref()
        .unwrap()
        .laplacian
        .vertex_areas()
        .iter()
        .sum();
    assert!(
        area_after < area_before,
        "Forward MeanCurvatureSmoothing must decrease surface area ({area_before} -> {area_after})"
    );
}

#[cfg(all(feature = "applied", feature = "biology"))]
#[test]
fn test_freesurfer_glm_and_cortical_hodgkin_huxley_regimes_produce_finite_maps() {
    let mut studio = RiemannianPdeStudio::default();
    studio.config.manifold = ManifoldKind::FoldedCorticalShell;
    studio.config.regime = PdeRegime::FreeSurferMorphometryGlm;
    studio.rebuild_all();
    studio.step_once();

    let fs = &studio.engine.as_ref().unwrap().freesurfer;
    assert_eq!(
        fs.thickness_mm.len(),
        studio.geometry.as_ref().unwrap().positions.len()
    );
    assert!(fs.thickness_mm.iter().all(|&t| t > 0.5 && t.is_finite()));
    assert!(fs.pial_internal_energy.is_finite() && fs.pial_external_energy.is_finite());
    assert!(fs.glm_t_stat.iter().any(|&t| t.abs() > 1.0));

    studio
        .import_preset_json(PRESET_CORTICAL_SPREADING_DEPRESSION_JSON)
        .unwrap();
    studio.rebuild_all();
    for _ in 0..8 {
        studio.step_once();
    }
    let hh_field = &studio.engine.as_ref().unwrap().scalar_field;
    assert!(hh_field.iter().all(|v| v.is_finite()));
    let max_mv = hh_field.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_mv = hh_field.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        max_mv > 10.0 && min_mv < -55.0,
        "Cortical Hodgkin-Huxley preset must ignite suprathreshold action potential wave (min={min_mv}, max={max_mv})"
    );
}

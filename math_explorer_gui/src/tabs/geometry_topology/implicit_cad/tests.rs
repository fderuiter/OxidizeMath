#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::mesher::{MAX_GRID_RES, MIN_GRID_RES};
use super::presets::{
    PRESET_IMPLICIT_GYROID, PRESET_IMPLICIT_SCHWARZ_P, PRESET_IMPLICIT_TORUS_BOX,
};
use super::*;
use crate::framework::InteractiveTool;
use crate::url_state::{
    MAX_DECODED_URL_STATE_BYTES, MAX_URL_FRAGMENT_BYTES, parse_share_location,
};
use oxidize_core::mesh::{export_mesh_to_obj_string, export_mesh_to_stl_bytes};

#[test]
fn test_all_three_presets_are_watertight_zero_nan_normals_positive_volume_and_exportable() {
    for (name, preset_json) in [
        ("gyroid", PRESET_IMPLICIT_GYROID),
        ("schwarz_p", PRESET_IMPLICIT_SCHWARZ_P),
        ("torus_box_csg", PRESET_IMPLICIT_TORUS_BOX),
    ] {
        let mut studio = ImplicitCadStudio::default();
        studio
            .import_preset_json(preset_json)
            .unwrap_or_else(|e| panic!("Preset {name} failed to import: {e}"));
        studio.refresh_mesh();
        assert!(
            studio.workspace.last_error.is_none(),
            "Unexpected error on {name}: {:?}",
            studio.workspace.last_error
        );

        let diag = studio.workspace.diagnostics;
        assert!(
            diag.is_watertight(),
            "Preset {name} must be watertight: {diag:?}"
        );
        assert_eq!(diag.open_edges, 0, "Preset {name} must have 0 open edges");
        assert_eq!(diag.nan_normals, 0, "Preset {name} must have 0 NaN normals");
        assert!(
            diag.signed_volume_mm3 > 0.0,
            "Preset {name} must enclose positive mm^3 volume, got {}",
            diag.signed_volume_mm3
        );

        let stl = export_mesh_to_stl_bytes(&studio.workspace.mesh).expect("stl bytes");
        assert_eq!(stl.len(), 84 + diag.triangle_count * 50);
        let obj = export_mesh_to_obj_string(&studio.workspace.mesh).expect("obj string");
        assert!(!obj.contains("NaN"));
    }
}

#[test]
fn test_share_links_roundtrip_all_parameters_and_camera_pose_within_url_bounds() {
    let mut studio = ImplicitCadStudio::default();
    studio
        .import_preset_json(PRESET_IMPLICIT_TORUS_BOX)
        .expect("import torus-box");
    studio.box_size_mm = 48.5;
    studio.shell_thickness_mm = 1.2;
    studio.resolution = 28;
    studio.camera = Camera3D::new(0.41, -0.73, 1.35);

    let url = studio.encode_share_url().expect("encode share url");
    let hash_pos = url.find('#').expect("fragment separator");
    let fragment = &url[hash_pos..];
    assert!(
        fragment.len() <= MAX_URL_FRAGMENT_BYTES,
        "Share fragment exceeded {MAX_URL_FRAGMENT_BYTES} bytes: {}",
        fragment.len()
    );

    let parsed = parse_share_location(fragment, "")
        .expect("valid share location")
        .expect("present share params");
    assert_eq!(parsed.tab, "geometry_topology");
    assert_eq!(parsed.tool.as_deref(), Some("ImplicitCadStudio"));
    let state_json = parsed.state_json.expect("decoded state json");
    assert!(state_json.len() <= MAX_DECODED_URL_STATE_BYTES);

    let mut restored = ImplicitCadStudio::default();
    restored
        .import_preset_json(&state_json)
        .expect("restore from URL JSON");
    assert_eq!(restored.equation, studio.equation);
    assert_eq!(restored.box_size_mm, 48.5);
    assert_eq!(restored.shell_thickness_mm, 1.2);
    assert_eq!(restored.resolution, 28);
    assert_eq!(restored.camera, studio.camera);
}

#[test]
fn test_resolution_bounds_and_viewport_plot_primitives() {
    let mut studio = ImplicitCadStudio::default();
    assert!(studio
        .try_load_state(
            r#"{"equation":"x^2+y^2+z^2-10","box_size_mm":40.0,"shell_thickness_mm":0.0,"resolution":128}"#
        )
        .is_err());
    assert!(studio
        .try_load_state(
            r#"{"equation":"x^2+y^2+z^2-10","box_size_mm":40.0,"shell_thickness_mm":0.0,"resolution":8}"#
        )
        .is_err());

    studio.resolution = 200;
    studio.refresh_mesh();
    assert_eq!(studio.resolution, MAX_GRID_RES);
    studio.resolution = 4;
    studio.refresh_mesh();
    assert_eq!(studio.resolution, MIN_GRID_RES);

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| studio.show(ctx));
    assert!(!egui_plot::get_last_plot_datasets().is_empty());
    assert!(!egui_plot::get_last_plot_primitives().is_empty());
}

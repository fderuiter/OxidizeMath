#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::preset::{
    PRESET_CARTESIAN_HELMHOLTZ, PRESET_CYLINDRICAL_STOKES, PRESET_SPHERICAL_GAUSS,
};
use super::{CoordinateFrameKind, CurvilinearVectorStudio};
use crate::framework::InteractiveTool;
use crate::url_state::URL_JSON_BOUNDS;
use eframe::egui;

#[test]
fn test_curvilinear_studio_all_frames_theorems_and_helmholtz() {
    let mut studio = CurvilinearVectorStudio::default();
    assert!(studio.diagnostics.gauss_residual < 1e-2);
    assert!(studio.diagnostics.stokes_residual < 1e-2);
    assert!(studio.diagnostics.helmholtz.max_irrotational_curl < 1e-9);
    assert!(studio.diagnostics.helmholtz.max_reconstruction_error < 1e-11);

    studio
        .import_preset_json(PRESET_CYLINDRICAL_STOKES)
        .unwrap();
    assert_eq!(studio.frame, CoordinateFrameKind::Cylindrical);
    assert!(studio.diagnostics.gauss_residual < 2e-2);
    assert!(studio.diagnostics.stokes_residual < 2e-2);

    studio.import_preset_json(PRESET_SPHERICAL_GAUSS).unwrap();
    assert_eq!(studio.frame, CoordinateFrameKind::Spherical);
    assert!(studio.diagnostics.gauss_residual < 3e-2);
    assert!(studio.diagnostics.stokes_residual < 3e-2);
}

#[test]
fn test_curvilinear_studio_state_roundtrip_and_url_bounds() {
    for raw_preset in [
        PRESET_CARTESIAN_HELMHOLTZ,
        PRESET_CYLINDRICAL_STOKES,
        PRESET_SPHERICAL_GAUSS,
    ] {
        let mut a = CurvilinearVectorStudio::default();
        a.import_preset_json(raw_preset).unwrap();
        let saved = a.save_state().expect("serializes state");
        assert!(
            saved.len() <= URL_JSON_BOUNDS.max_bytes,
            "State exceeded URL_JSON_BOUNDS: {} bytes",
            saved.len()
        );
        let exported = a.export_preset_json().unwrap();
        let mut b = CurvilinearVectorStudio::default();
        b.import_preset_json(&exported).unwrap();
        assert_eq!(a.save_state(), b.save_state());
    }
}

#[test]
fn test_curvilinear_studio_headless_render_and_exports() {
    let mut studio = CurvilinearVectorStudio::default();
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| studio.show(ctx));
    assert!(!studio.export_datasets().is_empty());
    assert!(!studio.export_plot_primitives().is_empty());
}

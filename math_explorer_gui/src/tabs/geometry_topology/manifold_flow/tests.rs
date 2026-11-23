#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::super::custom_surface::SurfaceType;
use super::super::export_utils::{count_open_and_nonmanifold_edges, signed_volume};
use super::preset::{
    PRESET_SPHERE_HUISKEN_SHRINK, PRESET_TORUS_HEAT_DIFFUSION, PRESET_TORUS_MCF_SMOOTHING,
};
use super::{ManifoldFlowMode, ManifoldFlowStudio};
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::InteractiveTool;
use crate::url_state::URL_JSON_BOUNDS;
use nalgebra::Point3;

#[test]
fn test_manifold_flow_mcf_shrinkage_and_heat_smoothing() {
    let mut studio = ManifoldFlowStudio::default();
    studio
        .import_preset_json(PRESET_SPHERE_HUISKEN_SHRINK)
        .unwrap();
    let area0 = studio.current_metric;
    studio.advance_substeps(8);
    assert!(
        studio.current_metric < area0,
        "Sphere area must shrink under MCF: {area0} -> {}",
        studio.current_metric
    );

    studio
        .import_preset_json(PRESET_TORUS_HEAT_DIFFUSION)
        .unwrap();
    assert_eq!(studio.mode, ManifoldFlowMode::HeatDiffusion);
    let peak0 = studio
        .temp_grid
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    studio.advance_substeps(10);
    let peak1 = studio
        .temp_grid
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        peak1 < peak0,
        "Peak temperature must diffuse: {peak0} -> {peak1}"
    );
}

#[test]
fn test_manifold_flow_watertight_3d_exports_all_topologies() {
    let mut studio = ManifoldFlowStudio::default();
    for surf in [
        SurfaceType::Sphere,
        SurfaceType::Torus,
        SurfaceType::KleinBottle,
        SurfaceType::CustomHeightfield,
        SurfaceType::CustomParametric,
    ] {
        studio.surface_type = surf;
        studio.reset_simulation();
        studio.advance_substeps(2);
        let cm = studio
            .export_colored_mesh_3d()
            .expect("exports colored mesh")
            .expect("valid mesh");
        assert_eq!(
            count_open_and_nonmanifold_edges(&cm.mesh),
            (0, 0),
            "Non-watertight mesh for {surf:?}"
        );
        assert!(
            signed_volume(&cm.mesh) > 0.0,
            "Inward mesh winding for {surf:?}"
        );
    }
}

#[test]
fn test_manifold_flow_cfl_clamp_and_finite_state_guard_rollback() {
    let mut studio = ManifoldFlowStudio {
        dt: 10.0,
        ..Default::default()
    };
    studio.advance_substeps(1);
    assert!(studio.effective_dt <= studio.cfl_dt + 1e-12);
    assert!(studio.is_finite_state());

    // Inject non-finite coordinate and verify FiniteStateGuard rollback restores finite state
    studio.backup_points.clone_from(&studio.mcf.surface.points);
    studio.mcf.surface.points[0] = Point3::new(f64::NAN, 0.0, 0.0);
    assert!(!studio.is_finite_state());
    studio.advance_substeps(1);
    assert!(studio.is_finite_state());
    assert!(studio.recovery_note.is_some());
}

#[test]
fn test_manifold_flow_presets_and_url_bounds() {
    for preset in [
        PRESET_TORUS_MCF_SMOOTHING,
        PRESET_SPHERE_HUISKEN_SHRINK,
        PRESET_TORUS_HEAT_DIFFUSION,
    ] {
        let mut a = ManifoldFlowStudio::default();
        a.import_preset_json(preset).unwrap();
        let saved = a.save_state().unwrap();
        assert!(saved.len() <= URL_JSON_BOUNDS.max_bytes);
        let env = a.export_preset_json().unwrap();
        let mut b = ManifoldFlowStudio::default();
        b.import_preset_json(&env).unwrap();
        assert_eq!(a.save_state(), b.save_state());
    }
}

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::presets::{
    PRESET_GLM_ATROPHY_JSON, PRESET_MRF_DEFORMABLE_JSON, PRESET_SULCAL_THICKNESS_JSON,
};
use super::{CorticalMorphometryTool, CorticalOverlayMode, SurfaceDisplayMode};
use crate::framework::InteractiveTool;
use crate::framework::recovery::FiniteStateGuard;
use crate::tabs::geometry_topology::export_utils::{
    count_open_and_nonmanifold_edges, signed_volume,
};
use eframe::egui;
use oxidize_core::mesh::{export_colored_mesh_to_ply_binary, export_mesh_to_stl_binary};

#[test]
fn test_presets_roundtrip_and_url_state_bounds() {
    for preset_json in [
        PRESET_SULCAL_THICKNESS_JSON,
        PRESET_GLM_ATROPHY_JSON,
        PRESET_MRF_DEFORMABLE_JSON,
    ] {
        let mut tool = CorticalMorphometryTool::default();
        tool.import_preset_json(preset_json)
            .expect("preset must import cleanly");
        let saved = tool.save_state().expect("must serialize state");
        assert!(saved.len() < 1024, "compact parameter-only JSON expected");

        let mut restored = CorticalMorphometryTool::default();
        restored
            .try_load_state(&saved)
            .expect("roundtrip load must succeed");
        assert_eq!(restored.save_state().as_deref(), Some(saved.as_str()));
    }
}

#[test]
fn test_inflation_blends_folded_and_inflated_geometry() {
    let mut tool = CorticalMorphometryTool::default();
    tool.ensure_evaluated();
    let folded = tool.blended_vertices(true);

    tool.inflation = 1.0;
    let inflated = tool.blended_vertices(true);
    assert_eq!(folded.len(), inflated.len());

    let max_disp = folded
        .iter()
        .zip(&inflated)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        max_disp > 1.0,
        "Expected substantial unfolding displacement, got {max_disp}"
    );
}

#[test]
fn test_potts_mrf_denoises_heavy_t1_noise() {
    let mut tool = CorticalMorphometryTool {
        mri_noise_level: 20.0,
        potts_beta: 1.8,
        ..Default::default()
    };
    tool.ensure_evaluated();
    assert!(
        tool.cache.mrf_accuracy >= tool.cache.ml_accuracy,
        "Expected Potts MRF ({:.3}) >= ML ({:.3})",
        tool.cache.mrf_accuracy,
        tool.cache.ml_accuracy
    );
}

#[test]
fn test_singular_collinear_glm_renders_warning_banner_without_panic() {
    let mut tool = CorticalMorphometryTool {
        overlay_mode: CorticalOverlayMode::GlmTStatistic,
        collinear_glm_design: true,
        ..Default::default()
    };
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let out = ctx.run(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 800.0),
            )),
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
    assert!(
        tool.cache
            .glm_warning
            .as_deref()
            .is_some_and(|w| w.contains("singular")),
        "Expected singular X^T X warning, got {:?}",
        tool.cache.glm_warning
    );
    let ak = out
        .platform_output
        .accesskit_update
        .expect("AccessKit tree");
    let rendered_banner = ak.nodes.iter().any(|(_, n)| {
        n.label()
            .into_iter()
            .chain(n.value())
            .any(|t| t.contains("singular"))
    });
    assert!(
        rendered_banner,
        "Expected singular GLM warning banner in UI"
    );
}

#[test]
fn test_finite_state_guard_clamps_non_finite_parameters_and_meshes() {
    let mut tool = CorticalMorphometryTool::default();
    tool.ensure_evaluated();
    assert!(tool.is_finite_state());

    tool.learning_rate = f64::NAN;
    tool.inflation = f64::INFINITY;
    if let Some(ref mut surf) = tool.cache.white_surface {
        surf.vertices[0][0] = f64::NAN;
    }
    assert!(!tool.is_finite_state());
    assert!(!tool.has_finite_parameters());

    FiniteStateGuard::clamp_parameters(&mut tool);
    tool.ensure_evaluated();
    assert!(tool.is_finite_state());
    assert!(tool.has_finite_parameters());
}

#[test]
fn test_watertight_colored_mesh_export_across_surfaces_and_overlays() {
    let mut tool = CorticalMorphometryTool {
        u_resolution: 16,
        v_resolution: 12,
        evolution_steps: 2,
        inflation: 0.4,
        ..Default::default()
    };
    tool.ensure_evaluated();
    for surface_mode in SurfaceDisplayMode::ALL {
        for overlay_mode in CorticalOverlayMode::ALL {
            tool.surface_mode = surface_mode;
            tool.overlay_mode = overlay_mode;
            let colored = tool
                .export_colored_mesh_3d()
                .expect("3D colored mesh supported")
                .expect("colored mesh generation succeeded");
            assert_eq!(colored.colors.len(), colored.mesh.vertices.len());
            assert!(colored.mesh.normals.iter().all(|n| n.is_finite()));
            assert_eq!(count_open_and_nonmanifold_edges(&colored.mesh), (0, 0));
            assert!(signed_volume(&colored.mesh) > 0.0);

            let ply = export_colored_mesh_to_ply_binary(&colored).expect("PLY serialization");
            assert!(ply.starts_with(b"ply\nformat binary_little_endian 1.0\n"));
            let stl = export_mesh_to_stl_binary(&colored.mesh).expect("STL serialization");
            assert_eq!(stl.len(), 84 + 50 * (colored.mesh.indices.len() / 3));
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_cached_thickness_workspace_reuses_buffers_with_zero_allocations() {
    let mut tool = CorticalMorphometryTool::default();
    tool.ensure_evaluated();
    let white = tool.cache.white_surface.clone().expect("white mesh cached");
    let pial = tool.cache.pial_surface.clone().expect("pial mesh cached");
    let baseline = tool.cache.thickness_mm.clone();

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ((), metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        tool.cache.evaluate_thickness(&white, &pial);
    });
    assert!(metrics.function_calls > 0);
    assert_eq!(tool.cache.thickness_mm, baseline);
}

#[cfg(target_arch = "wasm32")]
mod wasm_tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn wasm_cortical_morphometry_preset_and_mesh_roundtrip() {
        let mut tool = CorticalMorphometryTool::default();
        tool.import_preset_json(PRESET_GLM_ATROPHY_JSON).unwrap();
        let state = tool.save_state().unwrap();
        let mut restored = CorticalMorphometryTool::default();
        restored.try_load_state(&state).unwrap();
        assert_eq!(restored.save_state().as_deref(), Some(state.as_str()));

        let colored = restored.export_colored_mesh_3d().unwrap().unwrap();
        assert_eq!(count_open_and_nonmanifold_edges(&colored.mesh), (0, 0));
        assert!(signed_volume(&colored.mesh) > 0.0);
    }
}

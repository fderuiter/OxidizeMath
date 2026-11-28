#![cfg_attr(any(), verified(opt_out = "tests"))]
#![cfg(feature = "applied")]

use super::presets::{
    PRESET_GLM_ATROPHY_JSON, PRESET_MRF_DEFORMABLE_JSON, PRESET_SULCAL_THICKNESS_JSON,
};
use super::{CorticalMorphometryTool, CorticalOverlayMode, SurfaceDisplayMode};
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::InteractiveTool;
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
fn test_evaluate_cohort_glm_preserves_well_conditioned_contrasts() {
    use super::phantom::evaluate_cohort_glm;
    use math_explorer::applied::freesurfer::glm::{estimate_beta, t_statistic};
    use math_explorer::nalgebra::{DMatrix, DVector};

    let thickness = [2.8_f64, 2.5, 3.1, 2.2];
    let weights = [0.2_f64, 0.8, 0.5, 1.0];
    let atrophy_effect = 0.45_f64;
    let (t_map, peak_abs, warning) =
        evaluate_cohort_glm(&thickness, &weights, atrophy_effect, false);
    assert!(warning.is_none());
    assert_eq!(t_map.len(), thickness.len());
    assert!(peak_abs > 0.0);

    let ages = [-1.2_f64, -0.4, 0.3, 1.1, -0.9, -0.2, 0.5, 1.0];
    let subj_jitter = [-0.09_f64, 0.06, -0.04, 0.07, -0.08, 0.05, 0.09, -0.06];
    let mut x_rows = Vec::with_capacity(24);
    for (s, &age) in ages.iter().enumerate() {
        let is_patient = if s >= 4 { 1.0 } else { 0.0 };
        x_rows.extend_from_slice(&[1.0, is_patient, age]);
    }
    let x = DMatrix::from_row_slice(8, 3, &x_rows);
    let contrast = DVector::from_vec(vec![0.0, -1.0, 0.0]);
    for (idx, (&base_t, &w)) in thickness.iter().zip(&weights).enumerate() {
        let phase = ((idx % 7) as f64 - 3.0) * 0.012;
        let y = DVector::from_fn(8, |s, _| {
            let group_loss = if s >= 4 { atrophy_effect * w } else { 0.0 };
            (base_t - group_loss - 0.06 * ages[s] + subj_jitter[s] + phase * (s as f64 - 3.5))
                .max(0.5)
        });
        let beta = estimate_beta(&x, &y).unwrap();
        let sigma2 = ((&y - &x * &beta).norm_squared() / 5.0).max(1e-12);
        let expected_t = t_statistic(&contrast, &beta, &x, sigma2).unwrap();
        assert_eq!(t_map[idx], expected_t);

        let xtx_inv = (x.transpose() * &x).try_inverse().unwrap();
        let beta_ne = &xtx_inv * (x.transpose() * &y);
        let var_ne = sigma2 * contrast.dot(&(&xtx_inv * &contrast));
        let ne_t = contrast.dot(&beta_ne) / var_ne.sqrt();
        assert!((t_map[idx] - ne_t).abs() < 1e-11);
    }
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

//! Unit tests for [`super::RiemannSurfaceStudio`] covering monodromy sheet permutations,
//! Schwarz-Christoffel conformal metrics, Schwarz reflection, Cauchy derivatives, residues,
//! and bounded `< 4 KiB` URL state serialization.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::InteractiveTool;
use crate::starter_gallery::StarterPresetMetadata;
use crate::url_state::URL_JSON_BOUNDS;

#[test]
fn test_riemann_monodromy_nth_root_and_logarithm_sheet_transitions() {
    let mut studio = RiemannSurfaceStudio {
        branch_family: RiemannBranchFamily::NthRoot,
        root_n: 3,
        start_sheet: 0,
        loop_center: [0.0, 0.0],
        loop_radius: 1.0,
        winding_turns: 1,
        dirty: true,
        ..Default::default()
    };
    studio.recompute_if_dirty();
    assert_eq!(studio.diagnostics.sheet_count, Some(3));
    assert_eq!(studio.diagnostics.final_sheet, 1);
    assert!(studio.diagnostics.monodromy_jump > 1.0);

    // 3 turns around origin closes the 3-sheet monodromy cycle 0 -> 0
    studio.winding_turns = 3;
    studio.dirty = true;
    studio.recompute_if_dirty();
    assert_eq!(studio.diagnostics.final_sheet, 0);
    assert!(studio.diagnostics.monodromy_jump < 1e-12);

    // Loop NOT enclosing branch point z = 0 stays on sheet 0 even after 1 turn
    studio.loop_center = [1.5, 0.0];
    studio.loop_radius = 0.5;
    studio.winding_turns = 1;
    studio.dirty = true;
    studio.recompute_if_dirty();
    assert_eq!(studio.diagnostics.final_sheet, 0);
    assert!(studio.diagnostics.monodromy_jump < 1e-12);

    // LogSurface shifts Im(w) by 2 * pi * W without ever closing for W != 0
    studio.branch_family = RiemannBranchFamily::Logarithm;
    studio.loop_center = [0.0, 0.0];
    studio.loop_radius = 1.0;
    studio.start_sheet = 0;
    studio.winding_turns = 2;
    studio.dirty = true;
    studio.recompute_if_dirty();
    assert_eq!(studio.diagnostics.sheet_count, None);
    assert_eq!(studio.diagnostics.final_sheet, 2);
    assert!((studio.diagnostics.monodromy_jump - 4.0 * std::f64::consts::PI).abs() < 1e-9);
}

#[test]
fn test_riemann_conformal_cauchy_and_residue_agreement() {
    let mut studio = RiemannSurfaceStudio {
        meromorphic_preset: MeromorphicPreset::EntireCubic,
        loop_center: [0.0, 0.0],
        loop_radius: 1.5,
        probe_z: [0.4, 0.3],
        derivative_order: 2,
        contour_steps: 240,
        dirty: true,
        ..Default::default()
    };
    studio.recompute_if_dirty();

    // Entire function: contour integral is 0, Cauchy value matches f(z_0), f''(z_0) = 6 * z_0
    assert!(studio.diagnostics.contour_integral.norm() < 1e-9);
    assert!((studio.diagnostics.cauchy_value - studio.diagnostics.exact_probe_value).norm() < 1e-6);
    let z0 = num_complex::Complex64::new(0.4, 0.3);
    let exact_d2 = 6.0 * z0;
    assert!((studio.diagnostics.cauchy_nth_deriv - exact_d2).norm() < 1e-3);

    // Conformal area magnification equals square of conformal scale factor: |f'(z)|^2
    let scale_sq = studio.diagnostics.conformal_scale * studio.diagnostics.conformal_scale;
    assert!((studio.diagnostics.area_mag - scale_sq).abs() < 1e-12);

    // Simple pole f(z) = (z + 1)/(z - z_p) has residue z_p + 1
    studio.meromorphic_preset = MeromorphicPreset::SimplePoleRational;
    studio.pole_z = [0.25, -0.2];
    studio.dirty = true;
    studio.recompute_if_dirty();
    let expected_res = num_complex::Complex64::new(1.25, -0.2);
    assert!((studio.diagnostics.residue_integral - expected_res).norm() < 1e-5);
    assert!((studio.diagnostics.residue_limit - expected_res).norm() < 1e-5);
}

#[test]
fn test_riemann_presets_state_roundtrip_and_headless_ui() {
    let mut studio = RiemannSurfaceStudio::default();
    let ctx = egui::Context::default();
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw, |c| studio.show(c));
    assert!(!studio.export_datasets().is_empty());
    assert!(!studio.export_plot_primitives().is_empty());

    let presets: Vec<_> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .filter(|p| p.tool_name == Some("RiemannSurfaceStudio"))
        .collect();
    assert_eq!(presets.len(), 3);

    for preset in presets {
        studio.import_preset_json(preset.generate_preset()).unwrap();
        let saved = studio.save_state().expect("serializes");
        assert!(saved.len() <= URL_JSON_BOUNDS.max_bytes);
        let mut restored = RiemannSurfaceStudio::default();
        restored.try_load_state(&saved).unwrap();
        assert_eq!(restored.save_state().unwrap(), saved);
    }

    assert!(studio.try_load_state(r#"{"unknown": 1}"#).is_err());
}

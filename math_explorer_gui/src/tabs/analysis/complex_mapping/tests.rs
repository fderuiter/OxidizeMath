#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_custom_complex_mapping_evaluation() {
    let mut tool = ComplexMappingTool {
        function: MappingFunction::Custom,
        ..Default::default()
    };
    tool.math_input.set_expression("z^3 - 1/z");

    let z0 = Complex64::new(2.0, 0.0);
    let w0 = tool.eval_z(z0);
    assert!((w0.re - 7.5).abs() < 1e-6);
    assert!(w0.im.abs() < 1e-6);

    let deriv = tool.deriv_z(z0);
    // f'(z) = 3z^2 + 1/z^2 = 3*(4) + 0.25 = 12.25
    assert!((deriv.re - 12.25).abs() < 1e-4);
    assert!(deriv.im.abs() < 1e-4);
}

#[test]
fn test_custom_complex_mapping_division_by_zero_splits_contours_without_fallback_spikes() {
    let mut tool = ComplexMappingTool {
        function: MappingFunction::Custom,
        ..Default::default()
    };
    tool.math_input.set_expression("z^3 - 1/z");

    let z_zero = Complex64::new(0.0, 0.0);
    let w_zero = tool.eval_z(z_zero);
    assert!(w_zero.re.is_nan() && w_zero.im.is_nan());

    // Sweep real axis [-2, 2] across z = 0: must split into >= 2 segments and emit 0 (1e6, 1e6) vertices
    let segments = ui::split_contour_segments(161, 4.0, |j| {
        let x = -2.0 + 4.0 * (j as f64) / 160.0;
        tool.eval_z(Complex64::new(x, 0.0))
    });
    assert!(
        segments.len() >= 2,
        "Expected contour across z = 0 pole to split into at least 2 segments, got {}",
        segments.len()
    );
    for seg in &segments {
        for pt in seg {
            assert!(pt[0].is_finite() && pt[1].is_finite());
            assert_ne!(*pt, [1e6, 1e6]);
        }
    }
}

#[test]
fn test_custom_expr_and_params_state_roundtrip_and_legacy_compat() {
    let mut tool = ComplexMappingTool {
        function: MappingFunction::Custom,
        ..Default::default()
    };
    tool.math_input.set_expression("z + a^2 / z");
    let mut params = BTreeMap::new();
    params.insert("a".to_string(), 0.85);
    tool.math_input.set_params_map(&params);

    let saved = tool.save_state().expect("serialize ComplexMappingTool");
    let mut restored = ComplexMappingTool::default();
    restored.try_load_state(&saved).expect("load saved state");
    assert_eq!(restored.math_input.raw_text, "z + a^2 / z");
    assert_eq!(restored.math_input.params_map().get("a"), Some(&0.85));
    let w = restored.eval_z(Complex64::new(1.0, 0.0));
    assert!((w.re - (1.0 + 0.85 * 0.85)).abs() < 1e-12);
    assert!(restored.import_preset_json(STARTER_PRESET_JSON).is_ok());
}

#[test]
fn test_exponential_and_sine_guards_and_bounds_validation() {
    let exp_tool = ComplexMappingTool {
        function: MappingFunction::Exponential,
        ..Default::default()
    };
    let w_exp = exp_tool.eval_z(Complex64::new(1_000.0, 1.0));
    let dw_exp = exp_tool.deriv_z(Complex64::new(1_000.0, 1.0));
    assert!(w_exp.re.is_finite() && w_exp.im.is_finite());
    assert!(dw_exp.re.is_finite() && dw_exp.im.is_finite());

    let sin_tool = ComplexMappingTool {
        function: MappingFunction::Sine,
        ..Default::default()
    };
    let w_sin = sin_tool.eval_z(Complex64::new(0.5, 1_000.0));
    let dw_sin = sin_tool.deriv_z(Complex64::new(0.5, -1_000.0));
    assert!(w_sin.re.is_finite() && w_sin.im.is_finite());
    assert!(dw_sin.re.is_finite() && dw_sin.im.is_finite());

    let mut check = ComplexMappingTool::default();
    assert!(check.has_finite_parameters());
    assert!(check
        .try_load_state(
            r#"{"function":"Exponential","grid_min":-60.0,"grid_max":2.0,"grid_density":10,"inspection_point":[1.0,1.0]}"#
        )
        .is_err());
    assert!(check
        .try_load_state(
            r#"{"function":"Sine","grid_min":-2.0,"grid_max":2.0,"grid_density":10,"inspection_point":[55.0,1.0]}"#
        )
        .is_err());
}

#[test]
fn test_complex_mapping_clamps_grid_density_and_caches_contours_across_probe_moves() {
    let mut tool = ComplexMappingTool::default();
    tool.try_load_state(
        r#"{"function":"ZSquared","grid_min":-2.0,"grid_max":2.0,"grid_density":1000,"inspection_point":[1.0,0.5]}"#,
    )
    .expect("grid_density 1000 should load and clamp");
    assert_eq!(tool.grid_density, crate::presets::MAX_INTERACTIVE_GRID_SIZE);
    assert!(tool.has_finite_parameters());

    let mut live = ComplexMappingTool::default();
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |c| live.show(c));
    let _ = ctx.run(egui::RawInput::default(), |c| live.show(c));
    assert_eq!(live.cache.eval_count, 1);

    live.inspection_point = [-0.75, 1.25];
    let _ = ctx.run(egui::RawInput::default(), |c| live.show(c));
    assert_eq!(live.cache.eval_count, 1);
}

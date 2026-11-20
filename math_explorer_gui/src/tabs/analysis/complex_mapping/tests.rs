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
fn test_custom_complex_mapping_division_by_zero_safe() {
    let mut tool = ComplexMappingTool {
        function: MappingFunction::Custom,
        ..Default::default()
    };
    tool.math_input.set_expression("1/z");

    let z_zero = Complex64::new(0.0, 0.0);
    let w_zero = tool.eval_z(z_zero);
    assert_eq!(w_zero, Complex64::new(1e6, 1e6));
}

#[test]
fn test_custom_expr_state_roundtrip_and_legacy_compat() {
    let mut tool = ComplexMappingTool {
        function: MappingFunction::Custom,
        ..Default::default()
    };
    tool.math_input.set_expression("sin(z) + z^2");
    let saved = tool.save_state().expect("serialize ComplexMappingTool");

    let mut restored = ComplexMappingTool::default();
    restored.try_load_state(&saved).expect("load saved state");
    assert_eq!(restored.math_input.raw_text, "sin(z) + z^2");
    assert!(restored.math_input.is_valid());
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

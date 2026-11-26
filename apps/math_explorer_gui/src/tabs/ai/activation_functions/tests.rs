#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_custom_activation_curves_and_finite_difference_derivative() {
    let mut tool = ActivationFunctionsTool {
        selected_function: ActivationFunction::Custom,
        x_min: -2.0,
        x_max: 2.0,
        points: 41,
        ..Default::default()
    };
    tool.math_input.set_expression("clamp(a * x, -1, 1)");
    let mut params = BTreeMap::new();
    params.insert("a".to_string(), 2.0);
    tool.math_input.set_params_map(&params);

    let (f_pts, df_pts) = tool.evaluate_curves();
    assert_eq!(f_pts.len(), 41);
    assert_eq!(df_pts.len(), 41);

    // At x = 0 (index 20), f(0) = 0 and f'(0) = 2.0
    assert!((f_pts[20][1] - 0.0).abs() < 1e-9);
    assert!((df_pts[20][1] - 2.0).abs() < 1e-4);

    // At x = 1.5 (saturated region), f(1.5) = 1.0 and f'(1.5) = 0.0
    let sat_idx = f_pts
        .iter()
        .position(|p| (p[0] - 1.5).abs() < 1e-9)
        .unwrap();
    assert!((f_pts[sat_idx][1] - 1.0).abs() < 1e-9);
    assert!((df_pts[sat_idx][1] - 0.0).abs() < 1e-4);

    assert!(tool.import_preset_json(STARTER_PRESET_JSON).is_ok());
    assert!(tool.import_preset_json(CUSTOM_HARDTANH_PRESET_JSON).is_ok());
}

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

#[test]
fn test_clarke_subgradients_zero_spike_at_thresholds_and_symbolic_latex() {
    let mut tool = ActivationFunctionsTool {
        selected_function: ActivationFunction::Custom,
        x_min: -1.0,
        x_max: 1.0,
        points: 5, // samples [-1.0, -0.5, 0.0, 0.5, 1.0]
        ..Default::default()
    };

    // 1. Step Threshold: step(x - a) at a = 0.0 -> f'(x) = 0.0 everywhere (0.0 spike error at x = 0.0, not 50_000)
    tool.math_input.set_expression("step(x - a)");
    tool.math_input.set_param("a", 0.0);
    let (_, step_df) = tool.evaluate_curves();
    for &[_, dy] in &step_df {
        assert_eq!(dy, 0.0);
    }
    let (step_latex, _) = tool.custom_symbolic_derivative().expect("step latex");
    assert_eq!(step_latex, "f'(x) = 0");

    // 2. Parameterized HardTanh: clamp(a * x, -1, 1) at a = 2.0 -> kinks at x = -0.5 and x = 0.5 have Clarke 0.5 * a = 1.0
    tool.math_input.set_expression("clamp(a * x, -1, 1)");
    tool.math_input.set_param("a", 2.0);
    let (_, clamp_df) = tool.evaluate_curves();
    assert_eq!(clamp_df[0][1], 0.0); // x = -1.0 (saturated)
    assert_eq!(clamp_df[1][1], 1.0); // x = -0.5 (Clarke threshold)
    assert_eq!(clamp_df[2][1], 2.0); // x = 0.0 (linear)
    assert_eq!(clamp_df[3][1], 1.0); // x = 0.5 (Clarke threshold)
    assert_eq!(clamp_df[4][1], 0.0); // x = 1.0 (saturated)

    // 3. Leaky ReLU: max(a * x, x) at a = 0.2 -> kink at x = 0.0 has Clarke 0.5 * (0.2 + 1.0) = 0.6
    tool.math_input.set_expression("max(a * x, x)");
    tool.math_input.set_param("a", 0.2);
    let (_, max_df) = tool.evaluate_curves();
    assert!((max_df[1][1] - 0.2).abs() < 1e-12); // x = -0.5
    assert!((max_df[2][1] - 0.6).abs() < 1e-12); // x = 0.0 (Clarke threshold)
    assert!((max_df[3][1] - 1.0).abs() < 1e-12); // x = 0.5
    assert!(tool.custom_symbolic_derivative().is_some());
}

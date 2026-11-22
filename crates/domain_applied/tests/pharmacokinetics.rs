#![allow(missing_docs)]
// math_explorer/tests/pharmacokinetics.rs

// Import the module to be tested
use domain_applied::applied::pharmacokinetics::*;

const TOLERANCE: f64 = math_commons::registry::TOLERANCE_FAST;

// Helper function to create parameters for d-amphetamine based on the prompt.
#[verified_engine::verified]
fn get_d_amphetamine_params() -> PKParameters {
    // From prompt: d-amphetamine t_1/2 = 10h -> ke = ln(2)/10
    let ke_d = std::f64::consts::LN_2 / 10.0; // approx 0.0693
    // From prompt: IR T_max = 3h. We solve for ka.
    let t_max_target = 3.0;
    // The user prompt suggests ka ~ 1.0, which is a good initial guess.
    let ka_d = solve_ka(t_max_target, ke_d, 1.0, 100, TOLERANCE).expect("Failed to solve for ka_d");

    PKParameters::new(
        1.0,  // f
        20.0, // d
        ka_d, // ka
        ke_d, // ke
        1.0,  // v
    )
    .unwrap()
}

// Helper function to create parameters for l-amphetamine based on the prompt.
#[verified_engine::verified]
fn get_l_amphetamine_params() -> PKParameters {
    // From prompt: l-amphetamine t_1/2 = 13h -> ke = ln(2)/13
    let ke_l = std::f64::consts::LN_2 / 13.0; // approx 0.0533
    // As per prompt, assume same ka as d-amphetamine.
    let d_params = get_d_amphetamine_params();

    PKParameters::new(
        d_params.f(),
        d_params.d(),
        d_params.ka(),
        ke_l,
        d_params.v(),
    )
    .unwrap()
}

#[test]
#[verified_engine::verified]
fn test_half_life() {
    let ke = std::f64::consts::LN_2 / 10.0;
    assert!((half_life(ke) - 10.0).abs() < TOLERANCE);
    assert_eq!(half_life(0.0), f64::INFINITY);
}

#[test]
#[verified_engine::verified]
fn test_t_max_and_solve_ka() {
    let ke = 0.0693;
    let t_max_target = 3.0;
    let initial_guess = 1.0;

    let solved_ka = solve_ka(t_max_target, ke, initial_guess, 100, TOLERANCE)
        .expect("Solver should find a solution for ka");

    // Use the solved ka to calculate t_max and assert it matches the target.
    let calculated_t_max = t_max(solved_ka, ke);
    assert!((calculated_t_max - t_max_target).abs() < TOLERANCE);
}

#[test]
#[verified_engine::verified]
fn test_bateman_function() {
    let params = get_d_amphetamine_params();

    // At t=0, concentration must be 0.
    assert!(concentration_bateman(&params, 0.0).abs() < TOLERANCE);

    // At t=T_max, concentration should be at its peak.
    let tmax = t_max(params.ka(), params.ke());
    let c_max = concentration_bateman(&params, tmax);
    let c_before = concentration_bateman(&params, tmax - 0.1);
    let c_after = concentration_bateman(&params, tmax + 0.1);
    assert!(c_max > c_before && c_max > c_after);

    // Test the special case where ka == ke.
    let params_ka_eq_ke = PKParameters::new(
        params.f(),
        params.d(),
        params.ke(), // ka = ke
        params.ke(),
        params.v(),
    )
    .unwrap();

    let t = 5.0;
    let expected_c = (params_ka_eq_ke.f() * params_ka_eq_ke.d() * params_ka_eq_ke.ka() * t
        / params_ka_eq_ke.v())
        * (-params_ka_eq_ke.ka() * t).exp();
    let c = concentration_bateman(&params_ka_eq_ke, t);
    assert!((c - expected_c).abs() < TOLERANCE);
}

#[test]
#[verified_engine::verified]
fn test_superposition() {
    let params = get_d_amphetamine_params();
    let dose_times = &[0.0, 4.0]; // Two doses, 4 hours apart.
    let t = 6.0;

    let c_total = concentration_superposition(&params, dose_times, t);

    // Expected value is C(t - 0.0) + C(t - 4.0).
    let c1 = concentration_bateman(&params, 6.0);
    let c2 = concentration_bateman(&params, 2.0);
    let expected_c = c1 + c2;

    assert!((c_total - expected_c).abs() < TOLERANCE);
}

#[test]
#[verified_engine::verified]
fn test_enantiomer_model_ir() {
    let d_params = get_d_amphetamine_params();
    let l_params = get_l_amphetamine_params();
    let f_d = 0.75;
    let f_l = 0.25;

    let model = EnantiomerModel::new(d_params, l_params, f_d, f_l).unwrap();

    let t = 5.0;
    let c_total = model.concentration_ir_single_dose(t);

    // Expected value is C_d(t) + C_l(t) with scaled doses.
    let c_d = concentration_bateman(&d_params.with_dose(d_params.d() * f_d).unwrap(), t);
    let c_l = concentration_bateman(&l_params.with_dose(l_params.d() * f_l).unwrap(), t);
    let expected_c = c_d + c_l;

    assert!((c_total - expected_c).abs() < TOLERANCE);
}

#[test]
#[verified_engine::verified]
fn test_enantiomer_model_xr() {
    let d_params = get_d_amphetamine_params();
    let l_params = get_l_amphetamine_params();
    let f_d = 0.75;
    let f_l = 0.25;

    let model = EnantiomerModel::new(d_params, l_params, f_d, f_l).unwrap();

    let lag_time = 4.0;
    let f1 = 0.5;
    let f2 = 0.5;
    let t = 8.0;

    let c_xr = model.concentration_xr_single_dose(lag_time, f1, f2, t);

    // Expected: 0.5 * C_IR(t=8) + 0.5 * C_IR(t=8-4=4).
    let c_ir_at_8 = model.concentration_ir_single_dose(8.0);
    let c_ir_at_4 = model.concentration_ir_single_dose(4.0);
    let expected_c = f1 * c_ir_at_8 + f2 * c_ir_at_4;

    assert!((c_xr - expected_c).abs() < TOLERANCE);

    // Test multiple XR doses.
    let dose_times = &[0.0, 24.0]; // One dose now, one the next day.
    let t_multi = 26.0;
    let c_xr_multi = model.concentration_xr_multiple_doses(dose_times, lag_time, f1, f2, t_multi);

    // Expected: C_XR(t=26) + C_XR(t=2).
    let c_xr1 = model.concentration_xr_single_dose(lag_time, f1, f2, 26.0);
    let c_xr2 = model.concentration_xr_single_dose(lag_time, f1, f2, 2.0);
    let expected_c_multi = c_xr1 + c_xr2;

    assert!((c_xr_multi - expected_c_multi).abs() < TOLERANCE);
}

#[test]
#[verified_engine::verified]
fn test_pk_parameter_validation() {
    // Test negative dose
    let res = PKParameters::new(1.0, -10.0, 1.0, 1.0, 1.0);
    assert!(res.is_err());
    assert!(res.unwrap_err().to_string().contains("Dose"));

    // Test negative ka
    let res = PKParameters::new(1.0, 10.0, -1.0, 1.0, 1.0);
    assert!(res.is_err());

    // Test negative volume
    let res = PKParameters::new(1.0, 10.0, 1.0, 1.0, 0.0);
    assert!(res.is_err());

    // Test valid
    let res = PKParameters::new(1.0, 10.0, 1.0, 1.0, 1.0);
    assert!(res.is_ok());
}

#[test]
#[verified_engine::verified]
fn test_builder() {
    let params = PKParametersBuilder::new()
        .bioavailability(0.8)
        .dose(50.0)
        .absorption_rate(1.5)
        .elimination_rate(0.2)
        .volume(10.0)
        .build()
        .unwrap();

    assert_eq!(params.f(), 0.8);
    assert_eq!(params.d(), 50.0);
}

#[test]
#[verified_engine::verified]
fn test_loading_dose_improves_time_in_therapeutic_range() {
    let params = PKParameters::new(1.0, 100.0, 1.2, 0.1, 20.0).unwrap();
    let window = TherapeuticWindow::new(3.0, 8.5).unwrap();
    let times: Vec<f64> = (0..=960).map(|i| i as f64 * 0.05).collect();

    let maintenance_events = [
        DoseEvent::ir(0.0, 100.0).unwrap(),
        DoseEvent::ir(12.0, 100.0).unwrap(),
        DoseEvent::ir(24.0, 100.0).unwrap(),
        DoseEvent::ir(36.0, 100.0).unwrap(),
    ];
    let loading_events = [
        DoseEvent::ir(0.0, 200.0).unwrap(),
        DoseEvent::ir(12.0, 100.0).unwrap(),
        DoseEvent::ir(24.0, 100.0).unwrap(),
        DoseEvent::ir(36.0, 100.0).unwrap(),
    ];

    let maint_regimen =
        RegimenSuperposition::new(BatemanModel::new(params), &maintenance_events).unwrap();
    let load_regimen =
        RegimenSuperposition::new(BatemanModel::new(params), &loading_events).unwrap();

    let mut maint_conc = vec![0.0; times.len()];
    let mut load_conc = vec![0.0; times.len()];
    maint_regimen.sample_into(&times, &mut maint_conc).unwrap();
    load_regimen.sample_into(&times, &mut load_conc).unwrap();

    let maint_summary = summarize_exposure(&times, &maint_conc, window, 12.0, params.ke()).unwrap();
    let load_summary = summarize_exposure(&times, &load_conc, window, 12.0, params.ke()).unwrap();

    assert!((maint_summary.time_in_range_pct - 66.493236).abs() < 1e-4);
    assert!((load_summary.time_in_range_pct - 86.888657).abs() < 1e-4);
    assert_eq!(maint_summary.toxic_pct, 0.0);
    assert_eq!(load_summary.toxic_pct, 0.0);
}

#[test]
#[verified_engine::verified]
fn test_missed_dose_catch_up_peak_and_toxic_overshoot() {
    let params = PKParameters::new(1.0, 100.0, 1.2, 0.1, 20.0).unwrap();
    let window = TherapeuticWindow::new(3.0, 8.5).unwrap();
    let base = BatemanModel::new(params);

    let catchup_events = [
        DoseEvent::ir(0.0, 100.0).unwrap(),
        DoseEvent::ir(18.0, 150.0).unwrap(),
        DoseEvent::ir(24.0, 100.0).unwrap(),
        DoseEvent::ir(36.0, 100.0).unwrap(),
    ];
    let double_events = [
        DoseEvent::ir(0.0, 100.0).unwrap(),
        DoseEvent::ir(18.0, 200.0).unwrap(),
        DoseEvent::ir(24.0, 100.0).unwrap(),
        DoseEvent::ir(36.0, 100.0).unwrap(),
    ];

    let catchup = RegimenSuperposition::new(base, &catchup_events).unwrap();
    let double_dose = RegimenSuperposition::new(base, &double_events).unwrap();
    let uniform = SuperpositionModel::new(base, vec![0.0, 18.0, 24.0, 36.0]);

    let peak_t = 18.0 + t_max(params.ka(), params.ke());
    let expected_catchup_peak = concentration_bateman(&params, peak_t)
        + concentration_bateman(&params.with_dose(150.0).unwrap(), peak_t - 18.0);
    let actual_catchup_peak = catchup.concentration(peak_t);
    let uniform_peak = uniform.concentration(peak_t);

    assert!((actual_catchup_peak - expected_catchup_peak).abs() < TOLERANCE);
    assert!((actual_catchup_peak - 6.702799).abs() < 1e-5);
    let rel_err = (actual_catchup_peak - uniform_peak) / actual_catchup_peak;
    assert!((rel_err - 0.297561).abs() < 1e-4);

    let times: Vec<f64> = (0..=960).map(|i| i as f64 * 0.05).collect();
    let mut catchup_c = vec![0.0; times.len()];
    let mut double_c = vec![0.0; times.len()];
    catchup.sample_into(&times, &mut catchup_c).unwrap();
    double_dose.sample_into(&times, &mut double_c).unwrap();

    let catchup_summary =
        summarize_exposure(&times, &catchup_c, window, 12.0, params.ke()).unwrap();
    let double_summary = summarize_exposure(&times, &double_c, window, 12.0, params.ke()).unwrap();

    assert_eq!(catchup_summary.toxic_pct, 0.0);
    assert!((double_summary.toxic_pct - 8.324662).abs() < 1e-4);
}

#[test]
#[verified_engine::verified]
fn test_enantiomer_and_two_pulse_regimens_and_calibration() {
    let (ka_d, ke_d) = calibrate_ka_from_half_life(3.0, 10.0, 100, TOLERANCE).unwrap();
    assert!((t_max(ka_d, ke_d) - 3.0).abs() < TOLERANCE);
    assert!((half_life(ke_d) - 10.0).abs() < TOLERANCE);

    let d_params = get_d_amphetamine_params();
    let l_params = get_l_amphetamine_params();
    let chiral = EnantiomerModel::new(d_params, l_params, 0.75, 0.25).unwrap();
    let events = [
        DoseEvent::xr(0.0, 30.0, 4.0, 0.5, 0.5).unwrap(),
        DoseEvent::ir(14.0, 10.0).unwrap(),
    ];
    let regimen = RegimenSuperposition::new(chiral, &events).unwrap();
    let times = [0.0, 3.0, 7.0, 15.0, 20.0];
    let mut d_buf = [0.0; 5];
    let mut l_buf = [0.0; 5];
    let mut tot_buf = [0.0; 5];
    regimen
        .sample_isomers_into(&times, &mut d_buf, &mut l_buf, &mut tot_buf)
        .unwrap();
    for i in 0..times.len() {
        assert!((tot_buf[i] - regimen.concentration(times[i])).abs() < TOLERANCE);
        assert!((d_buf[i] + l_buf[i] - tot_buf[i]).abs() < TOLERANCE);
    }

    let xr_base = TwoPulseModel::new(BatemanModel::new(d_params), 4.0, 0.5, 0.5);
    let xr_regimen =
        RegimenSuperposition::new(xr_base, &[DoseEvent::ir(0.0, 40.0).unwrap()]).unwrap();
    let c_expected = xr_base.with_dose(40.0).unwrap().concentration(6.0);
    assert!((xr_regimen.concentration(6.0) - c_expected).abs() < TOLERANCE);
}
